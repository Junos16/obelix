from __future__ import annotations
import os
import random
import time
import json
import importlib
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_MODEL = None
_PREV_OBS_EVAL = None

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=36, n_actions=5):
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class Transition:
    def __init__(self, s, a, r, s2, done):
        self.s = s
        self.a = a
        self.r = r
        self.s2 = s2
        self.done = done

class PrioritizedReplayBuffer:
    """A simple Prioritized Experience Replay (PER) using pure NumPy for fast alpha/beta sampling."""
    def __init__(self, capacity=100_000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        
    def add(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        items = [self.buffer[i] for i in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        s = np.stack([it.s for it in items])
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items])
        d = np.array([it.done for it in items], dtype=np.float32)
        
        return s, a, r, s2, d, indices, np.array(weights, dtype=np.float32)
        
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)
            
    def __len__(self):
        return len(self.buffer)

def train(level: int, wall_obstacles: bool, episodes: int, seed: int = None, trial=None, config_file: str = None, render: bool = False):
    """
    Train D3QN agent. Supports parallel sweeping if trial is provided.
    Runs on CUDA if available, but weights load to CPU in policy().
    """
    global _CURRENT_LEVEL, _CURRENT_WALL, _MODEL
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _MODEL = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training D3QN on device: {device}")

    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    # Default hyperparameters
    config = {
        "gamma": 0.99,
        "lr": 5e-4,
        "batch_size": 128,
        "replay_size": 100000,
        "warmup": 2000,
        "target_sync": 1000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_steps": 150000,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)

    gamma = config["gamma"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    replay_size = config["replay_size"]
    warmup = config["warmup"]
    target_sync = config["target_sync"]
    eps_start = config["eps_start"]
    eps_end = config["eps_end"]
    eps_decay_steps = max(1, config["eps_decay_steps"])
    per_alpha = config["per_alpha"]
    per_beta_start = config["per_beta"]
    
    rng_seed = seed if seed is not None else 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    q = DuelingDQN().to(device)
    tgt = DuelingDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=lr)
    replay = PrioritizedReplayBuffer(replay_size, alpha=per_alpha)
    steps = 0

    def eps_by_step(t):
        if t >= eps_decay_steps: return eps_end
        return eps_start + (t / eps_decay_steps) * (eps_end - eps_start)
        
    def beta_by_step(t):
        if t >= eps_decay_steps: return 1.0
        return per_beta_start + (t / eps_decay_steps) * (1.0 - per_beta_start)

    for ep in range(episodes):
        env = OBELIX(
            scaling_factor=config["scaling_factor"],
            arena_size=config["arena_size"],
            max_steps=config["max_steps"],
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=config["box_speed"],
            seed=rng_seed + ep
        )
        obs_raw = env.reset(seed=rng_seed + ep)
        prev_obs_raw = obs_raw.copy()
        ep_ret = 0.0
        t_start = time.time()

        for _ in range(config["max_steps"]):
            # State representation: concat [obs, delta_obs]
            delta_obs = (obs_raw - prev_obs_raw).astype(np.float32)
            s = np.concatenate([obs_raw, delta_obs]).astype(np.float32)
            
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    qs = q(s_t).squeeze(0).cpu().numpy()
                a = int(np.argmax(qs))

            next_obs_raw, r, done = env.step(ACTIONS[a], render=render)
            nxt_delta_obs = (next_obs_raw - obs_raw).astype(np.float32)
            s2 = np.concatenate([next_obs_raw, nxt_delta_obs]).astype(np.float32)
            
            # Reward Shaping
            shaped_r = float(r)
            if np.sum(next_obs_raw[:17]) == 0:
                shaped_r -= 2.0  # Wandering penalty
            if np.sum(next_obs_raw[:17]) > np.sum(obs_raw[:17]):
                shaped_r += 5.0  # Intensity bonus
            if a == 2 and np.any(obs_raw[4:12] > 0):
                shaped_r += 0.5  # Forward momentum
            if a != 2 and np.any(obs_raw[1:16:2] > 0):
                shaped_r -= 0.5  # Anti-rotation
            
            ep_ret += float(r) # Track true unshaped reward
            replay.add(Transition(s=s, a=a, r=shaped_r, s2=s2, done=bool(done)))
            
            prev_obs_raw = obs_raw.copy()
            obs_raw = next_obs_raw
            steps += 1

            if len(replay) >= max(warmup, batch_size):
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, indices, weights = replay.sample(batch_size, beta=beta)
                
                sb_t = torch.tensor(sb, device=device)
                ab_t = torch.tensor(ab, device=device)
                rb_t = torch.tensor(rb, device=device)
                s2b_t = torch.tensor(s2b, device=device)
                db_t = torch.tensor(db, device=device)
                weights_t = torch.tensor(weights, device=device)

                # Double Q-Learning update
                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                
                # PER loss
                td_errors = torch.abs(pred - y)
                loss = (weights_t * nn.functional.smooth_l1_loss(pred, y, reduction='none')).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                
                # Update priorities
                replay.update_priorities(indices, td_errors.detach().cpu().numpy() + 1e-5)

                if steps % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        # Report to Optuna trial if available
        if trial is not None:
            trial.report(ep_ret, ep)
            if trial.should_prune():
                raise importlib.import_module("optuna.exceptions").TrialPruned()

        duration = time.time() - t_start
        print(f"Episode {ep+1}/{episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} ({duration:.2f}s)")

    os.makedirs("models", exist_ok=True)
    suffix = f"_trial{trial.number}" if trial else ""
    out_path = f"models/d3qn_level{level}{'_wall' if wall_obstacles else ''}{suffix}_weights.pth"
    torch.save(q.state_dict(), out_path)
    print(f"Training complete! Model saved to {out_path}")
    
    return policy

def _load_once():
    global _MODEL
    if _MODEL is not None: return

    # In sweep/eval we look for the generic (or trial-specific) name
    wpath = f"models/d3qn_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}_weights.pth"
    if not os.path.exists(wpath):
        # Graceful fallback to search for highest trial or just an error
        raise FileNotFoundError(f"Missing weights file at {wpath}.")

    model = DuelingDQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model

def policy(obs: np.ndarray, rng: np.random.Generator = None) -> str:
    global _PREV_OBS_EVAL
    _load_once()
    
    if _PREV_OBS_EVAL is None or np.sum(np.abs(obs - _PREV_OBS_EVAL)) > 5:
        delta_obs = np.zeros_like(obs, dtype=np.float32)
    else:
        delta_obs = (obs - _PREV_OBS_EVAL).astype(np.float32)
    
    _PREV_OBS_EVAL = obs.copy()
    
    s = np.concatenate([obs, delta_obs]).astype(np.float32)
    x = torch.from_numpy(s).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]

def get_optuna_params(trial, total_episodes):
    params = {}
    params["gamma"] = trial.suggest_float("gamma", 0.9, 0.999)
    params["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
    params["target_sync"] = trial.suggest_categorical("target_sync", [500, 1000, 2000])
    return params
