from __future__ import annotations
import os
import random
import time
import json
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_MODEL = None
_PREV_OBS_EVAL = None

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOActorCritic(nn.Module):
    def __init__(self, in_dim=36, n_actions=5):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def train(level: int, wall_obstacles: bool, episodes: int, seed: int = None, trial=None, config_file: str = None, render: bool = False):
    global _CURRENT_LEVEL, _CURRENT_WALL, _MODEL
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _MODEL = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PPO on device: {device}")

    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    # Default PPO Hyperparameters
    config = {
        "gamma": 0.99,
        "lr": 3e-4,
        "gae_lambda": 0.95,
        "update_epochs": 4,
        "batch_size": 64,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_freq": 1000, # Steps to accumulate before update
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
    gae_lambda = config["gae_lambda"]
    update_epochs = config["update_epochs"]
    batch_size = config["batch_size"]
    clip_coef = config["clip_coef"]
    ent_coef = config["ent_coef"]
    vf_coef = config["vf_coef"]
    max_grad_norm = config["max_grad_norm"]
    update_freq = config["update_freq"]

    rng_seed = seed if seed is not None else 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    agent = PPOActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    buffer_s = []
    buffer_a = []
    buffer_logp = []
    buffer_v = []
    buffer_r = []
    buffer_d = []
    
    global_step = 0
    
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

        for step in range(config["max_steps"]):
            global_step += 1
            delta_obs = (obs_raw - prev_obs_raw).astype(np.float32)
            s = np.concatenate([obs_raw, delta_obs]).astype(np.float32)
            
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(s_t)
            
            a = action.item()
            next_obs_raw, r, done = env.step(ACTIONS[a], render=render)
            
            # Reward Shaping
            shaped_r = float(r)
            if np.sum(next_obs_raw[:17]) == 0: shaped_r -= 2.0
            if np.sum(next_obs_raw[:17]) > np.sum(obs_raw[:17]): shaped_r += 5.0
            if a == 2 and np.any(obs_raw[4:12] > 0): shaped_r += 0.5
            if a != 2 and np.any(obs_raw[1:16:2] > 0): shaped_r -= 0.5

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_logp.append(logprob.item())
            buffer_v.append(value.item())
            buffer_r.append(shaped_r)
            buffer_d.append(1.0 if done else 0.0)

            ep_ret += float(r)
            prev_obs_raw = obs_raw.copy()
            obs_raw = next_obs_raw

            if done or len(buffer_s) == update_freq:
                # Calculate GAE
                with torch.no_grad():
                    if done:
                        next_value = 0.0
                    else:
                        nxt_delta_obs = (next_obs_raw - prev_obs_raw).astype(np.float32)
                        nxt_s = np.concatenate([next_obs_raw, nxt_delta_obs]).astype(np.float32)
                        nxt_s_t = torch.tensor(nxt_s, dtype=torch.float32, device=device).unsqueeze(0)
                        next_value = agent.get_value(nxt_s_t).item()
                        
                b_s = torch.tensor(np.array(buffer_s), device=device)
                b_a = torch.tensor(np.array(buffer_a), device=device)
                b_logp = torch.tensor(np.array(buffer_logp), device=device)
                b_v = torch.tensor(np.array(buffer_v), device=device)
                b_r = torch.tensor(np.array(buffer_r), device=device)
                b_d = torch.tensor(np.array(buffer_d), device=device)

                advantages = torch.zeros_like(b_r).to(device)
                lastgaelam = 0
                for t in reversed(range(len(buffer_s))):
                    if t == len(buffer_s) - 1:
                        nextnonterminal = 1.0 - (1.0 if done else 0.0)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - b_d[t]
                        nextvalues = b_v[t + 1]
                    delta = b_r[t] + gamma * nextvalues * nextnonterminal - b_v[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                
                returns = advantages + b_v
                
                # PPO Update
                b_inds = np.arange(len(buffer_s))
                for epoch in range(update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, len(buffer_s), batch_size):
                        end = start + batch_size
                        mb_inds = b_inds[start:end]
                        
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_s[mb_inds], b_a[mb_inds])
                        logratio = newlogprob - b_logp[mb_inds]
                        ratio = logratio.exp()

                        mb_advantages = advantages[mb_inds]
                        # Advantage Normalization
                        if len(mb_advantages) > 1:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                        v_clipped = b_v[mb_inds] + torch.clamp(newvalue - b_v[mb_inds], -clip_coef, clip_coef)
                        v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                        optimizer.step()
                        
                buffer_s, buffer_a, buffer_logp, buffer_v, buffer_r, buffer_d = [], [], [], [], [], []
                
            if done:
                break

        if trial is not None:
            trial.report(ep_ret, ep)
            if trial.should_prune():
                raise importlib.import_module("optuna.exceptions").TrialPruned()

        duration = time.time() - t_start
        print(f"Episode {ep+1}/{episodes} return={ep_ret:.1f} ({duration:.2f}s)")

    os.makedirs("models", exist_ok=True)
    suffix = f"_trial{trial.number}" if trial else ""
    out_path = f"models/ppo_level{level}{'_wall' if wall_obstacles else ''}{suffix}_weights.pth"
    torch.save(agent.state_dict(), out_path)
    print(f"Training complete! Model saved to {out_path}")
    
    return policy

def _load_once():
    global _MODEL
    if _MODEL is not None: return

    wpath = f"models/ppo_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}_weights.pth"
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing weights file at {wpath}.")

    model = PPOActorCritic()
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
        logits = _MODEL.actor(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]

def get_optuna_params(trial, total_episodes):
    params = {}
    params["gamma"] = trial.suggest_float("gamma", 0.9, 0.999)
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 0.99)
    params["clip_coef"] = trial.suggest_float("clip_coef", 0.1, 0.4)
    params["ent_coef"] = trial.suggest_float("ent_coef", 0.0, 0.05)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    params["update_epochs"] = trial.suggest_categorical("update_epochs", [3, 4, 10])
    return params
