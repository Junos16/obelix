from __future__ import annotations
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

import json

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None):
    """
    Standardized train function called by src/main.py
    """
    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    # Default hyperparameters
    config = {
        "gamma": 0.99,
        "lr": 1e-3,
        "batch_size": 256,
        "replay_size": 100000,
        "warmup": 2000,
        "target_sync": 2000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_steps": 200000,
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
            print(f"Loaded hyperparameters from {config_file}")

    gamma = config["gamma"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    replay_size = config["replay_size"]
    warmup = config["warmup"]
    target_sync = config["target_sync"]
    eps_start = config["eps_start"]
    eps_end = config["eps_end"]
    eps_decay_steps = config["eps_decay_steps"]
    seed = config["seed"]
    max_steps = config["max_steps"]
    scaling_factor = config["scaling_factor"]
    arena_size = config["arena_size"]
    box_speed = config["box_speed"]
    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    # Hardcoded hyperparameters from the original script
    gamma = 0.99
    lr = 1e-3
    batch_size = 256
    replay_size = 100000
    warmup = 2000
    target_sync = 2000
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 200000
    seed = 42
    max_steps = 1000
    scaling_factor = 5
    arena_size = 500
    box_speed = 2

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q = DQN()
    tgt = DQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=lr)
    replay = Replay(replay_size)
    steps = 0

    def eps_by_step(t):
        if t >= eps_decay_steps:
            return eps_end
        frac = t / eps_decay_steps
        return eps_start + frac * (eps_end - eps_start)

    for ep in range(episodes):
        env = OBELIX(
            scaling_factor=scaling_factor,
            arena_size=arena_size,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=box_speed,
            seed=seed + ep,
        )
        s = env.reset(seed=seed + ep)
        ep_ret = 0.0

        for _ in range(max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
            s = s2
            steps += 1

            if len(replay) >= max(warmup, batch_size):
                sb, ab, rb, s2b, db = replay.sample(batch_size)
                sb_t = torch.tensor(sb)
                ab_t = torch.tensor(ab)
                rb_t = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t = torch.tensor(db)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")

    os.makedirs("models", exist_ok=True)
    out_path = f"models/ddqn_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
    torch.save(q.state_dict(), out_path)
    print(f"Training complete! Model saved to {out_path}")


_MODEL = None

def _load_once(level, wall_obstacles):
    global _MODEL
    if _MODEL is not None:
        return

    # Look for weights using the standard naming convention set by our train logic above
    # Note: For Codabench submissions, this logic needs to be simplified to just target 'weights.pth'
    wpath = f"models/ddqn_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"

    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing weights file at {wpath}. Train the agent first.")

    model = DQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model

def policy(obs: np.ndarray, rng: np.random.Generator, level: int=1, wall_obstacles: bool=False) -> str:
    """Standardized policy function called by evaluate.py / Codabench"""
    _load_once(level, wall_obstacles)

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]
