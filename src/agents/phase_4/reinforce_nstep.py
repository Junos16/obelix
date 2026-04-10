"""
REINFORCE with n-step returns for OBELIX — phase 4.

Extends reinforce.py by replacing full Monte Carlo episode returns with
truncated n-step returns.

Motivation:
  Full MC returns have high variance for OBELIX's 1000-step episodes: every
  random event over the entire trajectory contributes noise to the return
  estimate for the first action. n-step truncation at n=10 reduces variance
  at the cost of a slight bias (credit assignment only looks n steps ahead).

Mechanism:
  The agent collects n steps, computes the discounted sum G = Σ_{i=0}^{n-1}
  γ^i r_{t+i}, then immediately performs a policy gradient update. There is
  NO bootstrapping (unlike DDQN n-step): the return is purely from actual
  rewards, not G + γ^n V(s_{t+n}). This keeps the update on-policy.

Tradeoff vs reinforce.py:
  - More frequent updates (every n steps vs every episode): better sample
    utilisation and more gradient steps per unit time.
  - Slightly biased estimate: the return for an action doesn't account for
    rewards beyond step n. For n=10 and γ=0.99, γ^10 ≈ 0.90 so the bias
    is modest.

Architecture and reward shaping: identical to reinforce.py.
Config adds: n_steps (default 10).
"""
from __future__ import annotations
import os
import json
import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

def _extract_features(obs, ts_seen, ts_stuck, last_fw):
    f = np.zeros(26, dtype=np.float32)
    f[0:18] = obs.astype(np.float32)
    f[18] = float(np.sum(obs[4:12]))
    f[19] = float(np.sum(obs[0:4]))
    f[20] = float(np.sum(obs[12:16]))
    f[21] = float(obs[16])
    f[22] = float(obs[17])
    f[23] = min(1.0, ts_seen / 50.0)
    f[24] = min(1.0, ts_stuck / 20.0)
    f[25] = last_fw
    return f

def _shaped_reward(raw, obs):
    r = raw
    if obs[17] > 0:
        r += 195.0
    if float(np.sum(obs[:17])) == 0.0 and obs[17] == 0:
        r -= 2.0
    if obs[16] > 0:
        r += 3.0
    return r

def _priv_shaping(env, prev_dist):
    curr_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                          (env.bot_center_y - env.box_center_y)**2)
    r = 0.0
    if not env.enable_push:
        r += 2.0 * (prev_dist - curr_dist)
        dx = env.box_center_x - env.bot_center_x
        dy = env.box_center_y - env.bot_center_y
        if abs(dx) + abs(dy) > 1e-3:
            angle_to_box = math.degrees(math.atan2(dy, dx)) % 360
            diff = abs(angle_to_box - env.facing_angle % 360)
            if diff > 180: diff = 360 - diff
            if diff < 45: r += 0.5
    else:
        r += 1.0
    return r, curr_dist


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


# Inference globals
_MODEL: Optional[PolicyNet] = None
_ts_seen: int = 100
_ts_stuck: int = 100
_last_action: Optional[int] = None

_CURRENT_LEVEL: int = 1
_CURRENT_WALL: bool = False
_CURRENT_PREFIX: Optional[str] = None


def train(level, wall_obstacles, episodes, config_file=None, render=False,
          prefix=None, trial=None):
    global _MODEL, _ts_seen, _ts_stuck, _last_action
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _MODEL = None

    difficulty = {1: 0, 2: 2, 3: 3}[level]

    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2,
        "use_privileged": True,
        "n_steps": 10,
    }

    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            config.update(json.load(f))

    if trial is not None:
        p = get_optuna_params(trial, episodes)
        config.update(p)

    seed = config["seed"]
    n_steps = config["n_steps"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    rolling_returns = []
    best_rolling = -float("inf")

    print(f"Training REINFORCE n-step (n={n_steps}) level={level} wall={wall_obstacles} episodes={episodes}")

    for ep in range(episodes):
        ep_seed = seed + ep * 1009
        env = OBELIX(
            scaling_factor=config["scaling_factor"],
            arena_size=config["arena_size"],
            max_steps=config["max_steps"],
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=config["box_speed"],
            seed=ep_seed,
        )
        obs = env.reset(seed=ep_seed)
        t_ep = time.time()

        ts_seen = 100
        ts_stuck = 100
        last_fw = 0.0

        if config["use_privileged"]:
            prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                                  (env.bot_center_y - env.box_center_y)**2)
        else:
            prev_dist = 0.0

        log_probs = []
        rewards = []

        for _ in range(config["max_steps"]):
            ts_seen = 0 if np.any(obs[:17] > 0) else ts_seen + 1
            ts_stuck = 0 if obs[17] > 0 else ts_stuck + 1

            feat = _extract_features(obs, ts_seen, ts_stuck, last_fw)
            feat_t = torch.from_numpy(feat).unsqueeze(0)
            logits = net(feat_t)
            dist = Categorical(logits=logits)
            act = dist.sample()
            log_probs.append(dist.log_prob(act))

            obs, raw_reward, done = env.step(ACTIONS[act.item()], render=render)
            last_fw = 1.0 if act.item() == 2 else 0.0

            r = _shaped_reward(raw_reward, obs)
            if config["use_privileged"]:
                priv_r, prev_dist = _priv_shaping(env, prev_dist)
                r += priv_r

            rewards.append(r)
            if done:
                break

        T = len(rewards)
        # Truncated n-step returns (no bootstrapping)
        G = np.zeros(T, dtype=np.float32)
        for t in range(T):
            g = 0.0
            for k in range(min(n_steps, T - t)):
                g += (config["gamma"] ** k) * rewards[t + k]
            G[t] = g

        G_t = torch.tensor(G)
        G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)

        loss = -torch.stack(log_probs).squeeze() @ G_t
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        ep_return = float(sum(rewards))
        rolling_returns.append(ep_return)
        if len(rolling_returns) > 10:
            rolling_returns.pop(0)
        rolling_mean = float(np.mean(rolling_returns))

        if rolling_mean > best_rolling:
            best_rolling = rolling_mean

        print(f"Ep {ep+1}/{episodes} | return={ep_return:.1f} | rolling10={rolling_mean:.1f} | gamma={config['gamma']:.3f} | n_steps={n_steps} | {time.time()-t_ep:.1f}s")

        if trial is not None:
            trial.report(rolling_mean, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    base = prefix if prefix else f"reinforce_nstep_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base}_weights.pth"
    torch.save(net.state_dict(), out_path)
    print(f"Saved weights to {out_path}  (best rolling mean={best_rolling:.1f})")

    _MODEL = net


def _load_once():
    global _MODEL
    if _MODEL is None:
        base = (_CURRENT_PREFIX if _CURRENT_PREFIX
                else f"reinforce_nstep_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
        wpath = f"models/{base}_weights.pth"
        _MODEL = PolicyNet()
        _MODEL.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        _MODEL.eval()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _ts_seen, _ts_stuck, _last_action
    _load_once()

    _ts_seen = 0 if np.any(obs[:17] > 0) else _ts_seen + 1
    _ts_stuck = 0 if obs[17] > 0 else _ts_stuck + 1

    last_fw = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    feat = _extract_features(obs, _ts_seen, _ts_stuck, last_fw)
    feat_t = torch.from_numpy(feat).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(feat_t).squeeze()

    act = int(torch.argmax(logits).item())
    _last_action = act
    return ACTIONS[act]


def get_optuna_params(trial, total_episodes):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.97, 0.99, 0.995, 0.999]),
        "use_privileged": trial.suggest_categorical("use_privileged", [True, False]),
        "n_steps": trial.suggest_categorical("n_steps", [5, 10, 20]),
    }
