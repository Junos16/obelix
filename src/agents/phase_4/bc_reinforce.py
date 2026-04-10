"""
Behavioural Cloning warm-start + REINFORCE for OBELIX — phase 4.

Two-phase training:
  Phase 1 — Behavioural Cloning (BC):
    PolicyNet 26→64→64→5 is pre-trained with cross-entropy loss on human
    demonstration (obs, action) pairs. This gives a starting policy that
    avoids the circle-running attractor before any gradient-based RL begins.

  Phase 2 — REINFORCE fine-tuning:
    RL optimises against actual env reward starting from BC weights.

Rationale for the two-phase approach:
  BC alone is ceiling-limited by what the human demonstrates; it cannot
  improve beyond the demonstrator. REINFORCE can discover strategies better
  than the human (e.g. optimal approach angles) but struggles to escape the
  circle-running attractor from random init. BC provides the escape; RL
  provides the improvement.

RL learning rate (1e-4, lower than reinforce.py default 3e-4):
  Prevents gradient updates from quickly overwriting the BC initialisation.
  With a high lr the first few poor-reward episodes (which happen when the
  RL policy departs from BC) can degrade the weights faster than the agent
  can recover.

Architecture matches reinforce.py exactly (26→64→64→5) so BC weights
transfer directly without reshaping.

Demo format: data/bc_demos.npz. Collect with: uv run src/record_demos.py
"""
from __future__ import annotations
import os
import json
import math
from typing import Optional

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None
_NET: Optional[nn.Module] = None

_time_since_seen: int = 100
_time_since_stuck: int = 100
_last_action: Optional[int] = None


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
            nn.Linear(26, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


def _bc_pretrain(net, config):
    demo_path = config["bc_demo_path"]
    if not os.path.exists(demo_path):
        print(f"Demo file not found: {demo_path}")
        print("Record demos first: uv run src/record_demos.py --output", demo_path)
        return False

    data = np.load(demo_path)
    raw_obs = data["observations"]  # (N, 18)
    actions = data["actions"]       # (N,)

    features = np.stack([_extract_features(o, 0, 0, 0.0) for o in raw_obs])
    feat_t = torch.tensor(features, dtype=torch.float32)
    act_t = torch.tensor(actions, dtype=torch.long)

    opt = optim.Adam(net.parameters(), lr=config["bc_lr"])
    bc_epochs = config["bc_epochs"]
    batch_size = 32
    N = len(feat_t)

    for epoch in range(bc_epochs):
        idx = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            b_idx = idx[start:start + batch_size]
            logits = net(feat_t[b_idx])
            loss = F.cross_entropy(logits, act_t[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  BC epoch {epoch+1}/{bc_epochs}  loss={total_loss/n_batches:.4f}")

    return True


def train(level, wall_obstacles, episodes, config_file=None, render=False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _NET

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _NET = None

    difficulty = {1: 0, 2: 2, 3: 3}[level]

    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "bc_demo_path": "data/bc_demos.npz",
        "bc_epochs": 30,
        "bc_lr": 1e-3,
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2,
        "use_privileged": True,
    }

    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            config.update(json.load(f))

    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = PolicyNet()

    # Phase 1: BC pre-training
    print(f"=== BC Pre-training (level={level}, wall={wall_obstacles}) ===")
    ok = _bc_pretrain(net, config)
    if not ok:
        return

    # Phase 2: REINFORCE fine-tuning
    print(f"=== REINFORCE fine-tuning for {episodes} episodes ===")
    opt = optim.Adam(net.parameters(), lr=config["lr"])
    gamma = config["gamma"]
    use_priv = config["use_privileged"]

    best_ret = -float("inf")

    for ep in range(episodes):
        ep_seed = seed + ep
        t_ep = time.time()
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

        ts_seen = 100
        ts_stuck = 100
        last_fw = 0.0

        prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                              (env.bot_center_y - env.box_center_y)**2)

        log_probs = []
        rewards = []
        ep_ret = 0.0

        for _ in range(config["max_steps"]):
            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1
            if obs[17] > 0:
                ts_stuck = 0
            else:
                ts_stuck += 1

            feat = _extract_features(obs, ts_seen, ts_stuck, last_fw)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            logits = net(feat_t).squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            log_probs.append(dist.log_prob(act))

            a_idx = int(act.item())
            obs, raw_r, done = env.step(ACTIONS[a_idx], render=render)
            last_fw = 1.0 if a_idx == 2 else 0.0

            r = _shaped_reward(raw_r, obs)
            if use_priv:
                priv_r, prev_dist = _priv_shaping(env, prev_dist)
                r += priv_r

            rewards.append(r)
            ep_ret += raw_r

            if done:
                break

        # Compute discounted returns
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        ret_t = torch.tensor(returns, dtype=torch.float32)
        # Whiten
        if ret_t.std() > 1e-8:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        lp_t = torch.stack(log_probs)
        loss = -(lp_t * ret_t).sum()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if ep_ret > best_ret:
            best_ret = ep_ret

        print(f"Episode {ep+1}/{episodes}  ret={ep_ret:.1f}  best={best_ret:.1f}  ({time.time()-t_ep:.1f}s)")

        if trial is not None:
            trial.report(ep_ret, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    base_name = (f"{prefix}" if prefix
                 else f"bc_reinforce_level{level}{'_wall' if wall_obstacles else ''}")
    out_path = (f"models/{base_name}_trial_{trial.number}_weights.pth"
                if trial is not None
                else f"models/{base_name}_weights.pth")
    torch.save(net.state_dict(), out_path)
    print(f"Saved to {out_path}")
    _NET = net


def _load_once():
    global _NET
    if _NET is not None:
        return
    base_name = (f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX
                 else f"bc_reinforce_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
    wpath = f"models/{base_name}_weights.pth"
    net = PolicyNet()
    net.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    net.eval()
    _NET = net


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _time_since_seen, _time_since_stuck, _last_action
    _load_once()

    if np.any(obs[:17] > 0):
        _time_since_seen = 0
    else:
        _time_since_seen += 1
    if obs[17] > 0:
        _time_since_stuck = 0
    else:
        _time_since_stuck += 1

    last_fw = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    feat = _extract_features(obs, _time_since_seen, _time_since_stuck, last_fw)
    feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
    logits = _NET(feat_t).squeeze(0).numpy()
    best = int(np.argmax(logits))
    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["bc_epochs"] = trial.suggest_categorical("bc_epochs", [20, 30, 50])
    params["lr"] = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    params["bc_lr"] = trial.suggest_float("bc_lr", 1e-4, 1e-2, log=True)
    return params
