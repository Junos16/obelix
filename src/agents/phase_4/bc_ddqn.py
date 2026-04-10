"""
Behavioural Cloning warm-start + n-step DDQN for OBELIX — phase 4.

Two-phase training:
  Phase 1 — Behavioural Cloning (BC):
    QNet 26→128→64→5 is pre-trained as a classifier with cross-entropy loss
    on demonstration (obs, action) pairs. The resulting logits are not
    calibrated Q-values, but the relative ordering of action preferences is
    directionally correct (i.e. the argmax action matches the demonstrator).

  Phase 2 — n-step DDQN fine-tuning:
    RL trains from BC weights with n_steps=8 bootstrapped targets.

eps_start=0.3 (vs 1.0 for plain DDQN):
  Starting with full random exploration (ε=1.0) would override the BC
  policy for the first ~50% of training, wasting the pre-training. ε=0.3
  means 70% of actions follow the BC-initialised Q-values from step 1,
  allowing the good BC behaviours to accumulate replay buffer experience
  early and seed the Bellman backup with useful transitions.

n-step returns in the RL phase (n=8):
  BC's approach behaviour (robot near box) needs to quickly link contact
  events to the downstream push reward (+500). n-step propagates this
  connection ~8× faster than 1-step TD.

Architecture matches ddqn.py (26→128→64→5) so BC weights transfer
directly without reshaping.

Demo format: data/bc_demos.npz. Collect with: uv run src/record_demos.py
"""
from __future__ import annotations
import os
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

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
_QNET: Optional[nn.Module] = None

_time_since_seen: int = 100
_time_since_stuck: int = 100
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.05


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


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5),
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


class ReplayBuffer:
    def __init__(self, cap: int):
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

    def __len__(self):
        return len(self.buf)


class NStepBuffer:
    def __init__(self, n_steps: int, gamma: float):
        self.n = n_steps
        self.gamma = gamma
        self.buf = deque(maxlen=n_steps)

    def push(self, s, a, r, s_n, done):
        self.buf.append((s, a, r, s_n, done))
        if len(self.buf) < self.n:
            return None
        return self._make_transition()

    def _make_transition(self):
        s0, a0 = self.buf[0][0], self.buf[0][1]
        G = 0.0
        for i, (_, _, ri, _, di) in enumerate(self.buf):
            G += (self.gamma ** i) * ri
            if di:
                return s0, a0, G, self.buf[i][3], True
        s_n = self.buf[-1][3]
        done_n = self.buf[-1][4]
        return s0, a0, G, s_n, done_n

    def flush(self):
        results = []
        while len(self.buf) > 0:
            s0, a0 = self.buf[0][0], self.buf[0][1]
            G = 0.0
            for i, (_, _, ri, _, di) in enumerate(self.buf):
                G += (self.gamma ** i) * ri
                if di:
                    results.append((s0, a0, G, self.buf[i][3], True))
                    self.buf.popleft()
                    break
            else:
                s_n = self.buf[-1][3]
                done_n = self.buf[-1][4]
                results.append((s0, a0, G, s_n, done_n))
                self.buf.popleft()
        return results

    def clear(self):
        self.buf.clear()


def _bc_pretrain(qnet, config):
    demo_path = config["bc_demo_path"]
    if not os.path.exists(demo_path):
        print(f"Demo file not found: {demo_path}")
        print("Record demos first: uv run src/record_demos.py --output", demo_path)
        return False

    data = np.load(demo_path)
    raw_obs = data["observations"]
    actions = data["actions"]

    features = np.stack([_extract_features(o, 0, 0, 0.0) for o in raw_obs])
    feat_t = torch.tensor(features, dtype=torch.float32)
    act_t = torch.tensor(actions, dtype=torch.long)

    opt = optim.Adam(qnet.parameters(), lr=config["bc_lr"])
    bc_epochs = config["bc_epochs"]
    batch_size = 32
    N = len(feat_t)

    for epoch in range(bc_epochs):
        idx = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            b_idx = idx[start:start + batch_size]
            logits = qnet(feat_t[b_idx])
            loss = F.cross_entropy(logits, act_t[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  BC epoch {epoch+1}/{bc_epochs}  loss={total_loss/n_batches:.4f}")

    return True


def train(level, wall_obstacles, episodes, config_file=None, render=False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _QNET

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _QNET = None

    difficulty = {1: 0, 2: 2, 3: 3}[level]

    config = {
        "lr": 1e-4,
        "gamma": 0.99,
        "eps_start": 0.3,
        "eps_end": 0.05,
        "eps_decay_frac": 0.4,
        "batch_size": 64,
        "replay_size": 50000,
        "target_update": 1000,
        "warmup_steps": 500,
        "n_steps": 8,
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    qnet = QNet()
    tgt = QNet()

    # Phase 1: BC pre-training
    print(f"=== BC Pre-training (level={level}, wall={wall_obstacles}) ===")
    ok = _bc_pretrain(qnet, config)
    if not ok:
        return

    tgt.load_state_dict(qnet.state_dict())
    tgt.eval()

    # Phase 2: n-step DDQN fine-tuning
    print(f"=== n-step DDQN fine-tuning for {episodes} episodes ===")
    opt = optim.Adam(qnet.parameters(), lr=config["lr"])
    gamma = config["gamma"]
    n_steps = config["n_steps"]
    gamma_n = gamma ** n_steps
    batch_size = config["batch_size"]
    use_priv = config["use_privileged"]

    replay = ReplayBuffer(config["replay_size"])
    total_steps = 0
    total_ep_steps = episodes * config["max_steps"]
    decay_steps = max(1, int(total_ep_steps * config["eps_decay_frac"]))

    def get_eps(t):
        if t >= decay_steps:
            return config["eps_end"]
        frac = t / decay_steps
        return config["eps_start"] + frac * (config["eps_end"] - config["eps_start"])

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

        nstep_buf = NStepBuffer(n_steps, gamma)
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
            eps = get_eps(total_steps)
            if np.random.rand() < eps:
                a_idx = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    qs = qnet(torch.tensor(feat, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a_idx = int(np.argmax(qs))

            obs2, raw_r, done = env.step(ACTIONS[a_idx], render=render)
            last_fw = 1.0 if a_idx == 2 else 0.0
            ep_ret += raw_r

            r = _shaped_reward(raw_r, obs2)
            if use_priv:
                priv_r, prev_dist = _priv_shaping(env, prev_dist)
                r += priv_r

            feat2 = _extract_features(obs2, ts_seen, ts_stuck, last_fw)
            result = nstep_buf.push(feat, a_idx, r, feat2, done)
            if result is not None:
                replay.add(Transition(*result))

            total_steps += 1
            obs = obs2

            if len(replay) >= max(config["warmup_steps"], batch_size):
                sb, ab, rb, s2b, db = replay.sample(batch_size)
                sb_t = torch.tensor(sb)
                ab_t = torch.tensor(ab)
                rb_t = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t = torch.tensor(db)

                with torch.no_grad():
                    next_a = qnet(s2b_t).argmax(dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + gamma_n * (1.0 - db_t) * next_val

                pred = qnet(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
                opt.step()

                if total_steps % config["target_update"] == 0:
                    tgt.load_state_dict(qnet.state_dict())

            if done:
                break

        # flush remaining n-step transitions
        for result in nstep_buf.flush():
            replay.add(Transition(*result))

        if ep_ret > best_ret:
            best_ret = ep_ret

        print(f"Episode {ep+1}/{episodes}  ret={ep_ret:.1f}  best={best_ret:.1f}  eps={get_eps(total_steps):.3f}  ({time.time()-t_ep:.1f}s)")

        if trial is not None:
            trial.report(ep_ret, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    base_name = (f"{prefix}" if prefix
                 else f"bc_ddqn_level{level}{'_wall' if wall_obstacles else ''}")
    out_path = (f"models/{base_name}_trial_{trial.number}_weights.pth"
                if trial is not None
                else f"models/{base_name}_weights.pth")
    torch.save(qnet.state_dict(), out_path)
    print(f"Saved to {out_path}")
    _QNET = qnet


def _load_once():
    global _QNET
    if _QNET is not None:
        return
    base_name = (f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX
                 else f"bc_ddqn_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
    wpath = f"models/{base_name}_weights.pth"
    net = QNet()
    net.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    net.eval()
    _QNET = net


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _time_since_seen, _time_since_stuck, _last_action, _repeat_count
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

    with torch.no_grad():
        qs = _QNET(torch.tensor(feat, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    order = np.argsort(-qs)
    best = int(order[0])

    if _last_action is not None:
        best_q, second_q = float(qs[order[0]]), float(qs[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["bc_epochs"] = trial.suggest_categorical("bc_epochs", [20, 30, 50])
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["n_steps"] = trial.suggest_categorical("n_steps", [4, 8, 16])
    params["eps_start"] = trial.suggest_float("eps_start", 0.1, 0.5)
    return params
