"""
Supervised Pretraining + n-step Double DQN for OBELIX — phase 4.

Workflow:
  1. Run sp_encoder.py to produce a frozen EncoderNet (144→64→32→10).
  2. This file runs n-step DDQN on a QNet (13→64→32→5) whose input is
     the 13-dim belief-policy vector produced by the frozen encoder.

Replay buffer design:
  The replay buffer stores 13-dim processed belief-state features, NOT
  raw 18-dim observations. Since the encoder is frozen (stationary), the
  mapping obs_window → belief is deterministic and time-invariant. It is
  therefore correct to pre-encode before storage — sampling a batch then
  re-encoding on the fly would produce identical results but waste CPU
  by running the encoder forward pass for every batch sample every step.

n-step returns (n=8):
  Same motivation as ddqn_nstep.py: the push reward (+500) must propagate
  quickly back through long episodes. Encoding compresses the observation
  but doesn't change the credit-assignment problem; n-step is still needed.

Advantage over plain ddqn.py:
  The belief state explicitly encodes box position/velocity even when the
  box is invisible (difficulty 2/3), giving the Q-function access to latent
  state information that raw observations cannot provide.

Saved weights: {"encoder": state_dict, "policy": state_dict}
Loaded by sp_ddqn_infer.py.
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
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
OBS_DIM = 18
WINDOW = 8
BELIEF_DIM = 10
POLICY_IN = 13

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None
_encoder = None
_qnet = None
_obs_window = None
_ts_seen = 100
_ts_stuck = 100
_last_fw = 0.0
_last_action: Optional[int] = None
_repeat: int = 0
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.02


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


def _make_encoder_input(obs_window):
    return np.concatenate(obs_window, axis=0).astype(np.float32)


def _make_policy_input(belief, ts_seen, ts_stuck, last_fw):
    f = np.zeros(POLICY_IN, dtype=np.float32)
    f[:BELIEF_DIM] = belief
    f[BELIEF_DIM] = min(1.0, ts_seen / 50.0)
    f[BELIEF_DIM+1] = min(1.0, ts_stuck / 20.0)
    f[BELIEF_DIM+2] = last_fw
    return f


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW * OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, BELIEF_DIM),
        )

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(POLICY_IN, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_ACTIONS),
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
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf = deque()

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def ready(self):
        return len(self.buf) >= self.n

    def flush(self, replay: ReplayBuffer):
        while self.ready():
            s0, a0, _, _, _ = self.buf[0]
            G = 0.0
            final_s2 = None
            final_done = False
            for i, (_, _, ri, s2i, di) in enumerate(self.buf):
                G += (self.gamma ** i) * ri
                final_s2 = s2i
                final_done = di
                if di:
                    break
            replay.add(Transition(s=s0, a=a0, r=G, s2=final_s2, done=final_done))
            self.buf.popleft()
            if final_done:
                self.buf.clear()
                break

    def flush_all(self, replay: ReplayBuffer):
        while self.buf:
            s0, a0, _, _, _ = self.buf[0]
            G = 0.0
            final_s2 = None
            final_done = False
            for i, (_, _, ri, s2i, di) in enumerate(self.buf):
                G += (self.gamma ** i) * ri
                final_s2 = s2i
                final_done = di
                if di:
                    break
            replay.add(Transition(s=s0, a=a0, r=G, s2=final_s2, done=final_done))
            self.buf.popleft()
            if final_done:
                self.buf.clear()
                break


def train(level: int, wall_obstacles: bool, episodes: int,
          config_file=None, render: bool = False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _encoder, _qnet

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix

    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    config = {
        "encoder_path": None,
        "lr": 1e-4,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_frac": 0.5,
        "batch_size": 64,
        "replay_size": 50000,
        "target_update": 1000,
        "warmup_steps": 1000,
        "n_steps": 8,
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

    encoder_path = config.get("encoder_path")
    if encoder_path is None:
        if prefix:
            encoder_path = f"models/{prefix}_encoder.pth"
        else:
            wall_tag = "_wall" if wall_obstacles else ""
            encoder_path = f"models/sp_encoder_level{level}{wall_tag}.pth"

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}. Run sp_encoder first.")

    encoder = EncoderNet()
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Loaded encoder from {encoder_path}")

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    qnet = QNet()
    tgt = QNet()
    tgt.load_state_dict(qnet.state_dict())
    tgt.eval()

    optimizer = torch.optim.Adam(qnet.parameters(), lr=config["lr"])
    replay = ReplayBuffer(config["replay_size"])

    total_steps = episodes * config["max_steps"]
    eps_decay_steps = max(1, int(total_steps * config["eps_decay_frac"]))
    steps = 0
    rolling_returns = []

    def get_eps(t):
        if t >= eps_decay_steps:
            return config["eps_end"]
        return config["eps_start"] + (t / eps_decay_steps) * (config["eps_end"] - config["eps_start"])

    print(f"DDQN SP: level={level}, wall={wall_obstacles}, episodes={episodes}, n_steps={config['n_steps']}")

    for ep in range(episodes):
        ep_seed = seed + ep * 1337
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
        obs_window = deque([np.zeros(OBS_DIM, dtype=np.float32)] * WINDOW, maxlen=WINDOW)
        obs_window.append(obs.astype(np.float32))

        ts_seen = 100
        ts_stuck = 100
        last_fw = 0.0
        ep_ret = 0.0

        nstep = NStepBuffer(config["n_steps"], config["gamma"])

        prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                              (env.bot_center_y - env.box_center_y)**2)

        enc_input = _make_encoder_input(list(obs_window))
        with torch.no_grad():
            belief = encoder(torch.tensor(enc_input).unsqueeze(0)).squeeze(0).numpy()
        feat = _make_policy_input(belief, ts_seen, ts_stuck, last_fw)

        for _ in range(config["max_steps"]):
            eps = get_eps(steps)
            if np.random.rand() < eps:
                act_idx = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    qs = qnet(torch.tensor(feat).unsqueeze(0)).squeeze(0).numpy()
                act_idx = int(np.argmax(qs))

            obs, raw, done = env.step(ACTIONS[act_idx], render=False)
            obs_window.append(obs.astype(np.float32))

            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1
            if obs[17] > 0:
                ts_stuck = 0
            else:
                ts_stuck += 1

            r_step = _shaped_reward(raw, obs)
            if config["use_privileged"]:
                pr, prev_dist = _priv_shaping(env, prev_dist)
                r_step += pr

            ep_ret += r_step
            last_fw = 1.0 if act_idx == 2 else 0.0

            enc_input2 = _make_encoder_input(list(obs_window))
            with torch.no_grad():
                belief2 = encoder(torch.tensor(enc_input2).unsqueeze(0)).squeeze(0).numpy()
            feat2 = _make_policy_input(belief2, ts_seen, ts_stuck, last_fw)

            nstep.add(feat, act_idx, r_step, feat2, done)
            nstep.flush(replay)

            feat = feat2
            steps += 1

            if len(replay) >= max(config["warmup_steps"], config["batch_size"]):
                sb, ab, rb, s2b, db = replay.sample(config["batch_size"])
                sb_t = torch.tensor(sb)
                ab_t = torch.tensor(ab)
                rb_t = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t = torch.tensor(db)

                with torch.no_grad():
                    next_a = torch.argmax(qnet(s2b_t), dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + config["gamma"] * (1.0 - db_t) * next_val

                pred = qnet(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
                optimizer.step()

                if steps % config["target_update"] == 0:
                    tgt.load_state_dict(qnet.state_dict())

            if done:
                break

        nstep.flush_all(replay)

        rolling_returns.append(ep_ret)
        if len(rolling_returns) > 10:
            rolling_returns.pop(0)
        rolling_mean = float(sum(rolling_returns) / len(rolling_returns))
        print(f"Episode {ep+1}/{episodes}  return={ep_ret:.1f}  rolling10={rolling_mean:.1f}  "
              f"eps={get_eps(steps):.3f}  replay={len(replay)}  ({time.time()-t_ep:.1f}s)")

        if trial is not None:
            trial.report(ep_ret, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    wall_tag = "_wall" if wall_obstacles else ""
    base = f"{prefix}" if prefix else f"sp_ddqn_level{level}{wall_tag}"
    out_path = f"models/{base}_weights.pth"
    torch.save({"encoder": encoder.state_dict(), "policy": qnet.state_dict()}, out_path)
    _encoder = encoder
    _qnet = qnet
    print(f"Saved to {out_path}")


def _load_once():
    global _encoder, _qnet, _obs_window
    if _encoder is not None and _qnet is not None:
        return
    wall_tag = "_wall" if _CURRENT_WALL else ""
    base = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"sp_ddqn_level{_CURRENT_LEVEL}{wall_tag}"
    wpath = f"models/{base}_weights.pth"
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing weights at {wpath}")
    d = torch.load(wpath, map_location="cpu", weights_only=True)
    enc = EncoderNet()
    enc.load_state_dict(d["encoder"])
    enc.eval()
    _encoder = enc
    q = QNet()
    q.load_state_dict(d["policy"])
    q.eval()
    _qnet = q
    _obs_window = deque([np.zeros(OBS_DIM, dtype=np.float32)] * WINDOW, maxlen=WINDOW)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _obs_window, _ts_seen, _ts_stuck, _last_fw, _last_action, _repeat
    _load_once()

    if _obs_window is None:
        _obs_window = deque([np.zeros(OBS_DIM, dtype=np.float32)] * WINDOW, maxlen=WINDOW)

    _obs_window.append(obs.astype(np.float32))

    if np.any(obs[:17] > 0):
        _ts_seen = 0
    else:
        _ts_seen += 1
    if obs[17] > 0:
        _ts_stuck = 0
    else:
        _ts_stuck += 1

    enc_input = _make_encoder_input(list(_obs_window))
    with torch.no_grad():
        belief = _encoder(torch.tensor(enc_input).unsqueeze(0)).squeeze(0).numpy()
    policy_input = torch.tensor(
        _make_policy_input(belief, _ts_seen, _ts_stuck, _last_fw),
        dtype=torch.float32)
    with torch.no_grad():
        qs = _qnet(policy_input.unsqueeze(0)).squeeze(0).numpy()

    order = np.argsort(-qs)
    best = int(order[0])

    if _last_action is not None:
        best_q, second_q = float(qs[order[0]]), float(qs[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat < _MAX_REPEAT:
                best = _last_action
                _repeat += 1
            else:
                _repeat = 0
        else:
            _repeat = 0

    _last_fw = 1.0 if best == 2 else 0.0
    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["n_steps"] = trial.suggest_categorical("n_steps", [1, 4, 8, 16])
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    return params
