"""
Supervised Pretraining + REINFORCE policy gradient for OBELIX — phase 4.

Workflow:
  1. Run sp_encoder.py to produce a frozen EncoderNet (144→64→32→10).
  2. This file runs REINFORCE on a PolicyNet (13→64→32→5) whose input is
     the 13-dim belief-policy vector produced by the frozen encoder.

Policy input (13-dim):
  [10 encoder belief dims | ts_seen_norm | ts_stuck_norm | last_fw]
  The encoder output is computed at each step by running the frozen
  EncoderNet over the rolling obs_window deque (maxlen=8).

Why 64 hidden units despite 13-dim input:
  The belief state encodes relational geometric features (angle, distance,
  velocity). A wider hidden layer gives the policy room to compose these
  into action strategies that depend on multiple conditions simultaneously
  (e.g. facing box AND close AND push enabled → go forward).

Advantage over plain reinforce.py:
  At difficulty 2/3 the box blinks on/off. The raw 26-dim features encode
  time_since_seen as a scalar, giving only a coarse memory signal. The
  encoder's belief state is trained to explicitly recover box position from
  the history window, providing a much richer signal for the policy.

Saved weights: {"encoder": state_dict, "policy": state_dict}
Loaded by sp_reinforce_infer.py.
"""
from __future__ import annotations
import os
import json
import math
import random
import time
from collections import deque
from typing import Optional

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
_policy_net = None
_obs_window = None
_ts_seen = 100
_ts_stuck = 100
_last_fw = 0.0
_last_action: Optional[int] = None


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


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(POLICY_IN, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def train(level: int, wall_obstacles: bool, episodes: int,
          config_file=None, render: bool = False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _encoder, _policy_net

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix

    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    config = {
        "encoder_path": None,
        "lr": 3e-4,
        "gamma": 0.99,
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

    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    gamma = config["gamma"]

    rolling_returns = []
    print(f"REINFORCE SP: level={level}, wall={wall_obstacles}, episodes={episodes}")

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

        log_probs = []
        rewards = []
        ep_ret = 0.0

        prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                              (env.bot_center_y - env.box_center_y)**2)

        for _ in range(config["max_steps"]):
            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1
            if obs[17] > 0:
                ts_stuck = 0
            else:
                ts_stuck += 1

            enc_input = _make_encoder_input(list(obs_window))
            with torch.no_grad():
                belief = encoder(torch.tensor(enc_input).unsqueeze(0)).squeeze(0)
            policy_input = torch.tensor(
                _make_policy_input(belief.numpy(), ts_seen, ts_stuck, last_fw),
                dtype=torch.float32)

            logits = net(policy_input.unsqueeze(0)).squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            log_probs.append(dist.log_prob(action_t))

            act_idx = int(action_t.item())
            obs, raw, done = env.step(ACTIONS[act_idx], render=False)
            obs_window.append(obs.astype(np.float32))

            r_step = _shaped_reward(raw, obs)
            if config["use_privileged"]:
                pr, prev_dist = _priv_shaping(env, prev_dist)
                r_step += pr

            rewards.append(r_step)
            ep_ret += r_step
            last_fw = 1.0 if act_idx == 2 else 0.0

            if done:
                break

        # Compute returns
        G = 0.0
        returns = []
        for rr in reversed(rewards):
            G = rr + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -torch.stack(log_probs) @ returns_t / len(log_probs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()

        rolling_returns.append(ep_ret)
        if len(rolling_returns) > 10:
            rolling_returns.pop(0)
        rolling_mean = float(sum(rolling_returns) / len(rolling_returns))
        print(f"Episode {ep+1}/{episodes}  return={ep_ret:.1f}  rolling10={rolling_mean:.1f}  loss={loss.item():.4f}  ({time.time()-t_ep:.1f}s)")

        if trial is not None:
            trial.report(ep_ret, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    wall_tag = "_wall" if wall_obstacles else ""
    base = f"{prefix}" if prefix else f"sp_reinforce_level{level}{wall_tag}"
    out_path = f"models/{base}_weights.pth"
    torch.save({"encoder": encoder.state_dict(), "policy": net.state_dict()}, out_path)
    _encoder = encoder
    _policy_net = net
    print(f"Saved to {out_path}")


def _load_once():
    global _encoder, _policy_net, _obs_window
    if _encoder is not None and _policy_net is not None:
        return
    wall_tag = "_wall" if _CURRENT_WALL else ""
    base = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"sp_reinforce_level{_CURRENT_LEVEL}{wall_tag}"
    wpath = f"models/{base}_weights.pth"
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing weights at {wpath}")
    d = torch.load(wpath, map_location="cpu", weights_only=True)
    enc = EncoderNet()
    enc.load_state_dict(d["encoder"])
    enc.eval()
    _encoder = enc
    net = PolicyNet()
    net.load_state_dict(d["policy"])
    net.eval()
    _policy_net = net
    _obs_window = deque([np.zeros(OBS_DIM, dtype=np.float32)] * WINDOW, maxlen=WINDOW)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _obs_window, _ts_seen, _ts_stuck, _last_fw, _last_action
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
        belief = _encoder(torch.tensor(enc_input).unsqueeze(0)).squeeze(0)
    policy_input = torch.tensor(
        _make_policy_input(belief.numpy(), _ts_seen, _ts_stuck, _last_fw),
        dtype=torch.float32)
    with torch.no_grad():
        logits = _policy_net(policy_input.unsqueeze(0)).squeeze(0).numpy()

    best = int(np.argmax(logits))
    _last_fw = 1.0 if best == 2 else 0.0
    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params["gamma"] = trial.suggest_float("gamma", 0.9, 1.0)
    return params
