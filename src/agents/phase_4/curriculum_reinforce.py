"""
Curriculum REINFORCE for OBELIX — phase 4.

Applies the same 6-stage curriculum structure as curriculum_cma.py but
using REINFORCE policy gradient rather than evolutionary search.

Curriculum stages:
  (diff=0, no wall) → (diff=0, wall) → (diff=2, no wall) → (diff=2, wall)
  → (diff=3, no wall) → (diff=3, wall)
  Budget is split across stages by _split_budget (later stages get more).

Stage transition mechanics:
  When advancing to the next stage:
    1. PolicyNet weights are KEPT (they encode behaviours learned so far).
    2. Adam optimiser state is RESET to a fresh instance with reduced lr
       (lr × warmstart_lr_factor, default 0.5).
    3. Baseline EMA is reset to 0.

Why reset Adam but keep weights:
  Adam's first and second moment estimates (m, v) encode the gradient
  distribution of the previous stage. Carrying them to a new stage causes
  the per-parameter trust regions to be miscalibrated for the new task —
  parameters that were stable before may need rapid change now, and Adam's
  accumulated second moment would suppress their updates. A fresh Adam
  starts unbiased. The weights themselves encode the POLICY (approach
  behaviour, turning strategies) which should be preserved; only the
  optimiser's internal state needs resetting.

Why not curriculum_ddqn:
  The replay buffer accumulates transitions from the current difficulty
  stage. When advancing to a harder stage, the buffer remains full of
  easy-stage transitions that no longer represent the new observation
  distribution. Training a value function on this stale data produces
  Q-values that are valid for the old env but misleading for the new one.
  REINFORCE is on-policy: each episode generates fresh experience from the
  current stage, making it inherently curriculum-safe.

Architecture and reward shaping: identical to reinforce.py.
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


# Curriculum definitions
ALL_STAGES = [(0, False), (0, True), (2, False), (2, True), (3, False), (3, True)]

ADVANCE_THRESHOLDS = {
    (0, False): 500,
    (0, True):  300,
    (2, False): 200,
    (2, True):  150,
    (3, False): 100,
    (3, True):   50,
}

STAGE_WEIGHTS = {s: 1.0 + 0.5 * i for i, s in enumerate(ALL_STAGES)}


def _build_curriculum(level, wall_obstacles):
    max_stage = {1: 1, 2: 3, 3: 5}[level]
    stages = ALL_STAGES[:max_stage + 1]
    if not wall_obstacles:
        stages = [(d, w) for d, w in stages if not w]
    return stages


def _split_budget(total, stages):
    weights = [STAGE_WEIGHTS[s] for s in stages]
    total_w = sum(weights)
    budgets = [max(5, int(total * w / total_w)) for w in weights]
    budgets[-1] += total - sum(budgets)
    return budgets


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

    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2,
        "use_privileged": True,
        "warmstart_lr_factor": 0.5,
    }

    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            config.update(json.load(f))

    if trial is not None:
        p = get_optuna_params(trial, episodes)
        config.update(p)

    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    stages = _build_curriculum(level, wall_obstacles)
    budgets = _split_budget(episodes, stages)

    print(f"Curriculum REINFORCE stages: {stages}")
    print(f"Episode budgets: {budgets}  (total={sum(budgets)})")

    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    baseline_ema = 0.0

    os.makedirs("models", exist_ok=True)
    base = prefix if prefix else f"curriculum_reinforce_level{level}{'_wall' if wall_obstacles else ''}"

    global_ep = 0

    for stage_idx, (difficulty, wall) in enumerate(stages):
        max_eps = budgets[stage_idx]
        threshold = ADVANCE_THRESHOLDS[(difficulty, wall)]

        print(f"\n=== Stage {stage_idx}/{len(stages)-1}: diff={difficulty} wall={wall} budget={max_eps} eps ===")

        if stage_idx > 0:
            # Warm-start: keep weights, reset optimizer with reduced lr, reset baseline
            warm_lr = config["lr"] * config["warmstart_lr_factor"]
            optimizer = torch.optim.Adam(net.parameters(), lr=warm_lr)
            baseline_ema = 0.0

        rolling_returns = []

        for ep in range(max_eps):
            ep_seed = seed + global_ep * 1009
            env = OBELIX(
                scaling_factor=config["scaling_factor"],
                arena_size=config["arena_size"],
                max_steps=config["max_steps"],
                wall_obstacles=wall,
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
            G = np.zeros(T, dtype=np.float32)
            running = 0.0
            for t in reversed(range(T)):
                running = rewards[t] + config["gamma"] * running
                G[t] = running

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

            print(f"  Ep {ep+1:4d}/{max_eps}  return={ep_return:.1f}  rolling={rolling_mean:.1f}  ({time.time()-t_ep:.1f}s)")

            if trial is not None:
                trial.report(rolling_mean, global_ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            global_ep += 1

            # Advance early if rolling mean exceeds threshold after warmup
            if len(rolling_returns) == 10 and ep >= 9:
                if rolling_mean >= threshold:
                    print(f"  [advance] rolling mean {rolling_mean:.1f} >= {threshold}")
                    break

        # Stage checkpoint
        stage_path = f"models/{base}_stage{stage_idx}_weights.pth"
        torch.save(net.state_dict(), stage_path)
        print(f"  -> Stage {stage_idx} checkpoint: {stage_path}")

    out_path = f"models/{base}_weights.pth"
    torch.save(net.state_dict(), out_path)
    print(f"\nDone. Final weights -> {out_path}")

    _MODEL = net


def _load_once():
    global _MODEL
    if _MODEL is None:
        base = (_CURRENT_PREFIX if _CURRENT_PREFIX
                else f"curriculum_reinforce_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
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
        "warmstart_lr_factor": trial.suggest_float("warmstart_lr_factor", 0.1, 1.0),
    }
