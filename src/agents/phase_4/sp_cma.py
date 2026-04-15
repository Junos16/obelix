"""
Supervised Pretraining + CMA-ES policy optimisation for OBELIX — phase 4.

Workflow:
  1. Run sp_encoder.py to produce a frozen EncoderNet (144→64→32→10).
  2. This file runs CMA-ES on a SMALL policy net (13→32→16→5, 1061 params)
     whose input is the 13-dim belief-policy vector:
       [10 encoder outputs | ts_seen_norm | ts_stuck_norm | last_fw]

Why smaller policy vs base cma_es (1477 params):
  The encoder compresses the 144-bit observation window to 10 semantically
  rich dimensions (distance, angle, velocity, push status). The policy only
  needs to map this compact belief to 5 actions — a much simpler function
  than mapping raw 26-dim features. A 1061-param policy:
    - Runs ~4× fewer eigendecompositions of the covariance matrix per
      generation (matrix size 1061 vs 1477).
    - Requires fewer CMA-ES samples to cover its search space well.

Policy input (13-dim):
  Belief state (10) + temporal memory (3): the encoder already encodes
  time_since_seen implicitly in the belief, but the extra scalars provide
  the policy with normalised, directly legible signals without requiring
  the encoder to represent them precisely.

Saved weights format: {"encoder": encoder_state_dict, "policy": flat_tensor}
Both components are loaded by sp_cma_infer.py at inference time.
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

SP_INPUT = 13
SP_H1 = 32
SP_H2 = 16
SP_OUTPUT = 5
N_PARAMS = SP_INPUT*SP_H1+SP_H1 + SP_H1*SP_H2+SP_H2 + SP_H2*SP_OUTPUT+SP_OUTPUT  # 1061

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None
_encoder = None
_pol_model: Optional[nn.Module] = None
_obs_window = None
_ts_seen = 100
_ts_stuck = 100
_last_fw = 0.0
_last_action: Optional[int] = None
_repeat_count = 0
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


def _get_belief_target(env, ts_seen):
    arena_diag = math.sqrt(env.arena_size**2 + env.arena_size**2)
    dx = env.box_center_x - env.bot_center_x
    dy = env.box_center_y - env.bot_center_y
    dist = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    corner_dists = [
        math.sqrt(env.box_center_x**2 + env.box_center_y**2),
        math.sqrt((env.arena_size - env.box_center_x)**2 + env.box_center_y**2),
        math.sqrt(env.box_center_x**2 + (env.arena_size - env.box_center_y)**2),
        math.sqrt((env.arena_size - env.box_center_x)**2 + (env.arena_size - env.box_center_y)**2),
    ]
    target = np.zeros(BELIEF_DIM, dtype=np.float32)
    target[0] = min(1.0, dist / arena_diag)
    target[1] = math.sin(angle)
    target[2] = math.cos(angle)
    target[3] = float(env.box_visible)
    target[4] = env._box_vx / max(env.box_speed, 1)
    target[5] = env._box_vy / max(env.box_speed, 1)
    target[6] = float(env.enable_push)
    target[7] = float(env.stuck_flag)
    target[8] = min(1.0, ts_seen / 50.0)
    target[9] = min(1.0, min(corner_dists) / arena_diag)
    return target


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
            nn.Linear(SP_INPUT, SP_H1),
            nn.Tanh(),
            nn.Linear(SP_H1, SP_H2),
            nn.Tanh(),
            nn.Linear(SP_H2, SP_OUTPUT),
        )

    def forward(self, x):
        return self.net(x)


def _flat_to_policy(flat: np.ndarray) -> PolicyNet:
    model = PolicyNet()
    idx = 0
    for layer in [model.net[0], model.net[2], model.net[4]]:
        n_w = layer.in_features * layer.out_features
        W = flat[idx:idx + n_w].reshape(layer.in_features, layer.out_features)
        layer.weight.data = torch.from_numpy(W.T.copy()).float()
        idx += n_w
        layer.bias.data = torch.from_numpy(flat[idx:idx + layer.out_features].copy()).float()
        idx += layer.out_features
    model.eval()
    return model


class CMAES:
    def __init__(self, n: int, sigma0: float = 0.5, pop_size: int = 48, seed: int = 42):
        self.n = n
        self.lam = pop_size
        self.mu = pop_size // 2
        self.rng = np.random.default_rng(seed)

        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.sigma = sigma0
        self.cs = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.ds = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.cc = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) /
                       ((n + 2.0) ** 2 + self.mu_eff))

        self.mean = np.zeros(n, dtype=np.float64)
        self.ps = np.zeros(n, dtype=np.float64)
        self.pc = np.zeros(n, dtype=np.float64)

        self.C = np.eye(n, dtype=np.float64)
        self.B = np.eye(n, dtype=np.float64)
        self.D = np.ones(n, dtype=np.float64)
        self.invsqrtC = np.eye(n, dtype=np.float64)
        self._eigen_update_gap = max(1, int(n / (10 * self.lam)))
        self._gen_since_eigen = 0

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        samples = self.mean[None, :] + self.sigma * (z @ np.diag(self.D) @ self.B.T)
        return samples

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray):
        n = self.n
        order = np.argsort(-fitnesses)
        solutions = solutions[order]

        old_mean = self.mean.copy()
        self.mean = self.weights @ solutions[:self.mu]

        mean_shift = (self.mean - old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * (self.invsqrtC @ mean_shift)

        h_sig = 1.0 if (np.linalg.norm(self.ps) /
                        math.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self._gen_since_eigen + 1))) <
                        (1.4 + 2.0 / (n + 1.0)) * self.chi_n) else 0.0

        self.pc = (1.0 - self.cc) * self.pc + \
                  h_sig * math.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * mean_shift

        artmp = (solutions[:self.mu] - old_mean[None, :]) / self.sigma
        self.C = ((1.0 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1.0 - h_sig) * self.cc * (2.0 - self.cc) * self.C) +
                  self.cmu * (self.weights[:, None] * artmp).T @ artmp)

        self.sigma *= math.exp((self.cs / self.ds) *
                               (np.linalg.norm(self.ps) / self.chi_n - 1.0))

        self._gen_since_eigen += 1
        if self._gen_since_eigen >= self._eigen_update_gap:
            self._update_eigen()
            self._gen_since_eigen = 0

    def _update_eigen(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T


def _evaluate_genome(flat_weights, encoder, difficulty, wall_obstacles, config, n_rollouts=3, use_privileged=True):
    pol_model = _flat_to_policy(flat_weights)
    seed = config["seed"]
    total_fitness = 0.0

    for r in range(n_rollouts):
        ep_seed = seed + r * 7919
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
        episode_reward = 0.0
        attached = False
        pushed = False

        prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                              (env.bot_center_y - env.box_center_y)**2)

        for step in range(config["max_steps"]):
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
                belief = encoder(torch.tensor(enc_input).unsqueeze(0)).squeeze(0).numpy()
            policy_input = _make_policy_input(belief, ts_seen, ts_stuck, last_fw)
            with torch.no_grad():
                logits = pol_model(torch.from_numpy(policy_input).unsqueeze(0)).squeeze(0).numpy()
            act_idx = int(np.argmax(logits))

            obs, raw, done = env.step(ACTIONS[act_idx], render=False)
            obs_window.append(obs.astype(np.float32))

            r_step = _shaped_reward(raw, obs)
            if use_privileged:
                pr, prev_dist = _priv_shaping(env, prev_dist)
                r_step += pr

            episode_reward += r_step
            last_fw = 1.0 if act_idx == 2 else 0.0

            if env.enable_push and not attached:
                attached = True
                episode_reward += 50.0
            if done and env.enable_push:
                pushed = True

            if done:
                break

        if pushed:
            episode_reward += 500.0

        total_fitness += episode_reward

    return total_fitness / n_rollouts


def train(level: int, wall_obstacles: bool, episodes: int,
          config_file=None, render: bool = False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _encoder, _pol_model

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix

    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    config = {
        "encoder_path": None,
        "sigma0": 0.5,
        "pop_size": 48,
        "n_rollouts": 3,
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
    print(f"Loaded encoder from {encoder_path}")

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    cma = CMAES(n=N_PARAMS, sigma0=config["sigma0"], pop_size=config["pop_size"], seed=seed)

    best_fitness = -float("inf")
    best_genome = cma.mean.copy()

    print(f"CMA-ES SP: level={level}, wall={wall_obstacles}, generations={episodes}, pop={config['pop_size']}")

    for gen in range(episodes):
        t_start = time.time()
        solutions = cma.ask()

        fitnesses = np.zeros(cma.lam)
        for i in range(cma.lam):
            eval_config = dict(config)
            eval_config["seed"] = seed + gen * cma.lam + i
            fitnesses[i] = _evaluate_genome(
                solutions[i].astype(np.float32),
                encoder=encoder,
                difficulty=difficulty,
                wall_obstacles=wall_obstacles,
                config=eval_config,
                n_rollouts=config["n_rollouts"],
                use_privileged=config["use_privileged"],
            )

        cma.tell(solutions, fitnesses)

        gen_best = float(np.max(fitnesses))
        gen_mean = float(np.mean(fitnesses))
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = solutions[np.argmax(fitnesses)].copy()

        duration = time.time() - t_start
        print(f"Gen {gen+1}/{episodes}  best={gen_best:.1f}  mean={gen_mean:.1f}  "
              f"all-time-best={best_fitness:.1f}  sigma={cma.sigma:.4f}  ({duration:.1f}s)")

        if trial is not None:
            trial.report(gen_best, gen)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    wall_tag = "_wall" if wall_obstacles else ""
    base = f"{prefix}" if prefix else f"sp_cma_level{level}{wall_tag}"
    out_path = (f"models/{base}_trial_{trial.number}_weights.pth"
                if trial is not None else f"models/{base}_weights.pth")

    best_genome_f32 = best_genome.astype(np.float32)
    torch.save({"encoder": encoder.state_dict(),
                "policy": torch.from_numpy(best_genome_f32)}, out_path)
    _pol_model = _flat_to_policy(best_genome_f32)
    _encoder = encoder
    print(f"Saved to {out_path} (fitness={best_fitness:.1f})")


def _load_once():
    global _encoder, _pol_model, _obs_window
    if _encoder is not None and _pol_model is not None:
        return
    wall_tag = "_wall" if _CURRENT_WALL else ""
    base = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"sp_cma_level{_CURRENT_LEVEL}{wall_tag}"
    wpath = f"models/{base}_weights.pth"
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing weights at {wpath}")
    d = torch.load(wpath, map_location="cpu", weights_only=True)
    enc = EncoderNet()
    enc.load_state_dict(d["encoder"])
    enc.eval()
    _encoder = enc
    _pol_model = _flat_to_policy(d["policy"].numpy())
    _obs_window = deque([np.zeros(OBS_DIM, dtype=np.float32)] * WINDOW, maxlen=WINDOW)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _obs_window, _ts_seen, _ts_stuck, _last_fw, _last_action, _repeat_count
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
    policy_input = _make_policy_input(belief, _ts_seen, _ts_stuck, _last_fw)

    with torch.no_grad():
        logits = _pol_model(torch.from_numpy(policy_input).unsqueeze(0)).squeeze(0).numpy()

    order = np.argsort(-logits)
    best = int(order[0])

    if _last_action is not None:
        best_q, second_q = float(logits[order[0]]), float(logits[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_fw = 1.0 if best == 2 else 0.0
    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["sigma0"] = trial.suggest_float("sigma0", 0.1, 2.0, log=True)
    params["pop_size"] = trial.suggest_categorical("pop_size", [24, 48, 64, 96])
    params["n_rollouts"] = trial.suggest_categorical("n_rollouts", [2, 3, 5])
    return params
