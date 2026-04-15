"""
Behavioural Cloning warm-start + CMA-ES for OBELIX — phase 4.

Two-phase training:
  Phase 1 — Behavioural Cloning (BC):
    A 26→32→16→5 PyTorch MLP is pre-trained with cross-entropy loss on
    (obs, action) pairs from human demonstrations (data/bc_demos.npz).
    This gives a starting policy that already demonstrates approach
    behaviour before any env interaction.

  Phase 2 — CMA-ES fine-tuning:
    Flat weights are extracted from the BC PyTorch net and used as the
    CMA-ES mean vector. CMA-ES then optimises the policy against the
    actual env reward + fitness bonuses.

Why BC before CMA-ES:
  From random initialisation, CMA-ES on OBELIX frequently converges to
  the circle-running local optimum within the first 20 generations: the
  fitness landscape near zero weights gives consistent (if low) reward
  from avoiding walls, and the evolutionary search never escapes.
  BC provides a starting policy that already contacts the box in most
  demonstrations, placing the CMA-ES mean in a basin where the push
  reward (+500) is achievable.

sigma0=0.3 (vs 0.5 default):
  BC weights already occupy a reasonable region of policy space; a smaller
  initial step size prevents CMA-ES from immediately jumping away from the
  BC solution into random territory.

Architecture must match cma_es.py exactly (26→32→16→5, 1477 params) so
that BC flat weights transfer directly as the CMA-ES mean vector without
reshaping.

Demo format: data/bc_demos.npz with keys 'observations' (N,18) float32
and 'actions' (N,) int32. Collect with: uv run src/record_demos.py
"""
from __future__ import annotations
import os
import json
import math
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

INPUT_DIM = 26
HIDDEN1 = 32
HIDDEN2 = 16
OUTPUT_DIM = N_ACTIONS
N_PARAMS = INPUT_DIM * HIDDEN1 + HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 + HIDDEN2 * OUTPUT_DIM + OUTPUT_DIM  # 1477

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None
_MODEL: Optional[nn.Module] = None

_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.02
_time_since_seen: int = 100
_time_since_stuck: int = 100


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


# BC network (PyTorch, same shape as CMA net)
class BCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.l2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.l3 = nn.Linear(HIDDEN2, OUTPUT_DIM)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)


def _net_to_flat(net: BCNet) -> np.ndarray:
    return torch.cat([p.data.flatten() for p in net.parameters()]).numpy()


def _flat_to_model(flat: np.ndarray) -> BCNet:
    model = BCNet()
    idx = 0
    for layer in [model.l1, model.l2, model.l3]:
        n_w = layer.in_features * layer.out_features
        W = flat[idx:idx + n_w].reshape(layer.in_features, layer.out_features)
        layer.weight.data = torch.from_numpy(W.T.copy()).float()
        idx += n_w
        layer.bias.data = torch.from_numpy(flat[idx:idx + layer.out_features].copy()).float()
        idx += layer.out_features
    model.eval()
    return model


def _evaluate_genome(flat_weights, difficulty, wall_obstacles, config, n_rollouts=3):
    model = _flat_to_model(flat_weights)
    seed = config["seed"]
    total_fitness = 0.0
    use_priv = config.get("use_privileged", True)

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

        ts_seen = 100
        ts_stuck = 100
        last_fw = 0.0
        attached = False
        pushed_to_boundary = False
        env_reward = 0.0

        prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                              (env.bot_center_y - env.box_center_y)**2)
        priv_total = 0.0

        for step in range(config["max_steps"]):
            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1
            if obs[17] > 0:
                ts_stuck = 0
            else:
                ts_stuck += 1

            features = _extract_features(obs, ts_seen, ts_stuck, last_fw)
            with torch.no_grad():
                logits = model(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
            act_idx = int(np.argmax(logits))

            obs, reward, done = env.step(ACTIONS[act_idx], render=False)
            env_reward += reward
            last_fw = 1.0 if act_idx == 2 else 0.0

            if use_priv:
                pr, prev_dist = _priv_shaping(env, prev_dist)
                priv_total += pr

            if env.enable_push and not attached:
                attached = True
            if done and env.enable_push:
                pushed_to_boundary = True

            if done:
                break

        fitness = env_reward + priv_total
        fitness += 50.0 if attached else 0.0
        fitness += 500.0 if pushed_to_boundary else 0.0
        total_fitness += fitness

    return total_fitness / n_rollouts


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
        return self.mean[None, :] + self.sigma * (z @ np.diag(self.D) @ self.B.T)

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray):
        n = self.n
        order = np.argsort(-fitnesses)
        solutions = solutions[order]

        old_mean = self.mean.copy()
        self.mean = self.weights @ solutions[:self.mu]

        mean_shift = (self.mean - old_mean) / self.sigma
        self.ps = ((1.0 - self.cs) * self.ps +
                   math.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * (self.invsqrtC @ mean_shift))

        h_sig = 1.0 if (np.linalg.norm(self.ps) /
                        math.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self._gen_since_eigen + 1))) <
                        (1.4 + 2.0 / (n + 1.0)) * self.chi_n) else 0.0

        self.pc = ((1.0 - self.cc) * self.pc +
                   h_sig * math.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * mean_shift)

        artmp = (solutions[:self.mu] - old_mean[None, :]) / self.sigma
        self.C = ((1.0 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1.0 - h_sig) * self.cc * (2.0 - self.cc) * self.C) +
                  self.cmu * (self.weights[:, None] * artmp).T @ artmp)

        self.sigma *= math.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1.0))

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


def _bc_pretrain(config):
    demo_path = config["bc_demo_path"]
    if not os.path.exists(demo_path):
        print(f"Demo file not found: {demo_path}")
        print("Record demos first: uv run src/record_demos.py --output", demo_path)
        return None

    data = np.load(demo_path)
    raw_obs = data["observations"]
    actions = data["actions"]

    features = np.stack([_extract_features(o, 0, 0, 0.0) for o in raw_obs])
    feat_t = torch.tensor(features, dtype=torch.float32)
    act_t = torch.tensor(actions, dtype=torch.long)

    net = BCNet()
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

    return net


def train(level, wall_obstacles, episodes, config_file=None, render=False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _MODEL

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _MODEL = None

    difficulty = {1: 0, 2: 2, 3: 3}[level]

    config = {
        "sigma0": 0.3,
        "pop_size": 48,
        "n_rollouts": 3,
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

    # Phase 1: BC pre-training
    print(f"=== BC Pre-training (level={level}, wall={wall_obstacles}) ===")
    bc_net = _bc_pretrain(config)
    if bc_net is None:
        return

    bc_weights = _net_to_flat(bc_net).astype(np.float64)
    print(f"BC weights extracted: {len(bc_weights)} params")

    # Phase 2: CMA-ES fine-tuning
    print(f"=== CMA-ES fine-tuning for {episodes} generations ===")
    cma = CMAES(n=N_PARAMS, sigma0=config["sigma0"], pop_size=config["pop_size"], seed=seed)
    cma.mean = bc_weights.copy()

    best_fitness = -float("inf")
    best_genome = cma.mean.copy()

    for gen in range(episodes):
        t_start = time.time()
        solutions = cma.ask()

        fitnesses = np.zeros(cma.lam)
        for i in range(cma.lam):
            eval_config = dict(config)
            eval_config["seed"] = seed + gen * cma.lam + i
            fitnesses[i] = _evaluate_genome(
                solutions[i].astype(np.float32),
                difficulty=difficulty,
                wall_obstacles=wall_obstacles,
                config=eval_config,
                n_rollouts=config["n_rollouts"],
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
    base_name = (f"{prefix}" if prefix
                 else f"bc_cma_level{level}{'_wall' if wall_obstacles else ''}")
    out_path = (f"models/{base_name}_trial_{trial.number}_weights.pth"
                if trial is not None
                else f"models/{base_name}_weights.pth")

    best_f32 = best_genome.astype(np.float32)
    _MODEL = _flat_to_model(best_f32)
    torch.save(torch.from_numpy(best_f32), out_path)
    print(f"Saved best genome ({N_PARAMS} params, fitness={best_fitness:.1f}) to {out_path}")


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return
    base_name = (f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX
                 else f"bc_cma_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
    wpath = f"models/{base_name}_weights.pth"
    flat = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    _MODEL = _flat_to_model(flat)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck
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
    features = _extract_features(obs, _time_since_seen, _time_since_stuck, last_fw)

    with torch.no_grad():
        logits = _MODEL(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()

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

    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["sigma0"] = trial.suggest_float("sigma0", 0.05, 1.0, log=True)
    params["bc_epochs"] = trial.suggest_categorical("bc_epochs", [20, 30, 50])
    params["n_rollouts"] = trial.suggest_categorical("n_rollouts", [2, 3, 5])
    return params
