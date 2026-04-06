"""
MODIFICATIONS:
- Base Algorithm: CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
- Policy: Small MLP neural network (26 → 32 → 16 → 5) with tanh activations.
- Privileged Fitness: Uses env.bot_center_x/y, env.box_center_x/y, env.enable_push
  during training for distance-based reward shaping. Inference sees only the 18-bit obs.
- Augmented Features: Raw 18-bit obs + sensor group sums + temporal memory
  (time_since_seen, time_since_stuck) + last_action_was_forward.
- CMA-ES: Full (μ/μ_w, λ)-CMA-ES with step-size adaptation (CSA) and
  covariance matrix rank-μ + rank-1 updates. Implemented from scratch (no pycma).
"""
from __future__ import annotations
import os
import json
import math
import time
import random
from typing import Optional

import numpy as np
import torch
import optuna

from obelix import OBELIX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

# Network shape
INPUT_DIM = 26
HIDDEN1 = 32
HIDDEN2 = 16
OUTPUT_DIM = N_ACTIONS

# Total number of parameters (weights + biases)
N_PARAMS = (INPUT_DIM * HIDDEN1 + HIDDEN1 +
            HIDDEN1 * HIDDEN2 + HIDDEN2 +
            HIDDEN2 * OUTPUT_DIM + OUTPUT_DIM)  # 1477

# ---------------------------------------------------------------------------
# Globals (same pattern as phase_2 agents)
# ---------------------------------------------------------------------------
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None
_WEIGHTS: Optional[np.ndarray] = None  # flat weight vector

# Inference state
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.02
_time_since_seen: int = 100
_time_since_stuck: int = 100

# ---------------------------------------------------------------------------
# MLP helpers (pure numpy, no autograd)
# ---------------------------------------------------------------------------

def _unpack_weights(flat: np.ndarray):
    """Unpack a flat parameter vector into weight matrices and biases."""
    idx = 0
    def take(shape):
        nonlocal idx
        size = 1
        for s in shape:
            size *= s
        w = flat[idx : idx + size].reshape(shape)
        idx += size
        return w

    W1 = take((INPUT_DIM, HIDDEN1))
    b1 = take((HIDDEN1,))
    W2 = take((HIDDEN1, HIDDEN2))
    b2 = take((HIDDEN2,))
    W3 = take((HIDDEN2, OUTPUT_DIM))
    b3 = take((OUTPUT_DIM,))
    return W1, b1, W2, b2, W3, b3


def _forward(features: np.ndarray, W1, b1, W2, b2, W3, b3) -> np.ndarray:
    """Forward pass through the MLP. Returns action logits (5-dim)."""
    h1 = np.tanh(features @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    return logits


def _extract_features(obs: np.ndarray,
                       time_since_seen: int,
                       time_since_stuck: int,
                       last_action_was_fw: float) -> np.ndarray:
    """Build the 26-dim augmented feature vector from raw 18-bit obs."""
    features = np.zeros(26, dtype=np.float32)
    features[0:18] = obs.astype(np.float32)
    # Aggregated sensor groups
    features[18] = float(np.sum(obs[4:12]))   # front activation
    features[19] = float(np.sum(obs[0:4]))    # left activation
    features[20] = float(np.sum(obs[12:16]))  # right activation
    features[21] = float(obs[16])             # IR on
    features[22] = float(obs[17])             # stuck flag
    # Temporal memory (normalised to 0..1)
    features[23] = min(1.0, time_since_seen / 50.0)
    features[24] = min(1.0, time_since_stuck / 20.0)
    features[25] = last_action_was_fw
    return features


# ---------------------------------------------------------------------------
# Fitness evaluation (training only – uses privileged env info)
# ---------------------------------------------------------------------------

def _evaluate_genome(flat_weights: np.ndarray,
                     difficulty: int,
                     wall_obstacles: bool,
                     config: dict,
                     n_rollouts: int = 3) -> float:
    """Run n_rollouts episodes, return the mean shaped fitness.

    Privileged shaping (only during training):
      - Per-step distance reduction: +1.5 per pixel closer to box.
      - Per-step heading alignment: +0.3 when facing within 45° of box.
      - Attach bonus: +50 on first attachment.
      - Push-to-boundary bonus: +500 on terminal success.
    The raw env reward already includes -200 per stuck step, -1 per step,
    one-time sensor bonuses, +100 attach, and +2000 boundary push.
    """
    W1, b1, W2, b2, W3, b3 = _unpack_weights(flat_weights)
    seed = config["seed"]
    total_fitness = 0.0

    for r in range(n_rollouts):
        ep_seed = seed + r * 7919  # spread seeds
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

        env_reward = 0.0
        distance_shaping = 0.0
        heading_shaping = 0.0
        attached = False
        pushed_to_boundary = False

        # Privileged: current distance to box
        prev_dist = math.sqrt(
            (env.bot_center_x - env.box_center_x) ** 2 +
            (env.bot_center_y - env.box_center_y) ** 2
        )

        for step in range(config["max_steps"]):
            # Update temporal memory
            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1
            if obs[17] > 0:
                ts_stuck = 0
            else:
                ts_stuck += 1

            features = _extract_features(obs, ts_seen, ts_stuck, last_fw)
            logits = _forward(features, W1, b1, W2, b2, W3, b3)
            act_idx = int(np.argmax(logits))

            obs, reward, done = env.step(ACTIONS[act_idx], render=False)
            env_reward += reward

            last_fw = 1.0 if act_idx == 2 else 0.0

            # ----- Per-step privileged shaping -----
            if not env.enable_push:
                # Distance shaping: reward getting closer to the box
                curr_dist = math.sqrt(
                    (env.bot_center_x - env.box_center_x) ** 2 +
                    (env.bot_center_y - env.box_center_y) ** 2
                )
                delta_dist = prev_dist - curr_dist  # positive = got closer
                distance_shaping += 1.5 * delta_dist
                prev_dist = curr_dist

                # Heading shaping: reward facing toward the box
                dx = env.box_center_x - env.bot_center_x
                dy = env.box_center_y - env.bot_center_y
                if abs(dx) + abs(dy) > 1e-3:
                    angle_to_box = math.degrees(math.atan2(dy, dx)) % 360
                    facing = env.facing_angle % 360
                    angle_diff = abs(angle_to_box - facing)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    if angle_diff < 45:
                        heading_shaping += 0.3

            if env.enable_push and not attached:
                attached = True
            if done and env.enable_push:
                pushed_to_boundary = True

            if done:
                break

        # Shaped fitness = raw env reward + privileged shaping
        fitness = env_reward
        fitness += distance_shaping
        fitness += heading_shaping
        fitness += 50.0 if attached else 0.0
        fitness += 500.0 if pushed_to_boundary else 0.0

        total_fitness += fitness

    return total_fitness / n_rollouts


# ---------------------------------------------------------------------------
# CMA-ES implementation (from scratch)
# ---------------------------------------------------------------------------

class CMAES:
    """Minimal (μ/μ_w, λ)-CMA-ES."""

    def __init__(self, n: int, sigma0: float = 0.5, pop_size: int = 48,
                 seed: int = 42):
        self.n = n
        self.lam = pop_size  # λ
        self.mu = pop_size // 2  # number of parents
        self.rng = np.random.default_rng(seed)

        # Recombination weights (log-linear)
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size control
        self.sigma = sigma0
        self.cs = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.ds = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # Covariance adaptation
        self.cc = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) /
                       ((n + 2.0) ** 2 + self.mu_eff))

        # State
        self.mean = np.zeros(n, dtype=np.float64)
        self.ps = np.zeros(n, dtype=np.float64)  # evolution path for σ
        self.pc = np.zeros(n, dtype=np.float64)  # evolution path for C

        # We store C in factored form: C = B D^2 B^T
        # For efficiency with n~1500, we keep full C and eigendecompose periodically.
        self.C = np.eye(n, dtype=np.float64)
        self.B = np.eye(n, dtype=np.float64)
        self.D = np.ones(n, dtype=np.float64)
        self.invsqrtC = np.eye(n, dtype=np.float64)
        self._eigen_update_gap = max(1, int(n / (10 * self.lam)))
        self._gen_since_eigen = 0

    def ask(self) -> np.ndarray:
        """Sample λ candidate solutions."""
        z = self.rng.standard_normal((self.lam, self.n))
        # x_k = mean + sigma * B D z_k
        samples = self.mean[None, :] + self.sigma * (z @ np.diag(self.D) @ self.B.T)
        return samples

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray):
        """Update CMA-ES state given evaluated solutions and fitnesses (higher = better)."""
        n = self.n

        # Sort by fitness (descending – we maximise)
        order = np.argsort(-fitnesses)
        solutions = solutions[order]

        # Weighted mean of top-μ
        old_mean = self.mean.copy()
        self.mean = self.weights @ solutions[:self.mu]

        # Evolution path for σ (CSA)
        mean_shift = (self.mean - old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * (self.invsqrtC @ mean_shift)

        # Heaviside function
        h_sig = 1.0 if (np.linalg.norm(self.ps) /
                        math.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self._gen_since_eigen + 1))) <
                        (1.4 + 2.0 / (n + 1.0)) * self.chi_n) else 0.0

        # Evolution path for C
        self.pc = (1.0 - self.cc) * self.pc + \
                  h_sig * math.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * mean_shift

        # Rank-1 + rank-μ covariance update
        artmp = (solutions[:self.mu] - old_mean[None, :]) / self.sigma
        self.C = ((1.0 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1.0 - h_sig) * self.cc * (2.0 - self.cc) * self.C) +
                  self.cmu * (self.weights[:, None] * artmp).T @ artmp)

        # Step-size update
        self.sigma *= math.exp((self.cs / self.ds) *
                               (np.linalg.norm(self.ps) / self.chi_n - 1.0))

        # Eigendecomposition (expensive, do periodically)
        self._gen_since_eigen += 1
        if self._gen_since_eigen >= self._eigen_update_gap:
            self._update_eigen()
            self._gen_since_eigen = 0

    def _update_eigen(self):
        """Eigendecompose C to update B, D, invsqrtC."""
        # Symmetrise
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T


# ---------------------------------------------------------------------------
# train() – follows the phase_2 workflow
# ---------------------------------------------------------------------------

def train(level: int, wall_obstacles: bool, episodes: int,
          config_file: str = None, render: bool = False,
          prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _WEIGHTS

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _WEIGHTS = None

    print(f"Training CMA-ES for level {level} with wall_obstacles={wall_obstacles} "
          f"for {episodes} generations")

    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3

    # Default config
    config = {
        "sigma0": 0.5,
        "pop_size": 48,
        "n_rollouts": 3,
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2,
    }

    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            config.update(json.load(f))

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    cma = CMAES(n=N_PARAMS, sigma0=config["sigma0"],
                pop_size=config["pop_size"], seed=seed)

    best_fitness = -float("inf")
    best_genome = cma.mean.copy()

    for gen in range(episodes):
        t_start = time.time()
        solutions = cma.ask()

        fitnesses = np.zeros(cma.lam)
        for i in range(cma.lam):
            # Shift seed each generation so episodes are diverse
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
            # The best solution is now the first after sorting inside tell()
            best_genome = solutions[np.argmax(fitnesses)].copy()

        duration = time.time() - t_start
        print(f"Gen {gen + 1}/{episodes}  best={gen_best:.1f}  "
              f"mean={gen_mean:.1f}  all-time-best={best_fitness:.1f}  "
              f"sigma={cma.sigma:.4f}  ({duration:.1f}s)")

        # Optuna pruning support
        if trial is not None:
            trial.report(gen_best, gen)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Save best weights
    os.makedirs("models", exist_ok=True)
    base_name = (f"{prefix}" if prefix
                 else f"cma_es_level{level}{'_wall' if wall_obstacles else ''}")
    out_path = (f"models/{base_name}_trial_{trial.number}_weights.pth"
                if trial is not None
                else f"models/{base_name}_weights.pth")

    _WEIGHTS = best_genome.astype(np.float32)
    torch.save(torch.from_numpy(_WEIGHTS), out_path)
    print(f"Saved best genome ({N_PARAMS} params, fitness={best_fitness:.1f}) to {out_path}")


# ---------------------------------------------------------------------------
# _load_once() / policy() – follows the phase_2 workflow
# ---------------------------------------------------------------------------

def _load_once():
    global _WEIGHTS
    if _WEIGHTS is None:
        base_name = (f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX
                     else f"cma_es_level{_CURRENT_LEVEL}"
                          f"{'_wall' if _CURRENT_WALL else ''}")
        wpath = f"models/{base_name}_weights.pth"
        _WEIGHTS = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    return _WEIGHTS


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck
    _load_once()

    # Update temporal memory
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

    W1, b1, W2, b2, W3, b3 = _unpack_weights(_WEIGHTS)
    logits = _forward(features, W1, b1, W2, b2, W3, b3)

    order = np.argsort(-logits)
    best = int(order[0])

    # Anti-oscillation smoothing
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


# ---------------------------------------------------------------------------
# Optuna hyperparameter interface
# ---------------------------------------------------------------------------

def get_optuna_params(trial, total_episodes):
    params = {}
    params["sigma0"] = trial.suggest_float("sigma0", 0.1, 2.0, log=True)
    params["pop_size"] = trial.suggest_categorical("pop_size", [24, 48, 64, 96])
    params["n_rollouts"] = trial.suggest_categorical("n_rollouts", [2, 3, 5])
    return params
