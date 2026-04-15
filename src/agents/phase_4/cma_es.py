"""
CMA-ES agent for OBELIX — phase 4.

Algorithm:
  Full (μ/μ_w, λ)-CMA-ES implemented from scratch (no pycma dependency).
  Improvements over the phase-3 flat evolution strategy:
    - Rank-1 + rank-μ covariance matrix adaptation tracks the local loss
      landscape curvature, rotating and scaling the sampling distribution to
      follow successful search directions.
    - Cumulative Step-size Adaptation (CSA) automatically adjusts σ so the
      search neither stagnates nor explodes.
  Previous phase used tournament selection with fixed σ; this is strictly
  better on ill-conditioned landscapes.

Policy:
  26→32→16→5 MLP (tanh), 1477 parameters. Small enough that 48 candidates ×
  3 rollouts per generation remain fast on CPU; large enough to encode
  directional sensor logic.

Feature engineering (26-dim):
  Raw 18-bit obs + 3 sonar group sums (front/left/right) + IR flag + stuck
  flag + time_since_seen (norm) + time_since_stuck (norm) + last_fw flag.
  Temporal features give the obs-only policy partial POMDP memory for the
  blinking-box difficulties.

Reward shaping (obs-only variant, use_privileged=False):
  +195 when obs[17] (stuck) > 0  — overrides -200 env penalty to net -5.
  -2   when all 17 sensors silent — penalises open-space circling.
  +3   when obs[16] (IR) > 0     — rewards physical proximity to box.
  Limitation: the -2 silent penalty can train wall-hugging at difficulty 2+
  where the box blinks off. Reduce to -0.5 if wall-hugging is observed.

Reward shaping (privileged variant, use_privileged=True):
  Above obs-only terms PLUS:
  +2.0 × (prev_dist - curr_dist)  — dense distance-reduction signal.
  +0.5 if facing within 45° of box — heading alignment bonus.
  +1.0/step while enable_push     — sustained push encouragement.
  Privileged info is available via env internals during training only;
  inference uses the identical 18-bit obs vector as all other agents.

Fitness: shaped_env_reward + 50 (if ever attached) + 500 (if pushed to wall).
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
import torch.nn as nn
import optuna

from obelix import OBELIX

# Constants
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


_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None
_MODEL: Optional[nn.Module] = None  # cached inference model

# Inference state
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.02
_time_since_seen: int = 100
_time_since_stuck: int = 100

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.Tanh(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.Tanh(),
            nn.Linear(HIDDEN2, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def _flat_to_model(flat: np.ndarray) -> MLP:
    """Load a flat CMA-ES parameter vector into an MLP (matches numpy packing order)."""
    model = MLP()
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


def _shaped_reward(raw, obs):
    r = raw
    if obs[17] > 0:
        r += 195.0          # override -200 stuck -> -5
    if float(np.sum(obs[:17])) == 0.0 and obs[17] == 0:
        r -= 2.0            # penalise open-space circling
    if obs[16] > 0:
        r += 3.0            # IR contact proximity bonus
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

def _evaluate_genome(flat_weights: np.ndarray,
                     difficulty: int,
                     wall_obstacles: bool,
                     config: dict,
                     n_rollouts: int = 3) -> float:
    model = _flat_to_model(flat_weights)
    seed = config["seed"]
    total_fitness = 0.0
    use_privileged = config.get("use_privileged", True)

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

        env_reward = 0.0
        attached = False
        pushed_to_boundary = False
        prev_dist = math.sqrt(
            (env.bot_center_x - env.box_center_x) ** 2 +
            (env.bot_center_y - env.box_center_y) ** 2
        ) if use_privileged else 0.0

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
            with torch.no_grad():
                logits = model(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
            act_idx = int(np.argmax(logits))

            obs, reward, done = env.step(ACTIONS[act_idx], render=False)
            r = _shaped_reward(reward, obs)
            if use_privileged:
                pr, prev_dist = _priv_shaping(env, prev_dist)
                r += pr
            env_reward += r

            last_fw = 1.0 if act_idx == 2 else 0.0

            if env.enable_push and not attached:
                attached = True
            if done and env.enable_push:
                pushed_to_boundary = True

            if done:
                break

        fitness = env_reward
        fitness += 50.0 if attached else 0.0
        fitness += 500.0 if pushed_to_boundary else 0.0

        total_fitness += fitness

    return total_fitness / n_rollouts

class CMAES:
    """Minimal (mu/mu_w, lambda)-CMA-ES."""

    def __init__(self, n: int, sigma0: float = 0.5, pop_size: int = 48,
                 seed: int = 42):
        self.n = n
        self.lam = pop_size  # lambda
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
        self.ps = np.zeros(n, dtype=np.float64)  # evolution path for sigma
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
        """Sample lambda candidate solutions."""
        z = self.rng.standard_normal((self.lam, self.n))
        # x_k = mean + sigma * B D z_k
        samples = self.mean[None, :] + self.sigma * (z @ np.diag(self.D) @ self.B.T)
        return samples

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray):
        """Update CMA-ES state given evaluated solutions and fitnesses"""
        n = self.n

        # Sort by fitness
        order = np.argsort(-fitnesses)
        solutions = solutions[order]

        # Weighted mean of top-mu
        old_mean = self.mean.copy()
        self.mean = self.weights @ solutions[:self.mu]

        # Evolution path for sigma (CSA)
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

        # Rank-1 + rank-mu covariance update
        artmp = (solutions[:self.mu] - old_mean[None, :]) / self.sigma
        self.C = ((1.0 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1.0 - h_sig) * self.cc * (2.0 - self.cc) * self.C) +
                  self.cmu * (self.weights[:, None] * artmp).T @ artmp)

        # Step-size update
        self.sigma *= math.exp((self.cs / self.ds) *
                               (np.linalg.norm(self.ps) / self.chi_n - 1.0))

        # Eigendecomposition
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

def train(level: int, wall_obstacles: bool, episodes: int,
          config_file: str = None, render: bool = False,
          prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _MODEL

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _MODEL = None

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
        "use_privileged": True,
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

    torch.save(torch.from_numpy(best_genome.astype(np.float32)), out_path)
    print(f"Saved best genome ({N_PARAMS} params, fitness={best_fitness:.1f}) to {out_path}")

def _load_once():
    global _MODEL
    if _MODEL is None:
        base_name = (f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX
                     else f"cma_es_level{_CURRENT_LEVEL}"
                          f"{'_wall' if _CURRENT_WALL else ''}")
        wpath = f"models/{base_name}_weights.pth"
        flat = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
        _MODEL = _flat_to_model(flat)
    return _MODEL


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck
    model = _load_once()

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

    with torch.no_grad():
        logits = model(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()

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

def get_optuna_params(trial, total_episodes):
    params = {}
    params["sigma0"] = trial.suggest_float("sigma0", 0.1, 2.0, log=True)
    params["pop_size"] = trial.suggest_categorical("pop_size", [24, 48, 64, 96])
    params["n_rollouts"] = trial.suggest_categorical("n_rollouts", [2, 3, 5])
    return params
