"""
Curriculum CMA-ES — phase 4.

Extends cma_es.py with a 6-stage progressive difficulty curriculum:
  Stage 0: diff=0 (static box),         no wall
  Stage 1: diff=0 (static box),         wall
  Stage 2: diff=2 (blinking box),       no wall
  Stage 3: diff=2 (blinking box),       wall
  Stage 4: diff=3 (blinking+moving),    no wall
  Stage 5: diff=3 (blinking+moving),    wall
  wall_obstacles=False skips wall stages. level caps the final stage.

Motivation:
  Level 3 cold-start with CMA-ES fails because the circle-running attractor
  dominates the search from random init and the harder difficulty compounds
  this. Curriculum provides stepping stones: each stage bootstraps from the
  previous one's best policy.

Warm-start between stages:
  The CMA mean is copied from the previous stage's best genome; the
  covariance matrix is RESET to identity with sigma_warmstart. Rationale:
  the mean encodes useful directional knowledge; the covariance would
  otherwise inherit search directions calibrated to the easier env, which
  can bias the search away from the harder task's optimum.

Anti-forgetting (review rollouts):
  Each generation also evaluates review_rollouts rollouts on a randomly
  chosen earlier stage, blended into fitness (blend_weight=0.3). This
  prevents the network from unlearning simpler skills as it advances.

Stagnation restart:
  If sigma collapses below 1e-4 for stagnation_limit generations without
  improvement, the covariance is reset (mean preserved). This escapes
  degenerate search distributions without discarding learned weights.

Budget split:
  The total `episodes` (=generations) budget is split across stages by
  _split_budget, with later/harder stages receiving proportionally more
  generations (STAGE_WEIGHTS = 1.0 + 0.5*stage_index).

Architecture: 26→64→32→5 (3973 params, tanh).
  Larger than base cma_es (1477) because curriculum provides richer
  gradient-free signal that can fill a bigger network.

Training on CPU is intentional: rollouts are OpenCV/numpy-bound; GPU adds
overhead for a ~4k-param network with no batching.
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

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

INPUT_DIM = 26
HIDDEN1   = 64
HIDDEN2   = 32
OUTPUT_DIM = N_ACTIONS
N_PARAMS = (INPUT_DIM * HIDDEN1 + HIDDEN1 +
            HIDDEN1  * HIDDEN2 + HIDDEN2 +
            HIDDEN2  * OUTPUT_DIM + OUTPUT_DIM)  # 3973

# Inference globals
_CURRENT_LEVEL:  int           = 1
_CURRENT_WALL:   bool          = False
_CURRENT_PREFIX: Optional[str] = None
_MODEL:          Optional[nn.Module] = None

_last_action:     Optional[int] = None
_repeat_count:    int = 0
_MAX_REPEAT       = 3
_CLOSE_Q_DELTA    = 0.02
_time_since_seen: int = 100
_time_since_stuck: int = 100


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

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


def _extract_features(obs, time_since_seen, time_since_stuck, last_fw):
    f = np.zeros(26, dtype=np.float32)
    f[0:18] = obs.astype(np.float32)
    f[18] = float(np.sum(obs[4:12]))   # front sonar group
    f[19] = float(np.sum(obs[0:4]))    # left sonar group
    f[20] = float(np.sum(obs[12:16]))  # right sonar group
    f[21] = float(obs[16])             # IR on
    f[22] = float(obs[17])             # stuck flag
    f[23] = min(1.0, time_since_seen  / 50.0)
    f[24] = min(1.0, time_since_stuck / 20.0)
    f[25] = last_fw
    return f


# ---------------------------------------------------------------------------
# Fitness (uses privileged env state during training only)
# ---------------------------------------------------------------------------

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


def _evaluate_genome(flat_weights, difficulty, wall_obstacles, config, n_rollouts=3):
    model = _flat_to_model(flat_weights)
    total = 0.0
    use_privileged = config.get("use_privileged", True)

    for rr in range(n_rollouts):
        ep_seed = config["seed"] + rr * 7919
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
        last_fw  = 0.0
        env_reward = 0.0
        attached = pushed = False
        prev_dist = math.sqrt(
            (env.bot_center_x - env.box_center_x) ** 2 +
            (env.bot_center_y - env.box_center_y) ** 2
        ) if use_privileged else 0.0

        for _ in range(config["max_steps"]):
            ts_seen  = 0 if np.any(obs[:17] > 0) else ts_seen  + 1
            ts_stuck = 0 if obs[17] > 0           else ts_stuck + 1

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
                pushed = True
            if done:
                break

        fitness = env_reward
        fitness += 50.0  if attached else 0.0
        fitness += 500.0 if pushed   else 0.0
        total += fitness

    return total / n_rollouts


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

class CMAES:
    def __init__(self, n, sigma0=0.5, pop_size=48, seed=42):
        self.n   = n
        self.lam = pop_size
        self.mu  = pop_size // 2
        self.rng = np.random.default_rng(seed)

        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff  = 1.0 / np.sum(self.weights ** 2)

        self.sigma  = sigma0
        self.cs     = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.ds     = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.chi_n  = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.cc     = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c1     = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.cmu    = min(1.0 - self.c1,
                         2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) /
                         ((n + 2.0) ** 2 + self.mu_eff))

        self.mean     = np.zeros(n, dtype=np.float64)
        self.ps       = np.zeros(n, dtype=np.float64)
        self.pc       = np.zeros(n, dtype=np.float64)
        self.C        = np.eye(n, dtype=np.float64)
        self.B        = np.eye(n, dtype=np.float64)
        self.D        = np.ones(n, dtype=np.float64)
        self.invsqrtC = np.eye(n, dtype=np.float64)
        self._eigen_gap    = max(1, int(n / (10 * self.lam)))
        self._since_eigen  = 0

    def warm_start(self, mean, sigma):
        """Load new mean, reset covariance to identity for a fresh directional search."""
        self.mean     = mean.astype(np.float64).copy()
        self.sigma    = sigma
        self.ps       = np.zeros(self.n, dtype=np.float64)
        self.pc       = np.zeros(self.n, dtype=np.float64)
        self.C        = np.eye(self.n, dtype=np.float64)
        self.B        = np.eye(self.n, dtype=np.float64)
        self.D        = np.ones(self.n, dtype=np.float64)
        self.invsqrtC = np.eye(self.n, dtype=np.float64)
        self._since_eigen = 0

    def reset_covariance(self, sigma):
        """Keep mean, reset covariance — use when sigma collapses without progress."""
        self.sigma    = sigma
        self.ps[:]    = 0
        self.pc[:]    = 0
        self.C        = np.eye(self.n, dtype=np.float64)
        self.B        = np.eye(self.n, dtype=np.float64)
        self.D        = np.ones(self.n, dtype=np.float64)
        self.invsqrtC = np.eye(self.n, dtype=np.float64)
        self._since_eigen = 0

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean[None, :] + self.sigma * (z @ np.diag(self.D) @ self.B.T)

    def tell(self, solutions, fitnesses):
        n = self.n
        order    = np.argsort(-fitnesses)
        solutions = solutions[order]

        old_mean    = self.mean.copy()
        self.mean   = self.weights @ solutions[:self.mu]
        mean_shift  = (self.mean - old_mean) / self.sigma

        self.ps = ((1.0 - self.cs) * self.ps +
                   math.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) *
                   (self.invsqrtC @ mean_shift))

        h_sig = float(
            np.linalg.norm(self.ps) /
            math.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self._since_eigen + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * self.chi_n
        )

        self.pc = ((1.0 - self.cc) * self.pc +
                   h_sig * math.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * mean_shift)

        artmp = (solutions[:self.mu] - old_mean[None, :]) / self.sigma
        self.C = ((1.0 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1.0 - h_sig) * self.cc * (2.0 - self.cc) * self.C) +
                  self.cmu * (self.weights[:, None] * artmp).T @ artmp)

        self.sigma *= math.exp((self.cs / self.ds) *
                               (np.linalg.norm(self.ps) / self.chi_n - 1.0))

        self._since_eigen += 1
        if self._since_eigen >= self._eigen_gap:
            self._update_eigen()
            self._since_eigen = 0

    def _update_eigen(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        eigenvalues  = np.maximum(eigenvalues, 1e-20)
        self.D       = np.sqrt(eigenvalues)
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T


# ---------------------------------------------------------------------------
# Curriculum definitions
# ---------------------------------------------------------------------------

# All possible stages in training order
ALL_STAGES = [
    (0, False),   # stage 0: static box, no wall
    (0, True),    # stage 1: static box, wall
    (2, False),   # stage 2: blinking box, no wall
    (2, True),    # stage 3: blinking box, wall
    (3, False),   # stage 4: blinking + moving box, no wall
    (3, True),    # stage 5: blinking + moving box, wall
]

# Rolling-mean fitness threshold to advance early to the next stage.
# Set conservatively; tighten if the agent advances too quickly.
ADVANCE_THRESHOLDS = {
    (0, False): 1800.0,
    (0, True):  1200.0,
    (2, False):  900.0,
    (2, True):   700.0,
    (3, False):  600.0,
    (3, True):   400.0,
}

# Relative budget per stage — later/harder stages get more gens
STAGE_WEIGHTS = {s: 1.0 + 0.5 * i for i, s in enumerate(ALL_STAGES)}


def _build_curriculum(level, wall_obstacles):
    max_stage = {1: 1, 2: 3, 3: 5}[level]
    stages = ALL_STAGES[:max_stage + 1]
    if not wall_obstacles:
        stages = [(d, w) for d, w in stages if not w]
    return stages


def _split_budget(total_gens, stages):
    weights = [STAGE_WEIGHTS[s] for s in stages]
    total_w = sum(weights)
    budgets = [max(5, int(total_gens * w / total_w)) for w in weights]
    budgets[-1] += total_gens - sum(budgets)   # absorb rounding remainder
    return budgets


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def train(level, wall_obstacles, episodes, config_file=None, render=False,
          prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _MODEL
    _CURRENT_LEVEL  = level
    _CURRENT_WALL   = wall_obstacles
    _CURRENT_PREFIX = prefix
    _MODEL          = None

    config = {
        "sigma0":           0.5,
        "sigma_warmstart":  0.3,   # sigma when warm-starting into a new stage
        "sigma_restart":    0.3,   # sigma after a local covariance restart
        "pop_size":         48,
        "n_rollouts":       3,     # rollouts on current stage per candidate
        "review_rollouts":  1,     # rollouts on a random previous stage (anti-forgetting)
        "blend_weight":     0.3,   # fraction of fitness from previous-stage review
        "stagnation_limit": 20,    # gens with no stage_best improvement before restart
        "seed":             42,
        "max_steps":        1000,
        "scaling_factor":   5,
        "arena_size":       500,
        "box_speed":        2,
    }
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            config.update(json.load(f))

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    stages  = _build_curriculum(level, wall_obstacles)
    budgets = _split_budget(episodes, stages)

    print(f"Curriculum stages : {stages}")
    print(f"Generation budgets: {budgets}  (total={sum(budgets)})")

    cma = CMAES(n=N_PARAMS, sigma0=config["sigma0"],
                pop_size=config["pop_size"], seed=seed)

    best_genome          = cma.mean.copy()
    best_fitness_overall = -float("inf")
    global_gen           = 0

    for stage_idx, (difficulty, wall) in enumerate(stages):
        max_gens  = budgets[stage_idx]
        threshold = ADVANCE_THRESHOLDS[(difficulty, wall)]
        prev_stages = stages[:stage_idx]

        print(f"\n=== Stage {stage_idx}/{len(stages)-1}: "
              f"diff={difficulty}  wall={wall}  budget={max_gens} gens ===")

        if stage_idx > 0:
            cma.warm_start(best_genome, config["sigma_warmstart"])

        stage_best       = -float("inf")
        stagnation_count = 0
        rolling_means    = []

        for gen in range(max_gens):
            t0        = time.time()
            solutions = cma.ask()
            fitnesses = np.zeros(cma.lam)

            for i in range(cma.lam):
                eval_cfg        = dict(config)
                eval_cfg["seed"] = seed + global_gen * cma.lam + i

                f_curr = _evaluate_genome(
                    solutions[i].astype(np.float32),
                    difficulty=difficulty,
                    wall_obstacles=wall,
                    config=eval_cfg,
                    n_rollouts=config["n_rollouts"],
                )

                # Mix in one random previous stage to prevent forgetting
                if prev_stages and config["blend_weight"] > 0:
                    p_diff, p_wall = random.choice(prev_stages)
                    f_prev = _evaluate_genome(
                        solutions[i].astype(np.float32),
                        difficulty=p_diff,
                        wall_obstacles=p_wall,
                        config=eval_cfg,
                        n_rollouts=config["review_rollouts"],
                    )
                    fitnesses[i] = ((1.0 - config["blend_weight"]) * f_curr +
                                    config["blend_weight"] * f_prev)
                else:
                    fitnesses[i] = f_curr

            cma.tell(solutions, fitnesses)

            gen_best = float(np.max(fitnesses))
            gen_mean = float(np.mean(fitnesses))

            if gen_best > stage_best:
                stage_best       = gen_best
                best_genome      = solutions[np.argmax(fitnesses)].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            if gen_best > best_fitness_overall:
                best_fitness_overall = gen_best

            rolling_means.append(gen_mean)
            if len(rolling_means) > 10:
                rolling_means.pop(0)

            print(f"  Gen {gen+1:4d}/{max_gens}  best={gen_best:8.1f}  "
                  f"mean={gen_mean:8.1f}  stage_best={stage_best:8.1f}  "
                  f"sigma={cma.sigma:.4f}  ({time.time()-t0:.1f}s)")

            # Local restart: if sigma collapses with no progress, keep mean and re-explore
            if cma.sigma < 1e-2 and stagnation_count >= config["stagnation_limit"]:
                print(f"  [restart] sigma={cma.sigma:.2e} stagnated {stagnation_count} gens — resetting covariance")
                cma.reset_covariance(config["sigma_restart"])
                stagnation_count = 0

            if trial is not None:
                trial.report(gen_best, global_gen)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            global_gen += 1

            # Adaptive advance: rolling mean exceeds threshold after a warmup period
            if len(rolling_means) == 10 and gen >= 9:
                if np.mean(rolling_means) >= threshold:
                    print(f"  [advance] rolling mean {np.mean(rolling_means):.1f} >= {threshold:.1f}")
                    break

        # Checkpoint after each stage
        os.makedirs("models", exist_ok=True)
        base       = prefix if prefix else f"curriculum_cma_level{level}{'_wall' if wall_obstacles else ''}"
        stage_path = f"models/{base}_stage{stage_idx}_weights.pth"
        torch.save(torch.from_numpy(best_genome.astype(np.float32)), stage_path)
        print(f"  -> Stage {stage_idx} checkpoint: {stage_path}")

    # Final weights file (used by policy() and the submission)
    base     = prefix if prefix else f"curriculum_cma_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = (f"models/{base}_trial_{trial.number}_weights.pth"
                if trial is not None else f"models/{base}_weights.pth")

    best_f32 = best_genome.astype(np.float32)
    _MODEL = _flat_to_model(best_f32)
    torch.save(torch.from_numpy(best_f32), out_path)
    print(f"\nDone. Final weights -> {out_path}  (best fitness={best_fitness_overall:.1f})")


# ---------------------------------------------------------------------------
# Inference (18-bit obs only, no privileged info)
# ---------------------------------------------------------------------------

def _load_once():
    global _MODEL
    if _MODEL is None:
        base  = (_CURRENT_PREFIX if _CURRENT_PREFIX
                 else f"curriculum_cma_level{_CURRENT_LEVEL}"
                      f"{'_wall' if _CURRENT_WALL else ''}")
        wpath = f"models/{base}_weights.pth"
        flat  = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
        _MODEL = _flat_to_model(flat)
    return _MODEL


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck
    _load_once()

    _time_since_seen  = 0 if np.any(obs[:17] > 0) else _time_since_seen  + 1
    _time_since_stuck = 0 if obs[17] > 0           else _time_since_stuck + 1

    last_fw  = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    features = _extract_features(obs, _time_since_seen, _time_since_stuck, last_fw)
    with torch.no_grad():
        logits = _MODEL(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
    order    = np.argsort(-logits)
    best     = int(order[0])

    if _last_action is not None:
        if (logits[order[0]] - logits[order[1]]) < _CLOSE_Q_DELTA:
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
# Optuna interface
# ---------------------------------------------------------------------------

def get_optuna_params(trial, total_episodes):
    return {
        "sigma0":          trial.suggest_float("sigma0",          0.2,  1.5,  log=True),
        "sigma_warmstart": trial.suggest_float("sigma_warmstart", 0.1,  0.5,  log=True),
        "pop_size":        trial.suggest_categorical("pop_size",  [32, 48, 64, 96]),
        "n_rollouts":      trial.suggest_categorical("n_rollouts", [2, 3, 5]),
        "blend_weight":    trial.suggest_float("blend_weight",    0.1,  0.4),
    }
