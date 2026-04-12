"""
Supervised pretraining of a belief encoder for OBELIX — phase 4.

Purpose:
  Trains a neural network that maps an observation window (last 8 × 18 =
  144 raw bits) to a 10-dimensional belief state using privileged env
  information as supervision. The resulting encoder is FROZEN and used as
  a feature extractor in sp_cma, sp_reinforce, sp_ddqn.

Belief state (10 dims):
  0: dist_norm         — normalised bot-to-box distance
  1: sin(angle)        — sin of direction from bot to box
  2: cos(angle)        — cos of direction from bot to box
  3: box_visible       — is box detectable by sensors (binary)
  4: box_vx_norm       — box x-velocity normalised by box_speed
  5: box_vy_norm       — box y-velocity normalised by box_speed
  6: enable_push       — is robot currently attached/pushing (binary)
  7: stuck_flag        — robot is stuck (binary)
  8: time_since_seen   — normalised time since box was last visible
  9: corner_dist_norm  — normalised distance of box to nearest corner

Training:
  - Data collected with a RANDOM policy to ensure diverse arena coverage,
    not just states visited by a good policy.
  - Mixed loss: MSE for continuous targets (dims 0,1,2,4,5,8,9); BCE for
    binary targets (dims 3,6,7).
  - EncoderNet: 144→64→32→10 (ReLU).

Why supervised pretraining instead of end-to-end RL:
  SP provides dense privileged supervision at every timestep. Training a
  belief state end-to-end with sparse RL rewards would require orders of
  magnitude more env interactions and often results in shallow features
  that exploit reward shortcuts rather than encoding true state.

Why freeze the encoder during RL:
  If encoder and policy are co-adapted with sparse RL rewards, the encoder
  can collapse to features that help the current (possibly poor) policy
  but lose the privileged information. Freezing the encoder ensures the
  belief state remains aligned with the privileged targets throughout RL.

Output: models/{prefix}_encoder.pth (state_dict) + _encoder_config.json
Must be run BEFORE any sp_* training.
"""
from __future__ import annotations
import os
import json
import math
import random
from collections import deque

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
_CURRENT_PREFIX = None


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


def train(level: int, wall_obstacles: bool, episodes: int,
          config_file=None, render: bool = False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix

    difficulty = 0 if level == 1 else 2 if level == 2 else 3

    config = {
        "encoder_epochs": 50,
        "encoder_lr": 1e-3,
        "encoder_batch_size": 256,
        "data_episodes": 200,
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
    torch.manual_seed(seed)

    import time
    t0_collect = time.time()
    n_data_eps = config["data_episodes"]
    log_interval = max(1, n_data_eps // 5)
    print(f"[encoder] Collecting {n_data_eps} episodes  (level={level}, wall={wall_obstacles})")

    inputs = []
    targets = []

    for ep in range(n_data_eps):
        ep_seed = seed + ep * 1337
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

        for _ in range(config["max_steps"]):
            enc_in = _make_encoder_input(list(obs_window))
            tgt = _get_belief_target(env, ts_seen)
            inputs.append(enc_in)
            targets.append(tgt)

            act = random.randint(0, N_ACTIONS - 1)
            obs, _, done = env.step(ACTIONS[act], render=False)
            obs_window.append(obs.astype(np.float32))

            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1

            if done:
                break

        if (ep + 1) % log_interval == 0 or ep == 0:
            elapsed = time.time() - t0_collect
            rate = (ep + 1) / elapsed
            eta = (n_data_eps - ep - 1) / rate if rate > 0 else float("inf")
            print(f"  collect {ep+1:>{len(str(n_data_eps))}}/{n_data_eps}  "
                  f"samples={len(inputs):,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    inputs_t = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets_t = torch.tensor(np.array(targets), dtype=torch.float32)

    collect_time = time.time() - t0_collect
    print(f"[encoder] Dataset: {len(inputs_t):,} samples collected in {collect_time:.0f}s")
    belief_names = ["dist_norm","sin_ang","cos_ang","box_vis","box_vx","box_vy","enable_push","stuck","ts_seen","corner_dist"]
    means = targets_t.mean(0).tolist()
    stds  = targets_t.std(0).tolist()
    print(f"  belief target stats (mean ± std):")
    for i, name in enumerate(belief_names):
        print(f"    [{i}] {name:<14} {means[i]:+.3f} ± {stds[i]:.3f}")

    encoder = EncoderNet()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config["encoder_lr"])

    continuous_idx = [0, 1, 2, 4, 5, 8, 9]
    binary_idx = [3, 6, 7]

    dataset = torch.utils.data.TensorDataset(inputs_t, targets_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["encoder_batch_size"], shuffle=True)

    n_epochs = config["encoder_epochs"]
    log_every = max(1, n_epochs // 20)
    t0_train = time.time()
    print(f"[encoder] Training for {n_epochs} epochs  "
          f"(lr={config['encoder_lr']}, batch={config['encoder_batch_size']})")

    for epoch in range(n_epochs):
        epoch_mse = 0.0
        epoch_bce = 0.0
        n_batches = 0
        for xb, yb in loader:
            pred = encoder(xb)
            mse = nn.functional.mse_loss(pred[:, continuous_idx], yb[:, continuous_idx])
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred[:, binary_idx], yb[:, binary_idx])
            loss = mse + bce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_mse += mse.item()
            epoch_bce += bce.item()
            n_batches += 1
        if (epoch + 1) % log_every == 0 or epoch == 0:
            avg_mse = epoch_mse / max(1, n_batches)
            avg_bce = epoch_bce / max(1, n_batches)
            elapsed = time.time() - t0_train
            eta = elapsed / (epoch + 1) * (n_epochs - epoch - 1)
            print(f"  epoch {epoch+1:>{len(str(n_epochs))}}/{n_epochs}  "
                  f"loss={avg_mse+avg_bce:.4f}  "
                  f"mse={avg_mse:.4f}  bce={avg_bce:.4f}  "
                  f"ETA={eta:.0f}s")

    # Post-training per-dimension eval on full dataset
    encoder.eval()
    with torch.no_grad():
        all_pred = encoder(inputs_t)
        print(f"[encoder] Final per-dimension errors (on training set):")
        for i, name in enumerate(belief_names):
            if i in binary_idx:
                prob = torch.sigmoid(all_pred[:, i])
                acc = ((prob > 0.5).float() == targets_t[:, i]).float().mean().item()
                print(f"    [{i}] {name:<14} accuracy={acc:.3f}")
            else:
                mae = (all_pred[:, i] - targets_t[:, i]).abs().mean().item()
                print(f"    [{i}] {name:<14} MAE={mae:.4f}")
    encoder.train()

    os.makedirs("models", exist_ok=True)
    if prefix:
        base = f"{prefix}_encoder"
    else:
        wall = wall_obstacles
        base = f"sp_encoder_level{level}{'_wall' if wall else ''}"

    enc_path = f"models/{base}.pth"
    cfg_path = f"models/{base}_config.json"
    torch.save(encoder.state_dict(), enc_path)
    with open(cfg_path, "w") as f:
        json.dump({"encoder_path": enc_path, "belief_dim": BELIEF_DIM, "window": WINDOW}, f)
    print(f"Encoder saved to {enc_path}")
    print(f"Encoder config saved to {cfg_path}")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    # encoder-only agent; use sp_cma/sp_reinforce/sp_ddqn for full policy.
    return ACTIONS[int(rng.integers(0, N_ACTIONS))]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["encoder_epochs"] = trial.suggest_categorical("encoder_epochs", [30, 50, 100])
    params["encoder_lr"] = trial.suggest_float("encoder_lr", 1e-4, 1e-2, log=True)
    params["data_episodes"] = trial.suggest_categorical("data_episodes", [100, 200, 500])
    return params
