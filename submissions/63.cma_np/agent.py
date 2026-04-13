"""
CMA-ES Inference Agent
Loads the evolved MLP weights and runs a forward pass to select actions.
Uses the same augmented 26-dim feature extraction as the training script.
"""
from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

# Network shape (must match training)
INPUT_DIM = 26
HIDDEN1 = 32
HIDDEN2 = 16
OUTPUT_DIM = N_ACTIONS

# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------

def _unpack_weights(flat: np.ndarray):
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
    h1 = np.tanh(features @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    return logits


def _extract_features(obs: np.ndarray,
                       time_since_seen: int,
                       time_since_stuck: int,
                       last_action_was_fw: float) -> np.ndarray:
    features = np.zeros(26, dtype=np.float32)
    features[0:18] = obs.astype(np.float32)
    features[18] = float(np.sum(obs[4:12]))   # front activation
    features[19] = float(np.sum(obs[0:4]))    # left activation
    features[20] = float(np.sum(obs[12:16]))  # right activation
    features[21] = float(obs[16])             # IR on
    features[22] = float(obs[17])             # stuck flag
    features[23] = min(1.0, time_since_seen / 50.0)
    features[24] = min(1.0, time_since_stuck / 20.0)
    features[25] = last_action_was_fw
    return features


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_WEIGHTS: Optional[np.ndarray] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.02
_time_since_seen: int = 100
_time_since_stuck: int = 100


def _load_once():
    global _WEIGHTS
    if _WEIGHTS is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. "
            "Train offline and include it in the submission zip."
        )

    _WEIGHTS = torch.load(wpath, map_location="cpu", weights_only=True).numpy()


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
