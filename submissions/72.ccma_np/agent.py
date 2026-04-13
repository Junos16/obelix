"""
Inference-only agent for curriculum_cma submissions.
Copy this file as agent.py into your submission folder alongside weights.pth.

weights.pth is produced by curriculum_cma.train() — it stores a flat numpy
array of 3973 parameters for a 26->64->32->5 MLP.

ZIP layout:
  submission.zip
    agent.py        (this file, renamed)
    weights.pth
"""
from __future__ import annotations
from typing import Optional
import os
import math
import numpy as np
import torch

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

INPUT_DIM  = 26
HIDDEN1    = 64
HIDDEN2    = 32
OUTPUT_DIM = 5

# Inference state
_weights:         Optional[np.ndarray] = None
_last_action:     Optional[int] = None
_repeat_count:    int = 0
_time_since_seen: int = 100
_time_since_stuck: int = 100

_MAX_REPEAT    = 3
_CLOSE_Q_DELTA = 0.02


def _unpack_weights(flat):
    idx = 0
    def take(shape):
        nonlocal idx
        size = 1
        for s in shape:
            size *= s
        w = flat[idx:idx + size].reshape(shape)
        idx += size
        return w
    W1 = take((INPUT_DIM, HIDDEN1))
    b1 = take((HIDDEN1,))
    W2 = take((HIDDEN1, HIDDEN2))
    b2 = take((HIDDEN2,))
    W3 = take((HIDDEN2, OUTPUT_DIM))
    b3 = take((OUTPUT_DIM,))
    return W1, b1, W2, b2, W3, b3


def _forward(features, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(features @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3


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


def _load_once():
    global _weights
    if _weights is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. "
            "Train with curriculum_cma and copy the output weights here."
        )
    _weights = torch.load(wpath, map_location="cpu", weights_only=True).numpy()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck
    _load_once()

    _time_since_seen  = 0 if np.any(obs[:17] > 0) else _time_since_seen  + 1
    _time_since_stuck = 0 if obs[17] > 0           else _time_since_stuck + 1

    last_fw  = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    features = _extract_features(obs, _time_since_seen, _time_since_stuck, last_fw)
    logits   = _forward(features, *_unpack_weights(_weights))
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
