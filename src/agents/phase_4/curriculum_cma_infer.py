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
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

INPUT_DIM  = 26
HIDDEN1    = 64
HIDDEN2    = 32
OUTPUT_DIM = 5

# Inference state
_MODEL:           Optional[nn.Module] = None
_last_action:     Optional[int] = None
_repeat_count:    int = 0
_time_since_seen: int = 100
_time_since_stuck: int = 100

_MAX_REPEAT    = 3
_CLOSE_Q_DELTA = 0.02


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
    global _MODEL
    if _MODEL is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. "
            "Train with curriculum_cma and copy the output weights here."
        )
    flat = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    _MODEL = _flat_to_model(flat)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck
    _load_once()

    _time_since_seen  = 0 if np.any(obs[:17] > 0) else _time_since_seen  + 1
    _time_since_stuck = 0 if obs[17] > 0           else _time_since_stuck + 1

    last_fw  = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    features = _extract_features(obs, _time_since_seen, _time_since_stuck, last_fw)

    with torch.no_grad():
        logits = _MODEL(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()

    order = np.argsort(-logits)
    best  = int(order[0])

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
