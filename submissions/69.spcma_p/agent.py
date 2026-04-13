from __future__ import annotations
from typing import Optional
from collections import deque
import os
import math
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

OBS_DIM    = 18
WIN_SIZE   = 8
ENC_IN     = OBS_DIM * WIN_SIZE   # 144
ENC_H1     = 64
ENC_H2     = 32
BELIEF_DIM = 10

POL_IN  = BELIEF_DIM + 3   # 13
POL_H1  = 32
POL_H2  = 16
POL_OUT = 5
N_PARAMS = POL_IN * POL_H1 + POL_H1 + POL_H1 * POL_H2 + POL_H2 + POL_H2 * POL_OUT + POL_OUT  # 1061

_CLOSE_Q_DELTA = 0.02
_MAX_REPEAT    = 3

_enc:             Optional[nn.Module] = None
_pol_weights:     Optional[np.ndarray] = None
_obs_window:      deque = deque(maxlen=WIN_SIZE)
_last_action:     Optional[int] = None
_repeat_count:    int = 0
_time_since_seen: int = 100
_time_since_stuck: int = 100


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENC_IN, ENC_H1),
            nn.ReLU(),
            nn.Linear(ENC_H1, ENC_H2),
            nn.ReLU(),
            nn.Linear(ENC_H2, BELIEF_DIM),
        )

    def forward(self, x):
        return self.net(x)


def _unpack_policy(flat):
    idx = 0
    def take(shape):
        nonlocal idx
        size = 1
        for s in shape: size *= s
        w = flat[idx:idx + size].reshape(shape)
        idx += size
        return w
    W1 = take((POL_IN, POL_H1))
    b1 = take((POL_H1,))
    W2 = take((POL_H1, POL_H2))
    b2 = take((POL_H2,))
    W3 = take((POL_H2, POL_OUT))
    b3 = take((POL_OUT,))
    return W1, b1, W2, b2, W3, b3


def _forward_policy(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3


def _load_once():
    global _enc, _pol_weights
    if _enc is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
    ckpt = torch.load(wpath, map_location="cpu", weights_only=True)
    enc = EncoderNet()
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    _enc = enc
    _pol_weights = ckpt["policy"].numpy()


@torch.no_grad()
def _encode(window: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
    return _enc(x).squeeze(0).numpy()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck

    _load_once()

    _time_since_seen  = 0 if np.any(obs[:17] > 0) else _time_since_seen  + 1
    _time_since_stuck = 0 if obs[17] > 0           else _time_since_stuck + 1

    if len(_obs_window) == 0:
        for _ in range(WIN_SIZE):
            _obs_window.append(np.zeros(OBS_DIM, dtype=np.float32))
    _obs_window.append(obs.astype(np.float32))

    window = np.concatenate(list(_obs_window))   # 144-dim
    belief = _encode(window)                      # 10-dim

    last_fw = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    pol_in  = np.concatenate([
        belief,
        [min(1.0, _time_since_seen  / 50.0),
         min(1.0, _time_since_stuck / 20.0),
         last_fw],
    ]).astype(np.float32)

    logits = _forward_policy(pol_in, *_unpack_policy(_pol_weights))
    order  = np.argsort(-logits)
    best   = int(order[0])

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
