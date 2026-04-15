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
_pol_model:       Optional[nn.Module] = None
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


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(POL_IN, POL_H1),
            nn.Tanh(),
            nn.Linear(POL_H1, POL_H2),
            nn.Tanh(),
            nn.Linear(POL_H2, POL_OUT),
        )

    def forward(self, x):
        return self.net(x)


def _flat_to_policy(flat: np.ndarray) -> PolicyNet:
    """Load a flat CMA-ES parameter vector into a PolicyNet (matches numpy packing order)."""
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


def _load_once():
    global _enc, _pol_model
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
    _pol_model = _flat_to_policy(ckpt["policy"].numpy())


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

    with torch.no_grad():
        logits = _pol_model(torch.from_numpy(pol_in).unsqueeze(0)).squeeze(0).numpy()
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
