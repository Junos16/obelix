from __future__ import annotations
from typing import Optional
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

OBS_DIM    = 18
WIN_SIZE   = 8
ENC_IN     = OBS_DIM * WIN_SIZE   # 144
BELIEF_DIM = 10
POL_IN     = BELIEF_DIM + 3       # 13

_CLOSE_Q_DELTA = 0.02
_MAX_REPEAT    = 3

_enc:             Optional[nn.Module] = None
_qnet:            Optional[nn.Module] = None
_obs_window:      deque = deque(maxlen=WIN_SIZE)
_last_action:     Optional[int] = None
_repeat_count:    int = 0
_time_since_seen: int = 100
_time_since_stuck: int = 100


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENC_IN, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, BELIEF_DIM),
        )

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(POL_IN, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        return self.net(x)


def _load_once():
    global _enc, _qnet
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

    qnet = QNet()
    qnet.load_state_dict(ckpt["policy"])
    qnet.eval()
    _qnet = qnet


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen, _time_since_stuck

    _load_once()

    _time_since_seen  = 0 if np.any(obs[:17] > 0) else _time_since_seen  + 1
    _time_since_stuck = 0 if obs[17] > 0           else _time_since_stuck + 1

    if len(_obs_window) == 0:
        for _ in range(WIN_SIZE):
            _obs_window.append(np.zeros(OBS_DIM, dtype=np.float32))
    _obs_window.append(obs.astype(np.float32))

    window = torch.from_numpy(
        np.concatenate(list(_obs_window)).astype(np.float32)
    ).unsqueeze(0)
    belief = _enc(window)   # (1, 10)

    last_fw = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    extras  = torch.tensor([[
        min(1.0, _time_since_seen  / 50.0),
        min(1.0, _time_since_stuck / 20.0),
        last_fw,
    ]], dtype=torch.float32)

    pol_in = torch.cat([belief, extras], dim=1)   # (1, 13)
    q_vals = _qnet(pol_in).squeeze(0).numpy()
    order  = np.argsort(-q_vals)
    best   = int(order[0])

    if _last_action is not None:
        if (q_vals[order[0]] - q_vals[order[1]]) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]
