from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def _extract_features(obs, ts_seen, ts_stuck, last_fw):
    f = np.zeros(26, dtype=np.float32)
    f[0:18] = obs.astype(np.float32)
    f[18] = float(np.sum(obs[4:12]))
    f[19] = float(np.sum(obs[0:4]))
    f[20] = float(np.sum(obs[12:16]))
    f[21] = float(obs[16])
    f[22] = float(obs[17])
    f[23] = min(1.0, ts_seen / 50.0)
    f[24] = min(1.0, ts_stuck / 20.0)
    f[25] = last_fw
    return f


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


_model: Optional[QNet] = None
_ts_seen = 100
_ts_stuck = 100
_last_action: Optional[int] = None
_repeat = 0


def _load_once():
    global _model
    if _model is None:
        wpath = os.path.join(os.path.dirname(__file__), "weights.pth")
        _model = QNet()
        _model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        _model.eval()


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _ts_seen, _ts_stuck, _last_action, _repeat
    _load_once()

    if np.any(obs[:17] > 0):
        _ts_seen = 0
    else:
        _ts_seen += 1
    if obs[17] > 0:
        _ts_stuck = 0
    else:
        _ts_stuck += 1

    last_fw = 1.0 if _last_action == 2 else 0.0
    feat = _extract_features(obs, _ts_seen, _ts_stuck, last_fw)
    q = _model(torch.tensor(feat).unsqueeze(0)).squeeze(0)

    order = q.argsort(descending=True)
    best = int(order[0].item())

    if _last_action is not None:
        top1 = float(q[order[0]])
        top2 = float(q[order[1]])
        if top1 - top2 < 0.02:
            if _repeat < 3:
                best = _last_action
                _repeat += 1
            else:
                _repeat = 0
        else:
            _repeat = 0

    _last_action = best
    return ACTIONS[best]
