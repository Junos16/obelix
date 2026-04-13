from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# Inference globals
_model = None
_ts_seen: int = 100
_ts_stuck: int = 100
_last_action: Optional[int] = None


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


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


def _load_once():
    global _model
    if _model is None:
        here = os.path.dirname(__file__)
        wpath = os.path.join(here, "weights.pth")
        if not os.path.exists(wpath):
            raise FileNotFoundError(
                "weights.pth not found next to agent.py. "
                "Train and copy the output weights here."
            )
        _model = PolicyNet()
        _model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        _model.eval()


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _ts_seen, _ts_stuck, _last_action
    _load_once()

    _ts_seen = 0 if np.any(obs[:17] > 0) else _ts_seen + 1
    _ts_stuck = 0 if obs[17] > 0 else _ts_stuck + 1

    last_fw = 1.0 if (_last_action is not None and _last_action == 2) else 0.0
    feat = _extract_features(obs, _ts_seen, _ts_stuck, last_fw)
    feat_t = torch.from_numpy(feat).unsqueeze(0)

    logits = _model(feat_t).squeeze()
    act = int(torch.argmax(logits).item())
    _last_action = act
    return ACTIONS[act]
