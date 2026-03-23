from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
STATE_SPACE_SIZE = 2**18

def obs_to_state(obs: np.ndarray, last_action: int) -> int:
    base_state = int(np.sum(2**np.where(obs > 0)[0]))
    return base_state * 5 + last_action

_Q_TABLE: Optional[np.ndarray] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

def _load_once():
    global _Q_TABLE
    if _Q_TABLE is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    _Q_TABLE = torch.load(wpath, map_location="cpu", weights_only=True).numpy()

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    _load_once()
    
    last_act = 2 if _last_action is None else _last_action
    stateID = obs_to_state(obs, last_act)
    q = _Q_TABLE[stateID]
    best = int(np.argmax(q))

    if _last_action is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
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
