from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
STATE_SPACE_SIZE = (2**18) * 4

def obs_to_state(obs: np.ndarray, decay_bin: int) -> int:
    base_state = int(np.sum(2**np.where(obs > 0)[0]))
    return base_state * 4 + decay_bin

_Q_TABLE: Optional[np.ndarray] = None
_last_action: Optional[int] = None
_time_since_seen: int = 100
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
        
    # Standard, highly secure PyTorch load
    _Q_TABLE = torch.load(wpath, map_location="cpu", weights_only=True).numpy()

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _time_since_seen
    _load_once()
    
    _time_since_seen = 0 if np.any(obs[:17] > 0) else _time_since_seen + 1
    decay_bin = 0 if _time_since_seen == 0 else (1 if _time_since_seen <= 10 else (2 if _time_since_seen <= 20 else 3))
    
    stateID = obs_to_state(obs, decay_bin)
    
    q_vals = _Q_TABLE[stateID]
    
    order = np.argsort(-q_vals)
    best = int(order[0])

    # Smoothing Logic (Anti-Oscillation)
    if _last_action is not None:
        best_q, second_q = float(q_vals[order[0]]), float(q_vals[order[1]])
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