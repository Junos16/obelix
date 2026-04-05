 from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class FrameStackAgent(nn.Module):
    def __init__(self, obs_dim=18, stack_size=4, n_actions=5):
        super().__init__()
        self.stack_size = stack_size
        self.in_dim = obs_dim * stack_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

_model: Optional[FrameStackAgent] = None
_obs_stack: List[np.ndarray] = []

def _load_once():
    global _model
    if _model is not None:
        return        

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
        
    m = FrameStackAgent()
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _obs_stack
    _load_once()

    # If the stack is empty (start of new eval episode), fill it with the initial observation
    if len(_obs_stack) == 0:
        _obs_stack = [np.zeros_like(obs) for _ in range(3)] + [obs.copy()]
    else:
        # Check if environment reset (drastic change in observation, or just rely on continuous pushing)
        # A simple heuristic to detect reset if needed:
        if np.all(obs == 0) and not np.all(_obs_stack[-1] == 0):
            _obs_stack = [np.zeros_like(obs) for _ in range(3)] + [obs.copy()]
        else:
            _obs_stack.pop(0)
            _obs_stack.append(obs.copy())

    flat_obs = np.concatenate(_obs_stack).astype(np.float32)
    tensor_obs = torch.from_numpy(flat_obs).unsqueeze(0)

    logits = _model(tensor_obs).squeeze(0).numpy()
    best_action = int(np.argmax(logits))

    return ACTIONS[best_action] 