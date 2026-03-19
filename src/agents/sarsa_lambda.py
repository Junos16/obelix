from __future__ import annotations
import random
import numpy as np

from src.obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45]
STATE_SPACE_SIZE = 2**18

class SarsaLambdaAgent:
    def __init__(self, n_actions=5):
        self.q_table = np.zeroes((STATE_SPACE_SIZE, n_actions), dtype=np.float32)
        self.e_table = np.zeroes((STATE_SPACE_SIZE, n_actions), dtype=np.float32)

    def reset_traces(self):
        self.e_table.fill(0.0)

def obs_to_state(obs: np.ndarray) -> int:
    return np.sum(2**np.where(obs > 0)[0])

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None):
    
