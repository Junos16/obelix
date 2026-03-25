"""
MODIFICATIONS:
- Feature Engineering: 37 features (18-bit Obs + 18-bit Delta Obs + 1 Bias).
- Model Storage: Saves and loads both A (covariance) and b (reward) matrices.
"""
from __future__ import annotations
import os
import random
import json
import time
import numpy as np
import torch
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_FEATURES = 37 

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX = None
_LINUCB_STATE = None

class LinUCBAgent:
    def __init__(self, n_actions=5, n_features=18):
        self.n_actions = n_actions
        self.n_features = n_features
        
        self.A = np.zeros((n_actions, n_features, n_features), dtype=np.float32)
        for a in range(n_actions):
            self.A[a] = np.eye(n_features, dtype=np.float32)
            
        self.b = np.zeros((n_actions, n_features, 1), dtype=np.float32)

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False, prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _LINUCB_STATE
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _LINUCB_STATE = None
    
    print("Training LinUCB agent for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    # Default hyperparameters
    config = {
        "alpha": 1.5, 
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config.update(json.load(f))

    alpha = config["alpha"]
    seed = config["seed"]
    
    random.seed(seed)
    np.random.seed(seed)    
    
    # Bias feature is already included in N_FEATURES (18 base + 18 delta + 1 bias)
    agent = LinUCBAgent(n_actions=len(ACTIONS), n_features=N_FEATURES)
    
    for episode in range(episodes):
        env = OBELIX(
            scaling_factor=config["scaling_factor"],
            arena_size=config["arena_size"],
            max_steps=config["max_steps"],
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=config["box_speed"],
            seed=seed+episode
        )

        obs = env.reset(seed=seed+episode)
        prev_obs = obs.copy()
        episode_return = 0.0

        t_start = time.time()
        for _ in range(config["max_steps"]):
            delta_obs = (obs - prev_obs).astype(np.float32)
            x_t = np.concatenate([obs, delta_obs, [1.0]]).reshape(-1, 1).astype(np.float32)
            
            p = np.zeros(len(ACTIONS))

            for a in range(len(ACTIONS)):
                A_inv = np.linalg.inv(agent.A[a])
                theta_a = A_inv @ agent.b[a]
                expected_r = (theta_a.T @ x_t).item()
                explore = alpha * np.sqrt((x_t.T @ A_inv @ x_t).item())
                p[a] = expected_r + explore
            
            action = np.argmax(p)
            next_obs, reward, done = env.step(ACTIONS[action], render=render)
            
            agent.A[action] = agent.A[action] + (x_t @ x_t.T)
            agent.b[action] = agent.b[action] + (reward * x_t) 

            prev_obs = obs.copy()
            obs = next_obs
            episode_return += reward
            
            if done:
                break
        
        duration = time.time() - t_start
        print(f"Episode {episode+1}/{episodes} return={episode_return:.1f} ({duration:.2f}s)")
        
        if trial is not None:
            trial.report(episode_return, episode)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    os.makedirs("models", exist_ok=True)
    base_name = f"{prefix}" if prefix else f"linUCB_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_weights.pth"
    
    state_dict = {
        "A": torch.from_numpy(agent.A),
        "b": torch.from_numpy(agent.b)
    }
    torch.save(state_dict, out_path)
    print(f"Saved LinUCB matrices to {out_path}")

_LINUCB_STATE = None
_PREV_OBS_EVAL = None

def _load_once():
    global _LINUCB_STATE
    if _LINUCB_STATE is None:
        base_name = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"linUCB_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}"
        wpath = f"models/{base_name}_weights.pth"
        loaded = torch.load(wpath, map_location="cpu", weights_only=True)
        _LINUCB_STATE = {
            "A": loaded["A"].numpy(),
            "b": loaded["b"].numpy()
        }
    return _LINUCB_STATE
    
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _PREV_OBS_EVAL
    state = _load_once()
    
    if _PREV_OBS_EVAL is None or np.sum(np.abs(obs - _PREV_OBS_EVAL)) > 5:
        delta_obs = np.zeros_like(obs, dtype=np.float32)
    else:
        delta_obs = (obs - _PREV_OBS_EVAL).astype(np.float32)
    
    _PREV_OBS_EVAL = obs.copy()
    
    x_t = np.concatenate([obs, delta_obs, [1.0]]).reshape(-1, 1).astype(np.float32)
    expected_rewards = np.zeros(len(ACTIONS))

    for a in range(len(ACTIONS)):
        A_inv = np.linalg.inv(state["A"][a])
        theta_a = A_inv @ state["b"][a]
        expected_rewards[a] = (theta_a.T @ x_t).item()
    
    best_action_idx = int(np.argmax(expected_rewards))
    return ACTIONS[best_action_idx]

def get_optuna_params(trial, total_episodes):
    params = {}
    params["alpha"] = trial.suggest_float("alpha", 0.05, 5.0, log=True)
    return params