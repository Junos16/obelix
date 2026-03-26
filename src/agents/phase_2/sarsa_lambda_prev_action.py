"""
MODIFICATIONS:
- Base Algorithm: SARSA-Lambda.
- State Augmentation: 18-bit sensor + Previous Action mapping to 1,310,720 integer states.
- POMDP Fix: Traces propagate rewards back through history to bridge the gap over blinking boxes.
- Trace Management: Dutch Traces updated via active trace sets for fast tabular convergence.
"""
from __future__ import annotations
import random
import os
import json
import time
import numpy as np
import torch
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX = None
_Q_TABLE = None
STATE_SPACE_SIZE = (2**18) * 5

_repeat_count: int = 0
_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05
_last_action: Optional[int] = None

class SarsaLambdaAgent:
    def __init__(self, n_actions=5):
        self.q_table = np.zeros((STATE_SPACE_SIZE, n_actions), dtype=np.float32)
        self.e_table = np.zeros((STATE_SPACE_SIZE, n_actions), dtype=np.float32)

    def reset_traces(self):
        self.e_table.fill(0.0)

def obs_to_state(obs: np.ndarray, last_action: int) -> int:
    base_state = int(np.sum(2**np.where(obs > 0)[0]))
    return base_state * 5 + last_action

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False, prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _Q_TABLE
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _Q_TABLE = None
    
    print("Training sarsa_lambda_prev_action for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    config = {
        "gamma": 0.99,
        "lambda_": 0.90,
        "alpha": 0.1,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_episodes": max(1, int(episodes * 0.8)),
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config.update(json.load(f))

    gamma = config["gamma"]
    lambda_ = config["lambda_"]
    alpha = config["alpha"]
    seed = config["seed"]
    
    random.seed(seed)
    np.random.seed(seed)    
    
    agent = SarsaLambdaAgent()

    def get_epsilon_greedy_action(stateID: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return random.choice(range(len(ACTIONS)))
        else:
            return int(np.argmax(agent.q_table[stateID]))
    
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
        last_action = 2 # Best guess for start is Forward
        stateID = obs_to_state(obs, last_action)
        epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
        action = get_epsilon_greedy_action(stateID, epsilon)

        agent.reset_traces()
        episode_return = 0.0
        q_old = agent.q_table[stateID, action]
        active_traces = set()

        t_start = time.time()
        for step in range(config["max_steps"]):
            # For next state mapping based on current logic

            next_obs, reward, done = env.step(ACTIONS[action], render=render)
            
            next_stateID = obs_to_state(next_obs, action)
            next_epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
            next_action = get_epsilon_greedy_action(next_stateID, next_epsilon)
            
            q_curr = agent.q_table[stateID, action]
            q_next = agent.q_table[next_stateID, next_action] if not done else 0.0
            delta = reward + gamma * q_next - q_curr
            
            agent.e_table[stateID, action] += (1.0 - alpha * gamma * lambda_ * agent.e_table[stateID, action])
            active_traces.add((stateID, action))

            traces_to_remove = []
            for s, a in active_traces:
                agent.q_table[s, a] += alpha * (delta + q_curr - q_old) * agent.e_table[s, a]
                if s == stateID and a == action:
                    agent.q_table[s, a] -= alpha * (q_curr - q_old)
                
                agent.e_table[s, a] *= gamma * lambda_
                if agent.e_table[s, a] < 1e-4:
                    agent.e_table[s, a] = 0.0
                    traces_to_remove.append((s, a))
            
            obs = next_obs
            last_action = action
            stateID = next_stateID
            action = next_action
            episode_return += reward
            if done:
                break
        
        duration = time.time() - t_start
        print(f"Episode {episode+1}/{episodes} return={episode_return:.1f} eps={epsilon:.3f} ({duration:.2f}s)")
        
        if trial is not None:
            trial.report(episode_return, episode)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    os.makedirs("models", exist_ok=True)
    base_name = f"{prefix}" if prefix else f"sarsa_lambda_prev_action_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_trial_{trial.number}_weights.pth" if trial is not None else f"models/{base_name}_weights.pth"
    
    _Q_TABLE = agent.q_table
    torch.save(torch.from_numpy(agent.q_table), out_path)
    print(f"Saved Q-table to {out_path}")

def _load_once():
    global _Q_TABLE
    if _Q_TABLE is None:
        base_name = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"sarsa_lambda_prev_action_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}"
        wpath = f"models/{base_name}_weights.pth"
        _Q_TABLE = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    return _Q_TABLE
    
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    _load_once()
    
    last_act = 2 if _last_action is None else _last_action
    stateID = obs_to_state(obs, last_act)

    q_vals = _Q_TABLE[stateID]
    order = np.argsort(-q_vals)
    best = int(order[0])

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

def get_optuna_params(trial, total_episodes):
    params = {}
    params["gamma"] = trial.suggest_float("gamma", 0.9, 1.0)
    params["lambda_"] = trial.suggest_float("lambda_", 0.5, 0.99)
    params["alpha"] = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    eps_fraction = trial.suggest_float("eps_decay_fraction", 0.4, 0.9)
    params["eps_decay_episodes"] = max(1, int(total_episodes * eps_fraction))
    return params
