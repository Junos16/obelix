"""
MODIFICATIONS:
- Tabular State Encoding: 18-bit sensor mapping to 2^18 integer states.
- Optimistic Initialization: Q-table initialized to +10 to encourage exploration.
- Trace Management: Active trace set tracking to optimize updates.
- True Online Update: Dutch traces for more stable and faster convergence.
- Reward Shaping: Wandering penalty (-2.0), Intensity bonus (+5.0), Forward momentum (+0.5), and Anti-rotation penalty (-0.5).
"""
from __future__ import annotations
import random
import os
import json
import numpy as np
import torch

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_Q_TABLE = None
STATE_SPACE_SIZE = 2**18

class SarsaLambdaAgent:
    def __init__(self, n_actions=5):
        # self.q_table = np.zeroes((STATE_SPACE_SIZE, n_actions), dtype=np.float32)
        self.q_table = np.ones((STATE_SPACE_SIZE, n_actions), dtype=np.float32) * 10
        self.e_table = np.zeros((STATE_SPACE_SIZE, n_actions), dtype=np.float32)

    def reset_traces(self):
        self.e_table.fill(0.0)

def obs_to_state(obs: np.ndarray) -> int:
    return np.sum(2**np.where(obs > 0)[0])

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False):
    global _CURRENT_LEVEL, _CURRENT_WALL, _Q_TABLE
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _Q_TABLE = None
    
    print("Training SARSA-Lambda agent for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    # Default hyperparameters
    config = {
        "gamma": 0.99,
        "lambda_": 0.90,  # Trace decay rate
        "alpha": 0.1,     # Learning rate
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
            return np.argmax(agent.q_table[stateID])
    
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
        stateID = obs_to_state(obs)
        epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
        action = get_epsilon_greedy_action(stateID, epsilon)

        agent.reset_traces()
        episode_return = 0.0
        q_old = agent.q_table[stateID, action]
        active_traces = set()

        for _ in range(config["max_steps"]):
            next_obs, reward, done = env.step(ACTIONS[action], render=render)
            
            # Wandering Penalty: penalize seeing nothing
            if np.sum(next_obs[:17]) == 0:
                reward -= 2.0
            
            # Intensity Bonus: reward getting closer to objects
            if np.sum(next_obs[:17]) > np.sum(obs[:17]):
                reward += 5.0
            
            # Forward Momentum: reward moving forward when something is seen
            if action == 2 and np.any(obs[4:12] > 0):
                reward += 0.5
            
            # Anti-Rotation: penalize spinning when already near target
            if action != 2 and np.any(obs[1:16:2] > 0):
                reward -= 0.5
                
            next_stateID = obs_to_state(next_obs)
            next_epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
            next_action = get_epsilon_greedy_action(next_stateID, next_epsilon)
            
            q_curr = agent.q_table[stateID, action]
            q_next = agent.q_table[next_stateID, next_action] if not done else 0.0
            delta = reward + gamma * q_next - q_curr
            
            # Dutch trace update for tabular case
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
            
            for s, a in traces_to_remove:
                active_traces.remove((s, a))
            
            q_old = q_next
            stateID = next_stateID
            action = next_action
            episode_return += reward
            
            if done:
                break

        print(f"Episode {episode+1}/{episodes} return={episode_return:.1f} eps={epsilon:.3f}")
    
    os.makedirs("models", exist_ok=True)
    out_path = f"models/sarsa_lambda_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
    torch.save(torch.from_numpy(agent.q_table), out_path)
    print(f"Saved Q-table to {out_path}")

_Q_TABLE = None

def _load_once():
    global _Q_TABLE
    if _Q_TABLE is None:
        wpath = f"models/sarsa_lambda_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}_weights.pth"
        _Q_TABLE = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    return _Q_TABLE
    
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    stateID = obs_to_state(obs)
    best_action_idx = int(np.argmax(_Q_TABLE[stateID]))
    return ACTIONS[best_action_idx]

def get_optuna_params(trial, total_episodes):
    params = {}
    params["gamma"] = trial.suggest_float("gamma", 0.9, 1.0)
    params["alpha"] = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    params["lambda_"] = trial.suggest_float("lambda_", 0.5, 0.99)
    eps_fraction = trial.suggest_float("eps_decay_fraction", 0.4, 0.9)
    params["eps_decay_episodes"] = max(1, int(total_episodes * eps_fraction))
    return params    
