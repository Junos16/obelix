"""
MODIFICATIONS:
- Tabular State Encoding: 18-bit sensor mapping to 2^18 integer states.
- Zero Initialization: Q-table initialized to 0.0 to prevent exploration oscillation.
- Prioritized Sweeping: TD-error based planning using a max-priority queue.
- Predecessor Modeling: Reverse mapping of state transitions for optimized sweeps.
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
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX = None
_Q_TABLE = None
STATE_SPACE_SIZE = 2**18

class DynaQAgent:
    def __init__(self, n_actions=5):
        self.q_table = np.zeros((STATE_SPACE_SIZE, n_actions), dtype=np.float32)
        self.model = {}
        self.predecessors = {} 

def obs_to_state(obs: np.ndarray) -> int:
    return np.sum(2**np.where(obs > 0)[0])

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False, prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _Q_TABLE
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _Q_TABLE = None
    
    print("Training Dyna-Q agent for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    # Default hyperparameters
    config = {
        "gamma": 0.99,
        "alpha": 0.1, 
        "planning_steps": 50,     
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
    alpha = config["alpha"]
    planning_steps = config["planning_steps"]
    seed = config["seed"]
    
    random.seed(seed)
    np.random.seed(seed)    
    
    agent = DynaQAgent(n_actions=len(ACTIONS))

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
        stateID = obs_to_state(obs)
        epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
        
        episode_return = 0.0

        import heapq
        theta = 1e-4

        t_start = time.time()
        for _ in range(config["max_steps"]):
            action = get_epsilon_greedy_action(stateID, epsilon)
            next_obs, reward, done = env.step(ACTIONS[action], render=render)
                
            next_stateID = obs_to_state(next_obs)
            
            td_error = reward + (gamma * np.max(agent.q_table[next_stateID]) if not done else 0.0) - agent.q_table[stateID, action]
            agent.q_table[stateID, action] += alpha * td_error
            
            if stateID not in agent.model:
                agent.model[stateID] = {}
            agent.model[stateID][action] = (reward, next_stateID, done)
            
            if next_stateID not in agent.predecessors:
                agent.predecessors[next_stateID] = []
            if (stateID, action, reward) not in agent.predecessors[next_stateID]:
                agent.predecessors[next_stateID].append((stateID, action, reward))

            queue = []
            if abs(td_error) > theta:
                heapq.heappush(queue, (-abs(td_error), stateID, action))

            steps = 0
            while queue and steps < planning_steps:
                _, s, a = heapq.heappop(queue)
                r_sim, s_prime_sim, done_sim = agent.model[s][a]
                
                td_sim = r_sim + (gamma * np.max(agent.q_table[s_prime_sim]) if not done_sim else 0.0) - agent.q_table[s, a]
                agent.q_table[s, a] += alpha * td_sim
                
                if s in agent.predecessors:
                    for s_pre, a_pre, r_pre in agent.predecessors[s]:
                        td_pre = r_pre + gamma * np.max(agent.q_table[s]) - agent.q_table[s_pre, a_pre]
                        if abs(td_pre) > theta:
                            heapq.heappush(queue, (-abs(td_pre), s_pre, a_pre))
                steps += 1
            
            stateID = next_stateID
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
    base_name = f"{prefix}" if prefix else f"dyna_q_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_weights.pth"
    torch.save(torch.from_numpy(agent.q_table), out_path)
    print(f"Saved Q-table to {out_path}")

_Q_TABLE = None

def _load_once():
    global _Q_TABLE
    if _Q_TABLE is None:
        base_name = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"dyna_q_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}"
        wpath = f"models/{base_name}_weights.pth"
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
    params["planning_steps"] = trial.suggest_categorical("planning_steps", [10, 50, 100, 250])
    eps_fraction = trial.suggest_float("eps_decay_fraction", 0.4, 0.9)
    params["eps_decay_episodes"] = max(1, int(total_episodes * eps_fraction))
    return params