"""
MODIFICATIONS:
- Base Algorithm: Dyna-Q with Prioritized Sweeping.
- State Augmentation: 18-bit sensor + Target Lock for last 30 steps
- POMDP Fix: Stochastic Model explicitly tracks transition probabilities (counts) to handle the blinking box without corrupting hallucinations.
- Planning Phase: Updates Q-values using the Expected Value of all stochastic outcomes.
"""
from __future__ import annotations
import os
from typing import Optional
import random
import json
import time
import heapq
import numpy as np
import torch
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX = None
_Q_TABLE = None
STATE_SPACE_SIZE = (2**18) * 2 # 18 bits + 1 target lock boolean

class StochDynaQAgent:
    def __init__(self, n_actions=5):
        self.q_table = np.zeros((STATE_SPACE_SIZE, n_actions), dtype=np.float32)
        self.model = {}
        self.predecessors = {} 

def obs_to_state(obs: np.ndarray, target_lock: int) -> int:
    base_state = int(np.sum(2**np.where(obs > 0)[0]))
    return base_state * 2 + target_lock

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False, prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX, _Q_TABLE
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix
    _Q_TABLE = None
    
    print("Training Stochastic Dyna-Q for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
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
    
    agent = StochDynaQAgent(n_actions=len(ACTIONS))

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
        lock_timer = 0
        epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
        
        episode_return = 0.0
        theta = 1e-4
        t_start = time.time()

        for _ in range(config["max_steps"]):
            if np.any(obs[:17] > 0): lock_timer = 30
            target_lock = 1 if lock_timer > 0 else 0
            stateID = obs_to_state(obs, target_lock)

            action = get_epsilon_greedy_action(stateID, epsilon)
            next_obs, reward, done = env.step(ACTIONS[action], render=render)
                
            next_lock_timer = 30 if np.any(next_obs[:17] > 0) else max(0, lock_timer - 1)
            next_target_lock = 1 if next_lock_timer > 0 else 0
            next_stateID = obs_to_state(next_obs, next_target_lock)
            
            # 1. Real Experience Update (Standard Q-Learning)
            td_error = reward + (gamma * np.max(agent.q_table[next_stateID]) if not done else 0.0) - agent.q_table[stateID, action]
            agent.q_table[stateID, action] += alpha * td_error
            
            # 2. Stochastic Model Update (Probability Counter)
            if stateID not in agent.model: 
                agent.model[stateID] = {}
            if action not in agent.model[stateID]: 
                agent.model[stateID][action] = {}
            if next_stateID not in agent.model[stateID][action]:
                agent.model[stateID][action][next_stateID] = {'r': reward, 'done': done, 'count': 0}
            agent.model[stateID][action][next_stateID]['count'] += 1
            
            # 3. Predecessor Update (For Prioritized Sweeping)
            if next_stateID not in agent.predecessors:
                agent.predecessors[next_stateID] = set()
            agent.predecessors[next_stateID].add((stateID, action))

            queue = []
            if abs(td_error) > theta:
                heapq.heappush(queue, (-abs(td_error), stateID, action))

            # 4. Expected Value Planning Phase
            steps = 0
            while queue and steps < planning_steps:
                _, s, a = heapq.heappop(queue)
                
                transitions = agent.model[s][a]
                total_count = sum(t['count'] for t in transitions.values())
                expected_target = 0.0
                
                # Calculate Expected Value across all observed outcomes
                for s_prime, data in transitions.items():
                    prob = data['count'] / total_count
                    val = data['r'] + (gamma * np.max(agent.q_table[s_prime]) if not data['done'] else 0.0)
                    expected_target += prob * val
                    
                td_sim = expected_target - agent.q_table[s, a]
                agent.q_table[s, a] += alpha * td_sim
                
                # Push predecessors to the queue
                if s in agent.predecessors:
                    for s_pre, a_pre in agent.predecessors[s]:
                        pre_transitions = agent.model[s_pre][a_pre]
                        pre_total = sum(t['count'] for t in pre_transitions.values())
                        pre_exp = 0.0
                        for pre_s_prime, pre_data in pre_transitions.items():
                            pre_prob = pre_data['count'] / pre_total
                            pre_val = pre_data['r'] + (gamma * np.max(agent.q_table[pre_s_prime]) if not pre_data['done'] else 0.0)
                            pre_exp += pre_prob * pre_val
                        pre_td = pre_exp - agent.q_table[s_pre, a_pre]
                        if abs(pre_td) > theta:
                            heapq.heappush(queue, (-abs(pre_td), s_pre, a_pre))
                steps += 1
            
            obs = next_obs
            lock_timer = next_lock_timer
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
    base_name = f"{prefix}" if prefix else f"stoch_dyna_q_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_trial_{trial.number}_weights.pth" if trial is not None else f"models/{base_name}_weights.pth"
    
    _Q_TABLE = agent.q_table
    # Clean, strict PyTorch Tensor save
    torch.save(torch.from_numpy(agent.q_table), out_path)
    print(f"Saved Q-table to {out_path}")

_last_action: Optional[int] = None
_lock_timer: int = 0

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
    global _last_action, _repeat_count, _lock_timer
    _load_once()
    
    if np.any(obs[:17] > 0):
        _lock_timer = 30
    else:
        _lock_timer = max(0, _lock_timer - 1)
        
    target_lock = 1 if _lock_timer > 0 else 0
    stateID = obs_to_state(obs, target_lock)
    
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

def get_optuna_params(trial, total_episodes):
    params = {}
    params["gamma"] = trial.suggest_float("gamma", 0.9, 1.0)
    params["alpha"] = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    params["planning_steps"] = trial.suggest_categorical("planning_steps", [10, 50, 100])
    eps_fraction = trial.suggest_float("eps_decay_fraction", 0.4, 0.9)
    params["eps_decay_episodes"] = max(1, int(total_episodes * eps_fraction))
    return params