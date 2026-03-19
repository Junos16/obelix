from __future__ import annotations
import random
import numpy as np
import torch

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
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

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None):
    print("Training SARSA-Lambda agent for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    # Default hyperparameters
    config = {
        "gamma": 0.99,
        "lambda_": 0.90,  # Trace decay rate
        "alpha": 0.1,     # Learning rate
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_episodes": int(episodes * 0.8),
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

        for _ in range(config["max_steps"]):
            next_obs, reward, done = env.step(ACTIONS[action])
            next_stateID = obs_to_state(next_obs)
            next_epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
            next_action = get_epsilon_greedy_action(next_stateID, next_epsilon)
            
            td_error = reward + gamma * agent.q_table[next_stateID, next_action] - agent.q_table[stateID, action]
            agent.e_table[stateID, action] = 1

            active_s, active_a = np.nonzero(agent.e_table)   
                     
            agent.q_table[active_s, active_a] += alpha * td_error * agent.e_table[active_s, active_a]
            agent.e_table[active_s, active_a] *= gamma * lambda_

            # agent.e_table[agent.e_table < 1e-4] = 0.0
            
            stateID = next_stateID
            action = next_action
            episode_return += reward
            
            if done:
                td_error = reward - agent.q_table[stateID, action]
            else:
                td_error = reward + gamma * agent.q_table[next_stateID, next_action] - agent.q_table[stateID, action]
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes} return={episode_return:.1f} eps={epsilon:.3f} replay={len(replay)}")
    
    os.makedirs("models", exist_ok=True)
    out_path = f"models/sarsa_lambda_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
    torch.save(torch.from_numpy(agent.q_table), out_path)
    print(f"Saved Q-table to {out_path}")

_Q_TABLE = None

def _load_once(level: int, wall_obstacles: bool):
    global _Q_TABLE
    if _Q_TABLE is None:
        wpath = f"models/sarsa_lambda_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
        _Q_TABLE = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    return _Q_TABLE
    
def policy(obs: np.ndarray, rng: np.random.Generator, level: int=1, wall_obstacles: bool=False) -> str:
    _load_once(level, wall_obstacles)
    stateID = obs_to_state(obs)
    best_action_idx = int(np.argmax(_Q_TABLE[stateID]))
    return ACTIONS[best_action_idx]

    