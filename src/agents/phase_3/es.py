"""
Agent Training Techniques:
1. Pure PyTorch Evolutionary Strategy (ES): Instead of backpropagation (which fails on sparse, long-horizon rewards), 
   we evolve a population of PyTorch MLPs. We evaluate their fitness, select the top 'elites', and add 
   Gaussian noise to their weights to breed the next generation.
2. Frame Stacking for Memory: To handle the POMDP 'blinking box' at higher difficulties, the agent receives 
   the last 4 sensor observations concatenated together (72 inputs). This provides implicit velocity and 
   trajectory tracking without needing complex RNN hidden state management.
3. Omniscient Fitness Shaping: We bypass the sparse environment rewards by extracting absolute coordinates 
   (`env.bot_center_x`, `env.box_center_x`, etc.) directly from the environment during training to create a 
   dense, smooth fitness gradient.
"""

from __future__ import annotations
import os
import json
import time
import math
import copy
import torch
import torch.nn as nn
import numpy as np
import optuna

from obelix import OBELIX

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX = None

_eval_model: Optional[FrameStackAgent] = None
_eval_obs_stack: List[np.ndarray] = []

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

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

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

def mutate_model(model: nn.Module, mutation_power: float) -> nn.Module:
    """Creates a mutated copy of the given model."""
    child = copy.deepcopy(model)

    with torch.no_grad():
        for param in child.parameters():
            # Add Gaussian noise to weights
            noise = torch.randn_like(param) * mutation_power
            param.add_(noise)

    return child

def evaluate_fitness(model, env_config, seed):
    """Evaluates a single model using Omniscient Fitness Shaping."""
    env = OBELIX(
        scaling_factor=env_config["scaling_factor"],
        arena_size=env_config["arena_size"],
        max_steps=env_config["max_steps"],
        wall_obstacles=env_config["wall_obstacles"],
        difficulty=env_config["difficulty"],
        box_speed=env_config["box_speed"],
        seed=seed
    )

    obs = env.reset()

    # Initialize frame stack
    obs_stack = [np.zeros_like(obs) for _ in range(3)] + [obs]
    
    fitness = 0.0
    min_box_dist = float('inf')
    min_bound_dist = float('inf')    

    model.eval()
    with torch.no_grad():
        for _ in range(env_config["max_steps"]):
            # Flatten stack and predict
            flat_obs = np.concatenate(obs_stack).astype(np.float32)
            tensor_obs = torch.from_numpy(flat_obs).unsqueeze(0)
            logits = model(tensor_obs).squeeze(0)
            action_idx = int(torch.argmax(logits))

            next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)

            # Update stack
            obs_stack.pop(0)
            obs_stack.append(next_obs)

            # --- OMNISCIENT FITNESS SHAPING ---
            fitness += reward  # Keep base reward

            if not env.enable_push:
                # Phase 1: Minimize distance between bot and box
                dist = math.hypot(env.bot_center_x - env.box_center_x, env.bot_center_y - env.box_center_y)
                if dist < min_box_dist:
                    fitness += 1.0  # Dense reward
                    min_box_dist = dist
            else:
                # Phase 2: Push box to edge
                fitness += 2.0  # Reward for being attached

                # Distance to nearest boundary (inner boundary offset is 10)
                bx, by = env.box_center_x, env.box_center_y
                width, height = env.frame_size[1], env.frame_size[0]
                dist_x = min(bx - 10, width - 10 - bx)
                dist_y = min(by - 10, height - 10 - by)
                bound_dist = min(dist_x, dist_y)

                if bound_dist < min_bound_dist:
                    fitness += 2.0
                    min_bound_dist = bound_dist
        
            if env.stuck_flag == 1:
                fitness -= 10.0

            if done:
                if env.enable_push and env._box_touches_boundary(env.box_center_x, env.box_center_y):
                    fitness += 1000.0  # Success completion bonus
                break

    return fitness

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False, prefix: str = None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX
    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix

    print(f"Training ES Agent for level {level} with wall obstacles {wall_obstacles} for {episodes} evaluations")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3

    config = {
        "pop_size": 30,
        "elite_fraction": 0.2,
        "mutation_power": 0.1,
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config.update(json.load(f))

    env_config = {
        "scaling_factor": config["scaling_factor"],
        "arena_size": config["arena_size"],
        "max_steps": config["max_steps"],
        "wall_obstacles": wall_obstacles,
        "difficulty": difficulty,
        "box_speed": config["box_speed"]
    }

    generations = max(1, episodes // config["pop_size"])
    n_elites = max(1, int(config["pop_size"] * config["elite_fraction"]))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Initialize Population
    population = [FrameStackAgent() for _ in range(config["pop_size"])]
    best_overall_model = None
    best_overall_fitness = -float('inf')

    for gen in range(generations):
        t_start = time.time()
        fitness_scores = []

        # Evaluate generation
        for i, model in enumerate(population):
            eval_seed = config["seed"] + gen * config["pop_size"] + i
            fit = evaluate_fitness(model, env_config, eval_seed)
            fitness_scores.append((fit, model))

        # Sort descending
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in fitness_scores[:n_elites]]
        gen_best_fit = fitness_scores[0][0]
        gen_avg_fit = sum(x[0] for x in fitness_scores) / len(fitness_scores)

        if gen_best_fit > best_overall_fitness:
            best_overall_fitness = gen_best_fit
            best_overall_model = copy.deepcopy(elites[0])

        duration = time.time() - t_start
        print(f"Gen {gen+1}/{generations} | Best: {gen_best_fit:.1f} | Avg: {gen_avg_fit:.1f} | {duration:.2f}s")

        if trial is not None:
            trial.report(gen_best_fit, gen)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # Breed next generation
        next_population = copy.deepcopy(elites) # Elitism: carry over the best

        while len(next_population) < config["pop_size"]:
            # Pick a random elite to mutate
            parent = elites[np.random.randint(0, len(elites))]
            child = mutate_model(parent, config["mutation_power"])
            next_population.append(child)

        population = next_population

    os.makedirs("models", exist_ok=True)
    base_name = f"{prefix}" if prefix else f"es_agent_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_trial_{trial.number}_weights.pth" if trial is not None else f"models/{base_name}_weights.pth"

    # Save PyTorch state dict
    torch.save(best_overall_model.state_dict(), out_path)
    print(f"Saved ES model weights to {out_path}")

def get_optuna_params(trial, total_episodes):
    params = {}
    params["pop_size"] = trial.suggest_categorical("pop_size", [20, 30, 50])
    params["elite_fraction"] = trial.suggest_float("elite_fraction", 0.1, 0.3)
    params["mutation_power"] = trial.suggest_float("mutation_power", 0.01, 0.5, log=True)
    return params

def _load_once():
    global _eval_model
    if _eval_model is not None:
        return

    base_name = f"{_CURRENT_PREFIX}" if _CURRENT_PREFIX else f"es_agent_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}"
    wpath = f"models/{base_name}_weights.pth"
    if not os.path.exists(wpath):
        print(f"Warning: {wpath} not found. Returning random actions.")
        return

    m = FrameStackAgent()
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    _eval_model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _eval_obs_stack, _eval_model
    _load_once()

    if _eval_model is None:
        return ACTIONS[int(rng.integers(0, len(ACTIONS)))]

    # Initialize stack on first step or reset
    if len(_eval_obs_stack) == 0 or (np.all(obs == 0) and not np.all(_eval_obs_stack[-1] == 0)):
        _eval_obs_stack = [np.zeros_like(obs) for _ in range(3)] + [obs.copy()]
    
    else:
        _eval_obs_stack.pop(0)
        _eval_obs_stack.append(obs.copy())

    flat_obs = np.concatenate(_eval_obs_stack).astype(np.float32)
    tensor_obs = torch.from_numpy(flat_obs).unsqueeze(0)

    logits = _eval_model(tensor_obs).squeeze(0).numpy()
    best_action = int(np.argmax(logits))

    return ACTIONS[best_action] 