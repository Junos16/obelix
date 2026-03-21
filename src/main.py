import argparse
import importlib.util
import json
import os
import sys

from evaluate import evaluate_agent, append_leaderboard
from sweeper import run_sweep

def load_agent_module(agent_name: str):
    """Dynamically loads an agent module from src/agents/ for training"""
    agent_path = os.path.join(os.path.dirname(__file__), "agents", f"{agent_name}.py")
    if not os.path.exists(agent_path):
        print(f"Error: Agent file not found at {agent_path}")
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location(agent_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_submission_module(submission_dir: str):
    """Dynamically loads a submission module from submissions/<submission_dir>/agent.py for evaluation"""
    # Go up one level from src/ to the project root, then into submissions/
    root_dir = os.path.dirname(os.path.dirname(__file__))
    agent_path = os.path.join(root_dir, "submissions", submission_dir, "agent.py")
    
    if not os.path.exists(agent_path):
        print(f"Error: Submission agent file not found at {agent_path}")
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location(f"submission_{submission_dir}", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sweep_agent(args):
    agent_mod = load_agent_module(args.agent)
    
    if not hasattr(agent_mod, "train") or not hasattr(agent_mod, "policy"):
        print(f"Error: {args.agent}.py must define both 'train' and 'policy' functions.")
        sys.exit(1)
        
    if not hasattr(agent_mod, "get_optuna_params"):
        print(f"Error: {args.agent}.py must define 'get_optuna_params(trial, total_episodes)'.")
        sys.exit(1)
        
    run_sweep(
        agent_name=args.agent,
        agent_mod=agent_mod,
        get_params_fn=agent_mod.get_optuna_params, # Pass the function dynamically
        level=args.level,
        wall_obstacles=args.wall,
        episodes=args.episodes,
        n_trials=args.trials,
        render=args.render
    )

def train_agent(args):
    print(f"Training agent '{args.agent}' on level {args.level} with wall_obstacles={args.wall}")
    agent_mod = load_agent_module(args.agent)
    
    if hasattr(agent_mod, "train"):
        # We expect the agent module to have a train() function
        agent_mod.train(level=args.level, wall_obstacles=args.wall, episodes=args.episodes, config_file=args.config, render=args.render)
    else:
        print(f"Error: {args.agent}.py does not define a 'train' function.")
        print("Please implement 'def train(level, wall_obstacles, episodes):' in your agent file.")
        sys.exit(1)

def eval_agent(args):
    print(f"Evaluating submission '{args.submission}' on level {args.level} with wall_obstacles={args.wall}")
    agent_mod = load_submission_module(args.submission)
    
    if not hasattr(agent_mod, "policy"):
        print(f"Error: agent.py in {args.submission} must define a 'policy(obs, rng) -> str' function for evaluation.")
        sys.exit(1)
        
    policy_fn = getattr(agent_mod, "policy")
    
    result = evaluate_agent(
        policy_fn,
        agent_name=args.submission,
        runs=args.episodes,
        base_seed=0,  # Default seed as in evaluate.py
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        wall_obstacles=args.wall,
        difficulty=0 if args.level == 1 else 2 if args.level == 2 else 3,
        box_speed=2,
    )
    
    print(
        f"Result for {result.agent_name}: Mean Reward={result.mean_score:.3f} ± {result.std_score:.3f} "
        f"across {result.runs} runs."
    )
    # Automatically log to leaderboard
    leaderboard_csv = "leaderboard.csv"
    append_leaderboard(leaderboard_csv, result)
    print(f"Appended results to {leaderboard_csv}")
    
    # Also save individual evaluation result to JSON for consistency
    wall_suffix = "_wall" if args.wall else ""
    eval_json_path = f"models/{args.submission}_level{args.level}{wall_suffix}_eval_results.json"
    eval_data = {
        "agent": args.submission,
        "level": args.level,
        "wall": args.wall,
        "mean_reward": result.mean_score,
        "std_reward": result.std_score,
        "runs": result.runs
    }
    with open(eval_json_path, "w") as f:
        json.dump(eval_data, f, indent=4)
    print(f"Individual evaluation results saved to {eval_json_path}")

def main():
    parser = argparse.ArgumentParser(description="OBELIX RL Agent Trainer & Evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Optuna Sweep parser
    sweep_parser = subparsers.add_parser("sweep", help="Run Optuna hyperparameter sweep")
    sweep_parser.add_argument("--agent", type=str, required=True, help="Name of the agent file in src/agents/ (without .py)")
    sweep_parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1, help="Difficulty level")
    sweep_parser.add_argument("--wall", action="store_true", help="Enable the static wall obstacle")
    sweep_parser.add_argument("--episodes", type=int, default=300, help="Number of episodes per trial")
    sweep_parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials to run")
    sweep_parser.add_argument("--render", action="store_true", help="Render the environment during the sweep")

    # Training parser
    train_parser = subparsers.add_parser("train", help="Train an RL agent")
    train_parser.add_argument("--agent", type=str, required=True, help="Name of the agent file in src/agents/ (without .py)")
    train_parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1, help="Difficulty level")
    train_parser.add_argument("--wall", action="store_true", help="Enable the static wall obstacle")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    train_parser.add_argument("--config", type=str, default=None, help="Path to config JSON file (e.g., submissions/configs/...)")
    train_parser.add_argument("--render", action="store_true", help="Render the environment during training")

    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate a packaged submission")
    eval_parser.add_argument("--submission", type=str, required=True, help="Directory name of the submission in the submissions/ folder")
    eval_parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1, help="Difficulty level")
    eval_parser.add_argument("--wall", action="store_true", help="Enable the static wall obstacle")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of runs to average score over")

    args = parser.parse_args()

    if args.command == "sweep":
        sweep_agent(args)
    elif args.command == "train":
        train_agent(args)
    elif args.command == "eval":
        eval_agent(args)

if __name__ == "__main__":
    main()
