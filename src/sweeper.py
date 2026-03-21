import os
import json
import optuna

# Disable Optuna's spammy terminal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

from evaluate import evaluate_agent

def run_sweep(agent_name, agent_mod, get_params_fn, level, wall_obstacles, episodes, n_trials, render=False):
    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions/configs/temp", exist_ok=True)

    def objective(trial):
        # Dynamically get params from the specific agent's function
        params = get_params_fn(trial, episodes)
        
        config_path = f"submissions/configs/temp/{agent_name}_trial_{trial.number}.json"
        with open(config_path, "w") as f:
            json.dump(params, f)
            
        # Train
        agent_mod.train(
            level=level, 
            wall_obstacles=wall_obstacles, 
            episodes=episodes, 
            config_file=config_path,
            render=render
        )
        
        # Evaluate
        difficulty = 0 if level == 1 else 2 if level == 2 else 3
        
        eval_result = evaluate_agent(
            agent_policy=agent_mod.policy,
            agent_name=f"{agent_name}_trial_{trial.number}",
            runs=5,
            base_seed=100, 
            scaling_factor=5,
            arena_size=500,
            max_steps=1000,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2
        )
        
        if os.path.exists(config_path):
            os.remove(config_path)
            
        return eval_result.mean_score

    wall_suffix = "_wall" if wall_obstacles else ""
    db_path = f"sqlite:///models/{agent_name}_level{level}{wall_suffix}_sweep.db"
    study = optuna.create_study(
        study_name=f"{agent_name}_level{level}", 
        storage=db_path, 
        load_if_exists=True,
        direction="maximize"
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    wall_suffix = "_wall" if wall_obstacles else ""
    
    # 1. Save the single best hyperparameters
    best_path = f"models/{agent_name}_level{level}{wall_suffix}_best_params.json"
    binary_best_path = f"models/{agent_name}_level{level}{wall_suffix}_best_params.json" # keeping same name for compatibility
    with open(best_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
        
    # 2. Save the top 4 hyperparameter configurations
    top_path = f"models/{agent_name}_level{level}{wall_suffix}_top4_params.json"
    
    # Get all completed trials, sorted by value (maximize)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)
    
    top_4_configs = []
    for i, trial in enumerate(sorted_trials[:4]):
        top_4_configs.append({
            "rank": i + 1,
            "trial_number": trial.number,
            "score": trial.value,
            "params": trial.params
        })
        
    with open(top_path, "w") as f:
        json.dump(top_4_configs, f, indent=4)
        
    print(f"Sweep complete. Best params saved to {best_path}")
    print(f"Top 4 configs saved to {top_path}")