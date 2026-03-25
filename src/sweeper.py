import os
import json
import optuna

# Disable Optuna's spammy terminal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

from evaluate import evaluate_agent

def run_sweep(agent_name, agent_mod, get_params_fn, level, wall_obstacles, episodes, n_trials, render=False, prefix=None):
    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions/configs/temp", exist_ok=True)

    def objective(trial):
        # Dynamically get params from the specific agent's function
        params = get_params_fn(trial, episodes)
        
        config_path = f"submissions/configs/temp/{agent_name}_trial_{trial.number}.json"
        with open(config_path, "w") as f:
            json.dump(params, f)
            
        # Train (pass trial for pruning support)
        agent_mod.train(
            level=level, 
            wall_obstacles=wall_obstacles, 
            episodes=episodes, 
            config_file=config_path,
            render=render,
            prefix=prefix,
            trial=trial
        )
        
        # Evaluate
        difficulty = 0 if level == 1 else 2 if level == 2 else 3
        
        eval_result = evaluate_agent(
            agent_policy=agent_mod.policy,
            agent_name=agent_name,
            runs=2,
            base_seed=trial.number * 42,
            scaling_factor=5,
            arena_size=500,
            max_steps=500,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2
        )
        
        if os.path.exists(config_path):
            os.remove(config_path)
            
        print(f"Trial {trial.number} finished. Mean Reward: {eval_result.mean_score:.2f}")
        return eval_result.mean_score

    wall_suffix = "_wall" if wall_obstacles else ""
    base_name = prefix if prefix else f"{agent_name}_level{level}{wall_suffix}"
    db_file = f"{base_name}_sweep.db"
    db_abs_path = os.path.abspath(os.path.join("models", db_file))
    db_path = f"sqlite:///{db_abs_path}"
    
    print(f"Using Optuna database at: {db_abs_path}")
    
    study = optuna.create_study(
        study_name=base_name, 
        storage=db_path, 
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    wall_suffix = "_wall" if wall_obstacles else ""
    
    # 1. Save the single best hyperparameters
    best_path = f"models/{base_name}_best_params.json"
    with open(best_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
        
    # 2. Save the top 4 hyperparameter configurations as individual files
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)
    
    for i, trial in enumerate(sorted_trials[:4]):
        config_path = f"models/{base_name}_config_{i + 1}.json"
        with open(config_path, "w") as f:
            json.dump(trial.params, f, indent=4)
        print(f"  Config #{i + 1} (trial {trial.number}, score {trial.value:.2f}) -> {config_path}")
        
    print(f"Sweep complete. Best params saved to {best_path}")
