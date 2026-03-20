import os
import json
import optuna

# Disable Optuna's spammy terminal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

from evaluate import evaluate_agent

def run_sweep(agent_name, agent_mod, get_params_fn, level, wall_obstacles, episodes, n_trials):
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
            config_file=config_path
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

    db_path = f"sqlite:///models/{agent_name}_sweep.db"
    study = optuna.create_study(
        study_name=f"{agent_name}_level{level}", 
        storage=db_path, 
        load_if_exists=True,
        direction="maximize"
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Save best hyperparameters to a JSON file (matching the .pth naming convention)
    wall_suffix = "_wall" if wall_obstacles else ""
    out_path = f"models/{agent_name}_level{level}{wall_suffix}_best_params.json"
    
    with open(out_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)