# run_experiments.py
import json, argparse, itertools

# --- Import the new experiment worker script ---
from debias_experiment import run_experiment

def load_config(config_path):
    """Loads the JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main(config_path):
    """
    Runs a batch of Python experiments based on a config file.
    """
    config = load_config(config_path)

    # Get run parameters from config
    models = config.get("models_to_run", [])
    tasks = config.get("tasks_to_run", [])
    debias_methods = config.get("debias_methods_to_run", [])
    
    # Get shared parameters
    k_folds = config.get("k_folds", 5)
    output_path_base = config.get("output_path_base", "../output")
    local_path_setup = config.get("local_path_setup", True)
    
    # Build parameter grid
    param_grid = [
        {
            "MODEL_NAME": model_name,
            "TASK": task,
            "DEBIAS": debias
        }
        for model_name, task, debias in itertools.product(models, tasks, debias_methods)
    ]

    if not param_grid:
        print("Warning: Parameter grid is empty. No experiments to run.")
        print("Please check the '*_to_run' lists in your config file.")
        return

    print(f"--- Starting Experiment Runner ---")
    print(f"Found {len(param_grid)} experiments to run from '{config_path}'.")

    # Loop and execute experiments
    for i, params in enumerate(param_grid, 1):
        
        # --- Specific exclusion logic from original script ---
        if params["DEBIAS"] == "blind" and params["TASK"] == "stsb":
            print('='*50)
            print(f"SKIPPING ({i}/{len(param_grid)}): 'blind' on 'stsb' is not supported.")
            print('='*50)
            continue
        # --- End of specific logic ---

        print('='*50)
        print(f"Executing ({i}/{len(param_grid)}): {params['MODEL_NAME']}, {params['TASK']}, {params['DEBIAS']}")
        print('='*50)

        try:
            # --- THIS IS THE KEY CHANGE ---
            # Instead of papermill, we call our Python function
            run_experiment(
                model_name=params["MODEL_NAME"],
                task_name=params["TASK"],
                debias_method=params["DEBIAS"],
                k_folds=k_folds,
                output_path_base=output_path_base,
                local_path_setup=local_path_setup
            )
            # --- END OF KEY CHANGE ---
            
            print(f"--- SUCCESS: Experiment {i} finished. ---")
        
        except Exception as e:
            print(f"!!! ERROR: Experiment {i} FAILED !!!")
            print(f"  Parameters: {params}")
            print(f"  Error: {e}")
            print("---------------------------------")
            # Continue to the next experiment
            pass

    print("--- All experiments complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run debiasing experiments from a config file.")
    parser.add_argument(
        "-c", "--config", 
        default="config_runner.json", 
        help="Path to the JSON configuration file (default: config_runner.json)"
    )
    args = parser.parse_args()
    main(args.config)