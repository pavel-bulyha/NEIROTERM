# launcher.py

import sys
import pkgutil
import importlib.util
from pathlib import Path

from utils.registry import get_model, list_models

# Ensure that ROOT (NeuralNetworks) is first in the path
ROOT = Path(__file__).parent.resolve()
sys.path[0] = str(ROOT)

def import_network_modules():
    """
    Scan each subfolder in ROOT (except 'utils'),
    load all .py files as modules.
    """
    for sub in ROOT.iterdir():
        if not sub.is_dir() or sub.name == "utils" or sub.name.startswith("__"):
            continue
        for py in sub.glob("*.py"):
            if py.name == "__init__.py":
                continue
            mod_name = f"{sub.name}.{py.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, str(py))
            module = importlib.util.module_from_spec(spec) # type: ignore
            spec.loader.exec_module(module) # type: ignore

if __name__ == "__main__":
    import_network_modules()

    models = list_models()
    if not models:
        print("No models registered. Nothing to run.")
        sys.exit(0)

    # Display models with numbers
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    # Get user input
    user_input = input("Enter model numbers separated by ';' or 'all' to run all: ").strip()

    if user_input.lower() == "all":
        selected_models = models
    else:
        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(";")]
            selected_models = [models[i] for i in indices if 0 <= i < len(models)]
            if not selected_models:
                print("No valid models selected.")
                sys.exit(0)
        except ValueError:
            print("Invalid input. Please enter numbers separated by ';' or 'all'.")
            sys.exit(0)

    print(f"Selected models: {selected_models}")

    # 1) Folder with CSV files for training
    train_root = ROOT.parent / "TrainData"
    if not train_root.exists():
        print(f"Error: TrainData folder not found: {train_root}")
        sys.exit(1)

    # 2) Recursive search for all CSV files
    csv_files = list(train_root.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {train_root}")
        sys.exit(0)

    for data_file in csv_files:
        print(f"\n=== Data file: {data_file} ===")
        for model_name in selected_models:
            run_fn = get_model(model_name)
            for use_weighted in (False, True):
                mode = "weightedBCE" if use_weighted else "BCE"
                print(f"\n--- Running {model_name} [{mode}] ---")
                try:
                    results = run_fn(str(data_file), use_weighted)
                except Exception as e:
                    print(f"Error running {model_name} on {data_file.name} [{mode}]: {e}")
                    continue
                print("Final metrics:")
                for metric, value in results.items():
                    print(f"  {metric:12s}: {value:.4f}")
