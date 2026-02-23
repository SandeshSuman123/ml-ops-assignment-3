import json
import os

def save_results(results, path="results/local_eval_results.json"):
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
