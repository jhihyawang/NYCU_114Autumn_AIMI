import os
import pandas as pd

RESULTS_DIR = "results"

def parse_config(config_path):
    """Extract experiment info from config.txt"""
    info = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                info[key.strip()] = val.strip()
    return info

def collect_results(results_dir=RESULTS_DIR):
    """Scan all experiment folders and collect metrics"""
    rows = []
    for exp_name in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_name)
        config_file = os.path.join(exp_path, "config.txt")
        if not os.path.isfile(config_file):
            continue
        cfg = parse_config(config_file)

        rows.append({
            "Experiment ID": cfg.get("Experiment ID", exp_name),
            "Model": cfg.get("Model"),
            "Optimizer": cfg.get("Optimizer"),
            "LR": cfg.get("Learning Rate"),
            "Batch Size": cfg.get("Batch Size"),
            "Dropout": cfg.get("Dropout"),
            "Epochs": cfg.get("Epochs"),
            "Best Test Accuracy (%)": cfg.get("Best Test Accuracy"),
            "Final Test Accuracy (%)": cfg.get("Final Test Accuracy"),
        })

    df = pd.DataFrame(rows)
    df.sort_values(by=["Model", "Best Test Accuracy (%)"], ascending=[True, False], inplace=True)
    return df

if __name__ == "__main__":
    df = collect_results()
    print("\n================= Summary of Experiments =================")
    print(df.to_string(index=False))
    print("==========================================================\n")

    # Save to CSV
    out_path = os.path.join(RESULTS_DIR, "summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Summary saved to {out_path}")
