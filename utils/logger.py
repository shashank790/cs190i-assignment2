# utils/logger.py

import json
import os
import time
import csv
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_metrics_json(metrics: dict, output_path: str):
    """
    Save a dictionary of metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Saved metrics to {output_path}")

def save_metrics_csv(metrics: dict, output_path: str):
    """
    Save a dictionary of metrics to a CSV file (1 row).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"✅ Saved metrics to {output_path}")

def log_training_run(metrics: dict, method: str, format: str = "json"):
    """
    Save training/evaluation results for a specific method (full, head, lora).
    Automatically creates timestamped log files.
    """
    timestamp = get_timestamp()
    output_dir = f"results/{method}"
    filename = f"{method}_metrics_{timestamp}.{format}"
    path = os.path.join(output_dir, filename)

    if format == "json":
        save_metrics_json(metrics, path)
    elif format == "csv":
        save_metrics_csv(metrics, path)
    else:
        raise ValueError("Unsupported format. Use 'json' or 'csv'.")

# Example usage
if __name__ == "__main__":
    dummy_metrics = {"accuracy": 0.87, "eval_loss": 0.34, "training_time": "142.5s"}
    log_training_run(dummy_metrics, method="lora", format="json")