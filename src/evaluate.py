"""Create evaluation metrics and visualizations from the saved training results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config.settings import ARTIFACT_DIR

def generate_artifacts() -> Path:
    metrics_path = ARTIFACT_DIR / "model_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}. Train models first with 'python -m src.train'.")
    metrics = pd.read_csv(metrics_path)
    required = {"model", "accuracy", "precision", "recall", "f1"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Metrics file is missing columns: {sorted(required - set(metrics.columns))}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_json(ARTIFACT_DIR / "model_metrics.json", orient="records", indent=2)
    plot_metrics = metrics.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    ax = plot_metrics.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=15)
    ax.set_title("Fake News Classifier Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    chart_path = ARTIFACT_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print(f"Metrics JSON: {ARTIFACT_DIR / 'model_metrics.json'}")
    print(f"Comparison chart: {chart_path}")
    return chart_path

if __name__ == "__main__":
    try:
        generate_artifacts()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
