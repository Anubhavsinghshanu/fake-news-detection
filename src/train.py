"""Train and persist three classical ML models."""
from __future__ import annotations
import json
import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from config.settings import DATA_DIR, MODEL_DIR, ARTIFACT_DIR, RANDOM_SEED
from src.data_loader import load_dataset, DatasetError
from src.preprocessing import clean_text

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
    "multinomial_nb": MultinomialNB(),
    "linear_svc": LinearSVC(random_state=RANDOM_SEED, class_weight="balanced"),
}

def train_all(data_dir: Path | str | None = None) -> pd.DataFrame:
    frame = load_dataset(data_dir)
    texts = frame["text"].map(clean_text)
    labels = frame["label"]
    if labels.value_counts().min() < 2:
        raise DatasetError("At least two examples per class are required for a stratified train/test split.")
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for name, estimator in MODELS.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=100_000,
                                      ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", estimator),
        ])
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        duration = round(time.perf_counter() - started, 3)
        metrics = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "training_seconds": duration,
            "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
            "seed": RANDOM_SEED,
        }
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
        records.append(metrics)
        print(f"Trained {name}: accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f} ({duration}s)")
    results = pd.DataFrame(records).sort_values("f1", ascending=False)
    results.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    metadata = {"label_mapping": {"0": "Real", "1": "Fake"}, "random_seed": RANDOM_SEED,
                "dataset_dir": str(Path(data_dir).resolve()) if data_dir else str(DATA_DIR),
                "models": [f"{name}.joblib" for name in MODELS]}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to: {ARTIFACT_DIR / 'model_metrics.csv'}")
    print(f"Models saved to: {MODEL_DIR}")
    return results

if __name__ == "__main__":
    try:
        print(train_all().to_string(index=False))
    except DatasetError as exc:
        raise SystemExit(f"DATASET ERROR: {exc}") from exc
