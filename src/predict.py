"""Load trained pipelines and make predictions."""
from __future__ import annotations
from pathlib import Path
import joblib
from config.settings import MODEL_DIR
from src.preprocessing import clean_text

SUPPORTED_MODELS = ("logistic_regression", "multinomial_nb", "linear_svc")

class ModelNotFoundError(FileNotFoundError):
    """Raised when a requested trained model is unavailable."""

def load_model(model_name: str = "linear_svc", model_dir: Path | str | None = None):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(SUPPORTED_MODELS)}.")
    root = Path(model_dir).expanduser().resolve() if model_dir else MODEL_DIR
    path = root / f"{model_name}.joblib"
    if not path.is_file():
        raise ModelNotFoundError(
            f"Trained model not found: {path}. Run 'python -m src.train' from the project root after adding the dataset."
        )
    return joblib.load(path)

def predict_text(text: str, model_name: str = "linear_svc", model_dir: Path | str | None = None) -> dict:
    cleaned = clean_text(text)
    if len(cleaned) < 20:
        raise ValueError("Enter at least 20 characters of article text for a more meaningful prediction.")
    model = load_model(model_name, model_dir)
    label = int(model.predict([cleaned])[0])
    result = {"label": label, "prediction": "Fake" if label == 1 else "Real", "model": model_name,
              "note": "This is a statistical text classification result, not proof of truth or falsity."}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        result["scores"] = {"Real": round(float(probabilities[0]), 4), "Fake": round(float(probabilities[1]), 4)}
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function([cleaned])[0])
        result["decision_score"] = round(score, 4)
    return result
