import pytest
from src.predict import predict_text, load_model, ModelNotFoundError

def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="Run 'python -m src.train'"):
        load_model("linear_svc", tmp_path)

def test_short_text_is_rejected():
    with pytest.raises(ValueError, match="at least 20 characters"):
        predict_text("too short")
