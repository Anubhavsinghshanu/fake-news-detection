from src.train import MODELS

def test_three_models_are_configured():
    assert set(MODELS) == {"logistic_regression", "multinomial_nb", "linear_svc"}
