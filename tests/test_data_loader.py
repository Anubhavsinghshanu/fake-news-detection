import pandas as pd
import pytest
from src.data_loader import load_dataset, DatasetError

def test_load_combined_dataset_normalizes_labels(tmp_path):
    rows = []
    for i in range(6):
        rows.append({"text": f"Fake article unique text {i}", "label": "Fake"})
        rows.append({"text": f"Real article unique text {i}", "label": "Real"})
    pd.DataFrame(rows).to_csv(tmp_path / "news.csv", index=False)
    result = load_dataset(tmp_path)
    assert set(result.columns) == {"text", "label"}
    assert set(result.label.unique()) == {0, 1}
    assert len(result) == 12

def test_missing_dataset_fails_clearly(tmp_path):
    with pytest.raises(DatasetError, match="No CSV dataset found"):
        load_dataset(tmp_path)

def test_fake_true_pair_is_supported(tmp_path):
    pd.DataFrame({"title": [f"Fake title {i}" for i in range(6)],
                  "text": [f"Fake article body with enough content {i}" for i in range(6)]}).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame({"title": [f"Real title {i}" for i in range(6)],
                  "text": [f"Real article body with enough content {i}" for i in range(6)]}).to_csv(tmp_path / "True.csv", index=False)
    result = load_dataset(tmp_path)
    assert int((result.label == 1).sum()) == 6
    assert int((result.label == 0).sum()) == 6
