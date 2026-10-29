"""Dataset discovery, normalization, and schema validation."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from config.settings import DATA_DIR, TEXT_COLUMN_CANDIDATES, LABEL_COLUMN_CANDIDATES

class DatasetError(ValueError):
    """Raised when the supplied dataset is missing or invalid."""


def _find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _normalize_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    mapping = {"fake": 1, "false": 1, "1": 1, "real": 0, "true": 0, "0": 0,
               "unreliable": 1, "reliable": 0}
    return mapping.get(text)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise DatasetError(f"Could not read CSV '{path}': {exc}") from exc


def load_dataset(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load a combined labeled CSV or the common Fake.csv + True.csv dataset.

    Label convention: 1 = fake, 0 = real. No synthetic fallback is used.
    """
    root = Path(data_dir).expanduser().resolve() if data_dir else DATA_DIR
    if not root.exists() or not root.is_dir():
        raise DatasetError(
            f"Dataset directory not found: {root}\n"
            "Create data/raw/ and place the real dataset CSV files there, or set "
            "FAKE_NEWS_DATA_DIR to the dataset directory. See README.md."
        )
    fake_path, real_path = root / "Fake.csv", root / "True.csv"
    if fake_path.is_file() and real_path.is_file():
        fake = _read_csv(fake_path)
        real = _read_csv(real_path)
        fake["label"] = 1
        real["label"] = 0
        frame = pd.concat([fake, real], ignore_index=True)
    else:
        candidates = sorted(p for p in root.glob("*.csv") if p.is_file())
        if not candidates:
            raise DatasetError(
                f"No CSV dataset found in {root}. Add a labeled CSV, or both Fake.csv and True.csv."
            )
        if len(candidates) > 1:
            raise DatasetError(
                f"Found multiple CSV files in {root}, but not the recognized Fake.csv + True.csv pair. "
                "Keep one combined labeled CSV in the directory or use the expected pair."
            )
        frame = _read_csv(candidates[0])

    text_col = _find_column(frame.columns, TEXT_COLUMN_CANDIDATES)
    label_col = _find_column(frame.columns, LABEL_COLUMN_CANDIDATES)
    if text_col is None:
        raise DatasetError(f"No article text column found. Expected one of: {', '.join(TEXT_COLUMN_CANDIDATES)}.")
    if label_col is None:
        raise DatasetError(f"No label column found. Expected one of: {', '.join(LABEL_COLUMN_CANDIDATES)}.")

    normalized = frame[label_col].map(_normalize_label)
    valid = frame[text_col].notna() & frame[text_col].astype(str).str.strip().ne("") & normalized.notna()
    cleaned = pd.DataFrame({"text": frame.loc[valid, text_col].astype(str).str.strip(),
                            "label": normalized.loc[valid].astype(int)})
    cleaned = cleaned.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if len(cleaned) < 10:
        raise DatasetError(
            f"Only {len(cleaned)} valid labeled rows remain after cleaning; at least 10 are required. "
            "Check the text and label columns and label values."
        )
    if cleaned["label"].nunique() != 2:
        raise DatasetError("Dataset must contain both classes. Use labels Fake/Real or 1/0 (1=fake, 0=real).")
    return cleaned


def validate_dataset(data_dir: Path | str | None = None) -> dict:
    frame = load_dataset(data_dir)
    return {"rows": int(len(frame)), "fake_rows": int((frame.label == 1).sum()),
            "real_rows": int((frame.label == 0).sum()), "columns": list(frame.columns)}
