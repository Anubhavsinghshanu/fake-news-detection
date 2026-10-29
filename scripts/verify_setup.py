"""Check runtime, dependencies, project imports, dataset, models, and output paths."""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config.settings import DATA_DIR, MODEL_DIR, ARTIFACT_DIR
from src.data_loader import load_dataset, DatasetError

REQUIRED = ("numpy", "pandas", "sklearn", "joblib", "matplotlib", "streamlit", "pytest")
def main() -> int:
    failures = []
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        failures.append("Use CPython 3.11 or 3.12; other versions are not the documented target.")
    for package in REQUIRED:
        try:
            module = importlib.import_module(package)
            print(f"[OK] package {package}: {getattr(module, '__version__', 'available')}")
        except Exception as exc:
            failures.append(f"Missing or broken package '{package}': {exc}")
    try:
        importlib.import_module("src.train")
        importlib.import_module("src.predict")
        importlib.import_module("src.evaluate")
        print("[OK] project module imports")
    except Exception as exc:
        failures.append(f"Project import failed: {exc}")
    for path in (MODEL_DIR, ARTIFACT_DIR):
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] output directory: {path}")
        except OSError as exc:
            failures.append(f"Cannot create output directory '{path}': {exc}")
    try:
        summary = {"rows": len(load_dataset()), "columns": ["text", "label"]}
        print(f"[OK] dataset: {summary['rows']} usable rows at {DATA_DIR}")
    except DatasetError as exc:
        print(f"[MISSING] dataset: {exc}")
        failures.append("Dataset validation failed; add the real dataset before training.")
    model_files = list(MODEL_DIR.glob("*.joblib"))
    if model_files:
        print(f"[OK] trained model files found: {len(model_files)}")
    else:
        print(f"[MISSING] trained models in {MODEL_DIR}; run 'python -m src.train' after dataset setup.")
    if failures:
        print("\nVerification completed with issues:")
        for item in failures:
            print(f" - {item}")
        return 1
    print("\nEnvironment verification passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
