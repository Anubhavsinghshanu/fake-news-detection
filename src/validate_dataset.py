"""Command-line dataset validation."""
from __future__ import annotations
from src.data_loader import validate_dataset, DatasetError

def main() -> int:
    try:
        summary = validate_dataset()
    except DatasetError as exc:
        print(f"DATASET INVALID: {exc}")
        return 1
    print("Dataset validation passed")
    print(f"Rows: {summary['rows']}")
    print(f"Fake (1): {summary['fake_rows']}")
    print(f"Real (0): {summary['real_rows']}")
    print(f"Normalized columns: {', '.join(summary['columns'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
