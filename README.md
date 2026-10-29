# Fake News Detection using NLP

A portable, editable Python project for classifying news text as **Fake (1)** or **Real (0)** using TF-IDF features and three classical machine-learning models: Logistic Regression, Multinomial Naive Bayes, and Linear SVM. The Streamlit interface is separate from the training pipeline.

> **Important:** This is an educational text classifier, not a source-verification system. Its prediction is not proof that a claim is true or false. Performance depends on the dataset and can be affected by bias, leakage, and changing news topics.

## 1. Requirements

- Target runtime: CPython 3.11. CPython 3.12 may work with the pinned dependencies but should be verified in your environment.
- Internet access is needed to install dependencies and obtain the dataset.
- No API keys, paid services, Docker, or external database are required.
- Dataset is **not bundled**. Obtain a dataset you are permitted to use; see Dataset setup below.

## 2. Project layout

```text
app.py                    Streamlit UI
config/settings.py        Portable configurable paths
src/data_loader.py        Dataset discovery and validation
src/preprocessing.py      Shared text cleanup
src/train.py              Train and save all three models
src/predict.py            Model loading and inference
src/evaluate.py           Export metrics and chart
src/validate_dataset.py   Dataset validation CLI
scripts/verify_setup.py   Environment diagnostic
 tests/                    Automated tests
 data/raw/                 Place original CSV dataset here
 models/                   Saved model pipelines (generated)
 artifacts/                Metrics and charts (generated)
```

## 3. Dataset setup

The default dataset directory is `data/raw/`. The loader supports either:

1. The common Kaggle-style pair `Fake.csv` and `True.csv`. The loader assigns labels: `Fake.csv` → 1 and `True.csv` → 0.
2. A single combined `.csv` with a text column and a label column. Recognized text columns: `text`, `content`, `article`, `news`, `body`. Recognized label columns: `label`, `class`, `target`, `fake`, `is_fake`. Supported labels include `Fake`/`Real`, `False`/`True`, and `1`/`0` (where 1 means Fake and 0 means Real).

For the commonly used Kaggle dataset named **Fake and Real News Dataset**, download it from its original Kaggle listing if you have access and are allowed to use it. Extract the CSV files into `data/raw/`; do not put the ZIP file there without extracting it. Dataset licensing and terms are your responsibility. If using another dataset, check its schema and licensing first.

Do not rename columns arbitrarily without updating `src/data_loader.py`. Do not add a synthetic dataset as a replacement for missing real data.

### Override the dataset path

Set `FAKE_NEWS_DATA_DIR` to the directory containing the CSV file(s). Relative values are resolved from the project root; absolute paths are accepted.

**Windows Command Prompt:**
```bat
set FAKE_NEWS_DATA_DIR=data\raw
```

**PowerShell:**
```powershell
$env:FAKE_NEWS_DATA_DIR = "data\raw"
```

**macOS/Linux:**
```bash
export FAKE_NEWS_DATA_DIR="data/raw"
```

Optional output overrides: `FAKE_NEWS_MODEL_DIR` and `FAKE_NEWS_ARTIFACT_DIR`. Defaults are `models/` and `artifacts/`.

## 4. Setup — Windows 10/11

Open a terminal in the project root.

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, install CPython 3.11 and ensure the Python launcher is available. Do not use the system interpreter if a virtual environment can be created.

## 5. Setup — macOS

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.11` is unavailable, install CPython 3.11 using your preferred trusted package manager and retry.

## 6. Setup — Linux / Ubuntu

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Ubuntu, if virtual-environment support is missing, install the matching `python3.11-venv` OS package using your administrator-approved package manager, then recreate the environment. OS package names vary by distribution and repository.

## 7. Validate, train, evaluate, test, run

Execute commands from the project root after activating the virtual environment and placing the dataset in `data/raw/`.

```bash
# Check runtime, imports, dependencies, dataset, and model state
python scripts/verify_setup.py

# Validate the real dataset schema and class counts
python -m src.validate_dataset

# Train Logistic Regression, Multinomial Naive Bayes, and Linear SVM
python -m src.train

# Generate metrics JSON and a model-comparison chart
python -m src.evaluate

# Run automated tests
python -m pytest

# Launch the UI
python -m streamlit run app.py
```

The same Python commands work in Windows PowerShell/Command Prompt, macOS, Linux, and IDE-integrated terminals after environment activation. In Windows Command Prompt, the commands above are the same except for environment activation and environment-variable syntax shown earlier.

Training saves model pipelines to `models/` and held-out test metrics to `artifacts/model_metrics.csv`. Evaluation creates `artifacts/model_metrics.json` and `artifacts/model_comparison.png`. The test split is stratified and uses a fixed random seed. Results can still vary across library versions or data revisions; the dataset itself must be kept consistent for meaningful comparisons.

## 8. IDEs and managed notebooks

### VS Code
Open the extracted project folder. Select `.venv` as the Python interpreter, then run commands in the integrated terminal. Install the Python extension if needed.

### PyCharm
Open the project folder and configure the interpreter to use `.venv`. Run the documented commands in the terminal. No source-path edits should be needed.

### Jupyter Notebook
Use the same project directory and environment. In a notebook, run shell commands with `%`/`!` syntax only in notebook cells, for example `!python -m src.validate_dataset`. The core code remains ordinary Python modules; do not copy the project logic into notebook-only cells.

### Google Colab
Upload the project ZIP and extract it, or clone the project repository. Install requirements in a cell with `!python -m pip install -r requirements.txt`, then change to the project root and run the documented Python commands. Upload or mount the real dataset and set `FAKE_NEWS_DATA_DIR` if it is outside `data/raw/`. Colab runtimes are ephemeral: download generated models and artifacts or save them to mounted persistent storage before the session ends. Pinned package compatibility should be rechecked when Colab changes its base runtime.

### GitHub Codespaces
Open the repository in a Codespace, verify Python 3.11, create and activate `.venv`, install requirements, then use the documented commands. Training performance depends on the selected Codespace resources.

### Replit
Support is conditional on the current Replit Python runtime and process model. Import the project, check the Python version, install `requirements.txt`, add the dataset through the workspace or supported storage, then run `python -m streamlit run app.py` if the workspace permits exposing Streamlit's port. This project does not include Replit-specific deployment configuration and does not claim verified Replit compatibility.

## 9. Environment verification and troubleshooting

`python scripts/verify_setup.py` reports Python version, required packages, project imports, output directories, dataset validation, and model availability. On a fresh clone without a dataset, a dataset warning/failure is expected; add the dataset before training.

- **`No CSV dataset found`**: Extract the dataset into `data/raw/`, or set `FAKE_NEWS_DATA_DIR`.
- **`No article text column found` / `No label column found`**: Rename columns to supported names or extend the candidate lists in `src/data_loader.py`.
- **`Trained model not found`**: Run dataset validation, then `python -m src.train`.
- **Package installation fails**: Verify CPython 3.11, activate `.venv`, upgrade pip, and inspect the first installation error. Do not mix global and virtual-environment packages.
- **App fails to start**: Run `python -m streamlit run app.py` from the project root with the environment activated.
- **Out of memory / slow training**: Reduce `max_features` in `src/train.py` or use a smaller representative dataset; document any such change in your submission.

## 10. Testing status and limitations

Automated tests cover text cleanup, dataset schema normalization, missing data/model errors, support for the `Fake.csv`/`True.csv` pair, and model configuration. Run `python -m pytest` in the installed environment. This package has not been claimed as tested on every operating system or cloud IDE. Record the actual Python version, OS/runtime, test command, and result before reporting a platform as verified.

The app is an educational baseline. Accuracy metrics alone do not establish real-world fact-checking quality. For a stronger study, add source-based verification, deduplication/leakage analysis, temporal evaluation, calibration, and a documented dataset citation.
