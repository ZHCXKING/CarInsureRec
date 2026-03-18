## CarInsureRec

Insurance product recommendation experiments (target column: `product_item`). This repo includes:

- Multiple recommenders (NN / tree / statistical baselines)
- Missingness injection + imputation (MICE/KNN/GAIN/MIWAE)
- A contrastive-learning recommender: `MICLRec` (`MICLRecommend`)

### Requirements

- Python 3.10+ (3.11 recommended)
- Dependencies: `requirements.txt`

Notes:

- `torch~=2.9.1+cu126` is a CUDA 12.6 build tag. If you are on CPU-only or a different CUDA version, install a compatible PyTorch for your machine and adjust that line if needed.
- Reading `*.parquet` via `pandas.read_parquet(...)` usually requires `pyarrow` (or `fastparquet`). If you see a parquet engine error, install one of them.

### Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you run `experiment/experiment.py` (writes to Excel with `engine='openpyxl'`), install:

```powershell
pip install openpyxl
```

### Data

Datasets live under `data/`:

- `data/AWM/`
- `data/HIP/`
- `data/VID/`

Key files per dataset:

- `AllData.parquet`: full data (may contain missing values)
- `DropNaData.parquet`: drop-NA version (`load(..., is_dropna=True)` uses this)
- `Metadata.json`: column type metadata (categorical vs numerical)

Source links are listed in `data/link.txt`.

### Quick Start

Run the default demo (loads `AWM`, injects missingness, trains/evaluates MICL variants):

```powershell
python main.py
```

Common knobs in `main.py`:

- `k`: Top-K for HR@K / NDCG@K
- `seed`: random seed
- `ratio`: missingness ratio for `inject_missingness(...)`
- `load('AWM', ...)`: switch dataset (`AWM/HIP/VID`)

### Reproduce Experiments

Run the main experiment driver (loops datasets/models/seeds and writes into `experiment/experiment.xlsx`):

```powershell
python experiment/experiment.py
```

Current `__main__` runs:

- `test_Perf()`: baseline performance across datasets/models
- `test_NaRatio()`: performance under different missing ratios / imputers

Plotting scripts (reads `experiment/experiment.xlsx`, outputs PDFs under `experiment/`):

```powershell
python experiment/figure_experiment.py
```

Parameter tuning with Optuna (updates `experiment/<DATASET>/*_param.json`):

```powershell
python experiment/parameter_tuning.py
```

### Layout

- `src/`
  - `models/`: recommenders (NN/tree/statistical/MICL/ensemble)
  - `utils/`: data loading, missingness injection, imputation
  - `evaluation.py`: AUC, logloss, HR@K, NDCG@K, etc.
- `data/`: datasets + metadata
- `experiment/`: experiment scripts, params, outputs

