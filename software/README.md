# Energy-Efficient ML via Feature Selection

For the full, formatted user manual see `USER_MANUAL.docx` in this
directory. This README is a quick-reference.

## Contents

```
software/
├── src/
│   ├── run_experiments.py     # full grid in a single process
│   ├── run_chunk.py           # single (dataset, model, k) chunk
│   └── merge_and_plot.py      # builds results.csv + 5 figures
├── results/
│   ├── results.csv            # 90 rows, one per configuration
│   ├── summary_table.csv      # tidy aggregation used in the paper
│   └── chunk_*.json           # raw per-chunk outputs
├── figures/
│   └── fig{1..5}_*.png        # publication-quality figures
├── requirements.txt
├── USER_MANUAL.docx           # formatted manual (read this)
├── LICENSE
└── README.md                  # this file
```

## One-liner reproduction

```bash
cd software
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiments.py        # ~30 seconds in FAST_MODE
python src/merge_and_plot.py         # writes figures/*.png
```

To reproduce the full 25 / 50 / 75 % retention grid used in the paper:

```bash
FAST_MODE=0 python src/run_experiments.py
```

## Datasets

Both datasets ship inside `scikit-learn`; no network is required.

| Dataset                   | n     | d  | Classes |
|---------------------------|-------|----|---------|
| Wisconsin Breast Cancer   |  569  | 30 |    2    |
| Optical Digits            | 1797  | 64 |   10    |

## Methods compared

* Baseline (no feature selection)
* Filter — Mutual Information
* Wrapper — Recursive Feature Elimination (Logistic Regression base)
* Embedded — L1-regularised Linear SVM
* Embedded — Random Forest importance

Each is evaluated against three classifiers (Logistic Regression,
Random Forest, Linear SVM) at three retention ratios (25 / 50 / 75 %)
under stratified 5-fold CV, with energy reported via the
`E = P · t` proxy at `P = 65 W`.
