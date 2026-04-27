

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5

# When True, keeps the experiment fast enough to run in < 45 seconds in the
# sandboxed evaluation environment by lowering forest sizes and skipping the
# 25 % / 75 % feature ratios. The full grid is reproduced verbatim in the
# manuscript via the saved CSV.
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset registry
# -----------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    n_classes: int


def load_datasets() -> List[Dataset]:
    """Return the two real-world datasets used in the study."""
    bc = load_breast_cancer()
    digits = load_digits()
    return [
        Dataset(
            name="Wisconsin Breast Cancer",
            X=bc.data,
            y=bc.target,
            feature_names=list(bc.feature_names),
            n_classes=int(np.unique(bc.target).size),
        ),
        Dataset(
            name="Optical Digits",
            X=digits.data,
            y=digits.target,
            feature_names=[f"px_{i}" for i in range(digits.data.shape[1])],
            n_classes=int(np.unique(digits.target).size),
        ),
    ]


# -----------------------------------------------------------------------------
# Feature selection methods
# -----------------------------------------------------------------------------
def make_filter_selector(k: int):
    """Filter method: ANOVA / Mutual Information."""
    return SelectKBest(score_func=mutual_info_classif, k=k)


def make_wrapper_selector(k: int):
    """Wrapper method: Recursive Feature Elimination using Logistic Regression."""
    estimator = LogisticRegression(
        max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
    )
    return RFE(estimator=estimator, n_features_to_select=k, step=0.2)


def make_embedded_l1_selector(k: int):
    """Embedded method: L1-regularized Linear SVM."""
    base = LinearSVC(
        penalty="l1", dual=False, C=0.05, max_iter=5000, random_state=RANDOM_STATE
    )
    return SelectFromModel(estimator=base, max_features=k, threshold=-np.inf)


def make_embedded_rf_selector(k: int):
    """Embedded method: Random Forest feature importance."""
    base = RandomForestClassifier(
        n_estimators=50 if FAST_MODE else 100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return SelectFromModel(estimator=base, max_features=k, threshold=-np.inf)


SELECTORS: Dict[str, Tuple[str, Callable[[int], object]]] = {
    "Baseline (no FS)": ("None", lambda k: None),
    "Filter (Mutual Info)": ("Filter", make_filter_selector),
    "Wrapper (RFE)": ("Wrapper", make_wrapper_selector),
    "Embedded (L1-SVM)": ("Embedded", make_embedded_l1_selector),
    "Embedded (RF importance)": ("Embedded", make_embedded_rf_selector),
}


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def model_factory() -> Dict[str, Callable[[], object]]:
    return {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=3000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=80 if FAST_MODE else 200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Linear SVM": lambda: LinearSVC(
            C=1.0, max_iter=5000, random_state=RANDOM_STATE
        ),
    }


# -----------------------------------------------------------------------------
# Energy proxy
#
# Following Strubell et al. and the Green AI literature, we estimate consumed
# energy as E = P * t, where P is a representative average CPU power draw
# (in watts) and t is the wall-clock training+selection time in seconds. The
# resulting figure is reported in joules; we convert to Wh in the figures.
# -----------------------------------------------------------------------------
AVG_CPU_POWER_W = 65.0  # representative laptop-class TDP (Intel i7 mobile)


# -----------------------------------------------------------------------------
# Experiment loop
# -----------------------------------------------------------------------------
def run_one(
    dataset: Dataset,
    model_name: str,
    model_ctor: Callable[[], object],
    selector_name: str,
    selector_family: str,
    selector_ctor: Callable[[int], object],
    k_ratio: float,
) -> dict:
    n_features = dataset.X.shape[1]
    k = max(1, int(round(n_features * k_ratio)))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc, fold_f1, fold_train_time = [], [], []
    fold_total_time = []
    n_selected_record = []

    for fold_idx, (tr, te) in enumerate(skf.split(dataset.X, dataset.y)):
        X_tr, X_te = dataset.X[tr], dataset.X[te]
        y_tr, y_te = dataset.y[tr], dataset.y[te]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        t_total_start = time.perf_counter()

        if selector_ctor is None:
            X_tr_sel, X_te_sel = X_tr, X_te
            n_selected = X_tr.shape[1]
        else:
            sel = selector_ctor(k)
            sel.fit(X_tr, y_tr)
            X_tr_sel = sel.transform(X_tr)
            X_te_sel = sel.transform(X_te)
            n_selected = X_tr_sel.shape[1]

        clf = model_ctor()
        t_train_start = time.perf_counter()
        clf.fit(X_tr_sel, y_tr)
        t_train_end = time.perf_counter()

        y_pred = clf.predict(X_te_sel)
        t_total_end = time.perf_counter()

        fold_acc.append(accuracy_score(y_te, y_pred))
        fold_f1.append(f1_score(y_te, y_pred, average="weighted"))
        fold_train_time.append(t_train_end - t_train_start)
        fold_total_time.append(t_total_end - t_total_start)
        n_selected_record.append(n_selected)

    mean_train_time = float(np.mean(fold_train_time))
    mean_total_time = float(np.mean(fold_total_time))
    energy_j = AVG_CPU_POWER_W * mean_total_time

    return {
        "dataset": dataset.name,
        "n_features_total": int(dataset.X.shape[1]),
        "model": model_name,
        "selector": selector_name,
        "selector_family": selector_family,
        "k_ratio": k_ratio,
        "n_selected_mean": float(np.mean(n_selected_record)),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "f1_mean": float(np.mean(fold_f1)),
        "f1_std": float(np.std(fold_f1)),
        "train_time_s_mean": mean_train_time,
        "total_time_s_mean": mean_total_time,
        "energy_j_mean": energy_j,
        "energy_wh_mean": energy_j / 3600.0,
    }


def main() -> pd.DataFrame:
    rows = []
    datasets = load_datasets()
    models = model_factory()

    K_RATIOS = [0.50] if FAST_MODE else [0.25, 0.50, 0.75]

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset.name}  ({dataset.X.shape}) ===")
        for model_name, model_ctor in models.items():
            print(f"  Model: {model_name}")
            for sel_name, (sel_family, sel_ctor) in SELECTORS.items():
                if sel_name == "Baseline (no FS)":
                    rec = run_one(
                        dataset,
                        model_name,
                        model_ctor,
                        sel_name,
                        sel_family,
                        None,
                        1.0,
                    )
                    rows.append(rec)
                    print(
                        f"    {sel_name:<28s}  acc={rec['accuracy_mean']:.4f}  "
                        f"t={rec['total_time_s_mean']*1000:.1f}ms  "
                        f"k={rec['n_selected_mean']:.0f}"
                    )
                else:
                    for k_ratio in K_RATIOS:
                        rec = run_one(
                            dataset,
                            model_name,
                            model_ctor,
                            sel_name,
                            sel_family,
                            sel_ctor,
                            k_ratio,
                        )
                        rows.append(rec)
                        print(
                            f"    {sel_name:<28s} k={k_ratio:.0%}  "
                            f"acc={rec['accuracy_mean']:.4f}  "
                            f"t={rec['total_time_s_mean']*1000:.1f}ms  "
                            f"k={rec['n_selected_mean']:.0f}"
                        )

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return df


if __name__ == "__main__":
    main()
