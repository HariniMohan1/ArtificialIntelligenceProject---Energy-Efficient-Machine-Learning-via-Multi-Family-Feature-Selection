"""Merge chunked JSON outputs into a single CSV and create publication figures.

Generates:
    results/results.csv
    results/summary_table.csv
    figures/fig1_accuracy_vs_features.png   - Accuracy vs. retained-feature ratio
    figures/fig2_energy_breakdown.png       - Energy per (dataset, FS family)
    figures/fig3_speedup_vs_acc_loss.png    - Speedup vs accuracy loss scatter
    figures/fig4_pareto.png                 - Energy-accuracy Pareto frontier
    figures/fig5_per_method_summary.png     - Bar chart per method (averaged)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- styling ----------
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

PALETTE = {
    "Baseline (no FS)": "#4A4A4A",
    "Filter (Mutual Info)": "#1F77B4",
    "Wrapper (RFE)": "#FF7F0E",
    "Embedded (L1-SVM)": "#2CA02C",
    "Embedded (RF importance)": "#9467BD",
}
FAMILY_PALETTE = {
    "None": "#4A4A4A",
    "Filter": "#1F77B4",
    "Wrapper": "#FF7F0E",
    "Embedded": "#2CA02C",
}


def load_all() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS_DIR.glob("chunk_*.json")):
        with open(f) as fp:
            rows.extend(json.load(fp))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    return df


# ----------------------------------------------------------------------
# Figure 1: Accuracy vs. retained-feature ratio (per dataset, averaged
# across models). Baseline shown as a horizontal reference line.
# ----------------------------------------------------------------------
def fig_accuracy_vs_features(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.4), sharex=True)

    for ax, ds in zip(axes, df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        baseline_acc = sub[sub["selector"] == "Baseline (no FS)"]["accuracy_mean"].mean()
        ax.axhline(
            baseline_acc,
            color=PALETTE["Baseline (no FS)"],
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline (all features)",
        )
        for sel in [
            "Filter (Mutual Info)",
            "Wrapper (RFE)",
            "Embedded (L1-SVM)",
            "Embedded (RF importance)",
        ]:
            ssub = (
                sub[sub["selector"] == sel]
                .groupby("k_ratio")
                .agg(acc=("accuracy_mean", "mean"), std=("accuracy_mean", "std"))
                .reset_index()
            )
            ax.errorbar(
                ssub["k_ratio"] * 100,
                ssub["acc"],
                yerr=ssub["std"],
                marker="o",
                linewidth=2,
                capsize=3,
                color=PALETTE[sel],
                label=sel,
            )
        ax.set_title(ds)
        ax.set_xlabel("Features retained (%)")
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    fig.suptitle(
        "Predictive Accuracy vs. Retained-Feature Ratio",
        y=0.98, fontsize=12,
    )
    axes[1].set_xlabel("Features retained (%)")
    fig.subplots_adjust(top=0.92, bottom=0.18, hspace=0.30)
    out = FIGURES_DIR / "fig1_accuracy_vs_features.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ----------------------------------------------------------------------
# Figure 2: Energy (Wh proxy) per FS family, per dataset.
# ----------------------------------------------------------------------
def fig_energy_breakdown(df: pd.DataFrame):
    g = (
        df.groupby(["dataset", "selector_family"])["energy_wh_mean"]
        .mean()
        .reset_index()
    )

    datasets = list(g["dataset"].unique())
    families_order = ["None", "Filter", "Wrapper", "Embedded"]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.18
    x = np.arange(len(families_order))

    for i, ds in enumerate(datasets):
        vals = [
            g[(g["dataset"] == ds) & (g["selector_family"] == fam)]["energy_wh_mean"]
            .mean()
            * 1000  # mWh for readability
            for fam in families_order
        ]
        offset = (i - (len(datasets) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=ds,
            color=["#264653", "#E76F51"][i],
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["No FS", "Filter", "Wrapper", "Embedded"])
    ax.set_ylabel("Estimated energy per fold (mWh)")
    ax.set_title("Estimated Energy Cost per Feature-Selection Family")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    out = FIGURES_DIR / "fig2_energy_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


# ----------------------------------------------------------------------
# Figure 3: Per-method summary - accuracy bar + training time bar
# ----------------------------------------------------------------------
def fig_per_method_summary(df: pd.DataFrame):
    sub = df[df["k_ratio"].isin([0.5, 1.0])]  # 50% retained vs baseline
    g = (
        sub.groupby(["selector"])
        .agg(
            acc=("accuracy_mean", "mean"),
            train=("train_time_s_mean", "mean"),
            energy=("energy_wh_mean", "mean"),
        )
        .reindex(
            [
                "Baseline (no FS)",
                "Filter (Mutual Info)",
                "Wrapper (RFE)",
                "Embedded (L1-SVM)",
                "Embedded (RF importance)",
            ]
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.4))
    colors = [PALETTE[s] for s in g.index]

    axes[0].barh(range(len(g)), g["acc"], color=colors, edgecolor="white")
    axes[0].set_yticks(range(len(g)))
    axes[0].set_yticklabels(g.index)
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(g["acc"]) - 0.02, 1.0)
    axes[0].set_xlabel("Mean accuracy (averaged across datasets/models)")
    axes[0].set_title("Accuracy")
    for i, v in enumerate(g["acc"]):
        axes[0].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)

    axes[1].barh(range(len(g)), g["train"] * 1000, color=colors, edgecolor="white")
    axes[1].set_yticks(range(len(g)))
    axes[1].set_yticklabels(g.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean training time (ms)")
    axes[1].set_title("Training time")
    for i, v in enumerate(g["train"] * 1000):
        axes[1].text(v + max(g["train"] * 1000) * 0.01, i, f"{v:.0f}", va="center", fontsize=10)

    fig.suptitle(
        "Per-Method Summary at 50 % Feature Retention",
        y=0.99, fontsize=12,
    )
    fig.subplots_adjust(hspace=0.45, left=0.30)
    out = FIGURES_DIR / "fig3_per_method_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


# ----------------------------------------------------------------------
# Figure 4: Pareto - energy vs accuracy
# ----------------------------------------------------------------------
def fig_pareto(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for sel in df["selector"].unique():
        sub = df[df["selector"] == sel]
        ax.scatter(
            sub["energy_wh_mean"] * 1000,
            sub["accuracy_mean"],
            label=sel,
            color=PALETTE[sel],
            s=70,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Estimated energy per fold (mWh, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Energy\u2013Accuracy Trade-off Across All Configurations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    out = FIGURES_DIR / "fig4_pareto.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


# ----------------------------------------------------------------------
# Figure 5: Speedup vs accuracy loss (per-FS-method, averaged across models)
# ----------------------------------------------------------------------
def fig_speedup_vs_loss(df: pd.DataFrame):
    rows = []
    for ds in df["dataset"].unique():
        base = df[(df["dataset"] == ds) & (df["selector"] == "Baseline (no FS)")]
        b_acc = base["accuracy_mean"].mean()
        b_train = base["train_time_s_mean"].mean()
        for sel in df["selector"].unique():
            if sel == "Baseline (no FS)":
                continue
            for k in [0.25, 0.5, 0.75]:
                sub = df[
                    (df["dataset"] == ds)
                    & (df["selector"] == sel)
                    & (df["k_ratio"] == k)
                ]
                if sub.empty:
                    continue
                acc = sub["accuracy_mean"].mean()
                train = sub["train_time_s_mean"].mean()
                rows.append(
                    {
                        "dataset": ds,
                        "selector": sel,
                        "k_ratio": k,
                        "acc_loss_pp": (b_acc - acc) * 100,
                        "speedup": b_train / max(train, 1e-9),
                    }
                )
    pd_rows = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    markers = {"Wisconsin Breast Cancer": "o", "Optical Digits": "s"}
    for sel in pd_rows["selector"].unique():
        for ds in pd_rows["dataset"].unique():
            sub = pd_rows[(pd_rows["selector"] == sel) & (pd_rows["dataset"] == ds)]
            ax.scatter(
                sub["acc_loss_pp"],
                sub["speedup"],
                color=PALETTE[sel],
                marker=markers[ds],
                s=80,
                edgecolor="white",
                linewidth=1,
                label=f"{sel}  ({ds.split()[0]})",
            )
    ax.axhline(1.0, color="grey", linewidth=1)
    ax.axvline(0.0, color="grey", linewidth=1)
    ax.set_xlabel("Accuracy loss vs. baseline (percentage points)")
    ax.set_ylabel("Training-time speedup (×)")
    ax.set_title(
        "Training-Speed Gain vs. Accuracy Loss (per FS Method)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    out = FIGURES_DIR / "fig5_speedup_vs_loss.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


# ----------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------
def write_summary_table(df: pd.DataFrame):
    g = (
        df.groupby(["dataset", "selector"])
        .agg(
            acc_mean=("accuracy_mean", "mean"),
            acc_std=("accuracy_mean", "std"),
            train_ms=("train_time_s_mean", lambda x: np.mean(x) * 1000),
            energy_mwh=("energy_wh_mean", lambda x: np.mean(x) * 1000),
            n_selected=("n_selected_mean", "mean"),
        )
        .reset_index()
    )
    g.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print(f"  {RESULTS_DIR / 'summary_table.csv'}")
    return g


def main():
    df = load_all()
    print(f"Loaded {len(df)} rows")
    print("Generating figures...")
    fig_accuracy_vs_features(df)
    fig_energy_breakdown(df)
    fig_per_method_summary(df)
    fig_pareto(df)
    fig_speedup_vs_loss(df)
    print("Writing summary table...")
    write_summary_table(df)
    print("Done.")


if __name__ == "__main__":
    main()
