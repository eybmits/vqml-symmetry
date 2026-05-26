"""Generate the final paper-style figure package from completed experiment CSVs."""

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from src.groups_d4 import COORDS, DIRECTED_PAIRS, WIN_LINE_TRIPLES
from src.utils import CSV_DIR, FIGURES_DIR


PAPER_FIGURES_DIR = FIGURES_DIR / "paper"
SUMMARY_PATH = CSV_DIR / "paper_plot_summary.csv"

EXPECTED_ROWS = {
    "results_reproduction.csv": 10,
    "results_partial_data_sweep_full_L3p2.csv": 150,
    "results_lowdata_goldilocks_c4_d4_L3p2.csv": 120,
    "results_random_sharing_control_full_L3p2_train450.csv": 55,
    "results_depth_sweep_all_groups_draft_L1234_p123.csv": 216,
    "results_d4_compression_sweep_L3p2.csv": 60,
    "results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv": 240,
    "results_rule_based_oracle.csv": 1,
}

SUBGROUP_ORDER = ["none", "Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"]
RANDOM_CONTROL_ORDER = ["Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"]
COMPRESSION_ORDER = ["D4_all_tied", "D4_qubit_tied", "D4_pair_tied", "D4"]
FAMILY_ORDER = [
    "edge",
    "line_zzz",
    "line_ccrz",
    "line_zzz_ccrz",
    "edge_line_zzz",
    "edge_line_ccrz",
    "edge_line_zzz_ccrz",
    "line_pair_crz",
]

SUBGROUP_COLORS = {
    "none": "#8a8a8a",
    "Z2_rot180": "#E69F00",
    "Z2_reflection": "#56B4E9",
    "C4": "#009E73",
    "D2_V4": "#CC79A7",
    "D4": "#0072B2",
    "D4_all_tied": "#bdbdbd",
    "D4_qubit_tied": "#969696",
    "D4_pair_tied": "#636363",
}
FAMILY_COLORS = {
    "edge": "#4D4D4D",
    "line_zzz": "#56B4E9",
    "line_ccrz": "#E69F00",
    "line_zzz_ccrz": "#009E73",
    "edge_line_zzz": "#D55E00",
    "edge_line_ccrz": "#CC79A7",
    "edge_line_zzz_ccrz": "#0072B2",
    "line_pair_crz": "#999999",
}
SUBGROUP_MARKERS = {
    "none": "o",
    "Z2_rot180": "s",
    "Z2_reflection": "D",
    "C4": "^",
    "D2_V4": "v",
    "D4": "P",
}
FAMILY_LABELS = {
    "edge": "edge",
    "line_zzz": "line ZZZ",
    "line_ccrz": "line CCRZ",
    "line_zzz_ccrz": "line both",
    "edge_line_zzz": "edge + ZZZ",
    "edge_line_ccrz": "edge + CCRZ",
    "edge_line_zzz_ccrz": "edge + both",
    "line_pair_crz": "line-pair CRZ",
}


@dataclass(frozen=True)
class FigureOutput:
    stem: str
    pdf: Path
    png: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--fig-dir", type=Path, default=PAPER_FIGURES_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--no-strict-counts", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ci95(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def sem95(std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return 0.0
    return float(1.96 * std / math.sqrt(n))


def mean_ci(
    df: pd.DataFrame,
    keys: list[str],
    metric: str = "test_accuracy",
) -> pd.DataFrame:
    aggregations = {
        "n": (metric, "count"),
        "mean": (metric, "mean"),
        "std": (metric, "std"),
        "ci95": (metric, ci95),
        "train_mean": ("train_accuracy", "mean") if "train_accuracy" in df.columns else (metric, "mean"),
        "gap_mean": (
            ("generalization_gap", "mean") if "generalization_gap" in df.columns else (metric, "mean")
        ),
    }
    if "num_parameters" in df.columns:
        aggregations["params"] = ("num_parameters", "mean")
    elif "params" in df.columns:
        aggregations["params"] = ("params", "mean")
    else:
        aggregations["params"] = (metric, "mean")
    grouped = df.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def read_csvs(csv_dir: Path, strict_counts: bool = True) -> dict[str, pd.DataFrame]:
    csvs: dict[str, pd.DataFrame] = {}
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = csv_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if strict_counts and len(df) != expected_rows:
            raise ValueError(f"{filename}: expected {expected_rows} rows, found {len(df)}")
        csvs[filename] = df
    return csvs


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> FigureOutput:
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / f"{stem}.pdf"
    png = fig_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return FigureOutput(stem=stem, pdf=pdf, png=png)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, alpha=0.22, linewidth=0.8)


def clean_label(label: str) -> str:
    return label.replace("_", " ")


def ordered_existing(values: Iterable[str], order: list[str]) -> list[str]:
    seen = set(values)
    return [value for value in order if value in seen]


def append_summary(
    rows: list[dict[str, object]],
    *,
    figure: str,
    source: str,
    df: pd.DataFrame,
    keys: list[str],
    metric: str = "test_accuracy",
) -> None:
    summary = mean_ci(df, keys, metric)
    for _, row in summary.iterrows():
        entry: dict[str, object] = {
            "figure": figure,
            "source": source,
            "metric": metric,
            "n": int(row["n"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "ci95": float(row["ci95"]),
            "params": float(row["params"]) if pd.notna(row["params"]) else np.nan,
        }
        for key in keys:
            entry[key] = row[key]
        rows.append(entry)


def rule_oracle_accuracy(rule_df: pd.DataFrame) -> float:
    if "accuracy" not in rule_df.columns or rule_df.empty:
        raise ValueError("Rule-oracle results must contain a non-empty accuracy column")
    return float(rule_df["accuracy"].iloc[0])


def append_rule_oracle_summary(rows: list[dict[str, object]], figure: str, rule_df: pd.DataFrame) -> None:
    row = rule_df.iloc[0]
    rows.append(
        {
            "figure": figure,
            "source": "results_rule_based_oracle.csv",
            "metric": "accuracy",
            "n": int(row.get("n_examples", 1)),
            "mean": float(row["accuracy"]),
            "std": 0.0,
            "ci95": 0.0,
            "params": float(row.get("num_trainable_parameters", 0)),
            "circuit_family": "deterministic_rule_oracle",
            "subgroup": "D4_invariant",
            "train_size": np.nan,
        }
    )


def plot_paired_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    labels: list[str],
    *,
    color_map: dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(labels), dtype=float)
    pivot = df.pivot(index="seed", columns="subgroup", values=metric)
    for _, seed_row in pivot.iterrows():
        y = [seed_row[label] for label in labels]
        ax.plot(x, y, color="#bdbdbd", linewidth=0.8, zorder=1)
    for idx, label in enumerate(labels):
        values = pivot[label].dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) else []
        ax.scatter(
            np.full(len(values), x[idx]) + jitter,
            values,
            s=18,
            color=color_map.get(label, "#333333"),
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.errorbar(
            x[idx],
            values.mean(),
            yerr=ci95(values),
            fmt="s",
            color="black",
            markersize=4.5,
            capsize=3,
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(label) for label in labels])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    add_grid(ax)


def make_fig01_reproduction(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    labels = ["none", "D4"]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), constrained_layout=True)
    plot_paired_metric(
        axes[0],
        df,
        "test_accuracy",
        labels,
        color_map=SUBGROUP_COLORS,
        title="Test accuracy",
        ylabel="accuracy",
    )
    plot_paired_metric(
        axes[1],
        df,
        "generalization_gap",
        labels,
        color_map=SUBGROUP_COLORS,
        title="Generalization gap",
        ylabel="train - test",
    )
    axes[0].set_ylim(0.52, 0.78)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    append_summary(rows, figure="fig01", source="results_reproduction.csv", df=df, keys=["subgroup"])
    append_summary(
        rows,
        figure="fig01",
        source="results_reproduction.csv",
        df=df,
        keys=["subgroup"],
        metric="generalization_gap",
    )
    return save_figure(fig, fig_dir, "fig01_reproduction_d4_vs_none")


def make_fig02_partial(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), constrained_layout=True)
    for metric, ax, ylabel in [
        ("test_accuracy", axes[0], "test accuracy"),
        ("generalization_gap", axes[1], "train - test"),
    ]:
        summary = mean_ci(df, ["subgroup", "train_size"], metric)
        for subgroup in ordered_existing(summary["subgroup"], SUBGROUP_ORDER):
            sub = summary[summary["subgroup"] == subgroup].sort_values("train_size")
            x = sub["train_size"].to_numpy()
            y = sub["mean"].to_numpy()
            err = sub["ci95"].to_numpy()
            ax.plot(
                x,
                y,
                marker=SUBGROUP_MARKERS.get(subgroup, "o"),
                color=SUBGROUP_COLORS.get(subgroup, "#333333"),
                label=clean_label(subgroup),
                linewidth=1.7 if subgroup in {"C4", "D4"} else 1.1,
                markersize=4.2,
            )
            ax.fill_between(x, y - err, y + err, color=SUBGROUP_COLORS.get(subgroup, "#333333"), alpha=0.10)
        ax.set_xscale("log")
        ax.set_xticks([30, 60, 120, 240, 450])
        ax.set_xticklabels(["30", "60", "120", "240", "450"])
        ax.set_xlabel("training examples")
        ax.set_ylabel(ylabel)
        add_grid(ax)
    axes[0].set_title("Accuracy across data regimes")
    axes[1].set_title("Generalization gap")
    axes[0].set_ylim(0.38, 0.73)
    axes[1].set_ylim(-0.02, 0.62)
    axes[0].legend(ncol=2, frameon=False, loc="lower right")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    append_summary(
        rows,
        figure="fig02",
        source="results_partial_data_sweep_full_L3p2.csv",
        df=df,
        keys=["subgroup", "train_size"],
    )
    append_summary(
        rows,
        figure="fig02",
        source="results_partial_data_sweep_full_L3p2.csv",
        df=df,
        keys=["subgroup", "train_size"],
        metric="generalization_gap",
    )
    return save_figure(fig, fig_dir, "fig02_partial_equivariance_data_sweep")


def make_fig03_lowdata(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    diffs = []
    for train_size, block in df.groupby("train_size"):
        pivot = block.pivot(index="seed", columns="subgroup", values="test_accuracy")
        delta = pivot["C4"] - pivot["D4"]
        diffs.append(
            {
                "train_size": int(train_size),
                "mean": float(delta.mean()),
                "std": float(delta.std(ddof=1)),
                "ci95": ci95(delta),
                "n": int(delta.count()),
            }
        )
    summary = pd.DataFrame(diffs).sort_values("train_size")
    fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.errorbar(
        summary["train_size"],
        summary["mean"],
        yerr=summary["ci95"],
        color=SUBGROUP_COLORS["C4"],
        marker="o",
        linewidth=1.6,
        capsize=2.8,
    )
    ax.fill_between(
        summary["train_size"].to_numpy(),
        0,
        summary["mean"].to_numpy(),
        where=summary["mean"].to_numpy() >= 0,
        color=SUBGROUP_COLORS["C4"],
        alpha=0.12,
        interpolate=True,
    )
    ax.fill_between(
        summary["train_size"].to_numpy(),
        0,
        summary["mean"].to_numpy(),
        where=summary["mean"].to_numpy() < 0,
        color=SUBGROUP_COLORS["D4"],
        alpha=0.12,
        interpolate=True,
    )
    ax.set_xscale("log")
    ax.set_xticks(summary["train_size"])
    ax.set_xticklabels([str(int(value)) for value in summary["train_size"]], rotation=35, ha="right")
    ax.set_ylabel("test accuracy difference")
    ax.set_xlabel("training examples")
    ax.set_title("C4 minus D4 in low-data regimes")
    add_grid(ax)
    for _, row in summary.iterrows():
        rows.append(
            {
                "figure": "fig03",
                "source": "results_lowdata_goldilocks_c4_d4_L3p2.csv",
                "metric": "C4_minus_D4_test_accuracy",
                "train_size": int(row["train_size"]),
                "n": int(row["n"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "ci95": float(row["ci95"]),
                "params": np.nan,
            }
        )
    return save_figure(fig, fig_dir, "fig03_lowdata_c4_d4_delta")


def make_fig04_random(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), constrained_layout=True)
    x = np.arange(len(RANDOM_CONTROL_ORDER))
    width = 0.34
    for ax, metric, ylabel, title in [
        (axes[0], "test_accuracy", "test accuracy", "Accuracy"),
        (axes[1], "generalization_gap", "train - test", "Generalization gap"),
    ]:
        summary = mean_ci(df[df["subgroup"].isin(RANDOM_CONTROL_ORDER)], ["subgroup", "sharing_type"], metric)
        for offset, sharing_type, color in [
            (-width / 2, "random", "#bdbdbd"),
            (width / 2, "symmetry", "#0072B2"),
        ]:
            values = []
            errs = []
            for subgroup in RANDOM_CONTROL_ORDER:
                row = summary[(summary["subgroup"] == subgroup) & (summary["sharing_type"] == sharing_type)]
                values.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                errs.append(float(row["ci95"].iloc[0]) if not row.empty else 0.0)
            ax.bar(
                x + offset,
                values,
                yerr=errs,
                width=width,
                label=sharing_type,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                capsize=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([clean_label(label) for label in RANDOM_CONTROL_ORDER], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        add_grid(ax)
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.48, 0.74)
    axes[1].set_ylim(0.0, 0.08)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    append_summary(
        rows,
        figure="fig04",
        source="results_random_sharing_control_full_L3p2_train450.csv",
        df=df,
        keys=["subgroup", "sharing_type"],
    )
    append_summary(
        rows,
        figure="fig04",
        source="results_random_sharing_control_full_L3p2_train450.csv",
        df=df,
        keys=["subgroup", "sharing_type"],
        metric="generalization_gap",
    )
    return save_figure(fig, fig_dir, "fig04_random_sharing_control")


def make_fig05_depth(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    df = df.copy()
    df["depth"] = df["L"] * df["p"]
    lp_summary = mean_ci(df, ["subgroup", "L", "p", "depth"], "test_accuracy")
    best_by_depth = (
        lp_summary.sort_values(["subgroup", "depth", "mean"], ascending=[True, True, False])
        .groupby(["subgroup", "depth"], as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.25), constrained_layout=True)
    for subgroup in ordered_existing(best_by_depth["subgroup"], SUBGROUP_ORDER):
        sub = best_by_depth[best_by_depth["subgroup"] == subgroup].sort_values("depth")
        ax.plot(
            sub["depth"],
            sub["mean"],
            marker=SUBGROUP_MARKERS.get(subgroup, "o"),
            color=SUBGROUP_COLORS.get(subgroup, "#333333"),
            label=clean_label(subgroup),
            linewidth=1.7 if subgroup in {"C4", "D4"} else 1.1,
            markersize=4.0,
        )
    ax.set_xlabel("effective depth L x p")
    ax.set_ylabel("best test accuracy")
    ax.set_title("Depth and expressivity")
    ax.set_xticks(sorted(best_by_depth["depth"].unique()))
    ax.set_ylim(0.45, 0.75)
    add_grid(ax)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    for _, row in best_by_depth.iterrows():
        rows.append(
            {
                "figure": "fig05",
                "source": "results_depth_sweep_all_groups_draft_L1234_p123.csv",
                "metric": "best_test_accuracy_at_depth",
                "n": int(row["n"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "ci95": float(row["ci95"]),
                "params": float(row["params"]),
                "subgroup": row["subgroup"],
                "depth": int(row["depth"]),
                "L": int(row["L"]),
                "p": int(row["p"]),
            }
        )
    return save_figure(fig, fig_dir, "fig05_depth_expressivity_tradeoff")


def make_fig06_compression(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    summary = mean_ci(df, ["subgroup"], "test_accuracy")
    summary["subgroup"] = pd.Categorical(summary["subgroup"], COMPRESSION_ORDER, ordered=True)
    summary = summary.sort_values("subgroup")
    fig, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    ax.errorbar(
        summary["params"],
        summary["mean"],
        yerr=summary["ci95"],
        color="#4D4D4D",
        marker="o",
        linewidth=1.5,
        capsize=3,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            clean_label(str(row["subgroup"])),
            (row["params"], row["mean"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("test accuracy")
    ax.set_title("Beyond-D4 compression")
    ax.set_ylim(0.49, 0.66)
    add_grid(ax)
    append_summary(
        rows,
        figure="fig06",
        source="results_d4_compression_sweep_L3p2.csv",
        df=df,
        keys=["subgroup"],
    )
    return save_figure(fig, fig_dir, "fig06_beyond_d4_compression")


def make_fig07_oracle(
    df: pd.DataFrame,
    rule_df: pd.DataFrame,
    fig_dir: Path,
    rows: list[dict[str, object]],
) -> FigureOutput:
    selected = [
        ("edge", "D4"),
        ("edge", "C4"),
        ("line_zzz_ccrz", "D4"),
        ("edge_line_zzz", "D4"),
        ("edge_line_zzz_ccrz", "D4"),
        ("edge_line_zzz_ccrz", "C4"),
    ]
    summary = mean_ci(df, ["circuit_family", "subgroup", "train_size"], "test_accuracy")
    fig, ax = plt.subplots(figsize=(6.9, 3.35), constrained_layout=True)
    for family, subgroup in selected:
        sub = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == subgroup)]
        if sub.empty:
            raise ValueError(f"Missing oracle comparison: {family}/{subgroup}")
        sub = sub.sort_values("train_size")
        label = f"{FAMILY_LABELS[family]} / {subgroup}"
        is_main = family == "edge_line_zzz_ccrz" and subgroup == "D4"
        ax.errorbar(
            sub["train_size"],
            sub["mean"],
            yerr=sub["ci95"],
            marker=SUBGROUP_MARKERS.get(subgroup, "o"),
            color=FAMILY_COLORS[family],
            linestyle="-" if subgroup == "D4" else "--",
            linewidth=2.4 if is_main else 1.4,
            markersize=5.3 if is_main else 4.2,
            capsize=2.6,
            label=label,
            alpha=1.0 if is_main else 0.9,
        )
    oracle_acc = rule_oracle_accuracy(rule_df)
    ax.axhline(
        oracle_acc,
        color="#D55E00",
        linestyle=":",
        linewidth=1.6,
        label="_nolegend_",
        zorder=1,
    )
    ax.text(
        750,
        oracle_acc - 0.015,
        "deterministic rule oracle",
        ha="right",
        va="top",
        color="#D55E00",
        fontsize=8,
    )
    ax.set_xlabel("training examples")
    ax.set_ylabel("test accuracy")
    ax.set_title("Oracle-inspired equivariant architectures")
    ax.set_xticks([450, 600, 750])
    ax.set_ylim(0.64, 1.02)
    ax.set_yticks([0.65, 0.75, 0.85, 1.00])
    add_grid(ax)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.72))
    append_summary(
        rows,
        figure="fig07",
        source="results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv",
        df=df,
        keys=["circuit_family", "subgroup", "train_size"],
    )
    append_rule_oracle_summary(rows, "fig07", rule_df)
    return save_figure(fig, fig_dir, "fig07_oracle_inspired_accuracy")


def make_fig08_oracle_ablation(df: pd.DataFrame, fig_dir: Path, rows: list[dict[str, object]]) -> FigureOutput:
    df600 = df[df["train_size"] == 600].copy()
    summary = mean_ci(df600, ["circuit_family", "subgroup"], "test_accuracy")
    x = np.arange(len(FAMILY_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.45), constrained_layout=True)
    for offset, subgroup in [(-width / 2, "C4"), (width / 2, "D4")]:
        values = []
        errs = []
        for family in FAMILY_ORDER:
            row = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == subgroup)]
            if row.empty:
                raise ValueError(f"Missing ablation comparison: {family}/{subgroup}")
            values.append(float(row["mean"].iloc[0]))
            errs.append(float(row["ci95"].iloc[0]))
        ax.bar(
            x + offset,
            values,
            yerr=errs,
            width=width,
            color=SUBGROUP_COLORS[subgroup],
            label=subgroup,
            edgecolor="white",
            linewidth=0.6,
            capsize=2,
        )
    labels = [
        "edge",
        "line\nZZZ",
        "line\nCCRZ",
        "line\nboth",
        "edge+\nZZZ",
        "edge+\nCCRZ",
        "edge+\nboth",
        "line-pair\nCRZ",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test accuracy")
    ax.set_title("Oracle-inspired ablation at train size 600")
    ax.set_ylim(0.64, 0.84)
    add_grid(ax)
    ax.legend(frameon=False, loc="upper left")
    append_summary(
        rows,
        figure="fig08",
        source="results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv",
        df=df600,
        keys=["circuit_family", "subgroup"],
    )
    return save_figure(fig, fig_dir, "fig08_oracle_inspired_ablation")


def make_fig09_params(
    partial_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    fig_dir: Path,
    rows: list[dict[str, object]],
) -> FigureOutput:
    partial = partial_df[partial_df["train_size"] == 450].copy()
    partial["circuit_family"] = "edge"
    partial["source_family"] = "Meyer-style edge"
    oracle = oracle_df[(oracle_df["train_size"] == 450) & (oracle_df["circuit_family"] != "edge")].copy()
    oracle["source_family"] = "oracle-inspired"
    combined = pd.concat([partial, oracle], ignore_index=True)
    summary = mean_ci(combined, ["source_family", "circuit_family", "subgroup"], "test_accuracy")
    fig, ax = plt.subplots(figsize=(7.8, 4.15), constrained_layout=True)
    for _, row in summary.iterrows():
        family = str(row["circuit_family"])
        subgroup = str(row["subgroup"])
        is_highlight = (
            (family == "edge" and subgroup in {"none", "D4"})
            or (family == "edge_line_zzz_ccrz" and subgroup == "D4")
        )
        ax.errorbar(
            row["params"],
            row["mean"],
            yerr=row["ci95"],
            fmt=SUBGROUP_MARKERS.get(subgroup, "o"),
            color=FAMILY_COLORS.get(family, "#333333"),
            markersize=8 if is_highlight else 5.2,
            markeredgecolor="black" if is_highlight else "white",
            markeredgewidth=0.8 if is_highlight else 0.35,
            capsize=2,
            alpha=1.0 if is_highlight else 0.75,
        )
    annotations = {
        ("edge", "none"): "edge / none",
        ("edge", "D4"): "edge / D4",
        ("edge_line_zzz_ccrz", "D4"): "edge + line both / D4",
    }
    for _, row in summary.iterrows():
        key = (str(row["circuit_family"]), str(row["subgroup"]))
        if key in annotations:
            ax.annotate(
                annotations[key],
                (row["params"], row["mean"]),
                textcoords="offset points",
                xytext=(8, 8 if key != ("edge", "D4") else -16),
                fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#595959", linewidth=0.7),
            )
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("test accuracy at train size 450")
    ax.set_title("Accuracy is not explained by parameter count alone")
    ax.set_ylim(0.50, 0.84)
    ax.set_xlim(35, 220)
    add_grid(ax)
    # Compact custom legends.
    family_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS[fam], label=FAMILY_LABELS[fam], markersize=6)
        for fam in ["edge", "line_zzz_ccrz", "edge_line_zzz", "edge_line_ccrz", "edge_line_zzz_ccrz"]
    ]
    subgroup_handles = [
        plt.Line2D([0], [0], marker=SUBGROUP_MARKERS[sg], color="#333333", linestyle="none", label=sg, markersize=5)
        for sg in ["none", "C4", "D4"]
    ]
    first = ax.legend(
        handles=family_handles,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d9d9d9",
        loc="upper right",
        ncol=1,
        title="family",
        handletextpad=0.4,
    )
    ax.add_artist(first)
    ax.legend(handles=subgroup_handles, frameon=False, loc="upper left", title="symmetry")
    append_summary(
        rows,
        figure="fig09",
        source="combined_partial_oracle_train450",
        df=combined,
        keys=["source_family", "circuit_family", "subgroup"],
    )
    return save_figure(fig, fig_dir, "fig09_accuracy_vs_parameters")


def farthest_endpoints(line: tuple[int, int, int]) -> tuple[int, int]:
    best = (line[0], line[1])
    best_distance = -1.0
    for i, a in enumerate(line):
        for b in line[i + 1 :]:
            ax, ay = COORDS[a]
            bx, by = COORDS[b]
            distance = (ax - bx) ** 2 + (ay - by) ** 2
            if distance > best_distance:
                best_distance = distance
                best = (a, b)
    return best


def draw_board_nodes(ax: plt.Axes) -> None:
    for idx, (x, y) in COORDS.items():
        ax.scatter([x], [y], s=185, color="white", edgecolor="#333333", linewidth=1.0, zorder=4)
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, zorder=5)
    for value in [-0.5, 0.5]:
        ax.plot([-1.25, 1.25], [value, value], color="#d0d0d0", linewidth=0.8, zorder=0)
        ax.plot([value, value], [-1.25, 1.25], color="#d0d0d0", linewidth=0.8, zorder=0)
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_edge_pairs(ax: plt.Axes, alpha: float = 0.55) -> None:
    for src, dst in DIRECTED_PAIRS:
        x1, y1 = COORDS[src]
        x2, y2 = COORDS[dst]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color="#666666",
            alpha=alpha,
            shrinkA=12,
            shrinkB=12,
            zorder=1,
        )
        ax.add_patch(arrow)


def draw_winning_lines(ax: plt.Axes, alpha: float = 0.78) -> None:
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for index, line in enumerate(WIN_LINE_TRIPLES):
        a, b = farthest_endpoints(line)
        x1, y1 = COORDS[a]
        x2, y2 = COORDS[b]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=colors[index % len(colors)],
            linewidth=3.0,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )


def make_fig10_schematic(fig_dir: Path) -> FigureOutput:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)
    titles = [
        "(a) Meyer-style edge CRY",
        "(b) winning-line triples",
        "(c) edge + line gates",
    ]
    for ax, title in zip(axes, titles):
        draw_board_nodes(ax)
        ax.set_title(title, fontsize=9)
    draw_edge_pairs(axes[0], alpha=0.70)
    draw_winning_lines(axes[1], alpha=0.82)
    draw_edge_pairs(axes[2], alpha=0.35)
    draw_winning_lines(axes[2], alpha=0.78)
    legend_lines = [
        plt.Line2D([0], [0], color="#666666", linewidth=1.2, label="directed CRY pair"),
        plt.Line2D([0], [0], color="#0072B2", linewidth=3.0, label="3-qubit winning-line gate"),
    ]
    axes[2].legend(handles=legend_lines, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.14))
    return save_figure(fig, fig_dir, "fig10_architecture_schematic")


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    leading_cols = ["figure", "source", "metric", "n", "mean", "std", "ci95", "params"]
    other_cols = [col for col in df.columns if col not in leading_cols]
    df = df[leading_cols + other_cols]
    df.to_csv(output_path, index=False)


def verify_outputs(outputs: list[FigureOutput], summary_path: Path) -> None:
    missing = []
    for output in outputs:
        for path in [output.pdf, output.png]:
            if not path.exists() or path.stat().st_size <= 0:
                missing.append(path)
    if not summary_path.exists() or summary_path.stat().st_size <= 0:
        missing.append(summary_path)
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Missing or empty outputs:\n{formatted}")


def main() -> None:
    args = parse_args()
    configure_style()
    csvs = read_csvs(args.csv_dir, strict_counts=not args.no_strict_counts)
    summary_rows: list[dict[str, object]] = []
    outputs = [
        make_fig01_reproduction(csvs["results_reproduction.csv"], args.fig_dir, summary_rows),
        make_fig02_partial(csvs["results_partial_data_sweep_full_L3p2.csv"], args.fig_dir, summary_rows),
        make_fig03_lowdata(csvs["results_lowdata_goldilocks_c4_d4_L3p2.csv"], args.fig_dir, summary_rows),
        make_fig04_random(csvs["results_random_sharing_control_full_L3p2_train450.csv"], args.fig_dir, summary_rows),
        make_fig05_depth(csvs["results_depth_sweep_all_groups_draft_L1234_p123.csv"], args.fig_dir, summary_rows),
        make_fig06_compression(csvs["results_d4_compression_sweep_L3p2.csv"], args.fig_dir, summary_rows),
        make_fig07_oracle(
            csvs["results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"],
            csvs["results_rule_based_oracle.csv"],
            args.fig_dir,
            summary_rows,
        ),
        make_fig08_oracle_ablation(
            csvs["results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"],
            args.fig_dir,
            summary_rows,
        ),
        make_fig09_params(
            csvs["results_partial_data_sweep_full_L3p2.csv"],
            csvs["results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"],
            args.fig_dir,
            summary_rows,
        ),
        make_fig10_schematic(args.fig_dir),
    ]
    write_summary(summary_rows, args.summary)
    verify_outputs(outputs, args.summary)
    print("Generated paper figure package:")
    for output in outputs:
        print(f"  {output.pdf}")
        print(f"  {output.png}")
    print(f"  {args.summary}")


if __name__ == "__main__":
    main()
