"""Matplotlib figure generation for qc_symmetry experiment CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import FIGURES_DIR


def _mean_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    return df


def _heatmap(ax, table: pd.DataFrame, title: str, value_label: str) -> None:
    if table.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return
    image = ax.imshow(table.values, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("p")
    ax.set_ylabel("L")
    ax.set_title(title)
    for row_idx, row_value in enumerate(table.index):
        for col_idx, col_value in enumerate(table.columns):
            value = table.loc[row_value, col_value]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(value_label)


def plot_reproduction(
    csv_path: Path,
    output_path: Path = FIGURES_DIR / "fig_reproduction_train_test_accuracy.pdf",
) -> None:
    df = _mean_frame(csv_path)
    grouped = (
        df.groupby(["subgroup", "L", "p"], as_index=False)[
            ["train_accuracy", "test_accuracy", "generalization_gap"]
        ]
        .mean()
        .sort_values(["subgroup", "L", "p"])
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row_idx, subgroup in enumerate(["none", "D4"]):
        sub = grouped[grouped["subgroup"] == subgroup]
        for col_idx, metric in enumerate(["train_accuracy", "test_accuracy"]):
            table = sub.pivot(index="L", columns="p", values=metric).sort_index()
            _heatmap(
                axes[row_idx, col_idx],
                table,
                f"{subgroup}: {metric.replace('_', ' ')}",
                "accuracy",
            )

    summary = df.groupby("subgroup", as_index=False)[["test_accuracy", "generalization_gap"]].mean()
    axes[0, 2].bar(summary["subgroup"], summary["test_accuracy"], color=["#4c78a8", "#f58518"])
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("Mean test accuracy")
    axes[0, 2].set_ylabel("accuracy")
    axes[1, 2].bar(summary["subgroup"], summary["generalization_gap"], color=["#4c78a8", "#f58518"])
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_title("Mean generalization gap")
    axes[1, 2].set_ylabel("train - test")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_partial_equivariance(
    csv_path: Path,
    output_path: Path = FIGURES_DIR / "fig_partial_equivariance_data_sweep.pdf",
) -> None:
    df = _mean_frame(csv_path)
    grouped = df.groupby(["subgroup", "train_size"], as_index=False)[
        ["train_accuracy", "test_accuracy", "generalization_gap", "num_parameters", "subgroup_order"]
    ].mean()
    subgroups = list(grouped["subgroup"].drop_duplicates())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for subgroup in subgroups:
        sub = grouped[grouped["subgroup"] == subgroup].sort_values("train_size")
        axes[0].plot(sub["train_size"], sub["test_accuracy"], marker="o", label=subgroup)
        axes[1].plot(sub["train_size"], sub["generalization_gap"], marker="o", label=subgroup)
        axes[2].plot(sub["train_size"], sub["train_accuracy"], marker="o", label=subgroup)
    axes[0].set_title("Test accuracy vs train size")
    axes[1].set_title("Generalization gap vs train size")
    axes[2].set_title("Train accuracy vs train size")
    for ax in axes:
        ax.set_xlabel("train size")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("train - test")
    axes[2].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_approximate_symmetry(
    csv_path: Path,
    output_path: Path = FIGURES_DIR / "fig_approximate_symmetry.pdf",
) -> None:
    df = _mean_frame(csv_path).copy()
    for column in ["epsilon", "train_size"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = (
        df.groupby(["subgroup", "epsilon", "train_size"], as_index=False)[
            ["train_accuracy", "test_accuracy", "generalization_gap"]
        ]
        .mean()
        .sort_values(["subgroup", "epsilon", "train_size"])
    )
    epsilon_levels = sorted(grouped["epsilon"].unique().tolist())
    unique_train_sizes = sorted(grouped["train_size"].unique().tolist())
    subgroup_levels = grouped["subgroup"].dropna().unique().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Collapse train size for main curves to keep the figure readable.
    mean_over_size = (
        grouped.groupby(["subgroup", "epsilon"], as_index=False)[
            ["train_accuracy", "test_accuracy", "generalization_gap"]
        ]
        .mean()
    )
    for subgroup in subgroup_levels:
        sub = mean_over_size[mean_over_size["subgroup"] == subgroup].sort_values("epsilon")
        axes[0].plot(sub["epsilon"], sub["test_accuracy"], marker="o", label=subgroup)
        axes[1].plot(sub["epsilon"], sub["generalization_gap"], marker="o", label=subgroup)

    # Heatmap of subgroup × epsilon test accuracy (averaged across train sizes).
    heat = (
        grouped.groupby(["subgroup", "epsilon"], as_index=False)["test_accuracy"]
        .mean()
        .pivot(index="subgroup", columns="epsilon", values="test_accuracy")
        .reindex(index=subgroup_levels)
    )
    img = axes[2].imshow(
        heat.values,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_xlabel("epsilon")
    axes[2].set_xticks(range(len(heat.columns)))
    axes[2].set_xticklabels([f"{float(x):.2f}" for x in heat.columns], rotation=30)
    axes[2].set_ylabel("subgroup")
    axes[2].set_yticks(range(len(heat.index)))
    axes[2].set_yticklabels(heat.index)
    axes[2].set_title("Test accuracy heatmap (avg over train sizes)")
    colorbar = plt.colorbar(img, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("accuracy")

    axes[0].set_title("Test accuracy vs epsilon")
    axes[1].set_title("Generalization gap vs epsilon")
    for axis in axes[:2]:
        axis.set_xlabel("epsilon")
        axis.set_ylabel(axis.get_title())
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    if unique_train_sizes:
        plt.suptitle(
            "Approximate symmetry sweep: epsilon={}".format(
                ", ".join(str(size) for size in unique_train_sizes)
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_parameter_count(
    csv_path: Path,
    output_path: Path = FIGURES_DIR / "fig_parameter_count.pdf",
) -> None:
    df = _mean_frame(csv_path)
    summary = (
        df.groupby(["subgroup", "subgroup_order"], as_index=False)["num_parameters"]
        .mean()
        .sort_values(["subgroup_order", "num_parameters"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(summary["subgroup"], summary["num_parameters"], color="#4c78a8")
    axes[0].set_title("Parameter count")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("parameters")

    acc_summary = df.groupby(["subgroup", "num_parameters", "subgroup_order"], as_index=False)[
        "test_accuracy"
    ].mean()
    axes[1].scatter(acc_summary["num_parameters"], acc_summary["test_accuracy"], s=60)
    for _, row in acc_summary.iterrows():
        axes[1].annotate(row["subgroup"], (row["num_parameters"], row["test_accuracy"]), fontsize=8)
    axes[1].set_title("Test accuracy vs parameters")
    axes[1].set_xlabel("parameters")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_ylim(0, 1)

    order_summary = df.groupby(["subgroup", "subgroup_order"], as_index=False)["test_accuracy"].mean()
    axes[2].scatter(order_summary["subgroup_order"], order_summary["test_accuracy"], s=60)
    for _, row in order_summary.iterrows():
        axes[2].annotate(row["subgroup"], (row["subgroup_order"], row["test_accuracy"]), fontsize=8)
    axes[2].set_title("Test accuracy vs subgroup order")
    axes[2].set_xlabel("subgroup order")
    axes[2].set_ylabel("test accuracy")
    axes[2].set_ylim(0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_depth_sweep(
    csv_path: Path,
    output_path: Path = FIGURES_DIR / "fig_depth_vs_symmetry.pdf",
) -> None:
    df = _mean_frame(csv_path)
    df = df.copy()
    df["depth"] = df["L"] * df["p"]
    grouped = df.groupby(["subgroup", "depth"], as_index=False)[
        ["train_accuracy", "test_accuracy", "generalization_gap"]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for subgroup in grouped["subgroup"].drop_duplicates():
        sub = grouped[grouped["subgroup"] == subgroup].sort_values("depth")
        axes[0].plot(sub["depth"], sub["test_accuracy"], marker="o", label=subgroup)
        axes[1].plot(sub["depth"], sub["train_accuracy"], marker="o", label=subgroup)
        axes[2].plot(sub["depth"], sub["generalization_gap"], marker="o", label=subgroup)
    axes[0].set_title("Test accuracy vs depth")
    axes[1].set_title("Train accuracy vs depth")
    axes[2].set_title("Generalization gap vs depth")
    for ax in axes:
        ax.set_xlabel("L * p")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("accuracy")
    axes[2].set_ylabel("train - test")
    axes[0].legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_random_sharing_control(
    csv_path: Path,
    output_path: Path = FIGURES_DIR / "fig_random_sharing_control.pdf",
) -> None:
    df = _mean_frame(csv_path)
    grouped = df.groupby(["subgroup", "sharing_type"], as_index=False)[
        ["test_accuracy", "generalization_gap"]
    ].mean()
    labels = sorted(grouped["subgroup"].unique())
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for idx, metric in enumerate(["test_accuracy", "generalization_gap"]):
        for offset, sharing_type in [(-width / 2, "symmetry"), (width / 2, "random")]:
            values = []
            for subgroup in labels:
                row = grouped[(grouped["subgroup"] == subgroup) & (grouped["sharing_type"] == sharing_type)]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            axes[idx].bar(x + offset, values, width=width, label=sharing_type)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels, rotation=30, ha="right")
        axes[idx].set_title(metric.replace("_", " "))
        axes[idx].grid(True, axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("train - test")
    axes[0].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
