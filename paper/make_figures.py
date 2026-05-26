"""Generate compact manuscript figures from checked experiment CSV files."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from src.groups_d4 import COORDS, DIRECTED_PAIRS, WIN_LINE_TRIPLES


CSV_DIR = ROOT / "results" / "csv"
GFX_DIR = Path(__file__).resolve().parent / "gfx"

SUBGROUP_ORDER = ["none", "Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"]
SUBGROUP_LABELS = {
    "none": "none",
    "Z2_rot180": "Z2 rot",
    "Z2_reflection": "Z2 refl",
    "C4": "C4",
    "D2_V4": "D2",
    "D4": "D4",
}
SUBGROUP_COLORS = {
    "none": "#8a8a8a",
    "Z2_rot180": "#E69F00",
    "Z2_reflection": "#56B4E9",
    "C4": "#009E73",
    "D2_V4": "#CC79A7",
    "D4": "#0072B2",
}
FAMILY_COLORS = {
    "edge": "#666666",
    "line_zzz": "#56B4E9",
    "line_ccrz": "#E69F00",
    "line_zzz_ccrz": "#009E73",
    "edge_line_zzz": "#D55E00",
    "edge_line_ccrz": "#CC79A7",
    "edge_line_zzz_ccrz": "#0072B2",
    "line_pair_crz": "#999999",
}
FAMILY_LABELS = {
    "edge": "edge",
    "line_zzz": "line ZZZ",
    "line_ccrz": "line CCRZ",
    "line_zzz_ccrz": "line both",
    "edge_line_zzz": "edge+ZZZ",
    "edge_line_ccrz": "edge+CCRZ",
    "edge_line_zzz_ccrz": "edge+both",
    "line_pair_crz": "pair CRZ",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.8,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ci95(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def mean_ci(df: pd.DataFrame, keys: list[str], metric: str = "test_accuracy") -> pd.DataFrame:
    grouped = (
        df.groupby(keys, dropna=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            n=(metric, "count"),
            ci95=(metric, ci95),
            gap=("generalization_gap", "mean"),
            params=("num_parameters", "mean"),
        )
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def save(fig: plt.Figure, name: str) -> None:
    GFX_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GFX_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8.5)


def grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, alpha=0.22, linewidth=0.55)


def board_xy(index: int) -> tuple[float, float]:
    x, y = COORDS[index]
    return float(x), float(y)


def setup_board(ax: plt.Axes) -> None:
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for value in [-0.5, 0.5]:
        ax.plot([value, value], [-1.3, 1.3], color="#d0d0d0", lw=0.8, zorder=0)
        ax.plot([-1.3, 1.3], [value, value], color="#d0d0d0", lw=0.8, zorder=0)
    for idx in range(9):
        x, y = board_xy(idx)
        ax.scatter([x], [y], s=115, facecolor="white", edgecolor="#333333", lw=1.0, zorder=3)
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=7.5, zorder=4)


def draw_arrow(ax: plt.Axes, start: int, end: int, color: str = "#777777", lw: float = 0.8) -> None:
    x0, y0 = board_xy(start)
    x1, y1 = board_xy(end)
    vec = np.array([x1 - x0, y1 - y0], dtype=float)
    length = float(np.linalg.norm(vec))
    if length == 0:
        return
    unit = vec / length
    shrink = 0.18
    p0 = np.array([x0, y0]) + unit * shrink
    p1 = np.array([x1, y1]) - unit * shrink
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=7.0,
        lw=lw,
        color=color,
        alpha=0.82,
        zorder=2,
    )
    ax.add_patch(arrow)


def draw_line(ax: plt.Axes, line: tuple[int, int, int], color: str, lw: float = 2.2) -> None:
    coords = np.asarray([board_xy(i) for i in line])
    center = coords.mean(axis=0)
    # Sort the line coordinates by projection so lines are drawn end-to-end.
    direction = coords[np.argmax(coords[:, 0] + 0.37 * coords[:, 1])] - coords[
        np.argmin(coords[:, 0] + 0.37 * coords[:, 1])
    ]
    if np.linalg.norm(direction) > 0:
        scores = coords @ direction
        coords = coords[np.argsort(scores)]
    ax.plot(coords[:, 0], coords[:, 1], color=color, lw=lw, alpha=0.90, zorder=1)
    ax.scatter([center[0]], [center[1]], s=11, color=color, zorder=2)


def make_fig1_protocol() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.95), constrained_layout=True)

    setup_board(axes[0])
    for control, target in DIRECTED_PAIRS:
        draw_arrow(axes[0], control, target)
    axes[0].set_title("edge CRY")
    add_panel_label(axes[0], "(a)")

    ax = axes[1]
    ax.set_title("subgroup tying")
    add_panel_label(ax, "(b)")
    ax.set_xlim(0.6, 8.55)
    ax.set_ylim(35, 220)
    orders = {"none": 1, "Z2_rot180": 2, "Z2_reflection": 2, "C4": 4, "D2_V4": 4, "D4": 8}
    params = {"none": 204, "Z2_rot180": 108, "Z2_reflection": 126, "C4": 60, "D2_V4": 78, "D4": 54}
    for subgroup in SUBGROUP_ORDER:
        jitter = -0.12 if subgroup == "Z2_rot180" else 0.12 if subgroup == "Z2_reflection" else 0.0
        jitter += -0.12 if subgroup == "C4" else 0.12 if subgroup == "D2_V4" else 0.0
        ax.scatter(
            orders[subgroup] + jitter,
            params[subgroup],
            s=28,
            color=SUBGROUP_COLORS[subgroup],
            zorder=3,
        )
        ax.text(
            orders[subgroup] + jitter,
            params[subgroup] + 7,
            SUBGROUP_LABELS[subgroup],
            ha="center",
            fontsize=5.8,
        )
    ax.set_xlabel("group order")
    ax.set_ylabel("parameters")
    ax.set_xticks([1, 2, 4, 8])
    grid(ax)

    setup_board(axes[2])
    for control, target in DIRECTED_PAIRS:
        draw_arrow(axes[2], control, target, color="#888888", lw=0.55)
    line_colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
    for idx, line in enumerate(WIN_LINE_TRIPLES):
        draw_line(axes[2], line, line_colors[idx % len(line_colors)])
    axes[2].set_title("edge + line gates")
    add_panel_label(axes[2], "(c)")

    save(fig, "fig1_protocol")


def make_fig2_main_evidence() -> None:
    repro = pd.read_csv(CSV_DIR / "results_reproduction.csv")
    partial = pd.read_csv(CSV_DIR / "results_partial_data_sweep_full_L3p2.csv")
    oracle = pd.read_csv(CSV_DIR / "results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.15), constrained_layout=True)

    ax = axes[0]
    summary = mean_ci(repro, ["subgroup"])
    order = ["none", "D4"]
    x = np.arange(len(order))
    means = [float(summary.loc[summary["subgroup"] == g, "mean"].iloc[0]) for g in order]
    errs = [float(summary.loc[summary["subgroup"] == g, "ci95"].iloc[0]) for g in order]
    ax.bar(x, means, yerr=errs, color=[SUBGROUP_COLORS[g] for g in order], capsize=2.5, width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "D4"])
    ax.set_ylim(0.50, 0.75)
    ax.set_ylabel("test accuracy")
    ax.set_title("reproduction anchor")
    grid(ax)
    add_panel_label(ax, "(a)")

    ax = axes[1]
    summary = mean_ci(partial, ["subgroup", "train_size"])
    for subgroup in SUBGROUP_ORDER:
        sub = summary[summary["subgroup"] == subgroup].sort_values("train_size")
        ax.errorbar(
            sub["train_size"],
            sub["mean"],
            yerr=sub["ci95"],
            lw=1.05 if subgroup not in {"C4", "D4"} else 1.45,
            marker="o",
            markersize=2.8,
            capsize=1.6,
            color=SUBGROUP_COLORS[subgroup],
            label=SUBGROUP_LABELS[subgroup],
        )
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120, 240, 450])
    ax.set_xticklabels(["30", "60", "120", "240", "450"], rotation=30, ha="right")
    ax.set_ylim(0.38, 0.73)
    ax.set_xlabel("training examples")
    ax.set_title("subgroup sweep")
    grid(ax)
    ax.legend(ncol=2, frameon=False, loc="lower right", columnspacing=0.8, handlelength=1.2)
    add_panel_label(ax, "(b)")

    ax = axes[2]
    summary = mean_ci(oracle, ["circuit_family", "subgroup", "train_size"])
    specs = [
        ("edge", "D4", "edge/D4", "#666666", "-"),
        ("edge", "C4", "edge/C4", "#666666", "--"),
        ("edge_line_zzz_ccrz", "D4", "edge+both/D4", "#0072B2", "-"),
        ("edge_line_zzz_ccrz", "C4", "edge+both/C4", "#0072B2", "--"),
    ]
    for family, subgroup, label, color, style in specs:
        sub = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == subgroup)]
        sub = sub.sort_values("train_size")
        ax.errorbar(
            sub["train_size"],
            sub["mean"],
            yerr=sub["ci95"],
            lw=1.35,
            ls=style,
            marker="o",
            markersize=3.0,
            capsize=1.8,
            color=color,
            label=label,
        )
    ax.set_xticks([450, 600, 750])
    ax.set_ylim(0.64, 0.84)
    ax.set_xlabel("training examples")
    ax.set_title("task-aligned gates")
    grid(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    add_panel_label(ax, "(c)")

    save(fig, "fig2_main_evidence")


def make_fig3_controls() -> None:
    random_df = pd.read_csv(CSV_DIR / "results_random_sharing_control_full_L3p2_train450.csv")
    oracle = pd.read_csv(CSV_DIR / "results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv")
    partial = pd.read_csv(CSV_DIR / "results_partial_data_sweep_full_L3p2.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.15), constrained_layout=True)

    ax = axes[0]
    summary = mean_ci(random_df, ["subgroup", "sharing_type"])
    groups = ["Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"]
    x = np.arange(len(groups))
    width = 0.34
    for offset, sharing, color, label in [
        (-width / 2, "random", "#bbbbbb", "random"),
        (width / 2, "symmetry", "#0072B2", "group"),
    ]:
        means = []
        errs = []
        for group in groups:
            row = summary[(summary["subgroup"] == group) & (summary["sharing_type"] == sharing)].iloc[0]
            means.append(float(row["mean"]))
            errs.append(float(row["ci95"]))
        ax.bar(x + offset, means, width=width, yerr=errs, color=color, capsize=1.8, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBGROUP_LABELS[g] for g in groups], rotation=32, ha="right")
    ax.set_ylim(0.50, 0.73)
    ax.set_ylabel("test accuracy")
    ax.set_title("matched sharing control")
    ax.legend(frameon=False, loc="upper left")
    grid(ax)
    add_panel_label(ax, "(a)")

    ax = axes[1]
    train600 = oracle[oracle["train_size"] == 600]
    families = [
        "edge",
        "line_zzz",
        "line_ccrz",
        "line_zzz_ccrz",
        "edge_line_zzz",
        "edge_line_ccrz",
        "edge_line_zzz_ccrz",
        "line_pair_crz",
    ]
    summary = mean_ci(train600, ["circuit_family", "subgroup"])
    for i, subgroup in enumerate(["C4", "D4"]):
        offset = -0.17 if subgroup == "C4" else 0.17
        means = []
        errs = []
        for family in families:
            row = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == subgroup)].iloc[0]
            means.append(float(row["mean"]))
            errs.append(float(row["ci95"]))
        ax.bar(
            np.arange(len(families)) + offset,
            means,
            width=0.32,
            yerr=errs,
            capsize=1.5,
            color=SUBGROUP_COLORS[subgroup],
            alpha=0.88,
            label=subgroup,
        )
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=52, ha="right")
    ax.set_ylim(0.62, 0.84)
    ax.set_title("oracle-inspired ablation")
    ax.legend(frameon=False, loc="upper left")
    grid(ax)
    add_panel_label(ax, "(b)")

    ax = axes[2]
    part450 = partial[partial["train_size"] == 450].copy()
    psummary = mean_ci(part450, ["subgroup"])
    osummary = mean_ci(oracle[oracle["train_size"] == 600], ["circuit_family", "subgroup"])
    for subgroup in SUBGROUP_ORDER:
        row = psummary[psummary["subgroup"] == subgroup].iloc[0]
        ax.scatter(
            row["params"],
            row["mean"],
            color=SUBGROUP_COLORS[subgroup],
            s=22,
            marker="o",
            label=SUBGROUP_LABELS[subgroup] if subgroup in {"none", "C4", "D4"} else None,
            alpha=0.9,
        )
    for family in ["line_zzz_ccrz", "edge_line_zzz", "edge_line_ccrz", "edge_line_zzz_ccrz"]:
        for subgroup in ["C4", "D4"]:
            row = osummary[(osummary["circuit_family"] == family) & (osummary["subgroup"] == subgroup)].iloc[0]
            ax.scatter(
                row["params"],
                row["mean"],
                color=FAMILY_COLORS[family],
                s=34 if family == "edge_line_zzz_ccrz" else 24,
                marker="^" if subgroup == "C4" else "s",
                edgecolor="white",
                linewidth=0.4,
                alpha=0.95,
            )
    ax.annotate("edge/D4", xy=(54, 0.688), xytext=(74, 0.675), arrowprops={"arrowstyle": "->", "lw": 0.5}, fontsize=6)
    ax.annotate("edge+both/D4", xy=(90, 0.805), xytext=(112, 0.807), arrowprops={"arrowstyle": "->", "lw": 0.5}, fontsize=6)
    ax.set_xlabel("parameters")
    ax.set_title("not just parameter count")
    ax.set_ylim(0.50, 0.84)
    ax.set_xlim(40, 215)
    grid(ax)
    add_panel_label(ax, "(c)")

    save(fig, "fig3_controls")


def main() -> None:
    configure_style()
    make_fig1_protocol()
    make_fig2_main_evidence()
    make_fig3_controls()
    for name in ["fig1_protocol.pdf", "fig2_main_evidence.pdf", "fig3_controls.pdf"]:
        path = GFX_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure: {path}")
        print(path)


if __name__ == "__main__":
    main()
