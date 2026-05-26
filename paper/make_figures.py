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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.groups_d4 import COORDS, DIRECTED_PAIRS, WIN_LINE_TRIPLES, apply_transform_to_board


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


def rule_oracle_accuracy() -> float:
    df = pd.read_csv(CSV_DIR / "results_rule_based_oracle.csv")
    if "accuracy" not in df.columns or df.empty:
        raise ValueError("results_rule_based_oracle.csv must contain a non-empty accuracy column")
    return float(df["accuracy"].iloc[0])


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


def rounded_panel(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "#f7f7f7",
    edgecolor: str = "#8f8f8f",
    linewidth: float = 0.8,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=0,
    )
    ax.add_patch(patch)


def draw_ttt_board(
    ax: plt.Axes,
    center: tuple[float, float],
    size: float,
    board: list[int] | np.ndarray | None = None,
    *,
    indices: bool = False,
    colors: bool = False,
    line_overlay: bool = False,
    edge_overlay: bool = False,
) -> None:
    cx, cy = center
    cell = size / 3.0
    for delta in [-0.5, 0.5]:
        ax.plot([cx + delta * cell, cx + delta * cell], [cy - 1.5 * cell, cy + 1.5 * cell], color="#222222", lw=0.75)
        ax.plot([cx - 1.5 * cell, cx + 1.5 * cell], [cy + delta * cell, cy + delta * cell], color="#222222", lw=0.75)

    def pos(index: int) -> tuple[float, float]:
        x, y = COORDS[index]
        return cx + x * cell, cy + y * cell

    if colors:
        for group, color in [((0, 2, 4, 6), "#d8f1df"), ((1, 3, 5, 7), "#fde0dd"), ((8,), "#fff8b5")]:
            for idx in group:
                x, y = pos(idx)
                ax.add_patch(
                    FancyBboxPatch(
                        (x - 0.31 * cell, y - 0.31 * cell),
                        0.62 * cell,
                        0.62 * cell,
                        boxstyle="round,pad=0.004,rounding_size=0.006",
                        facecolor=color,
                        edgecolor="#555555",
                        linewidth=0.55,
                        zorder=1,
                    )
                )

    if edge_overlay:
        for control, target in DIRECTED_PAIRS:
            x0, y0 = pos(control)
            x1, y1 = pos(target)
            vec = np.array([x1 - x0, y1 - y0], dtype=float)
            length = float(np.linalg.norm(vec))
            if length:
                unit = vec / length
                start = np.array([x0, y0]) + unit * cell * 0.22
                end = np.array([x1, y1]) - unit * cell * 0.22
                ax.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="-|>",
                        mutation_scale=5.6,
                        lw=0.55,
                        color="#777777",
                        alpha=0.72,
                        zorder=2,
                    )
                )

    if line_overlay:
        line_colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
        for i, line in enumerate(WIN_LINE_TRIPLES):
            coords = np.asarray([pos(idx) for idx in line])
            direction = coords[np.argmax(coords[:, 0] + 0.37 * coords[:, 1])] - coords[
                np.argmin(coords[:, 0] + 0.37 * coords[:, 1])
            ]
            if np.linalg.norm(direction) > 0:
                coords = coords[np.argsort(coords @ direction)]
            ax.plot(coords[:, 0], coords[:, 1], color=line_colors[i % len(line_colors)], lw=1.65, alpha=0.86, zorder=2)

    if board is not None:
        arr = np.asarray(board)
        for idx, value in enumerate(arr):
            x, y = pos(idx)
            if value == 1:
                ax.text(x, y, r"$\times$", color="#0072B2", fontsize=9.5, ha="center", va="center", fontweight="bold", zorder=5)
            elif value == -1:
                ax.text(x, y, r"$\circ$", color="#E69F00", fontsize=10.5, ha="center", va="center", fontweight="bold", zorder=5)

    if indices:
        for idx in range(9):
            x, y = pos(idx)
            ax.text(x, y, rf"$x_{idx}$", fontsize=6.3, ha="center", va="center", zorder=5)


def draw_data_grid(ax: plt.Axes, center: tuple[float, float], size: float, values: list[int]) -> None:
    cx, cy = center
    cell = size / 3.0
    for r in range(3):
        for c in range(3):
            x0 = cx - 1.5 * cell + c * cell
            y0 = cy + 1.5 * cell - (r + 1) * cell
            ax.add_patch(Rectangle((x0, y0), cell, cell, facecolor="white", edgecolor="#bdbdbd", linewidth=0.45))
    rows = [[0, 1, 2], [7, 8, 3], [6, 5, 4]]
    for r, row in enumerate(rows):
        for c, idx in enumerate(row):
            val = values[idx]
            color = "#0072B2" if val == 1 else "#E69F00" if val == -1 else "#777777"
            text = "+1" if val == 1 else "-1" if val == -1 else "0"
            ax.text(cx - cell + c * cell, cy + cell - r * cell, text, ha="center", va="center", fontsize=6.2, color=color)


def draw_gate_box(ax: plt.Axes, xy: tuple[float, float], text: str, *, color: str = "#9b6bd3", width: float = 0.145) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.055,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor="#f2ecfa",
            edgecolor=color,
            linewidth=0.8,
            zorder=2,
        )
    )
    ax.text(x + width / 2, y + 0.0275, text, ha="center", va="center", fontsize=6.7, color="#222222", zorder=3)


def make_fig1_protocol() -> None:
    fig = plt.figure(figsize=(7.2, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.02, 1.55], wspace=0.11)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

    # Panel 1: the task symmetry.
    ax = axes[0]
    rounded_panel(ax, (0.04, 0.18), 0.92, 0.74, facecolor="#f7f7f7", edgecolor="#8a8a8a", linewidth=0.9)
    board = np.array([1, 1, -1, 1, -1, 0, -1, 0, -1])
    draw_ttt_board(ax, (0.30, 0.62), 0.22, board)
    draw_ttt_board(ax, (0.70, 0.62), 0.22, apply_transform_to_board(board, "rot90"))
    draw_ttt_board(ax, (0.70, 0.34), 0.22, apply_transform_to_board(board, "reflect_vertical"))
    ax.add_patch(FancyArrowPatch((0.42, 0.62), (0.57, 0.62), arrowstyle="->", mutation_scale=12, lw=0.9, color="#333333"))
    ax.add_patch(FancyArrowPatch((0.42, 0.46), (0.57, 0.36), arrowstyle="->", mutation_scale=12, lw=0.9, color="#333333"))
    ax.text(0.30, 0.81, "game", ha="center", fontsize=7.0)
    ax.text(0.70, 0.81, r"$90^\circ$ rotation", ha="center", fontsize=7.0)
    ax.text(0.70, 0.20, "reflection", ha="center", fontsize=7.0)
    ax.text(0.50, 0.895, r"$D_4=\langle r,f\rangle$", ha="center", va="center", fontsize=9.0, color="#444444")
    ax.text(0.50, 0.095, "Symmetry", ha="center", va="center", fontsize=10.8)
    ax.text(0.08, 0.49, "same label", ha="center", va="center", fontsize=7.0, color="#E69F00", rotation=90)

    # Panel 2: data encoding and parameter orbits.
    ax = axes[1]
    rounded_panel(ax, (0.04, 0.18), 0.92, 0.74, facecolor="#f9f7ff", edgecolor="#9b6bd3", linewidth=0.9)
    draw_ttt_board(ax, (0.28, 0.68), 0.21, board)
    draw_data_grid(ax, (0.70, 0.68), 0.20, board.tolist())
    ax.add_patch(FancyArrowPatch((0.42, 0.68), (0.56, 0.68), arrowstyle="->", mutation_scale=12, lw=0.9, color="#333333"))
    ax.text(0.28, 0.84, "board", ha="center", fontsize=7.0)
    ax.text(0.70, 0.84, r"data $g_i$", ha="center", fontsize=7.0)
    draw_gate_box(ax, (0.18, 0.455), r"$x_i=\frac{2\pi}{3}g_i$", color="#9b6bd3", width=0.25)
    draw_gate_box(ax, (0.57, 0.455), r"$R_X(x_i)$", color="#9b6bd3", width=0.20)
    ax.text(0.50, 0.483, "=", ha="center", va="center", fontsize=10.0)
    draw_ttt_board(ax, (0.50, 0.335), 0.185, None, colors=True)
    ax.text(0.50, 0.205, "corner / edge / center orbits", ha="center", fontsize=6.4, color="#5a3a8a")
    ax.text(0.50, 0.095, "Encoding", ha="center", va="center", fontsize=10.8)

    # Panel 3: equivariant ansatz and task-aligned extension.
    ax = axes[2]
    rounded_panel(ax, (0.025, 0.18), 0.95, 0.74, facecolor="#f4ecfb", edgecolor="#9b6bd3", linewidth=0.9)
    ax.text(0.275, 0.84, "edge ansatz", ha="center", fontsize=7.0, color="#5a3a8a")
    ax.text(0.745, 0.84, "winning-line gates", ha="center", fontsize=7.0, color="#1b7f35")
    draw_ttt_board(ax, (0.275, 0.61), 0.285, None, edge_overlay=True)
    draw_ttt_board(ax, (0.745, 0.61), 0.285, None, edge_overlay=True, line_overlay=True)

    draw_gate_box(ax, (0.115, 0.365), r"$R_XR_Y$", color="#9b6bd3", width=0.125)
    draw_gate_box(ax, (0.302, 0.365), r"$CRY$", color="#9b6bd3", width=0.105)
    ax.text(0.275, 0.305, "orbit sharing", ha="center", fontsize=6.3, color="#5a3a8a")

    draw_gate_box(ax, (0.600, 0.385), r"$ZZZ$", color="#2ca02c", width=0.100)
    draw_gate_box(ax, (0.780, 0.385), r"$CCR_Z$", color="#2ca02c", width=0.120)
    draw_gate_box(ax, (0.670, 0.305), r"$+$ edge CRY", color="#2ca02c", width=0.165)
    ax.text(0.745, 0.255, "equivariant extension", ha="center", fontsize=6.3, color="#1b7f35")

    ax.add_patch(FancyArrowPatch((0.445, 0.60), (0.575, 0.60), arrowstyle="->", mutation_scale=12, lw=0.9, color="#333333"))
    ax.text(0.50, 0.095, "Equivariant gateset", ha="center", va="center", fontsize=10.8)
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
    oracle_acc = rule_oracle_accuracy()
    ax.axhline(oracle_acc, color="#D55E00", lw=1.05, ls=":", zorder=1)
    ax.text(
        750,
        oracle_acc - 0.014,
        "rule oracle\n100%",
        ha="right",
        va="top",
        fontsize=6.0,
        color="#D55E00",
    )
    ax.set_xticks([450, 600, 750])
    ax.set_ylim(0.64, 1.02)
    ax.set_yticks([0.65, 0.75, 0.85, 1.00])
    ax.set_xlabel("training examples")
    ax.set_title("task-aligned gates")
    grid(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.78), handlelength=1.8)
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
