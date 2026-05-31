"""Generate compact manuscript figures from checked experiment CSV files.

Figure 1 is maintained as the standalone four-panel TikZ/PDF composite
``fig1_4panel_standalone.pdf`` and is only verified here.
"""

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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.groups_d4 import COORDS, DIRECTED_PAIRS, WIN_LINE_TRIPLES, apply_transform_to_board


CSV_DIR = ROOT / "results" / "csv"
GFX_DIR = Path(__file__).resolve().parent / "gfx"
TRAIN_SIZES = [30, 60, 120, 240, 450, 600]
EDGE_RESULTS = CSV_DIR / "results_paper_consistent_edge_L3p2.csv"
EDGE_LINES_RESULTS = CSV_DIR / "results_paper_consistent_edge_lines_L3p2.csv"
ABLATION_RESULTS = CSV_DIR / "results_paper_consistent_ablation_L3p2_train600.csv"
RANDOM_RESULTS = CSV_DIR / "results_paper_consistent_random_sharing_L3p2_train600.csv"
FIG1_PROTOCOL = Path(__file__).resolve().parent / "fig1_4panel_standalone.pdf"

SUBGROUP_ORDER = ["none", "Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"]
SUBGROUP_LABELS = {
    "none": "none",
    "Z2_rot180": r"$Z_2 r$",
    "Z2_reflection": r"$Z_2 f$",
    "C4": r"$C_4$",
    "D2_V4": r"$D_2$",
    "D4": r"$D_4$",
}
SUBGROUP_COLORS = {
    "none": "#9B9B9B",
    "Z2_rot180": "#7D9441",
    "Z2_reflection": "#5AA6BF",
    "C4": "#C9473A",
    "D2_V4": "#337FAE",
    "D4": "#1F1F1F",
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
    "line_zzz": "ZZZ",
    "line_ccrz": "CCRZ",
    "line_zzz_ccrz": "ZZZ+CCRZ",
    "edge_line_zzz": "edge+ZZZ",
    "edge_line_ccrz": "edge+CCRZ",
    "edge_line_zzz_ccrz": "edge+lines",
    "line_pair_crz": "pair-CRZ",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "axes.labelcolor": "#000000",
            "text.color": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "legend.labelcolor": "#000000",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
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
    fig.savefig(GFX_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.12, label, transform=ax.transAxes, fontweight="bold", fontsize=11.0, color="#000000")


def grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, alpha=0.25, linewidth=0.55, color="#B8B8B8")


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(length=3.8, width=1.0, colors="#000000")
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)


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


def verify_fig1_protocol() -> None:
    """Figure 1 is authored as the standalone four-panel TikZ/PDF composite."""
    if not FIG1_PROTOCOL.exists() or FIG1_PROTOCOL.stat().st_size == 0:
        raise RuntimeError(f"Missing canonical Figure 1 PDF: {FIG1_PROTOCOL}")


def make_fig2_main_evidence() -> None:
    edge = pd.read_csv(EDGE_RESULTS)
    edge_lines = pd.read_csv(EDGE_LINES_RESULTS)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.62))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.205, wspace=0.30)

    ax = axes[0]
    e600 = edge[edge["train_size"] == 600]
    order = ["none", "D4"]
    xbase = {"none": 0.0, "D4": 1.0}
    fill_color = {"none": "#B6B6B2", "D4": "#C9473A"}
    dot_color = {"none": "#4D4D4D", "D4": "#9E3229"}
    vals = {g: e600[e600["subgroup"] == g]["test_accuracy"].to_numpy(dtype=float) for g in order}

    # Half-violin (kernel density) on the outer side of each group.
    try:
        from scipy.stats import gaussian_kde

        _kde_ok = True
    except Exception:
        _kde_ok = False
    ygrid = np.linspace(0.565, 0.745, 240)
    for g in order:
        side = -1.0 if g == "none" else 1.0
        base = xbase[g]
        if _kde_ok and np.unique(vals[g]).size > 1:
            dens = gaussian_kde(vals[g])(ygrid)
            dens = dens / dens.max() * 0.30
            ax.fill_betweenx(
                ygrid,
                base + side * 0.07,
                base + side * (0.07 + dens),
                color=fill_color[g],
                alpha=0.20,
                lw=0.9,
                edgecolor=fill_color[g],
                zorder=2,
            )

    # Paired per-seed connectors with rain dots: each seed trains both none and D4
    # on the same split, so the links show that nearly every run improves.
    none_by_seed = e600[e600["subgroup"] == "none"].set_index("seed")["test_accuracy"]
    d4_by_seed = e600[e600["subgroup"] == "D4"].set_index("seed")["test_accuracy"]
    paired_seeds = sorted(set(none_by_seed.index) & set(d4_by_seed.index))
    jitter_rng = np.random.default_rng(1)
    for seed in paired_seeds:
        xn = xbase["none"] + 0.07 + jitter_rng.random() * 0.11
        xd = xbase["D4"] - 0.07 - jitter_rng.random() * 0.11
        y0 = float(none_by_seed[seed])
        y1 = float(d4_by_seed[seed])
        ax.plot([xn, xd], [y0, y1], color="#8C8C8C", lw=0.7, alpha=0.30, solid_capstyle="round", zorder=3)
        ax.scatter([xn], [y0], s=17, color=dot_color["none"], edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter([xd], [y1], s=17, color=dot_color["D4"], edgecolor="white", linewidth=0.4, zorder=4)

    # Mean +/- 95% CI marker at each group centre.
    for g in order:
        m = float(np.mean(vals[g]))
        ci = ci95(pd.Series(vals[g]))
        base = xbase[g]
        ax.plot([base, base], [m - ci, m + ci], color="#1F1F1F", lw=2.2, solid_capstyle="round", zorder=5)
        ax.scatter([base], [m], s=40, color=fill_color[g], edgecolor="#1F1F1F", linewidth=1.0, zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["none", r"$D_4$"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.555, 0.755)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75])
    ax.set_ylabel("test accuracy")
    ax.set_xlabel("")
    grid(ax)
    style_axes(ax)
    add_panel_label(ax, "a")

    ax = axes[1]
    summary = mean_ci(edge, ["subgroup", "train_size"])
    x_values = TRAIN_SIZES
    x_pos = np.arange(len(x_values), dtype=float)
    line_styles = {
        "D4": {"marker": "o", "ls": "-", "lw": 2.5, "zorder": 5},
        "C4": {"marker": "^", "ls": "-", "lw": 2.4, "zorder": 5},
        "D2_V4": {"marker": "D", "ls": "-", "lw": 2.0, "zorder": 4},
        "Z2_rot180": {"marker": "s", "ls": "-", "lw": 1.9, "zorder": 3},
        "Z2_reflection": {"marker": "v", "ls": "-", "lw": 1.9, "zorder": 3},
        "none": {"marker": "x", "ls": "--", "lw": 1.8, "zorder": 2},
    }
    for subgroup in ["D4", "C4", "D2_V4", "Z2_rot180", "Z2_reflection", "none"]:
        sub = summary[summary["subgroup"] == subgroup].sort_values("train_size")
        xpos = [x_values.index(int(value)) for value in sub["train_size"]]
        style = line_styles[subgroup]
        mean = sub["mean"].to_numpy(dtype=float)
        ci = sub["ci95"].to_numpy(dtype=float)
        ax.fill_between(
            xpos,
            mean - ci,
            mean + ci,
            color=SUBGROUP_COLORS[subgroup],
            alpha=0.11,
            linewidth=0.0,
            zorder=style["zorder"] - 1.5,
        )
        ax.plot(
            xpos,
            mean,
            lw=style["lw"],
            ls=style["ls"],
            marker=style["marker"],
            markersize=5.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=SUBGROUP_COLORS[subgroup],
            label=SUBGROUP_LABELS[subgroup],
            zorder=style["zorder"],
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_xlim(-0.18, len(x_values) - 0.82)
    ax.set_ylim(0.42, 0.71)
    ax.set_yticks([0.45, 0.55, 0.65])
    ax.set_xlabel("training examples", labelpad=3.0)
    grid(ax)
    style_axes(ax)
    add_panel_label(ax, "b")
    legend_b = ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=9.5,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.0,
        labelspacing=0.3,
        borderaxespad=0.5,
    )
    legend_b.set_zorder(10)

    ax = axes[2]
    combined = pd.concat([edge, edge_lines], ignore_index=True)
    summary = mean_ci(combined, ["circuit_family", "subgroup", "train_size"])
    specs = [
        ("edge", "D4", r"edge/$D_4$", "#6F6F6F", "-"),
        ("edge", "C4", r"edge/$C_4$", "#8B8B8B", "--"),
        ("edge_line_zzz_ccrz", "D4", r"edge+lines/$D_4$", "#1F1F1F", "-"),
        ("edge_line_zzz_ccrz", "C4", r"edge+lines/$C_4$", "#C9473A", "--"),
    ]
    x_values_c = TRAIN_SIZES
    x_pos_c = np.arange(len(x_values_c), dtype=float)
    for family, subgroup, label, color, style in specs:
        sub = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == subgroup)]
        sub = sub.sort_values("train_size")
        xpos = [x_values_c.index(int(value)) for value in sub["train_size"]]
        mean = sub["mean"].to_numpy(dtype=float)
        ci = sub["ci95"].to_numpy(dtype=float)
        zbase = 4 if family == "edge_line_zzz_ccrz" else 3
        ax.fill_between(
            xpos,
            mean - ci,
            mean + ci,
            color=color,
            alpha=0.13,
            linewidth=0.0,
            zorder=zbase - 1.5,
        )
        ax.plot(
            xpos,
            mean,
            lw=2.5,
            ls=style,
            marker="o" if family == "edge_line_zzz_ccrz" else "s",
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=color,
            label=label,
            zorder=zbase,
        )
    ax.set_xticks(x_pos_c)
    ax.set_xticklabels([str(value) for value in x_values_c])
    ax.set_xlim(-0.18, len(x_values_c) - 0.18)
    ax.set_ylim(0.50, 0.81)
    ax.set_yticks([0.55, 0.65, 0.75])
    ax.set_xlabel("training examples", labelpad=3.0)
    grid(ax)
    style_axes(ax)
    add_panel_label(ax, "c")

    order_c = [r"edge+lines/$D_4$", r"edge+lines/$C_4$", r"edge/$D_4$", r"edge/$C_4$"]
    handles_c, labels_c = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels_c, handles_c))
    ordered_c = [(label_to_handle[name], name) for name in order_c if name in label_to_handle]
    legend_c = ax.legend(
        [handle for handle, _ in ordered_c],
        [name for _, name in ordered_c],
        loc="lower right",
        ncol=1,
        fontsize=8.4,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.24,
        borderaxespad=0.35,
    )
    legend_c.set_zorder(10)

    for panel_ax in axes:
        panel_ax.set_box_aspect(1.0)

    save(fig, "fig2_main_evidence")


def make_fig3_controls() -> None:
    random_df = pd.read_csv(RANDOM_RESULTS)
    ablation = pd.read_csv(ABLATION_RESULTS)
    edge = pd.read_csv(EDGE_RESULTS)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.95))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.30, wspace=0.30)

    ax = axes[0]
    summary = mean_ci(random_df, ["subgroup", "sharing_type"])
    groups = ["Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"]
    x = np.arange(len(groups))
    for idx, group in enumerate(groups):
        random_row = summary[(summary["subgroup"] == group) & (summary["sharing_type"] == "random")].iloc[0]
        orbit_row = summary[(summary["subgroup"] == group) & (summary["sharing_type"] == "symmetry")].iloc[0]
        ax.plot(
            [idx - 0.13, idx + 0.13],
            [random_row["mean"], orbit_row["mean"]],
            color="#C9C9C4",
            linewidth=1.15,
            zorder=1.0,
        )
        ax.errorbar(
            idx - 0.13,
            random_row["mean"],
            yerr=random_row["ci95"],
            fmt="o",
            markersize=4.9,
            markerfacecolor="white",
            markeredgecolor="#A9A9A2",
            markeredgewidth=1.4,
            ecolor="#777777",
            elinewidth=1.0,
            capsize=2.2,
            zorder=3.0,
        )
        ax.errorbar(
            idx + 0.13,
            orbit_row["mean"],
            yerr=orbit_row["ci95"],
            fmt="o",
            markersize=4.9,
            markerfacecolor="#1F1F1F",
            markeredgecolor="#1F1F1F",
            markeredgewidth=1.2,
            ecolor="#1F1F1F",
            elinewidth=1.0,
            capsize=2.2,
            zorder=4.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SUBGROUP_LABELS[g] for g in groups])
    ax.set_ylim(0.50, 0.795)
    ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    ax.set_ylabel("test accuracy")
    grid(ax)
    style_axes(ax)
    add_panel_label(ax, "a")
    handles_a = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#A9A9A2",
            markeredgewidth=1.4,
            markersize=7.0,
            label="random",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1F1F1F",
            markeredgecolor="#1F1F1F",
            markersize=7.0,
            label="group orbit",
        ),
    ]
    legend_a = ax.legend(
        handles=handles_a,
        loc="upper left",
        ncol=2,
        fontsize=9.5,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.32,
        borderaxespad=0.4,
    )
    legend_a.set_zorder(10)

    ax = axes[1]
    train600 = ablation[ablation["train_size"] == 600]
    family_candidates = [
        "edge",
        "line_pair_crz",
        "line_zzz",
        "line_ccrz",
        "line_zzz_ccrz",
        "edge_line_zzz",
        "edge_line_ccrz",
        "edge_line_zzz_ccrz",
    ]
    summary = mean_ci(train600, ["circuit_family", "subgroup"])
    family_rank = (
        summary[summary["circuit_family"].isin(family_candidates) & summary["subgroup"].isin(["C4", "D4"])]
        .groupby("circuit_family")["mean"]
        .mean()
        .sort_values(kind="mergesort")
    )
    families = [family for family in family_rank.index if family in family_candidates]
    for idx, family in enumerate(families):
        c4_row = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == "C4")].iloc[0]
        d4_row = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == "D4")].iloc[0]
        ax.plot(
            [idx - 0.13, idx + 0.13],
            [c4_row["mean"], d4_row["mean"]],
            color="#D9D9D9",
            linewidth=1.1,
            zorder=1.0,
        )
        ax.errorbar(
            idx - 0.13,
            c4_row["mean"],
            yerr=c4_row["ci95"],
            fmt="o",
            markersize=4.8,
            markerfacecolor="#6F9EAA",
            markeredgecolor="#547F88",
            markeredgewidth=1.0,
            ecolor="#202020",
            elinewidth=1.0,
            capsize=2.0,
            zorder=3.0,
        )
        ax.errorbar(
            idx + 0.13,
            d4_row["mean"],
            yerr=d4_row["ci95"],
            fmt="o",
            markersize=4.8,
            markerfacecolor="#C9473A",
            markeredgecolor="#9E3229",
            markeredgewidth=1.0,
            ecolor="#202020",
            elinewidth=1.0,
            capsize=2.0,
            zorder=4.0,
        )
    if "edge" in families:
        edge_idx = families.index("edge")
        edge_d4_row = summary[(summary["circuit_family"] == "edge") & (summary["subgroup"] == "D4")].iloc[0]
        edge_d4_x = edge_idx + 0.13
        edge_d4_y = edge_d4_row["mean"]
        ax.scatter(
            [edge_d4_x],
            [edge_d4_y],
            s=132,
            facecolor="none",
            edgecolor="#000000",
            linewidth=0.65,
            zorder=5.0,
        )
        ax.annotate(
            r"original edge/$\mathbf{D}_4$",
            xy=(edge_d4_x, edge_d4_y),
            xytext=(edge_d4_x + 0.25, edge_d4_y - 0.034),
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.55,
                "color": "#000000",
                "shrinkA": 3.5,
                "shrinkB": 5.5,
                "mutation_scale": 7.0,
                "connectionstyle": "arc3,rad=0.0",
            },
            fontsize=9.0,
            color="#000000",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.28},
            ha="left",
            va="top",
            zorder=8.0,
        )
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=52, ha="right")
    ax.set_ylim(0.62, 0.84)
    ax.set_yticks([0.62, 0.65, 0.70, 0.75, 0.80, 0.84])
    ax.set_ylabel("test accuracy")
    grid(ax)
    style_axes(ax)
    add_panel_label(ax, "b")
    handles_b = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6F9EAA",
            markeredgecolor="#547F88",
            markeredgewidth=1.0,
            markersize=7.0,
            label=SUBGROUP_LABELS["C4"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#C9473A",
            markeredgecolor="#9E3229",
            markeredgewidth=1.0,
            markersize=7.0,
            label=SUBGROUP_LABELS["D4"],
        ),
    ]
    legend_b = ax.legend(
        handles=handles_b,
        loc="upper left",
        ncol=1,
        fontsize=9.5,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
        labelspacing=0.32,
        borderaxespad=0.5,
    )
    legend_b.set_zorder(10)

    ax = axes[2]
    part600 = edge[edge["train_size"] == 600].copy()
    psummary = mean_ci(part600, ["subgroup"])
    osummary = mean_ci(ablation[ablation["train_size"] == 600], ["circuit_family", "subgroup"])
    for subgroup in SUBGROUP_ORDER:
        row = psummary[psummary["subgroup"] == subgroup].iloc[0]
        ax.scatter(
            row["params"],
            row["mean"],
            facecolor="white",
            edgecolor="#7A7A7A",
            s=78,
            marker="o",
            linewidth=1.25,
            alpha=1.0,
            zorder=4.2,
        )
    for family in ["edge_line_zzz", "edge_line_ccrz", "edge_line_zzz_ccrz"]:
        for subgroup in ["C4", "D4"]:
            row = osummary[(osummary["circuit_family"] == family) & (osummary["subgroup"] == subgroup)].iloc[0]
            ax.scatter(
                row["params"],
                row["mean"],
                color="#C9473A",
                s=96 if family == "edge_line_zzz_ccrz" else 66,
                marker="^",
                edgecolor="white",
                linewidth=0.6,
                alpha=1.0 if family == "edge_line_zzz_ccrz" else 0.88,
                zorder=4.4,
            )
    edge_d4 = psummary[psummary["subgroup"] == "D4"].iloc[0]
    edge_lines_d4 = osummary[
        (osummary["circuit_family"] == "edge_line_zzz_ccrz") & (osummary["subgroup"] == "D4")
    ].iloc[0]
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.45}
    ax.scatter(
        [edge_d4["params"]],
        [edge_d4["mean"]],
        s=132,
        marker="o",
        facecolor="none",
        edgecolor="#000000",
        linewidth=0.65,
        zorder=5.2,
    )
    ax.scatter(
        [edge_lines_d4["params"]],
        [edge_lines_d4["mean"]],
        s=214,
        marker="o",
        facecolor="none",
        edgecolor="#000000",
        linewidth=0.65,
        zorder=5.2,
    )
    ax.annotate(
        r"original edge/$\mathbf{D}_4$",
        xy=(edge_d4["params"], edge_d4["mean"]),
        xytext=(58, 0.652),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.55,
            "color": "#000000",
            "shrinkA": 4.0,
            "shrinkB": 8.0,
            "mutation_scale": 7.0,
            "connectionstyle": "arc3,rad=0.06",
        },
        fontsize=9.0,
        color="#000000",
        fontweight="bold",
        bbox=label_box,
        ha="left",
        va="top",
        zorder=8,
    )
    ax.annotate(
        r"edge+lines/$\mathbf{D}_4$",
        xy=(edge_lines_d4["params"], edge_lines_d4["mean"]),
        xytext=(111, 0.818),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.55,
            "color": "#000000",
            "shrinkA": 2.5,
            "shrinkB": 8.0,
            "mutation_scale": 7.0,
            "connectionstyle": "arc3,rad=0.14",
        },
        fontsize=9.0,
        color="#000000",
        fontweight="bold",
        bbox=label_box,
        ha="left",
        va="center",
        zorder=8,
    )
    ax.set_xlabel("parameters")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.55, 0.83)
    ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83])
    ax.set_xlim(40, 225)
    ax.set_xticks([40, 100, 150, 215])
    grid(ax)
    style_axes(ax)
    add_panel_label(ax, "c")

    handles_c = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#7A7A7A", markeredgewidth=1.35, markersize=8.5, label="edge / baseline"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#C9473A", markeredgecolor="white", markersize=8.5, label="edge+lines"),
    ]
    legend_c = ax.legend(
        handles=handles_c,
        loc="lower left",
        ncol=1,
        fontsize=9.5,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.5,
        labelspacing=0.34,
        borderaxespad=0.5,
    )
    legend_c.set_zorder(10)

    for panel_ax in axes:
        panel_ax.set_box_aspect(1.0)

    save(fig, "fig3_controls")


def main() -> None:
    configure_style()
    verify_fig1_protocol()
    make_fig2_main_evidence()
    make_fig3_controls()
    print(FIG1_PROTOCOL)
    for name in ["fig2_main_evidence.pdf", "fig3_controls.pdf"]:
        path = GFX_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure: {path}")
        print(path)


if __name__ == "__main__":
    main()
