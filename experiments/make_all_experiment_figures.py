"""Generate main-paper and appendix figures for all meaningful experiment families."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from experiments import make_paper_figures as paper
from src.utils import CSV_DIR, FIGURES_DIR


APPENDIX_FIGURES_DIR = FIGURES_DIR / "appendix"
ALL_SUMMARY_PATH = CSV_DIR / "all_experiment_plot_summary.csv"
MANIFEST_PATH = CSV_DIR / "all_experiment_plot_manifest.csv"

APPENDIX_EXPECTED_ROWS = {
    "results_approximate_symmetry.csv": 72,
    "results_approximate_symmetry_stage2.csv": 27,
    "results_approximate_symmetry_stage2_fast.csv": 35,
    "results_line_ansatz_draft_L3p2.csv": 108,
    "results_lowdata_linezzz_all_groups_L3p2.csv": 216,
    "results_rule_based_oracle.csv": 1,
}

CANONICAL_CSVS = {
    "results_reproduction.csv": ("main", "reproduction anchor"),
    "results_partial_data_sweep_full_L3p2.csv": ("main", "full partial-equivariance data sweep"),
    "results_lowdata_goldilocks_c4_d4_L3p2.csv": ("main", "low-data C4 vs D4 zoom"),
    "results_random_sharing_control_full_L3p2_train450.csv": ("main", "parameter-matched random-sharing control"),
    "results_depth_sweep_all_groups_draft_L1234_p123.csv": ("main", "depth and expressivity sweep"),
    "results_d4_compression_sweep_L3p2.csv": ("main", "beyond-D4 compression sweep"),
    "results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv": (
        "main",
        "robust oracle-inspired architecture sweep",
    ),
    "results_approximate_symmetry.csv": ("appendix", "exploratory approximate-symmetry sweep"),
    "results_approximate_symmetry_stage2.csv": ("appendix", "exploratory approximate-symmetry stage 2"),
    "results_approximate_symmetry_stage2_fast.csv": (
        "appendix",
        "5-seed exploratory approximate-symmetry train-450 run",
    ),
    "results_line_ansatz_draft_L3p2.csv": ("appendix", "early line-ansatz draft sweep"),
    "results_lowdata_linezzz_all_groups_L3p2.csv": ("appendix", "low-data line-zzz all-groups sweep"),
    "results_rule_based_oracle.csv": ("appendix", "deterministic rule oracle sanity check"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--paper-fig-dir", type=Path, default=paper.PAPER_FIGURES_DIR)
    parser.add_argument("--appendix-fig-dir", type=Path, default=APPENDIX_FIGURES_DIR)
    parser.add_argument("--paper-summary", type=Path, default=paper.SUMMARY_PATH)
    parser.add_argument("--summary", type=Path, default=ALL_SUMMARY_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--no-strict-counts", action="store_true")
    return parser.parse_args()


def read_csv(csv_dir: Path, filename: str, expected_rows: int | None, strict: bool) -> pd.DataFrame:
    path = csv_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if strict and expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{filename}: expected {expected_rows} rows, found {len(df)}")
    return df


def classify_csv(filename: str) -> tuple[str, str]:
    if filename in CANONICAL_CSVS:
        return CANONICAL_CSVS[filename]
    stem = filename.removesuffix(".csv")
    if "smoke" in stem:
        return "excluded_smoke", "smoke/debug run"
    if stem.endswith("_old") or "_old" in stem:
        return "excluded_duplicate", "old duplicate result"
    if "quick" in stem or "fast" in stem:
        return "excluded_quick", "quick/fast exploratory duplicate"
    if stem in {
        "results_repro_L3p2",
        "results_oracle_inspired_robust_C4D4_L3p2",
        "results_partial_equivariance",
        "results_partial_equivariance_450_fast",
        "results_partial_equivariance_mid",
        "results_partial_main_L3p2",
        "results_depth_sweep",
        "results_depth_focus_fast",
        "results_medium_paper_l1p1",
        "results_paperlike_l1p1_seed0",
        "results_paper_minimum",
        "results_quick_compare",
        "results_quick_compare_paper_block",
        "results_random_sharing_control",
        "results_random_sharing_control_subset",
    }:
        return "excluded_duplicate", "superseded by canonical main or appendix run"
    return "excluded_other", "not part of meaningful plot package"


def write_manifest(csv_dir: Path, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(csv_dir.glob("results_*.csv")):
        df = pd.read_csv(path)
        category, reason = classify_csv(path.name)
        rows.append(
            {
                "filename": path.name,
                "rows": len(df),
                "category": category,
                "reason": reason,
            }
        )
    manifest = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def save(fig: plt.Figure, fig_dir: Path, stem: str) -> paper.FigureOutput:
    return paper.save_figure(fig, fig_dir, stem)


def make_a1_approximate(
    csvs: dict[str, pd.DataFrame],
    fig_dir: Path,
    rows: list[dict[str, object]],
) -> paper.FigureOutput:
    specs = [
        ("results_approximate_symmetry.csv", "early exploratory"),
        ("results_approximate_symmetry_stage2.csv", "stage 2 exploratory"),
        ("results_approximate_symmetry_stage2_fast.csv", "5-seed train-450"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), constrained_layout=True)
    for ax, (filename, title) in zip(axes, specs):
        df = csvs[filename].copy()
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        summary = paper.mean_ci(df, ["subgroup", "epsilon"], "test_accuracy")
        for subgroup in paper.ordered_existing(summary["subgroup"], paper.SUBGROUP_ORDER):
            sub = summary[summary["subgroup"] == subgroup].sort_values("epsilon")
            ax.errorbar(
                sub["epsilon"],
                sub["mean"],
                yerr=sub["ci95"],
                marker=paper.SUBGROUP_MARKERS.get(subgroup, "o"),
                color=paper.SUBGROUP_COLORS.get(subgroup, "#333333"),
                linewidth=1.2,
                markersize=3.5,
                capsize=2,
                label=paper.clean_label(subgroup),
            )
        ax.set_title(title)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("test accuracy")
        paper.add_grid(ax)
        paper.append_summary(rows, figure="A1", source=filename, df=df, keys=["subgroup", "epsilon"])
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    return save(fig, fig_dir, "appx_a1_approximate_symmetry_collection")


def make_a2_line_draft(
    df: pd.DataFrame,
    fig_dir: Path,
    rows: list[dict[str, object]],
) -> paper.FigureOutput:
    summary = paper.mean_ci(df, ["circuit_family", "subgroup", "train_size"], "test_accuracy")
    families = ["edge", "line_zzz"]
    subgroups = ["none", "C4", "D4"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True, sharey=True)
    for ax, family in zip(axes, families):
        for subgroup in subgroups:
            sub = summary[(summary["circuit_family"] == family) & (summary["subgroup"] == subgroup)]
            if sub.empty:
                continue
            sub = sub.sort_values("train_size")
            ax.errorbar(
                sub["train_size"],
                sub["mean"],
                yerr=sub["ci95"],
                marker=paper.SUBGROUP_MARKERS.get(subgroup, "o"),
                color=paper.SUBGROUP_COLORS.get(subgroup, "#333333"),
                linewidth=1.4,
                markersize=4,
                capsize=2,
                label=paper.clean_label(subgroup),
            )
        ax.set_title(paper.FAMILY_LABELS.get(family, family))
        ax.set_xlabel("training examples")
        ax.set_xticks(sorted(df["train_size"].unique()))
        paper.add_grid(ax)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0.38, 0.72)
    axes[1].legend(frameon=False, loc="lower right")
    paper.append_summary(
        rows,
        figure="A2",
        source="results_line_ansatz_draft_L3p2.csv",
        df=df,
        keys=["circuit_family", "subgroup", "train_size"],
    )
    return save(fig, fig_dir, "appx_a2_line_ansatz_draft")


def make_a3_line_lowdata(
    df: pd.DataFrame,
    fig_dir: Path,
    rows: list[dict[str, object]],
) -> paper.FigureOutput:
    summary = paper.mean_ci(df, ["subgroup", "train_size"], "test_accuracy")
    fig, ax = plt.subplots(figsize=(7.0, 3.25), constrained_layout=True)
    for subgroup in paper.ordered_existing(summary["subgroup"], paper.SUBGROUP_ORDER):
        sub = summary[summary["subgroup"] == subgroup].sort_values("train_size")
        ax.errorbar(
            sub["train_size"],
            sub["mean"],
            yerr=sub["ci95"],
            marker=paper.SUBGROUP_MARKERS.get(subgroup, "o"),
            color=paper.SUBGROUP_COLORS.get(subgroup, "#333333"),
            linewidth=1.4 if subgroup in {"C4", "D4"} else 1.0,
            markersize=3.6,
            capsize=1.8,
            label=paper.clean_label(subgroup),
        )
    ax.set_xscale("log")
    train_sizes = sorted(df["train_size"].unique())
    ax.set_xticks(train_sizes)
    ax.set_xticklabels([str(int(value)) for value in train_sizes], rotation=35, ha="right")
    ax.set_xlabel("training examples")
    ax.set_ylabel("test accuracy")
    ax.set_title("Low-data line-zzz sweep")
    ax.set_ylim(0.25, 0.66)
    paper.add_grid(ax)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    paper.append_summary(
        rows,
        figure="A3",
        source="results_lowdata_linezzz_all_groups_L3p2.csv",
        df=df,
        keys=["subgroup", "train_size"],
    )
    return save(fig, fig_dir, "appx_a3_lowdata_linezzz_all_groups")


def make_a4_oracle(
    df: pd.DataFrame,
    fig_dir: Path,
    rows: list[dict[str, object]],
) -> paper.FigureOutput:
    row = df.iloc[0]
    labels = ["circle", "draw", "cross", "overall"]
    values = [
        float(row["accuracy_circle"]),
        float(row["accuracy_draw"]),
        float(row["accuracy_cross"]),
        float(row["accuracy"]),
    ]
    counts = [int(row["n_circle"]), int(row["n_draw"]), int(row["n_cross"]), int(row["n_examples"])]
    colors = ["#56B4E9", "#009E73", "#E69F00", "#0072B2"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.7)
    for xi, value, count in zip(x, values, counts):
        ax.text(xi, value - 0.045, f"n={count}", ha="center", va="top", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title("Deterministic rule oracle sanity check")
    paper.add_grid(ax)
    ax.text(
        0.02,
        0.06,
        "0 trainable parameters; D4-invariant",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    for label, value, count in zip(labels, values, counts):
        rows.append(
            {
                "figure": "A4",
                "source": "results_rule_based_oracle.csv",
                "metric": "oracle_accuracy",
                "class": label,
                "n": count,
                "mean": value,
                "std": 0.0,
                "ci95": 0.0,
                "params": 0.0,
            }
        )
    return save(fig, fig_dir, "appx_a4_rule_based_oracle")


def wrap_reason(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def make_a5_manifest(
    manifest: pd.DataFrame,
    fig_dir: Path,
) -> paper.FigureOutput:
    display = manifest.copy()
    display["label"] = display["filename"].str.replace("results_", "", regex=False).str.replace(".csv", "", regex=False)
    display["reason_wrapped"] = display["reason"].map(wrap_reason)
    category_order = {
        "main": 0,
        "appendix": 1,
        "excluded_duplicate": 2,
        "excluded_quick": 3,
        "excluded_smoke": 4,
        "excluded_other": 5,
    }
    display["category_order"] = display["category"].map(category_order).fillna(99)
    display = display.sort_values(["category_order", "filename"]).reset_index(drop=True)
    height = max(5.0, 0.27 * len(display) + 1.0)
    fig, ax = plt.subplots(figsize=(10.5, height), constrained_layout=True)
    ax.set_axis_off()
    category_colors = {
        "main": "#0072B2",
        "appendix": "#009E73",
        "excluded_quick": "#E69F00",
        "excluded_duplicate": "#999999",
        "excluded_smoke": "#CC79A7",
        "excluded_other": "#d9d9d9",
    }
    y_positions = np.arange(len(display))[::-1]
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(display) + 1)
    ax.text(0.01, len(display) + 0.35, "Result CSV", fontweight="bold")
    ax.text(0.49, len(display) + 0.35, "Rows", fontweight="bold")
    ax.text(0.58, len(display) + 0.35, "Category", fontweight="bold")
    ax.text(0.76, len(display) + 0.35, "Reason", fontweight="bold")
    for y, (_, row) in zip(y_positions, display.iterrows()):
        color = category_colors.get(str(row["category"]), "#d9d9d9")
        ax.add_patch(Rectangle((0.0, y - 0.38), 1.0, 0.72, color="#f7f7f7" if y % 2 else "white", zorder=0))
        ax.add_patch(Rectangle((0.58, y - 0.20), 0.035, 0.40, color=color, zorder=1))
        ax.text(0.01, y, str(row["label"]), va="center", fontsize=7.2)
        ax.text(0.50, y, str(int(row["rows"])), va="center", ha="right", fontsize=7.2)
        ax.text(0.62, y, str(row["category"]).replace("_", " "), va="center", fontsize=7.2)
        ax.text(0.76, y, str(row["reason_wrapped"]), va="center", fontsize=7.0)
    ax.set_title("Experiment coverage map")
    return save(fig, fig_dir, "appx_a5_experiment_coverage_map")


def append_paper_summary(summary_path: Path, rows: list[dict[str, object]]) -> None:
    if not summary_path.exists():
        return
    paper_summary = pd.read_csv(summary_path)
    for _, row in paper_summary.iterrows():
        entry = row.to_dict()
        entry["section"] = "paper"
        rows.append(entry)


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    leading = ["section", "figure", "source", "metric", "n", "mean", "std", "ci95", "params"]
    other = [column for column in df.columns if column not in leading]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[leading + other].to_csv(output_path, index=False)


def verify_outputs(outputs: list[paper.FigureOutput], summary_path: Path, manifest_path: Path) -> None:
    missing: list[Path] = []
    for output in outputs:
        for path in [output.pdf, output.png]:
            if not path.exists() or path.stat().st_size <= 0:
                missing.append(path)
    for path in [summary_path, manifest_path]:
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(path)
    if missing:
        raise RuntimeError("Missing or empty outputs:\n" + "\n".join(str(path) for path in missing))


def main() -> None:
    args = parse_args()
    strict = not args.no_strict_counts
    paper.configure_style()

    paper_csvs = paper.read_csvs(args.csv_dir, strict_counts=strict)
    paper_rows: list[dict[str, object]] = []
    paper_outputs = [
        paper.make_fig01_reproduction(paper_csvs["results_reproduction.csv"], args.paper_fig_dir, paper_rows),
        paper.make_fig02_partial(paper_csvs["results_partial_data_sweep_full_L3p2.csv"], args.paper_fig_dir, paper_rows),
        paper.make_fig03_lowdata(paper_csvs["results_lowdata_goldilocks_c4_d4_L3p2.csv"], args.paper_fig_dir, paper_rows),
        paper.make_fig04_random(paper_csvs["results_random_sharing_control_full_L3p2_train450.csv"], args.paper_fig_dir, paper_rows),
        paper.make_fig05_depth(paper_csvs["results_depth_sweep_all_groups_draft_L1234_p123.csv"], args.paper_fig_dir, paper_rows),
        paper.make_fig06_compression(paper_csvs["results_d4_compression_sweep_L3p2.csv"], args.paper_fig_dir, paper_rows),
        paper.make_fig07_oracle(
            paper_csvs["results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"],
            paper_csvs["results_rule_based_oracle.csv"],
            args.paper_fig_dir,
            paper_rows,
        ),
        paper.make_fig08_oracle_ablation(
            paper_csvs["results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"],
            args.paper_fig_dir,
            paper_rows,
        ),
        paper.make_fig09_params(
            paper_csvs["results_partial_data_sweep_full_L3p2.csv"],
            paper_csvs["results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"],
            args.paper_fig_dir,
            paper_rows,
        ),
        paper.make_fig10_schematic(args.paper_fig_dir),
    ]
    paper.write_summary(paper_rows, args.paper_summary)
    paper.verify_outputs(paper_outputs, args.paper_summary)

    appendix_csvs = {
        filename: read_csv(args.csv_dir, filename, rows, strict)
        for filename, rows in APPENDIX_EXPECTED_ROWS.items()
    }
    manifest = write_manifest(args.csv_dir, args.manifest)
    appendix_rows: list[dict[str, object]] = []
    appendix_outputs = [
        make_a1_approximate(appendix_csvs, args.appendix_fig_dir, appendix_rows),
        make_a2_line_draft(appendix_csvs["results_line_ansatz_draft_L3p2.csv"], args.appendix_fig_dir, appendix_rows),
        make_a3_line_lowdata(
            appendix_csvs["results_lowdata_linezzz_all_groups_L3p2.csv"],
            args.appendix_fig_dir,
            appendix_rows,
        ),
        make_a4_oracle(appendix_csvs["results_rule_based_oracle.csv"], args.appendix_fig_dir, appendix_rows),
        make_a5_manifest(manifest, args.appendix_fig_dir),
    ]

    all_rows: list[dict[str, object]] = []
    append_paper_summary(args.paper_summary, all_rows)
    for row in appendix_rows:
        row["section"] = "appendix"
        all_rows.append(row)
    write_summary(all_rows, args.summary)
    verify_outputs(paper_outputs + appendix_outputs, args.summary, args.manifest)

    print("Generated complete experiment figure suite:")
    for output in paper_outputs:
        print(f"  {output.pdf}")
        print(f"  {output.png}")
    for output in appendix_outputs:
        print(f"  {output.pdf}")
        print(f"  {output.png}")
    print(f"  {args.summary}")
    print(f"  {args.manifest}")


if __name__ == "__main__":
    main()
