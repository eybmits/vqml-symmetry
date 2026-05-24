"""Generate paper-oriented tables and plots from an experiment CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.plotting import (
    plot_depth_sweep,
    plot_parameter_count,
    plot_partial_equivariance,
    plot_approximate_symmetry,
    plot_random_sharing_control,
    plot_reproduction,
)
from src.reporting import (
    write_approximate_symmetry_summary_table,
    write_main_results_table,
)
from src.utils import CSV_DIR, FIGURES_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--kind",
        choices=("partial", "reproduction", "depth", "random", "approximate"),
        default="partial",
    )
    parser.add_argument("--table-name", default="table_main_results")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.kind == "partial":
        table = write_main_results_table(
            args.csv,
            output_csv=CSV_DIR / f"{args.table_name}.csv",
            output_md=CSV_DIR / f"{args.table_name}.md",
        )
        print(table.to_string(index=False))
        if not args.no_plots:
            plot_partial_equivariance(args.csv)
            plot_parameter_count(args.csv)
    elif args.kind == "reproduction":
        if not args.no_plots:
            plot_reproduction(args.csv)
    elif args.kind == "depth":
        if not args.no_plots:
            plot_depth_sweep(args.csv)
    elif args.kind == "random":
        if not args.no_plots:
            plot_random_sharing_control(args.csv)
    elif args.kind == "approximate":
        if not args.no_plots:
            plot_approximate_symmetry(args.csv)
        table = write_approximate_symmetry_summary_table(
            args.csv,
            output_csv=CSV_DIR / f"{args.table_name}.csv",
            output_md=CSV_DIR / f"{args.table_name}.md",
        )
        print(table.to_string(index=False))
    else:
        raise ValueError(args.kind)
    print(f"Wrote summaries under {CSV_DIR} and figures under {FIGURES_DIR}.")


if __name__ == "__main__":
    main()
