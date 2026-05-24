"""Small, fast comparison for first-look Tic-Tac-Toe symmetry results."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_common_args, default_csv_path, run_configs, seeds_from_args
from src.plotting import plot_parameter_count
from src.utils import FIGURES_DIR, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(epochs=20, steps_per_epoch=10, test_size=300, single_qubit_block="ry")
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--subgroups", default="none,D4")
    parser.add_argument("--output", default=str(default_csv_path("results_quick_compare.csv")))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    subgroups = [item.strip() for item in args.subgroups.split(",") if item.strip()]
    configs = [
        ExperimentConfig(
            subgroup=subgroup,
            L=args.L,
            p=args.p,
            seed=seed,
            train_size=args.train_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            pl_device=args.pl_device,
            diff_method=args.diff_method,
            single_qubit_block=args.single_qubit_block,
            circuit_family=args.circuit_family,
            allow_overlap_if_needed=not args.strict_disjoint,
        )
        for subgroup in subgroups
        for seed in seeds_from_args(args)
    ]
    csv_path = Path(args.output)
    run_configs(configs, csv_path, resume=args.resume)
    if not args.no_plots:
        plot_parameter_count(csv_path, FIGURES_DIR / "fig_quick_compare_parameter_count.pdf")


if __name__ == "__main__":
    main()
