"""Stage 1: reproduce baseline vs full D4 Tic-Tac-Toe comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_common_args, default_csv_path, run_configs, seeds_from_args
from src.plotting import plot_reproduction
from src.utils import ExperimentConfig, parse_int_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(single_qubit_block="paper")
    parser.add_argument("--L-values", default="1,2,3")
    parser.add_argument("--p-values", default="1,2,3")
    parser.add_argument("--full-grid", action="store_true", help="Use L,p in 1..5.")
    parser.add_argument("--train-size", type=int, default=450)
    parser.add_argument("--output", default=str(default_csv_path("results_reproduction.csv")))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = seeds_from_args(args)
    l_values = [1, 2, 3, 4, 5] if args.full_grid else parse_int_list(args.L_values)
    p_values = [1, 2, 3, 4, 5] if args.full_grid else parse_int_list(args.p_values)
    configs = [
        ExperimentConfig(
            subgroup=subgroup,
            L=L,
            p=p,
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
        for subgroup in ("none", "D4")
        for L in l_values
        for p in p_values
        for seed in seeds
    ]
    csv_path = Path(args.output)
    run_configs(configs, csv_path, resume=args.resume)
    if not args.no_plots:
        plot_reproduction(csv_path)


if __name__ == "__main__":
    main()
