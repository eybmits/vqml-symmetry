"""Parameter-matched random-sharing controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_common_args, default_csv_path, run_configs, seeds_from_args
from src.plotting import plot_random_sharing_control
from src.utils import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=450)
    parser.add_argument("--output", default=str(default_csv_path("results_random_sharing_control.csv")))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = seeds_from_args(args)
    configs: list[ExperimentConfig] = []
    for seed in seeds:
        configs.append(
            ExperimentConfig(
                subgroup="none",
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
        )
    for subgroup in ("Z2_reflection", "Z2_rot180", "C4", "D2_V4", "D4"):
        for random_sharing in (False, True):
            for seed in seeds:
                configs.append(
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
                        random_sharing=random_sharing,
                        allow_overlap_if_needed=not args.strict_disjoint,
                    )
                )
    csv_path = Path(args.output)
    run_configs(configs, csv_path, resume=args.resume)
    if not args.no_plots:
        plot_random_sharing_control(csv_path)


if __name__ == "__main__":
    main()
