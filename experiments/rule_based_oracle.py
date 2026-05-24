"""Evaluate the deterministic Tic-Tac-Toe rule oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.rule_based import (
    evaluate_oracle,
    format_reversible_oracle_blueprint,
    oracle_is_d4_invariant,
)
from src.utils import CSV_DIR, ensure_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(CSV_DIR / "results_rule_based_oracle.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--print-blueprint",
        action="store_true",
        help="Print the exact reversible-circuit blueprint.",
    )
    args = parser.parse_args()

    ensure_results_dirs()
    row = {
        "model": "deterministic_rule_oracle",
        "dataset": "all_legal_states",
        "d4_invariant": oracle_is_d4_invariant(),
        **evaluate_oracle(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)

    print(pd.DataFrame([row]).to_string(index=False))
    print(f"Wrote {output}")
    if args.print_blueprint:
        print()
        print(format_reversible_oracle_blueprint())


if __name__ == "__main__":
    main()
