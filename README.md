# qc_symmetry

Reproducible experiments for:

**Partial Equivariance as a Tunable Inductive Bias in Variational Quantum Learning**

This project reproduces and extends the Tic-Tac-Toe experiment from Meyer et al.,
“Exploiting symmetry in variational quantum machine learning”. It compares
unconstrained variational quantum classifiers with subgroup-equivariant models
under partial and full subgroups of the Tic-Tac-Toe board symmetry group `D4`.

The goal is a controlled benchmark of inductive bias and expressivity. The code
does not claim quantum advantage.

## Setup

The local Anaconda Python 3.11 environment already has the core dependencies.
From the repository root:

```bash
/Users/markus/anaconda3/bin/python -m pip install -r requirements.txt
```

Run all sanity checks plus a tiny smoke training run:

```bash
/Users/markus/anaconda3/bin/python -m src.sanity_checks
```

Skip the smoke training if you only want structural checks:

```bash
/Users/markus/anaconda3/bin/python -m src.sanity_checks --skip-smoke
```

The default simulator backend is `lightning.qubit` with adjoint gradients.
Use `--pl-device default.qubit --diff-method backprop` to fall back to the
pure Python simulator.

## Short Paper Draft

The IEEE-style short-paper draft is in `paper/`:

- `paper/main.pdf` is the current compiled manuscript.
- `paper/main.tex` is the LaTeX entry point.
- `paper/txt/` contains the section text.
- `paper/gfx/` contains tracked vector figures used by the manuscript.
- `paper/make_figures.py` regenerates the manuscript-specific figures from the checked CSV results.

Build the manuscript from the repository root:

```bash
cd paper
make paper
```

Or run the steps explicitly:

```bash
/Users/markus/anaconda3/bin/python make_figures.py
latexmk -pdf -interaction=nonstopmode main.tex
```

The main manuscript story is:

1. Meyer-style full `D4` equivariance improves over no symmetry.
2. Partial equivariance, especially `C4`, closely tracks full `D4` on this benchmark.
3. Parameter-matched random sharing underperforms group-orbit sharing.
4. The strongest current result is the oracle-inspired equivariant `edge_line_zzz_ccrz` ansatz, which adds winning-line interactions while preserving equivariance.

## Experiments

Paper plan and experiment roadmap:

```bash
open docs/paper_plan.md
```

Minimum paper-draft sweep over all six symmetry variants:

```bash
/Users/markus/anaconda3/bin/python -m experiments.paper_minimum_sweep
```

Generate the core paper table and plots from a partial-sweep CSV:

```bash
/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  results/csv/results_paper_minimum.csv \
  --kind partial \
  --table-name table_paper_minimum
```

Stage 1 reproduction, debug grid with five seeds:

```bash
/Users/markus/anaconda3/bin/python -m experiments.reproduce_original
```

Fast first-look comparison with minimal depth and fewer optimizer steps:

```bash
/Users/markus/anaconda3/bin/python -m experiments.quick_compare
```

Full 20-seed reproduction over `L,p in {1,...,5}`:

```bash
/Users/markus/anaconda3/bin/python -m experiments.reproduce_original --final --full-grid
```

Partial-equivariance train-size sweep:

```bash
/Users/markus/anaconda3/bin/python -m experiments.partial_equivariance_sweep
```

The same data-regime sweep is also available under the requested alias:

```bash
/Users/markus/anaconda3/bin/python -m experiments.data_regime_sweep
```

Depth-vs-symmetry sweep:

```bash
/Users/markus/anaconda3/bin/python -m experiments.depth_sweep
```

Parameter-matched random-sharing control:

```bash
/Users/markus/anaconda3/bin/python -m experiments.random_sharing_control
```

Deterministic rule-based oracle sanity baseline:

```bash
/Users/markus/anaconda3/bin/python -m experiments.rule_based_oracle --print-blueprint
```

Winner-line 3-qubit ansatz comparison:

```bash
/Users/markus/anaconda3/bin/python -m experiments.line_ansatz_sweep \
  --circuit-families edge,line_zzz \
  --subgroups none,C4,D4 \
  --train-sizes 30,120,450 \
  --L 3 --p 2
```

Main paper experiments use `--single-qubit-block paper` by default. The
minimal `RY` block remains available via `--single-qubit-block ry` for quick
ablations. Scripts also accept `--epochs`, `--steps-per-epoch`, `--batch-size`,
`--lr`, `--seeds`, `--final`, `--test-size`, and `--no-plots`.

## Outputs

CSV files are written to `results/csv/`:

- `results_reproduction.csv`
- `results_partial_equivariance.csv`
- `results_depth_sweep.csv`
- `results_random_sharing_control.csv`
- `results_paper_minimum.csv`
- `results_approximate_symmetry.csv`
- `results_rule_based_oracle.csv`
- `results_line_ansatz_sweep.csv`
- `table_paper_minimum.csv`
- `table_paper_minimum.md`
- `table_approximate_symmetry.csv`
- `table_approximate_symmetry.md`

Figures are written to `results/figures/`:

- `fig_reproduction_train_test_accuracy.pdf`
- `fig_partial_equivariance_data_sweep.pdf`
- `fig_depth_vs_symmetry.pdf`
- `fig_parameter_count.pdf`
- `fig_random_sharing_control.pdf`
- `fig_approximate_symmetry.pdf`

## Data Split Note

The unique legal board-state dataset contains 316 circle-win states, 626
cross-win states, and 4536 draw-labeled states. Therefore an exactly disjoint
balanced `450` train / `600` test split is impossible for unique board states:
it would require 350 circle-win states. The splitter uses disjoint balanced
splits whenever feasible and otherwise falls back to original-like independent
balanced train/test sampling, logging `actual_disjoint`, `split_mode`, and
`overlap_count` in each CSV row. Use `--strict-disjoint` to fail instead of
falling back.

## Model

Each model uses 9 qubits, RX data encoding with angle `2*pi/3*x_i`, and
data-reuploading layers. A layer applies the data encoding followed by `p`
trainable blocks. The paper-style block uses subgroup-tied `RX` and `RY`
single-qubit gates plus directed `CRY` gates. The lower-parameter `RY`-only
block is kept as an ablation.

The invariant output vector is:

- circle win: average `Z` over corners
- draw: `Z` over center
- cross win: average `Z` over edges

Loss is mean squared vector error against labels in `{-1,+1}^3`.

## Exact Rule Oracle

The task also has a deterministic symbolic solution: check the 8 winning
lines for three crosses or three circles and otherwise output draw. The module
`src.rule_based` implements this exact oracle with zero trainable parameters
and verifies 100% accuracy on all legal generated boards. It is not a competing
learning model; it is an oracle sanity check that makes clear that the paper is
about inductive bias and generalization, not quantum advantage.

The same module includes a reversible-circuit blueprint using an orthogonal
two-bit encoding per board cell, with 3-control winner-line checks via
`MultiControlledX`. This exact blueprint cannot be implemented on top of the
current one-qubit RX ternary embedding without changing the data encoding, but
the winner-line triples are useful candidates for future trainable 3-qubit
equivariant ansatz blocks.

Trainable oracle-inspired variants are exposed through `--circuit-family`.
They keep the same RX data embedding, single-qubit paper block, invariant
observables, and subgroup parameter sharing, but replace directed `CRY` edge
entanglers with task-aligned 3-qubit motifs on Tic-Tac-Toe winning-line triples:

- `line_zzz`: trainable `MultiRZ(theta)` interactions.
- `line_ccrz`: trainable two-control phase rotations on each line.
- `line_zzz_ccrz`: combines both line interactions.
- `edge_line_zzz`: original edge `CRY` entanglers plus `line_zzz`.
- `edge_line_ccrz`: original edge `CRY` entanglers plus `line_ccrz`.
- `edge_line_zzz_ccrz`: original edge `CRY` entanglers plus both line motifs.
- `line_pair_crz`: controlled phase rotations on directed pairs inside each winning line.

The controlled-phase variants are closer to the logical line-checking oracle
while remaining exactly subgroup-invariant because the diagonal line operations
commute.

### Approximate symmetry sweep (novel extension)

Add controlled `ε`-corruption to emulate approximate symmetry:

- generate all legal states as usual,
- group states by full `D4` orbits,
- select a fraction `ε` of those orbits using a seed,
- swap circle and cross labels on selected orbits (draw labels unchanged).

Run:

```bash
/Users/markus/anaconda3/bin/python -m experiments.approximate_symmetry_sweep \
  --eps-values 0.0,0.05,0.10,0.20,0.40 \
  --train-sizes 30,60,120,240,450 \
  --L 3 --p 2 \
  --output results/csv/results_approximate_symmetry.csv \
  --seeds 0,1,2,3,4
```

Summarize and plot:

```bash
/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  results/csv/results_approximate_symmetry.csv \
  --kind approximate \
  --table-name table_approximate_symmetry
```

## Parameter Counts

Parameters per trainable block for the edge-CRY family:

| subgroup | `RY` block | paper block |
|---|---:|---:|
| `none` | 25 | 34 |
| `Z2_rot180` | 13 | 18 |
| `Z2_reflection` | 15 | 21 |
| `C4` | 7 | 10 |
| `D2_V4` | 9 | 13 |
| `D4` | 6 | 9 |

For the winner-line and hybrid families with the paper single-qubit block:

| subgroup | `line_zzz` / `line_ccrz` / `line_pair_crz` | `line_zzz_ccrz` | `edge_line_zzz` / `edge_line_ccrz` | `edge_line_zzz_ccrz` |
|---|---:|---:|---:|---:|
| `none` | 26 | 34 | 42 | 50 |
| `Z2_rot180` | 16 | 22 | 24 | 30 |
| `Z2_reflection` | 18 | 24 | 27 | 33 |
| `C4` | 9 | 12 | 13 | 16 |
| `D2_V4` | 13 | 18 | 18 | 23 |
| `D4` | 9 | 12 | 12 | 15 |

Total trainable parameters are `L * p * parameters_per_block`.
