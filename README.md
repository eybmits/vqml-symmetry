# qc_symmetry

Reproducibility package for the short paper:

**Symmetry as a Design Axis in Variational Quantum Learning**

The repository contains the final checked results, figure-generation code,
LaTeX source, generated manuscript PDFs, and generated figure PDFs needed to
inspect and reproduce the paper package. The benchmark is Tic-Tac-Toe board
classification with exact labels and known `D4` board symmetries. The goal is
to study symmetry-aware ansatz design in a controlled setting, not to claim
quantum advantage.

The checked-in manuscript outputs are `paper/main.pdf` with author metadata and
`paper/main_anonymous.pdf` for anonymous review.

## Repository layout

- `paper/`: manuscript source, generated PDFs, and figure-generation code.
- `results/csv/`: final checked CSV/JSON artifacts used by the figures and tables.
- `src/`: Tic-Tac-Toe data, symmetry groups, circuits, training, and checks.
- `experiments/`: final experiment matrix and artifact validation.
- `scripts/`: one-command reproduction and optional full rerun entry points.

## Setup

Use Python 3.11 or a compatible recent Python 3 environment:

```bash
python3 -m pip install -r requirements.txt
```

The default simulation path uses PennyLane with `lightning.qubit` and adjoint
gradients.

## Reproduce the paper from checked artifacts

The primary reviewer path validates the committed CSV/JSON artifacts,
regenerates the paper figures and summary table, and rebuilds the manuscript:

```bash
scripts/reproduce_paper_from_artifacts.sh
```

This runs:

```bash
python3 -m src.sanity_checks --skip-smoke
python3 -m experiments.consistent_paper_experiments --stage validate --epochs 100
python3 paper/make_figures.py
cd paper && make paper
```

The expected manuscripts are `paper/main.pdf` and `paper/main_anonymous.pdf`.

## Final evidence artifacts

The final paper evidence lives in `results/csv/`:

- `results_paper_consistent_edge_L3p2.csv`
- `results_paper_consistent_edge_lines_L3p2.csv`
- `results_paper_consistent_ablation_L3p2_train600.csv`
- `results_paper_consistent_random_sharing_L3p2_train600.csv`
- `results_paper_training_budget_audit_L3p2_train600.csv`
- `table_paper_consistent_summary.csv`
- `paper_training_budget_decision_L3p2.json`

Validation checks the shared optimizer protocol and expected row counts:

- edge subgroup sweep: `6` subgroups x `6` train sizes x `10` seeds = `360`
- edge+lines sweep: `2` subgroups x `6` train sizes x `10` seeds = `120`
- ablation sweep: `8` families x `2` subgroups x `10` seeds = `160`
- random-sharing control: `110` rows at train size `600`

The generated paper figures are:

- `paper/fig1_4panel_standalone.pdf`
- `paper/gfx/fig2_main_evidence.pdf`
- `paper/gfx/fig3_controls.pdf`

## Optional full experiment rerun

The committed CSV/JSON artifacts are sufficient for review and exact figure
reproduction. To rerun the final experiment matrix from scratch, use:

```bash
EPOCHS=100 SHARDS=5 scripts/run_consistent_paper_experiments.sh
```

Set `RUN_AUDIT=1` to rerun the training-budget audit before the final matrix.
The rerun script writes the same final CSV names, validates them, regenerates
figures, and rebuilds the manuscript.

## Paper build

From a clean clone, regenerate the figures before building the manuscript:

```bash
python3 paper/make_figures.py
cd paper && make paper
```

If the generated figure PDFs already exist locally, the manuscript alone can be
rebuilt with either author metadata or anonymous metadata:

```bash
cd paper
make paper
make anonymous
```

## Model and protocol

Each legal board is encoded on nine qubits with `RX(2*pi*g_i/3)`. All final
runs use `L=3`, `p=2`, the paper single-qubit block, Adam with learning rate
`0.01`, batch size `15`, `30` minibatch updates per epoch, `100` epochs, and
test size `600`.

The baseline `edge` ansatz uses orbit-shared single-qubit gates and directed
`CRY` edge interactions. The main `edge+lines` ansatz keeps the edge circuit and
adds orbit-shared `ZZZ` and `CCRZ` interactions on the Tic-Tac-Toe winning
triples.
