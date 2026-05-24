# Paper Plan: Partial Equivariance as a Tunable Inductive Bias in Variational Quantum Learning

## Core Idea

Symmetry in variational quantum learning is usually treated as binary: either no
symmetry, or the full task symmetry group `G`. We argue that the subgroup
lattice of `G` defines a continuous family of inductive biases through parameter
sharing. For Tic-Tac-Toe, `G = D4`. Every subgroup `H <= D4` ties parameters
over a different orbit decomposition and therefore picks a different point on
the bias-variance curve.

Meyer et al. (2023) show that full `D4` equivariance generalizes better than a
non-invariant baseline. We extend that result along three axes:

1. **The full subgroup lattice of `D4`** — six points between `none` and `D4`.
2. **Approximate symmetry** — partial subgroups beat full equivariance when the
   data is only partially symmetric. This is the first concrete setting where a
   "smaller" inductive bias is provably the right choice for a VQLM.
3. **Quantitative symmetrization gain** — the Caro et al. generalization bound
   predicts gap scaling `O(√(T/N))` with `T` the number of independent
   parameters. We test this prediction subgroup by subgroup.

Conservative claim:

> Partial equivariance provides a controllable trade-off between expressivity
> and inductive bias. It matches full `D4` on a fully symmetric task, beats
> full `D4` once the symmetry is broken, and the symmetrization gain `γ_H`
> ranks subgroups in the same order as the measured generalization gap.

We do not claim quantum advantage. The paper is a controlled study of inductive
bias, expressivity, and generalization on a well-defined toy task.

## Subgroup Variants

| Variant | Category | Order | Paper-block params/block | Purpose |
|---|---|---:|---:|---|
| `none` | no symmetry | 1 | 34 | high-expressivity baseline |
| `Z2_rot180` | partial | 2 | 18 | minimal rotation bias |
| `Z2_reflection` | partial | 2 | 21 | minimal reflection bias |
| `C4` | partial | 4 | 10 | rotation-only symmetry |
| `D2_V4` | partial | 4 | 13 | medium rotation/reflection subgroup |
| `D4` | full | 8 | 9 | full Tic-Tac-Toe symmetry |

Total trainable parameters are `L * p * params_per_block`. All main-paper
experiments use `single_qubit_block=paper`, i.e. the paper-style single-qubit
block `RY(θ2) RX(θ1)` with directed `CRY` entanglers and the corner-edge-center
"cemoid" layout. The `RY`-only block is kept as a low-parameter ablation.

## Stage 0 — Reproduction Gate (must pass before anything else)

The previous quick runs at `(L,p)=(1,1)` returned test accuracies near chance
(0.36–0.43). Meyer et al. report 0.59–0.80 for invariant models in their
`(l,p)∈{1..5}²` sweep. Before any partial-equivariance claim is made, we must
hit Meyer's band on `none` vs `D4`.

**Gate criterion (5 seeds, `L=3, p=2`, train 450, test 600, paper block,
100 epochs × 30 steps × batch 15, Adam lr=0.01):**

- mean `D4` test accuracy ≥ 0.65
- mean `none` test accuracy ≥ 0.50
- mean test gap (`D4 - none`) ≥ 0.05

A 20-epoch single-seed pilot already reached `D4` test 0.632, so the gate is
expected to pass.

```bash
/Users/markus/anaconda3/bin/python -m experiments.reproduce_original \
  --L-values 3 --p-values 2 --seeds 0,1,2,3,4 \
  --train-size 450 --test-size 600 --epochs 100 \
  --output results/csv/results_repro_L3p2.csv
```

If the gate fails: debug `init_scale`, encoding factor `2π/3`, observable
weights, optimizer schedule. **Do not proceed to Stage 1 until the gate
passes.**

## Stage 1 — Core Partial Sweep

All six variants on the symmetric data, exactly the regime the reproduction
gate validates.

- `(L,p) = (3,2)`, paper block.
- train sizes `{30, 60, 120, 240, 450}`, fixed `test_size = 600`.
- `≥ 10` seeds (final paper run: 20 seeds).
- Output: `results/csv/results_partial_main.csv`.

```bash
/Users/markus/anaconda3/bin/python -m experiments.partial_equivariance_sweep \
  --L 3 --p 2 --train-sizes 30,60,120,240,450 --final \
  --output results/csv/results_partial_main.csv
```

Headline figures from this CSV:

- Test accuracy vs train size, one curve per subgroup.
- Generalization gap vs train size, one curve per subgroup.
- Test accuracy vs subgroup order at fixed train size.
- Test accuracy vs parameter count.

## Stage 2 — Approximate Symmetry (primary novel result)

Real data is rarely exactly symmetric. We construct an `ε`-broken Tic-Tac-Toe
dataset whose **true** symmetry is a chosen proper subgroup `K ⊊ D4`: pick a
fraction `ε` of `K`-orbits of legal states, swap circle ↔ cross within each
selected `K`-orbit. Since `K`-orbits are strict subsets of `D4`-orbits whenever
`K ⊊ D4`, the corruption preserves `K`-invariance by construction but breaks
`D4`-invariance.

Verified empirically (`broken_K=C4, ε=0.2`): 0 `C4`-invariance violations,
232 `D4`-invariance violations across 5 478 states.

This gives a clean cross-over prediction:

- `H ⊆ K`: correct inductive bias → should match or beat all alternatives.
- `H ⊃ K` or `H ⊄ K`: over-constrained → degrades with `ε`.

Primary configuration: `K = C4` (rotations preserved, reflections broken).
Models with `H ∈ {none, Z2_rot180, C4}` are well-specified; `D4`, `D2_V4`,
`Z2_reflection` are over-constrained and should lose accuracy as `ε` grows.

Sweep:

- `ε ∈ {0.00, 0.05, 0.10, 0.20, 0.40}`.
- All six subgroups `H`.
- `(L,p) = (3,2)`, train 450, ≥ 5 seeds first pass / 10 seeds for paper.
- Optional ablation: rerun with `K = Z2_reflection` to confirm the cross-over
  generalizes beyond a single choice of broken symmetry.

Code (already in repo, fixed 2026-05-11 from a symmetry-preserving bug):

- `src/data_tictactoe.py::make_subgroup_orbits` — `K`-orbits of legal states.
- `src/data_tictactoe.py::make_corrupted_labels` — takes `broken_K` argument.
- `src/data_tictactoe.py::make_approximate_symmetry_split` — split with
  `broken_K`-aware corruption.
- `src/utils.py::ExperimentConfig.broken_K` — new field, default `"C4"`.
- `experiments/approximate_symmetry_sweep.py` — new `--broken-K` CLI flag.
- `src/plotting.py::plot_approximate_symmetry` — accuracy vs `ε`, one line
  per subgroup.

```bash
/Users/markus/anaconda3/bin/python -m experiments.approximate_symmetry_sweep \
  --L 3 --p 2 --train-sizes 450 --broken-K C4 \
  --eps-values 0.0,0.05,0.10,0.20,0.40 --seeds 0,1,2,3,4 \
  --output results/csv/results_approx_C4.csv
```

## Stage 3 — Quantitative Symmetrization Gain

Meyer et al. cite Caro et al. (Theorem 1 in the paper, Eq. 65) for the
generalization-gap scaling `Õ(√(T/N))`, where `T` is the number of independent
parameters. Symmetrization gain `γ_H = T_none / T_H` (Eq. 66).

We compute `γ_H` for each subgroup analytically (already in `groups_d4.py` via
`SharingPattern.n_parameters_per_block`) and test whether the empirical
generalization gap from Stage 1 ranks subgroups in the same order as
`√(T_H / N)`.

Deliverables:

- Table: `(H, T_H, γ_H, measured gap @ N=450, measured gap @ N=120)`.
- Scatter plot: predicted `√(T_H/N)` vs measured gap, with Spearman rank
  correlation.

This stage is pure post-processing of the Stage 1 CSV, no new training.

## Stage 4 — Random-Sharing Control (sanity)

Re-run of the existing `experiments/random_sharing_control.py` at the same
`(L,p) = (3,2)` regime, 10 seeds, train 450. Confirms that the partial-subgroup
gains are not just from "fewer parameters" but specifically from
**symmetry-aware** sharing. Already coded; just needs to be run at the new
depth.

## Stage 5 — Optional Extensions

Picked only if Stage 0–3 yield clear results in time:

- **Barren-plateau / gradient-variance per subgroup**: compute `Var(∂_θ C)` at
  random init for each `H` (Meyer Fig. 11 only does this for VQE). Test
  whether partial subgroups preserve gradient magnitude where `D4` flattens.
- **Classical CNN baseline**: small CNN with vs without `D4` augmentation on
  the same dataset and splits. Calibrates the absolute scale of the QML gain.
- **Depth-vs-symmetry curve**: existing `experiments/depth_sweep.py`, run on
  the same regime with the four representative subgroups.

## Required Figures and Tables (final paper)

Core figures:

- F1 — Reproduction sanity: train/test accuracy curves for `none` vs `D4` at
  `(L,p)=(3,2)`, overlaid with Meyer's reported band for context.
- F2 — Test accuracy vs train size, one curve per subgroup (Stage 1).
- F3 — Generalization gap vs train size (Stage 1).
- F4 — Test accuracy vs `ε` for all subgroups, with cross-over marked
  (Stage 2). **Headline figure.**
- F5 — Predicted `√(T_H/N)` vs measured gap, with Spearman ρ (Stage 3).
- F6 — Symmetry-aware vs random-sharing control (Stage 4).

Core table:

- Rows: subgroup variants.
- Columns: category, group order, parameter count `T_H`, `γ_H`, test accuracy
  per train size, mean generalization gap, accuracy at `ε=0`, accuracy at
  `ε=0.2`, seed count.

## Interpretation Rules

- **Stage 1 only**: if `D4` wins, the conclusion is that full symmetry is best
  for fully-symmetric data, but partial subgroups form a near-optimal tunable
  family with substantially fewer parameters.
- **Stage 2 cross-over**: if a partial subgroup wins at any `ε > 0`, the
  conclusion is that full task symmetry is *not* the right inductive bias for
  approximately-symmetric VQLMs. This is the strongest claim and the paper's
  main contribution.
- **Stage 3 rank correlation**: if Spearman ρ is high, the Caro bound is
  predictive at this scale. If low, we report the discrepancy as an open
  question.

We do not claim quantum advantage anywhere.

## Status (as of 2026-05-04)

- Pipeline code complete: 6 subgroups, paper + RY blocks, random-sharing
  control, 4 sweep scripts, plotting/reporting.
- Reproduction gate launched in background (`results_repro_L3p2.csv`); pilot
  20-epoch run hit `D4` test 0.632 → gate likely to pass.
- Stages 1–3 are scripted-but-not-yet-run at production depth.
- Stage 2 (approximate symmetry) code additions completed: label-corruption helper,
  sweep script, plot) — primary implementation work after the gate passes.
