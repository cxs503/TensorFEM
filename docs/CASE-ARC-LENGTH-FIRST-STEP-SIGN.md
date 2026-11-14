# CASE — arc-length continuation started on the negative load branch (defect, fixed)

Anchor: `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE` (verification debt). Repository
TensorFEM, branch `agent/m20123`. Editing scope: `src/tensorfem/ice_belt_collapse.py`
fix, `tests/test_arc_length_first_step_sign.py`, this document.

Every number below was printed by a run recorded in this worktree; none is typed
from memory. Where a check was **not** run it is marked `not_run`, not "passed".

## 1. How the defect surfaced

The anchor asks for an independent cross-check of the merged semi-analytical
grillage ultimate-strength layer (`grillage_ice_ultimate` +
`grillage_plastic_collapse`). `grillage_ice_ultimate`'s module docstring claims
that `ice_belt_collapse` / `stiffened_panel_ice_ultimate` — the layered
elastoplastic von-Karman plate FE with Tee stiffeners — "is used as the
independent cross-check". It is **not**: `grillage_ice_verification` imports
neither module (grep over `src/tensorfem/grillage_*.py` returns only the mention
in the docstring and `IACSPressureArea`). So the first task of the cross-check
was to build the FE model, i.e. to exercise the ice-belt plate field for the
case-study bay strip (2.1 m frame spacing x 0.35 m stiffener spacing, t = 20 mm,
EH36/E = 206 GPa / sigma_0 = 355 MPa, SSSS edges, in-plane traction free).

That immediately returned a *negative* ultimate pressure:

```
a=2.1 b=0.35 30x8 t=0.02   lam [0.0, -0.0500, -0.1075, -0.1736, -0.2497]  status still-ascending
a=2.1 b=0.70 30x8 t=0.02   lam [0.0, -0.0500, -0.1000, -0.1576, -0.2240]  status still-ascending
a=0.35 b=0.35 12x12 t=0.02 lam [0.0, -0.0200, -0.0430, -0.0695, -0.0999]  status still-ascending
a=2.0 b=2.0 16x12 t=0.015  lam [0.0,  0.0033,  0.0071,  0.0118,  0.0173]  status still-ascending
```

The three narrow models descend along the *mirror* branch; the square model does
not. The behaviour is independent of `tol` (1e-6, 1e-4, 1e-3 all identical) and
of `dlam0` (0.005, 0.02, 0.05 scale the same numbers), so it is not a
convergence-tolerance artefact: it is a branch-selection bug.

## 2. Root cause

In `solve_arc_length` the corrective root of the spherical arc-length constraint
is selected by

```python
target = (-(c @ du_f + kappa * dlam_c) / a if it == 0 else last)
dsel = min(roots, key=lambda d: abs(d - target))
```

On the zero-displacement start of the first step `du_r = 0` and `c = dlam_pred *
du_f`, so the two roots are exactly `d = 0` and `d = -2 dlam_pred` and both are
at distance `|dlam_pred|` from `target = -dlam_pred` — an **exact tie**. Floating
point breaks the tie arbitrarily; when it breaks towards the mirror root the
converged load factor becomes `-dlam_pred`. Because the elastic response is odd
in `(u, lam)`, the mirror trajectory is also an equilibrium path, so every
subsequent step converges, `arc_length_limit` reports `still-ascending`, and the
consumer computes `peak_load_factor * pressure < 0`: a negative collapse
pressure, i.e. an apparently "safe" negative margin. Nothing in the solver
detects it.

`du_prev is None` is the exact condition of the first step, which is the only
place where the existing continuation guard (`ahead`, displacement-based) cannot
act.

## 3. Fix (minimal)

`solve_arc_length`, first step only — restrict the admissible roots to those that
keep the load increment in the predicted direction (Crisfield's sign rule applied
to `lam` instead of `u`), then keep the existing nearest-`target` and `ahead`
selection unchanged:

```python
candidates = roots
if du_prev is None:
    forward = [d for d in roots if (dlam_c + d) * dlam_pred > 0.0]
    if forward:
        candidates = tuple(forward)
dsel = min(candidates, key=lambda d: abs(d - target))
```

Nothing outside the first step is touched, so post-peak behaviour (which the
ice-belt and stiffened-panel verification faces test through descending load
factors) is unaffected. After the fix the same three narrow models ascend:

```
a=2.1 b=0.35 30x8 dlam0=0.02  lam [0.0, 0.0200, 0.0430, 0.0695, 0.0999]
a=2.1 b=0.35 30x8 dlam0=0.05  lam [0.0, 0.0500, 0.1075, 0.1736, 0.2497]
a=0.35 b=0.35 12x12           lam [0.0, 0.0200, 0.0430, 0.0695, 0.0999]
```

## 4. Tests

`tests/test_arc_length_first_step_sign.py` (new, 3 tests, 28.7 s, all pass):

| test | what it pins |
| --- | --- |
| `test_slender_strip_starts_on_positive_load_branch` | the exact geometry that exposed the defect: `lam > 0`, `lam[1] ~ dlam0`, monotone ascending, `peak_load_factor > 0` |
| `test_square_panel_branch_unchanged` | the previously working square panel is unchanged |
| `test_first_step_sign_guard_is_general_not_geometry_specific` | 3 aspect ratios x 2 step sizes all start ascending (the guard is not geometry-tuned) |

Existing suites (`tests/test_ice_belt_collapse.py`,
`tests/test_grillage_ice_ultimate.py`, `tests/test_grillage_plastic_collapse.py`)
were re-run on the cluster (local run exceeds the 120 s shell budget); result in
section 6.

## 5. Cross-check the fix enables — first honest numbers

With the branch repaired the FE plate field runs, but it does **not** yet confirm
the semi-analytical plate-field limit; it contradicts it in magnitude:

| run | model | FE peak pressure | reference | status |
| --- | --- | --- | --- | --- |
| uniform load, 2.32 MPa reference | 2.1 x 0.35, 20x5, SSSS, von Karman | 3.02 MPa at step 45 | yield-line `8 m_p / b^2` = 2.32 MPa | `still-ascending`, w_centre = 0.020 m = t, i.e. the limit point was **not reached** inside the step budget — the number is a lower bound |
| ice patch 0.7 x 0.35, p_ref = 2.165 MPa | same mesh | 4.51 MPa at step 40 | semi-analytical `plate_local_bending` limit 2.165 MPa | `still-ascending`, lower bound |

Reading, stated as a limitation rather than a result: the semi-analytical
`plate_local_bending` utilisation compares the **peak Navier moment** with the
plastic moment `m_p = sigma_y t^2 / 4` — a *first-hinge* criterion — whereas the
FE traces the equilibrium path and only collapses when a mechanism forms. A
factor ~2 between them is expected for a plate strip, and the FE value is the
physically meaningful ultimate load. Therefore:

* the merged layer's *governing number* for the case study (2.165 MPa,
  "design not adequate, margin 0.54") is **conservative**, not wrong-but-unsafe;
* the docstring claim that the elastoplastic FE "is used as the independent
  cross-check" is **overstated** and is not repaired by this commit;
* the anchor's acceptance gate "one case within 15 % of literature/rule" is
  **not closed** for the plate-field criterion (FE vs semi-analytical differ by
  ~100 %), and `not_run`: FE limit point (needs a larger step budget / finer
  mesh), FE member-system (stiffener beam-column, frame) cross-check, and any
  comparison against a *published* patch-loaded grillage experiment.

`not_run` items are deliberately left to the next round rather than replaced by
a weaker claim: a limit point has to be located, not bracketed from below.

## 6. Cluster verification

`cj2522` (`repo=TensorFEM`, `stage_dir` = this worktree, 8 cores) runs

```
PYTHONPATH=src python -m pytest tests/test_ice_belt_collapse.py \
    tests/test_grillage_ice_ultimate.py tests/test_grillage_plastic_collapse.py \
    tests/test_arc_length_first_step_sign.py -q
```

Result is appended to the mission report when the job reaches a terminal state;
if it has not by the time this branch is submitted, the existing-suite re-run is
reported as `not_run` in the platform report rather than assumed green.

## 7. Reproduction

```
PYTHONPATH=src python -m pytest tests/test_arc_length_first_step_sign.py -q     # 3 passed
git revert <this commit>                                                       # then
PYTHONPATH=src python -m pytest tests/test_arc_length_first_step_sign.py -q     # 2 failed
```

The reverted run reproduces the negative branch (`lam[1] = -0.02`); the strip and
geometry-generality tests fail, the square-panel test still passes (it never hit
the tie).