# ADDENDUM — real `nx = 8` dense-ramp run (anchor GAP-PP-3-ICE-STRUCT-PLASTIC-SET)

**Why this file exists.** While the three-mesh campaign was queued
(`run_compute` job `cj3810`, still `queued` after 60 min), a real single-mesh run
finished in the work tree and left `artifacts/patch_plate_plastic_ladder_nx8_free.json`.
Its numbers are **not** compatible with the `nx = 8` row of
`docs/CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md`. Rather than leave an untracked
artifact that silently disagrees with the committed document, the measurement is
recorded here with its settings and its failures.

## 1. Run identity and settings

| item | value |
|---|---|
| driver | `scripts/patch_plate_plastic_mesh_ladder.py --nx 8` (in-plane free) |
| layers | 4 (exact through-thickness integration, shear correction 5/6) |
| ramp | 0.2 … 3.0 $p_y$, step 0.1 $p_y$, adaptive bisection tol 1e-2 |
| unload | 3 steps to 0 MPa |
| wall time | 502.5 s (single core) |
| artifact | `artifacts/patch_plate_plastic_ladder_nx8_free.json` |

References (from `tensorfem.plate_patch_plastic_limit`, DH36 @ −40 ℃):
$p_y$ = 0.86923 MPa, strip lower 0.81877, Ilyushin sine 1.51518,
yield line 1.84224, Johansen 2.37998 MPa.

## 2. Measured

| quantity | value | note |
|---|---|---|
| first yield | 0.87501 MPa | **+0.66 %** vs analytic $p_y$ → acceptance (<5 %) **met** |
| ramp end | 1.99924 MPa, 22 points | **truncated by the runaway guard** (`ramp_truncated = True`) |
| limit, two-line | 1.9123 MPa | uses the *last* interval, i.e. not the fully plastic tangent |
| limit, 2× deflection | 1.7526 MPa | |
| limit, stiffness 0.10 | 1.7378 MPa | stiffness 0.05 / 0.02 never fired (ramp ended first) |
| criteria spread (tight) | **0.100** | 10 % between definitions — the honest band |
| permanent set | **withheld** | `permanent_set_valid = False` (unload leg did not converge) |
| vs Ilyushin sine | **+26.2 %** | |
| vs yield-line upper | **+3.80 %** | |
| vs strip lower | +133.6 % | |

`bracket_verdict()` classifies this value as **`violated (above upper bound)`**.

## 3. What this does and does not prove

* **Does**: the elastic→plastic transition is confirmed on this mesh (+0.66 %, well
  inside the 5 % acceptance) — consistent with the earlier +0.79 %.
* **Does**: the three limit-load definitions disagree by 10 % on a **truncated**
  ramp, and two of them never fire. That is direct evidence for the claim the
  verdict layer encodes: a limit load from a ramp that never reaches the plastic
  plateau is a convention, not a measurement, and must be reported with its band —
  not as a single number.
* **Does not**: settle the plastic limit load. A single mesh with a 10 % criteria
  spread and no valid unload gives no converged value and no permanent set.
* **Open interpretation (not resolved here)**: the extracted value sits +3.8 %
  **above** the ideal-plastic yield-line upper bound. Two candidate explanations,
  both unexcluded by this run:
  (i) the two-line rule is biased high on a truncated ramp — supported by the fact
  that the other two definitions land 8–9 % lower and inside the bracket;
  (ii) hardening $H$ = 1 MPa legitimately carries the load past the *ideal*-plastic
  bound, i.e. the rigid-plastic bracket is only a strict gate in the $H \to 0$ limit.
  Distinguishing them needs a ramp that reaches the plateau (denser/longer ramp or
  a deflection-controlled leg), which is what the queued campaign is for.
  Consequence for the verdict layer: `bracket_verdict()` is a *necessary*
  admissibility test, not a sufficient accuracy test — it is reported next to the
  criteria spread for exactly this reason.

## 4. Discrepancy with the committed `nx = 8` row (flagged, unexplained)

`docs/CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md` §3 reports `nx = 8`,
in-plane free: limit **1.1997 MPa** (−20.8 % vs Ilyushin, −34.9 % vs yield line),
first yield 0.8761 MPa, permanent set 5.21 mm. This run, on the same case and the
same driver, gives first yield 0.8750 MPa (agrees to 0.1 %) but a limit load
**1.9123 MPa** (+59 % relative). The artifact behind the document's row is **not in
this branch**, so the difference cannot be diagnosed here; the settings that differ
are unrecorded ramp length/unload and the fact that this run truncated at the
runaway guard while the document's did not report truncation. Until the three-mesh
campaign re-runs both, the document's `nx = 8` limit load should be treated as
**unreproduced**, not as confirmed.

## 5. Status of the acceptance items

| criterion | status after this addendum |
|---|---|
| elastic → plastic transition < 5 % | **met** (two independent ramps: +0.79 %, +0.66 %) |
| plastic limit vs Ilyushin / upper bound < 10 % | **not_run** (no converged value; single-mesh values disagree with the bound in sign and by 26 %) |
| 3-level mesh convergence | **not_run** — `run_compute` cj3810 queued |
| pytest ≥ 10 green | **met** — 34 passed |

No number above is a converged limit load, and none is presented as one.