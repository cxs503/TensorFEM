# Board audit — m15372 · CASE-PATCH-TEST (TensorFEM, Robinson patch test A/B/C)

**Deliverable audited:** mission m15372, branch `agent/m15372`, commit `d3a33bd0`
("TensorFEM: Robinson patch test A/B/C suite (Q4/Q8/TET10) with distortion and h studies"),
merged to `main` as `4c6552f8` (Merge #699). Status `succeeded`.
**Audit mission:** m15490 (this one). **Method:** independent re-derivation, not a re-run of the
author's driver only — see §5. **Verdict:** `tensor_maturity_verdicts.id=332`, level **4**,
`verdict_by='board-audit'`, `gate_state=pending` at insert time.

---

## 1. Object under audit (physical problem / geometry / parameters)

The deliverable is a *verification* capability, not a new solver: a reusable patch-test suite
qualifying the element families that ship in TensorFEM.

* Criterion (Robinson 1987; restated Zienkiewicz & Taylor vol. 1 §3.6): an **irregular** patch
  with free interior nodes, loaded **only through boundary displacements** lying in the element
  span, must reproduce the interior field and recovered stresses to round-off.
* Three load states: **A** uniaxial tension `σxx=σ0` (constant strain), **B** pure shear
  `τxy=τ0` (constant strain), **C** pure bending `σxx=-Eκy` (linear stress, quadratic
  displacement); 3-D variant **C3**.
* Geometry: `[0,1]²` (2-D, thickness 1) / `[0,1]³` (3-D), 3×3 cells (2×2 hexes for TET10),
  interior nodes displaced off-grid (irregular). Element families: bilinear **Q4**,
  8-node serendipity **Q8** (3×3 Gauss), **TET10**.

| material | value | load | value |
|---|---|---|---|
| E | 210000 MPa | σ0 (A) | 100 MPa |
| ν | 0.30 | τ0 (B) | 50 MPa |
| G = E/2(1+ν) | 80769.2308 MPa | κ (C) | 0.001 1/mm |
| thickness | 1 mm | tolerance | relative error ≤ 1e-10 |

**Reference values are analytic** (`σ = D B u`), not tabulated — so the "zero hand-copying"
requirement of the anchor is structurally satisfied. Robinson's paper supplies the *criterion*
only; the report states honestly that the paper was not retrievable in the sandbox.

## 2. What was done in this audit (method / tool chain)

1. **Evidence check** against the three-piece contract (real code / run artifact / repro cmd)
   in the *merged* tree (`main`, i.e. what the ecosystem actually owns), not only in the
   author's worktree.
2. **Re-run** in a clean tree: `PYTHONPATH=src python3 benchmarks/patch_test_robinson.py`
   (49.5 s) and `pytest tests/test_patch_test_robinson.py -q` → **9 passed in 12.36 s**
   (cross-checked against `.github/workflows/ci.yml`, which runs `pytest -q` over `tests/`,
   so the suite is CI-collected).
3. **Independent solver** (`artifacts/board-audit-m15372/independent_patch_check.py`,
   ≈ 300 lines): own Q4/Q8 shape functions, own 2×2 / 3×3 Gauss rules, own Jacobian,
   own assembly and Dirichlet elimination, own stress recovery, **numpy** dense solve —
   no TensorFEM stiffness or solver code is executed. Only the *mesh* is imported so that
   numbers are comparable (a second, fully independent sine-perturbed mesh is also run).
4. **Referee against the board's judge**: `SELECT case_name,metric,tolerance FROM
   tensor_benchmark_cases WHERE anchor_slug='CASE-PATCH-TEST'` → **no row** (see F4), so the
   tolerance used is the mission-goal one, "constant strain / linear stress exact reproduction".
5. Claim binding: every number below is a key of
   `runs/patch_test_robinson/result.json` (sha256 `9b33a93b35732489`, regenerated here
   bit-identically) or of `artifacts/board-audit-m15372/independent_patch_check.json`.

**Two bugs were found and fixed in my own verifier before it was allowed to judge the
deliverable** (recorded because they show how easy a false accusation would have been):
my first version used the Jacobian convention `J[a][b]=∂x_b/∂ξ_a` instead of
`∂x_a/∂ξ_b` (invisible on axis-aligned affine meshes, 2–7 % stiffness error on distorted
quads), and a wrong 3×3 Gauss weight set (`25/81` everywhere instead of `25/81, 40/81, 64/81`).
Both bugs made the *deliverable* look wrong (Q4/A 8.9e-2) while it was right: after the fix
Q4/A became 2.8e-16 and Q4/C 1.1703e-2, matching the author to round-off.

## 3. Results (numbers, programmatic injection)

| case | family | mesh | free DOFs | delivered stress error | independent solver | agree? |
|---|---|---|---|---|---|---|
| A | Q4 | distorted | 8 | 4.263e-16 | 2.842e-16 | round-off |
| B | Q4 | distorted | 8 | 5.005e-16 | 6.551e-16 | round-off |
| C | Q4 | distorted | 8 | **1.1703e-02** | **1.1703e-02** | rel. diff 0.0 |
| A | Q8 | distorted | 32 | 3.084e-12 | 4.263e-15 | round-off |
| B | Q8 | distorted | 32 | 6.950e-12 | 7.257e-15 | round-off |
| C | Q8 | affine | 32 | 5.414e-15 | 7.308e-15 | round-off |
| C | Q8 | distorted | 32 | **8.8457e-04** | **8.8457e-04** | rel. diff 1.15e-09 |
| C3 | TET10 | distorted | 81 | 1.624e-15 | (not re-implemented) | torch path only |

On my own independent sine-distorted mesh the same qualitative split appears (A/B round-off,
Q4-C 2.02e-01, Q8-distorted-C 5.09e-02, Q8-affine-C 7.3e-15), so the split is an element-span
property, not an artefact of the author's one mesh.

Figures: `docs/assets/patch_test_robinson/{q8_distortion_sensitivity,case_C_refinement,
case_C_stress_profile}.png` (delivered, regenerated locally) and
`docs/board-audit-m15372/errors_compare.png` (this audit: delivered vs independent, log scale).

## 4. Reproducibility and durability

* Rerun is **bit-identical**: sha256 of `runs/patch_test_robinson/result.json`
  = `9b33a93b35732489` both in m15372's worktree and in this clean tree.
* `docs/patch-test-robinson.md` is *almost* generator-owned: see F3.

## 5. Findings (severity-ranked, with reproduction)

* **F1 (medium, durability)** — `runs/patch_test_robinson/result.json` did **not** survive the
  merge: `git show --stat 4c6552f8` land 7 of the 8 committed files, `git cat-file` shows no
  `runs/` entry in `main`. The provenance file every report number is injected from was absent
  from the repo of record until this audit regenerated it (bit-identical, so no number changed).
  Repro: `git ls-tree -r HEAD --name-only | grep patch_test_robinson`.
  Remediation: un-ignore / whitelist `runs/**/result.json` in the push filter, or emit the JSON
  under `docs/assets/…` (as the PNGs already are).
* **F2 (low, reporting data integrity)** — the Q8 rows of the report table print
  `free DOFs = -16`. Cause: `src/tensorfem/patch_test.py:330`
  `2*len(nodes) - 2*len(dirichlet)` where `dirichlet` holds one entry per *DOF* (2·n_bnd = 48),
  so 80−96. True value: 32 free DOFs on 40 nodes / 24 boundary nodes (independently counted).
  Not a solver defect; the negative number is meaningless in a published table.
* **F3 (low, reproducibility hygiene)** — `docs/patch-test-robinson.md` is not idempotent with
  its generator: rerunning the driver **deletes** the hand-written "Honest note on the rates"
  paragraph (which contains hand-typed numbers 5.0 / −4.3 / 1.4 / 7.5 / −10.7 / 3.8) while the
  document claims "every number below is injected from result.json". Repro: run the driver in a
  clean tree → `git diff docs/patch-test-robinson.md` (7 lines removed). Remediation: move the
  note (with rate numbers pulled from the data) into `report_markdown()`.
* **F4 (medium, board bookkeeping)** — anchor `CASE-PATCH-TEST` has **no row** in
  `tensor_benchmark_cases` (16 rows, ids 1–16, none this anchor), so the board's own judge
  ("got vs tolerance") cannot be applied mechanically to this case. Either the pool mission was
  never registered or the row was lost; the audit had to fall back on the mission-goal tolerance.
* **No blocking correctness defect found.** The negative controls are real gates: Q4 cannot
  reproduce case C (1.17e-2 > 1e-10) and distorted Q8 cannot either (8.85e-4), and the
  constant-strain cases stay at round-off on the *same* meshes — so the suite can fail, which is
  the property that makes a patch test worth anything.

**Overstatement check.** Three claims were tested and hold (`A/B exact`; `C exact on affine Q8`;
`C exact for distorted TET10` — the last only via the torch path, not independently
re-implemented here). One claim is diluted by F1/F3: "every number injected from result.json",
because the paragraph in §5 of the report is hand-written and the JSON was not in `main`.
`L5` is **not** claimed: the external anchor is a criterion, not a numeric reference, and the
paper was unreachable.

**Checks not run / independence limits.** TET10 (C3) was not re-implemented in numpy
(3-D quadratic tetrahedra, 3×3 exactness rules) — its 1.62e-15 is the author's number only.
`h`-refinement *rates* were not re-derived asymptotically: the delivered report itself warns the
meshes are not nested, so the rate column must not be read as an order — I confirm this reading
and did not certify any rate. Curved (non-straight-sided) elements were not exercised.

## 6. Level and disposition

| level | requirement (maturity gate scheme) | evidence |
|---|---|---|
| L1 | tests | 9 gates incl. 2 negative controls, `9 passed` |
| L2 | impl + entry + run artifact | `src/tensorfem/patch_test.py` (441 lines, no placeholders), driver CLI, `result.json` |
| L3 | + integration | harness drives production kernels `solve_continuum` / `solve_q8_plane` / `solve_tet10`; suite is repo-shipped and CI-collected. **Direction is harness→solver**; no `src/` module imports `patch_test` (grep-verified) — declared, not hidden. |
| L4 | + benchmark ref vs got + repro cmd | ref/got/exact-error for 8 cases, error analysis (distortion, h, numbering), bit-identical rerun, independent solver agreement to 1.2e-9 |
| L5 | + external anchor | not claimed |

**Disposition: `confirmed` at L4 recommended**, with F1–F4 as follow-up defects (F1/F4 worth a
small successor mission). The substance — real code, real solves, failing negative controls,
bit-reproducible numbers, independently reproduced constants — is what a benchmark verification
is supposed to look like; the defects are in artifact durability and reporting bookkeeping, not
in the numerics.