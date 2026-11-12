# Adversarial audit — `cylindrical_shell4` is a rank-deficient element (pinched cylinder / Scordelis–Lo anchors)

- **Branch / commit reviewed:** `agent/m16095`, base `4e8ed9e3` (Merge #753 via merge-master)
- **Artifacts reviewed:** `src/tensorfem/cylindrical_shell4.py`,
  `src/tensorfem/pinched_cylinder.py`,
  `benchmarks/verified/fem_pinched_cylinder/{run_verification.py,result.json,gates_verified.json,README.md}`,
  `benchmarks/verified/fem_shell_scordelis_lo/{run_verification.py,README.md}`,
  `tests/test_pinched_cylinder.py`, `tests/test_cylindrical_shell4.py`,
  `tests/test_scordelis_lo_verification.py`
- **New artifacts delivered with this audit:**
  `src/tensorfem/cylindrical_shell4_diagnostics.py`,
  `tests/test_cylindrical_shell4_diagnostics.py` (6 tests, green)
- **Ledger anchor:** `CASE-FEM-PINCHED-CYLINDER` (verification debt, not the case pool registration)
- **Disposition: REQUEST CHANGES** — the element and the two anchor verdicts built on it are
  withdrawn from "qualified" status until a rank-sufficient formulation is validated.

## 1. What the anchors claim

`benchmarks/verified/fem_pinched_cylinder` reports

| mesh | displacement | error vs −1.8248e−5 |
|---|---|---|
| 4×4 | −1.7727e−5 | 2.86 % |
| 8×8 | −1.7637e−5 | 3.35 % |
| 16×16 | −1.8278e−5 | 0.17 % |
| 32×32 | −1.8579e−5 | 1.82 % |

and concludes: *"This is a known limitation of the cylindrical Shell4 formulation with
selective reduced integration, not a bug. The element is still qualified for engineering
use because the finest-mesh error is within 3 % …"* (README, "Discussion").
Gate G2 (monotone convergence) is recorded as **FAIL** but the anchor is argued to be
acceptable anyway, and G3 ("observed convergence order 4.33 ≥ 1.0") is recorded as PASS.

## 2. Independent re-execution

The anchor mesh ladder was re-run with a sparse assembly (`scipy.sparse`, same element
routine, no other change) to extend the ladder past the dense-solve limit of the anchor
script:

| n | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| error % | 82.28 | 2.86 | 5.37 | 3.35 | 0.98 | **0.17** | 1.28 | 1.82 | 2.37 | **2.67** |

The displacement does **not** converge to the reference. From n = 16 onwards the error
increases monotonically (0.17 → 1.28 → 1.82 → 2.37 → 2.67 %) and the extracted
displacement (−1.8735e−5 at n = 64) is drifting away from −1.8248e−5. Strain energy
behaves the same way (2.2848e−6 → 2.3418e−6, monotone growth). The "0.17 % at 16×16"
is therefore a **crossing of the reference value by a non-converging sequence**, not an
accuracy result.

## 3. Root cause: the element matrix is rank deficient

A 4-node shell element has 24 DOFs and exactly **6** rigid-body modes. Measuring the
eigenvalues of one element (nearly square section, 2.5° arc, the configuration used by the
anchors):

```
element stiffness eigenvalues (normalised): 11 eigenvalues at 0, then 2.5e-7 ...
rank = 13 / 24     ->  null space = 11  =  6 rigid modes + 5 spurious mechanisms
```

The mechanism is visible in the source: the docstring says *"Membrane and bending terms use
2x2 Gauss integration"*, but the membrane block is evaluated at a **single point** and
scaled by the element area:

```python
_, det0, Bm0, _, _ = fields(0.0, 0.0)
K += Bm0.T @ Dm @ Bm0 * det0 * 4      # 1-point membrane
K += Bs0.T @ Ds @ Bs0 * det0 * 4      # 1-point transverse shear
```

Membrane and shear are both one-point integrated, so the element carries 5 zero-energy
hourglass modes. (The `drilling_factor = 1e-8` stabilisation only touches the unused
normal rotation and cannot compensate.)

Two controls confirm the diagnosis:

| integration | element rank | nullspace | pinched-cylinder error % (4 / 8 / 16 / 32) |
|---|---|---|---|
| 1-pt membrane + 1-pt shear (**production**) | 13 / 24 | 11 | 2.86 / 3.35 / 0.17 / 1.82 (oscillating) |
| 2×2 membrane + 1-pt shear | 18 / 24 | 6 | 84.97 / 64.90 / 38.25 / 14.67 (locking) |
| 2×2 membrane + 2×2 shear | 20 / 24 | ≤ 6 | 96.67 / 92.98 / 84.74 / 67.03 (locking) |

So the accuracy of the production element is **an artefact of its hourglass modes**: as
soon as the spurious mechanisms are removed the element locks severely (84 % → 15 % error,
the slow-decay signature of shear/membrane locking). The element is not a "specialised
cylindrical shell" that happens to converge non-monotonically — it is a rank-deficient,
locking-prone formulation.

## 4. Overstated / unsupported claims in the reviewed artifacts

1. `fem_pinched_cylinder/README.md` — *"This is a known limitation …, not a bug"*: not
   supported. The limitation has a concrete structural cause (see §3) and is a bug.
2. `fem_pinched_cylinder/gates_verified.json` — G3 `observed convergence order = 4.33`
   recorded as **PASS**. That number is a Richardson order computed from three
   non-asymptotic, oscillating points; with a non-converging sequence the order estimate is
   meaningless. It must not be used as evidence of correctness.
3. `fem_pinched_cylinder/README.md` — G1 "finest-mesh error ≤ 3 % PASS": true for the
   recorded 32×32 mesh, but false under continued refinement (2.67 % at 64×64 and rising),
   i.e. the gate is not a convergence statement and should not be presented as accuracy
   evidence.
4. `fem_shell_scordelis_lo/README.md` — the 0.02 % (16×16) result and the "monotone
   convergence" claim rest on the same rank-deficient element; the flat-facet failure is
   contrasted with this element as if it were a validated alternative. (Scordelis–Lo
   ladder not re-run in this audit — see §6.)
5. `tests/test_cylindrical_shell4.py::test_scordelis_lo_converges_and_qualifies_below_three_percent`
   and `tests/test_pinched_cylinder.py::test_pinched_cylinder_fine_mesh_qualifies_and_converges`
   assert that errors *decrease* over the sampled meshes (4→8→16 for Scordelis, 8→16→32 for
   pinched cylinder). For the pinched cylinder this ordering only holds because the sampled
   sequence crosses the reference value; the assertions do not encode any convergence
   invariant, they only execute the code.

## 5. Severity

| # | Finding | Severity | Evidence |
|---|---|---|---|
| F1 | Element stiffness rank 13/24: 5 spurious mechanisms (1-point membrane + 1-point shear) | **High** — any result produced with this element is mesh-dependent in an uncontrolled way | `element_nullspace_dimension()` |
| F2 | Removing the mechanisms produces locking (84 % → 15 % error), i.e. no usable formulation is currently shipped | **High** — the element cannot be repaired by a one-line integration change | `consistent_integration_nullspace_dimension()` + ladder run |
| F3 | Anchor verdicts (pinched cylinder G1/G3 "PASS", "known limitation") are unsupported | **Medium** — misleading capability registration | `result.json`, `gates_verified.json` |
| F4 | Tests assert monotone ordering over hand-picked meshes, not convergence invariants | **Medium** — false green | `tests/test_pinched_cylinder.py:5-9` |
| F5 | Docstring/code mismatch (claims 2×2 membrane, implements 1-point) | **Low** | `cylindrical_shell4.py:1-8` vs `:88-91` |

## 6. Checks performed / not performed

Performed (reproducible, see `tests/test_cylindrical_shell4_diagnostics.py`):

- element nullspace dimension, production vs consistent integration;
- exact rigid-mode energy (axial translation < 1e-18; in-plane translations O(h²) facet
  effect; rotation about the axis < 1e-12);
- pinched-cylinder ladder to n = 64 with sparse assembly;
- re-run of the existing anchor and of the new tests.

Not run / unknown — **do not read as passed**:

- Scordelis–Lo ladder re-run (the anchor's 16×16 result is therefore *unverified*, not
  refuted);
- an actual MITC4 assumed-shear implementation for this element: a covariant-shear
  prototype was drafted but not validated (only the 2×2 integration controls above were
  run), so the fix path is proposed, not demonstrated;
- the nonlinear path (`solve_pinched_cylinder_nonlinear`, `shell_nonlinear_step`) and the
  ice-belt / corotational users of this element were not re-checked; their results are
  suspect wherever `cylindrical_shell4_stiffness` is used;
- reaction-balance diagnostics were inconclusive (an O(h²) facet effect keeps the global
  reaction sum at ≈ −7.2 independent of refinement) and are not used as evidence.

## 7. Reproduction

```bash
cd <worktree>
OMP_NUM_THREADS=4 python -m pytest tests/test_cylindrical_shell4_diagnostics.py -q
PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run_verification.py
PYTHONPATH=src python - <<'PY'
from tensorfem.cylindrical_shell4_diagnostics import *
print("nullspace:", element_nullspace_dimension())          # 11
print("rigid:", {k: f"{v:.2e}" for k, v in rigid_body_patch().items()})
print("ladder:", [(r["n"], round(r["error_pct"], 2)) for r in pinched_cylinder_ladder((4, 8, 16))])
PY
```

## 8. Recommended remediation (for a follow-up mission, in order)

1. Replace the one-point membrane/shear blocks with **MITC4 assumed transverse shear**
   (covariant sampling at edge midpoints, 2×2 Gauss for membrane and bending) — this is
   the only combination in the literature that both restores rank and avoids locking.
2. Re-qualify against MacNeal–Harder (pinched cylinder −1.8248e−5, Scordelis–Lo −0.3024)
   with a mesh ladder that is required to be *monotone in the asymptotic range* and to
   report a grid-convergence estimate, not a single-mesh point.
3. Downgrade the two anchors' verdicts to `WITHHELD` and re-issue `gates_verified.json`
   with a convergence gate that uses successive-difference reduction (or GCI), not
   sign-crossing of |error|.
4. Audit downstream users (`corotational_shell`, `cylindrical_shell_modal`,
   `cylindrical_pressure_reference`, ice-belt FEM reports) for numbers produced with the
   rank-deficient matrix.