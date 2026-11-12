# `fem_shell_modal` anchor: verification-debt closure and honest re-grade

Mission **m16863** (verifier seat), 2026-11-12. Anchor
`benchmarks/verified/fem_shell_modal`, certificate `fem_shell_modal` in
`benchmarks/verified/anchor_certificates.json`.

## 1. What the anchor claimed

`result.json` declared `overall_verdict: PASS` and `gates_verified.json` was a
bare list with three gates. The shell gate read

```
{"id": "G3.shell", "description": "Shell: frequencies positive and in expected range",
 "value": 60.500108050743094, "limit": null, "passed": true}
```

i.e. a gate whose value is a 60.5 % deviation from the reference, whose
`limit` is `null`, and which nevertheless passes. The audit tool classifies that
as `E_PASS_WITHOUT_CRITERION` — unfalsifiable: nothing could ever turn it red.

## 2. Why the reference ladder itself was wrong

The anchor built its shell reference from
`tensorfem.cylindrical_shell_modal.dmv_frequency_hz` over `m = 1..7, n = 0..14`.
That helper contradicted its own docstring in two places:

| defect | evidence (R = 0.5 m, L = 1 m, t = 5 mm, E = 210 GPa, ν = 0.3, ρ = 7800) |
|---|---|
| `n = 1` branch **dropped the membrane term** `λ⁴/(λ²+n²)²` | returned **17.33 Hz**; retained-term Donnell **1175.43 Hz**; Euler-Bernoulli beam limit `(π/L)²√(E R²/2ρ)` = **2881.62 Hz** → 166× low vs the beam limit |
| `n = 0` axisymmetric branch was **dead code** (validator required `n_circ ≥ 1` while the docstring says `n ≥ 0`) | `dmv_frequency_hz(n_circ=0)` raised `ValueError: n_circ must be at least 1` |

Because the gate's oracle list was sorted, the unphysical 17.33 Hz mode became
the anchor's "lowest expected frequency" and the mode matching it performed
(34.6 %, 39.6 %, 60.5 % on its first three rows) was disclosed only in a prose
`note`, not in the verdict.

**Repair** (`src/tensorfem/cylindrical_shell_modal.py`): one Donnell expression
for every `n ≥ 0` (`span = λ²+n²`, `ω² = E/(ρR²)[λ⁴/span² + k·span²]`), `n = 0`
admissible, and the accuracy statement in the docstring now says that `n = 1` is
bracketed by the beam limit rather than claiming thin-shell accuracy there. The
`n ≥ 2` values are unchanged (relative difference 0 against the pre-repair code),
so no other consumer moves.

Independent check: `fsi_wet_modal.donnell_shell_frequency_hz` (which always kept
the membrane term) now agrees with the legacy helper to 1e-12, and
`tests/test_fsi_wet_modal.py::test_donnell_keeps_n1_membrane_term` was converted
from pinning the defect (`legacy < 0.3·beam`) to pinning the repair.

## 3. What the FE evidence actually is

`run_verification.py` now classifies every FE mode by radial participation
`‖w‖²/‖φ‖²` and runs a three-mesh ladder (4×8, 8×8, 8×16). Result:

| mesh | first six FE frequencies [Hz] | radial-only participation of the first three | mechanisms in the first ten |
|---|---|---|---|
| 4×8 (the anchor's mesh) | 75.85, 75.85, 79.84, 100.35, 100.35, 112.92 | 6.9e-26, 2.2e-25, 1.2e-22 | 8 |
| 8×8 | 39.17, 75.85, 75.85, 79.84, 86.23, 86.23 | 6.8e-26, 1.1e-25, 3.3e-26 | 8 |
| 8×16 | 0.000, 0.001, 33.38, 33.60, 33.91, 47.96 | 6.4e-25, 4.5e-25, 4.3e-16 | 10 |

The three "lowest frequencies" the anchor reported are **rotation-only
mechanisms**: their rotation participation is 1.000, their transverse
displacement participation is ~1e-25, and their frequency is mesh dependent
(75.85 → 39.17 → 0.001 Hz). The first mode that carries transverse displacement
is 100.35 Hz (4×8) / 98.87 Hz (8×8) / above 67.6 Hz (8×16, no physical mode in
the first ten of twenty requested).

The Donnell oracle minimum over `m = 1..3, n = 0..8` is
**202.14 Hz at (m, n) = (1, 5)**. So the FE spectrum neither reproduces the
oracle nor converges toward it under refinement: the physical-mode count in the
low spectrum *decreases* with mesh refinement. This agrees with, and now
quantifies, the defect already recorded in
`docs/fsi-wet-modal-dual-criteria.md` §7.2 ("the lowest modes are
radial-dominated spurious modes … fixing the element is a separate node").

## 4. Re-grade

`gates_verified.json` is now a canonical `tensorfem.verified-gates.v1` envelope
with four machine-checkable gates:

| gate | quantity | value | operator | threshold | verdict |
|---|---|---:|---|---:|---|
| `G1.beam` | cantilever modes 1–3, max error [%] | 6.4e-06 | `<=` | 3.0 | PASS |
| `G2.plate` | SS plate modes 1–3, max error [%] | 6.8425 | `<=` | 10.0 | PASS |
| `G3.shell_lowest_physical_vs_donnell` | lowest physical FE shell mode vs Donnell minimum [%], mesh 4×8 | 50.356 | `<=` | 15.0 | **FAIL** |
| `G4.shell_mechanism_free_low_spectrum` | rotation-only modes in the first ten | 8 | `==` | 0.0 | **FAIL** |

`passed: false`, `failed_gates: [G3…, G4…]`. The recomputed overall agrees with
the declared one, so the anchor is now *certified as failing* instead of
*contradicted*: it left the shrink-only debt registry
(`anchor_certificate_debt.json`, 27 → 26 anchors) because it no longer hides an
unfalsifiable pass. `tests/test_shell_modal_anchor_verification.py` pins the
oracle against an independent transcription (rel < 1e-12 for `n = 0..8`), pins
the two repaired branches and pins the honesty of the envelope.

## 5. Limitations / follow-up

* The root cause of the rotation-only DOFs is **not** fixed here. Candidates are
  the shell4 drilling-rotation stiffness or the simply-supported constraint set
  (`apply_simply_supported_bc` constrains `v, w, rx` at both ends only). This is
  element-level work, filed as a follow-up node.
* G3 is graded on the 4×8 mesh because the finer meshes carry no physical mode
  in the first ten, so a "finer is better" reading of this gate would be wrong.
* Donnell thin-shell theory is the oracle for `n ≥ 2`; the `n = 1` and `n = 0`
  rows are now computable but are not gated.
* `G2` at 6.84 % is discretisation on a 10×8 mesh, not physics.

Absolute paths of the evidence: `benchmarks/verified/fem_shell_modal/result.json`,
`…/gates_verified.json`, `…/shell_modal_audit.png`,
`benchmarks/verified/anchor_certificates.json`,
`benchmarks/verified/anchor_certificate_debt.json`.
