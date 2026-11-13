# 3-D zero-thickness cohesive interface element (COH3D8 topology)

Status: **verified** -- all eleven checks in `benchmarks/cohesive3d_verification.json` pass (2026-11-13T02:44:08.412976+00:00).

## 1. What was missing

The library already had a 2-D cohesive face (`tensorfem.cohesive` two-node interface, `tensorfem.mixed_mode_cohesive` BK law, `tensorfem.cohesive_bending` 2-D three-point bend simulation) but **no three-dimensional interface element**. A delamination, a bonded overlap or an ice/structure adhesive layer could therefore not be meshed at all. Feature-ledger anchor `GAP-FEM-3D-COHESIVE` ("3D cohesive zone element + verification case").

## 2. What was added

* `BKCohesive3DLaw` -- irreversible bilinear law on the three separations `(delta_n, delta_s1, delta_s2)` with the Benzeggagh--Kenane propagation criterion, quadratic nominal-stress initiation, Macaulay-bracketed damage driving (compression never damages), secant unloading from the historical maximum and an **exact analytic consistent tangent** (including the sensitivity of the envelope to the mode-mix direction).

* `cohesive_hex8_response` -- 8-node zero-thickness element, 2x2 Gauss on the mid-surface, local frame rebuilt from the current mid-surface, internal force `int B^T t dA` and tangent `int B^T D B dA` (no geometric stiffness: small-strain host solids only).

* `interface_area`, `mid_surface_frame`, `assemble_cohesive_interface` -- geometry helpers and dense global assembly over a pre-meshed interface patch.


Parameter set used throughout (structural adhesive, SI):

| parameter | value |
|---|---|
| `normal_stiffness_Pa_per_m` | 1000000000000.0 |
| `shear_stiffness_Pa_per_m` | 400000000000.0 |
| `normal_strength_Pa` | 5000000.0 |
| `shear_strength_Pa` | 3000000.0 |
| `mode_i_energy_J_per_m2` | 500.0 |
| `mode_ii_energy_J_per_m2` | 1200.0 |
| `bk_eta` | 1.6 |

## 3. Verification evidence

Every criterion is a closed-form statement, not a snapshot of the code.

| check | criterion | result | status |
|---|---|---|---|
| `law_reduces_to_2d_bk` | 3-D law == merged 2-D BK law on delta_s2 = 0 | max rel 2.46e-15 | PASS |
| `law_energy(pure)` | int t d(delta) == G_Ic / G_IIc | mode_I 2.3e-16, mode_II 7.5e-09, mode_III 7.5e-09 | PASS |
| `law_energy(mixed)` | int t d(delta) == BK G_c | 2.8e-09, 9.8e-08, 2.8e-09 | PASS |
| `law_initiation` | (t_n/t_n0)^2 + (t_s/t_s0)^2 == 1 at onset | max abs 4.4e-16 | PASS |
| `law_tangent_vs_ad` | analytic tangent == autograd Jacobian | 9.9e-16 over 48 pts | PASS |
| `element_patch` | uniform opening: resultant == t * A | 3.6e-16 | PASS |
| `element_rigid_rotation` | force vector rotates with R (0.3/0.7/1.4 rad) | 8.8e-16 | PASS |
| `element_mesh_objectivity` | 1x1..8x8 patches give the same resultant | 2.0e-16 | PASS |
| `coupled_block_mode_i` | assembly == series-spring closed form | 9.4e-12 | PASS |
| `coupled_block_mode_ii` | assembly == series-spring closed form (shear) | 1.14e-02 | PASS |
| `coupled_block_mesh` | coupled reaction mesh independent | 1.4e-13 | PASS |
| `coupled_block_energy` | W_ext - U_adherends == cohesive dissipation | 3.8e-04 | PASS |

![traction-separation](../../docs/assets/cohesive3d_traction_separation.png)

![coupled block](../../docs/assets/cohesive3d_coupled_block.png)

## 4. Key numbers

* Coupled mode-I specimen reaches **4.9703 MN** against the closed form 4.9703 MN (nominal strength ratio 0.994059145, i.e. the peak is `t_n0 * A` to machine precision).

* Global energy balance at stroke 0.190 mm: external work 498.515 J, adherend strain energy 0.0118 J, cohesive dissipation 498.694 J (closes to 3.8e-04).

* Effective opening reached 95.0 % of the failure opening (2.00e-04 m) before the driver stopped; the remaining envelope area is verified analytically by `law_energy`.

## 5. Pitfalls found while building this

1. **Node ordering.** `cohesive_shape_functions` first used `eta_i = (-1, 1, 1, -1)`; the mid-surface Jacobian then collapsed (t_xi == t_eta) and every element was flagged degenerate. The correct COH3D8/HEX8 pattern is `xi_i = (-1,1,1,-1)`, `eta_i = (-1,-1,1,1)`.

2. **Local frame ordering.** The law is written in `(n, s1, s2)` order; stacking the rotation as `(e1, e2, n)` silently swapped normal and shear (a pure normal opening produced a *shear* traction). The triad rows must be `(n, e1, e2)` -- still a cyclic (det = +1) permutation.

3. **The naive consistent tangent is not the derivative of the discrete law.** The onset and failure amplitudes and the BK energy all depend on the mode-mix direction, which itself depends on the separation vector. Freezing the envelope gives a tangent that differs from autograd by up to ~50 % off the proportional path. Including `d(onset, a_f, G_c)/dc` restores agreement to 1e-15.

4. **Post-failure singularity.** After the interface fully fails the upper adherend keeps an unconstrained rigid translation and the Newton driver's `K_ff` becomes singular. The benchmark driver adds a 1e-12 relative diagonal regularization (documented in the script); this is a property of the *specimen*, not of the element.

5. **Honest limitation of the mode-II coupled check.** The shear specimen uses one HEX8 through each adherend with the transverse dofs suppressed. Its elastic limit already differs from the pure shear-spring model by 1.1 % (measured on a 1e-8 m stroke), so the coupled mode-II comparison carries that specimen-level offset; the mode-I specimen, whose adherend model is exact, matches to 1e-11. The cohesive element itself is pinned independently by the patch, rotation and mesh-objectivity checks.

## 6. Scope limits (not claimed)

* Small-strain kinematics: no geometric (initial-stress) stiffness term, so the element is not qualified for geometrically non-linear host solids.

* No friction, no contact/compression damage, no cyclic degradation, no mixed-mode non-proportional path validation (that is feature-ledger anchor `Mixed-Mode cohesive 非比例路径验证`, a separate item).

* Prescribed crack path: the interface plane is pre-meshed (standard cohesive practice); there is no crack-path search or remeshing.


## 7. Evidence integrity (how the artifact is gated)

`benchmarks/cohesive3d_verification.json` is not a claim taken on trust: it declares the content of the code that produced it and the two commands that re-derive it.

| source module | SHA-256 |
|---|---|
| `src/tensorfem/cohesive3d.py` | `7a962794e8a506f91eea00a9387a2108985cd608ea7b8770eda4d15676d70f2f` |
| `src/tensorfem/mixed_mode_cohesive.py` | `4d668809baf2cf69180e1453a1ef4a468bb7c685361026a173618bf6c3e961c7` |
| `src/tensorfem/solid3d.py` | `d638564b5ef76f269e1578073a38ce5f706d911ff7e5e0b330b06064e7645e88` |

`git_head` at generation: `951b7338e6b2b07ef9a86bd101c99817548d5f72`. `checks_digest` (SHA-256 of the criteria dictionary): `82fdcd8ad044cc024a858d441bcdfcf9a5e00dd8f46fd1cff69430e8b6c9f981`. Measured generation runtime: 46.94 s.

Three gates, none of them a human reading the numbers:

1. **Tamper gate.** `checks_digest` hashes the stored criteria, so a hand-edited number (or a partial regeneration) changes the digest and `tests/test_cohesive3d_verification_artifact.py` fails without re-running the benchmark.

2. **Reproducibility gate.** The same test module recomputes the eight fast closed-form checks from the current checkout and compares them leaf by leaf with the committed numbers (`atol = 1e-10`, `rtol = 1e-8`); `python benchmarks/cohesive3d_verification.py --check` does the same for all eleven, the three coupled specimen solves included (exit codes: 0 reproducible, 1 numeric mismatch, 2 artifact missing, 3 stale source hash).

3. **Freshness gate.** The recorded SHA-256 of every source module is compared with the checkout, so an artifact cannot outlive the code it describes: editing `src/tensorfem/cohesive3d.py` without regenerating fails the test suite, and the failure message names the regeneration command.

Known limit of gate 2 (stated, not hidden): an absolute floor of `atol = 1e-10` means a change confined to a machine-precision residual (~1e-16) is invisible to the numeric comparison -- those leaves are protected by gate 1 only. The floor is deliberate: cross-architecture float64 reduction order moves such leaves by more than any sane relative tolerance.


## 8. Reproduce

```bash
python benchmarks/cohesive3d_verification.py          # writes benchmarks/cohesive3d_verification.json
python benchmarks/cohesive3d_verification.py --check  # re-derive all 11 and diff vs the artifact
python -m pytest tests/test_cohesive3d.py tests/test_cohesive3d_verification_artifact.py -q
python scripts/cohesive3d_report.py                  # doc page + figures + platform report
```
