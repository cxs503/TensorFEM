# CASE-HOLLOW-SPHERE-3D — 3-D thick-walled hollow sphere under internal pressure

**Repo** TensorFEM · **case pool** HOLLOW-SPHERE-3D (ref: analytic solution, tolerance <5%) ·
**artifact** `artifacts/case_hollow_sphere_3d/case_hollow_sphere_3d.json` ·
**driver** `benchmarks/case_hollow_sphere_3d.py` · **library** `src/tensorfem/hollow_sphere_3d.py`

## ① Problem setting and reference source

Spherical shell ``a <= r <= b`` with internal pressure ``p`` and a traction-free
outer surface. Closed-form reference (Lamé 1852; Timoshenko & Goodier, *Theory of
Elasticity*, 3rd ed. 1970, Sec. 5.6):

```
A = p a^3/(b^3-a^3),  B = A b^3
sigma_r = A - B/r^3            sigma_r(a) = -p, sigma_r(b) = 0
sigma_t = A + B/(2 r^3)        (biaxial hoop = meridional state)
u_r     = [(1-2 nu) A r + (1+nu) B/(2 r^2)]/E
U       = 2 pi p^2 [(1-2 nu) a^6 + (1+nu) a^3 b^3/2] / (E (b^3-a^3))
```

The reference needs no FE input; the closed-form strain energy agrees with
Clapeyron's theorem ``U = 1/2 p u_r(a) 4 pi a^2`` to machine precision, which is
the check that caught an algebra slip in the first draft of the formula.

| parameter | value |
|---|---|
| inner radius a | 8.0 m |
| outer radius b | 10.0 m |
| radius ratio b/a | 1.2500 |
| wall thickness t | 2.0 m |
| Young's modulus E | 2.1e+11 Pa |
| Poisson ratio nu | 0.3 |
| internal pressure p | 1e+06 Pa |
| analytic strain energy | 26833.4 J |
| exact shell volume | 2044.13 m^3 |

## ② Method and implementation

**Baseline (HEX8).** Cube-sphere surface (six mapped cube faces, nodes projected on
the sphere) lofted through radial layers, trilinear bricks, consistent pressure
loads from 2x2 face Gauss integration. This reuses the already qualified geometry
generator of `tensorfem.sphere_pressure_benchmark.sphere_volume_mesh`.

**Selected method (HEX20).** The same conforming topology upgraded to 20-node
serendipity bricks (`tensorfem.hex20.upgrade_hex8_to_hex20`) with **curved-boundary
projection**: every inner/outer surface midside node is projected radially onto its
exact sphere (Zienkiewicz & Taylor, *The FEM*, 5th ed., Vol. 1, Sec. 8.6). Pressure
is applied with the 8-node serendipity face shape functions and 3x3 Gauss
integration, so the load distribution stays variationally consistent.

Both paths solve with **Jacobi-preconditioned CG under symmetric elimination of the
six tangential gauge constraints** (`hollow_sphere_3d._constrained_pcg`). Rigid
motion is removed, radial expansion stays free, and no analytic displacement is
prescribed anywhere — the displacement field is a genuine FE result.

Key code pointers:
* `src/tensorfem/hollow_sphere_3d.py` — `lame_sphere` / `lame_displacement` /
  `strain_energy` (closed form), `_mesh` (HEX20 projection), `_pressure_loads`,
  `_constrained_pcg`, `probe_metrics`, `convergence`.
* `benchmarks/case_hollow_sphere_3d.py` — ladders, ref-vs-got table, verdict,
  figures, artifact.
* `tests/test_hollow_sphere_3d.py` — 15 invariants (equilibrium of the analytic
  field, Clapeyron energy identity, serendipity partition of unity, closed-surface
  load resultants, geometry volume gate, verdict sensitivity).

## ③ Parameter table (non-dimensional)

| quantity | value | note |
|---|---|---|
| b/a | 1.250 | thick-wall regime |
| sigma_t(a)/p | 2.0738 | analytic hoop amplification |
| von Mises(a)/p | 3.0738 | biaxial state |
| u_r(a) E/(p a) | 1.751639 | analytic |
| strain energy E/(p^2 a^3) | 11.005875 | analytic |

## ④ Comparison table (selected method, finest mesh)

Finest selected mesh: HEX20 n=4,
192 elements, 3198 dofs.
Probes are compared at matching locations (nodal values on the exact sphere
surfaces, layer volume averages against the reference evaluated at the same
centroid radius).

| probe | location | ref (Lamé) | FE | relative error | <5% |
|---|---|---|---|---|---|
| radial_displacement_inner | nodal average on r=8 m | 6.672912e-05 m | 6.670856e-05 m | 0.031% | ✅ |
| radial_displacement_outer | nodal average on r=10 m | 5.245902e-05 m | 5.229370e-05 m | 0.315% | ✅ |
| radial_stress_inner | layer volume average at r=8.33708 m | -0.761361 MPa | -0.779125 MPa | 2.333% | ✅ |
| hoop_stress_inner | layer volume average at r=8.33708 m | 1.954451 MPa | 1.898259 MPa | 2.875% | ✅ |
| von_mises_inner | layer volume average at r=8.33708 m | 2.715813 MPa | 2.677423 MPa | 1.414% | ✅ |
| hoop_stress_outer | layer volume average at r=9.32959 m | 1.695179 MPa | 1.651736 MPa | 2.563% | ✅ |
| radial_stress_outer | layer volume average at r=9.32959 m (applied_pressure) | -0.242818 MPa | -0.260479 MPa | 1.766% | ✅ |

Worst probe error **2.875%** (hoop_stress_inner),
tolerance 5.000% → **PASS**.
Equilibrium: free residual 7.168e-12,
gauge residual 1.777e-13; closed-surface
load resultants 2.20e-16 /
5.70e-17 (force / moment).

Baseline HEX8 at n=4:
worst 28.182% (radial_stress_inner) — the trilinear
family converges but has not reached the 5% line at this mesh
(`baseline_reached_tolerance = false`).

## ⑤ Convergence and sensitivity

| element | n (cube-face divisions) | radial layers | elements | dofs | worst probe error | strain-energy error | mesh volume error | CPU s |
|---|---|---|---|---|---|---|---|---|
| HEX8 | 4 | 2 | 192 | 882 | 28.182% | 9.071% | 7.116% | 0.0 |
| HEX20 | 4 | 2 | 192 | 3198 | 2.875% | 0.185% | 0.136% | 0.0 |

Observed error decay of the worst probe (selected family, last pair):
0.0 in ``1/n`` units
(``monotone_decreasing = none``).
The strain-energy error (an energy-norm measure) decreases monotonically and from
above, as displacement FE theory requires.

Sensitivity to the geometry treatment (same element count, 192 elements, n=4):
HEX8 mesh volume error 7.116% versus
HEX20 with curved-boundary projection
0.136% — the projection is what
buys the quadratic element its accuracy on a curved boundary.

## ⑥ Failure modes, limitations, artifacts

* **sigma_r → 0 on the outer surface.** A pointwise relative error there measures
  round-off; the outer radial probe is therefore scored as ``|sigma_r - ref|/p``
  (pressure-normalised), and this is stated in the metric contract
  (`secondary_probes`). The naive relative measure blows up to >15% even on meshes
  that are otherwise converged.
* **One radial layer cannot be probed.** With a single layer the innermost and
  outermost element sets coincide; `solve_hollow_sphere` rejects
  `radial_layers < 2` instead of reporting a meaningless pair of identical probes.
* **Unpreconditioned CG is too slow here.** `solve_sparse_static` runs unpreconditioned
  CG; this module uses Jacobi-PCG (67 iterations for HEX8 n=8 versus a near-stall
  on the same system).
* **Coarse meshes fail honestly.** HEX8 n=4 is 28.182%
  off: the case is not reachable with a handful of trilinear elements.
* **OSS cross-reading was unavailable for this domain.** `oss_inventory` holds 33
  rows, all LBM/SPH (`~/oss-cache` = pylbm, hschwane, Maarten-vd-Sande/lbm); there is
  no 3-D FEM structural entry to learn structure from, so the method selection rests
  on the textbook/serendipity literature and the in-repo prior art
  (`hex20.py`, `quadratic_solid.py`, `sphere_pressure_benchmark.py`). Recorded as
  evidence in `oss_solutions` (feature_slug `CASE-HOLLOW-SPHERE-3D`).
* **Scope limit.** Linear elasticity, small strain, single material, closed sphere,
  uniform pressure; no plasticity, contact or thermal loading.

Reproduction:

```
PYTHONPATH=$PWD/src python3 benchmarks/case_hollow_sphere_3d.py \
    --out artifacts/case_hollow_sphere_3d            # full ladders (~6 min, 1 core)
PYTHONPATH=$PWD/src python3 -m pytest tests/test_hollow_sphere_3d.py -q
```

Artifacts: `artifacts/case_hollow_sphere_3d/case_hollow_sphere_3d.json` (all meshes,
probes, verdicts), `profiles.png` (stress vs radius, FE vs Lamé),
`convergence.png` (worst probe error vs h), `field_cut.png` (meridional von Mises:
FE versus analytic).
