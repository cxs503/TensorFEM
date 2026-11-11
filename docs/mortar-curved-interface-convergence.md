# Curved-interface convergence statement for the penalty mortar coupling

Anchor: **GAP-FEM-MORTAR** (acceptance "mortar contact algorithm + convergence
verification"). Artifacts: `benchmarks/mortar_curved_interface_convergence.py`,
`tests/test_mortar_curved_convergence.py`,
`docs/mortar-curved-convergence/convergence.json`, figures
`docs/figures/mortar_curved_{convergence_budget,faceting_order,convergence_profile,convergence_tilt}.png`.

## 1 Why this study exists

`src/tensorfem/mortar_contact3d.py` ships *planar* patch tests (uniform
pressure transfer, non-matching quad meshes, role interchange, tangent vs
directional difference). `docs/mortar-two-block-segment-validation.md` states of
them, correctly: *"a patch test, not a convergence statement"*. Until now
nothing in the repository quantified how the low-order penalty mortar transfer
error decays when the interface is **curved**, nor which part of that error a
user can reduce by refining the slave mesh and which part they cannot.

## 2 Case, geometry, parameters (all numbers from `convergence.json`)

One undeformed configuration, frictionless penalty contact, no Newton iteration
(the quantity studied is the transfer/discretisation error, not the solver):

| quantity | value |
|---|---|
| master surface | circular arc, R = 1.0 m, apex at z = δ, inscribed chords (n_master) |
| slave surface | planar plate, x ∈ [−0.20, 0.20] m, y ∈ [0, 0.02] m (W = 0.02), z = φx |
| apex penetration δ | 0.01 m (a = √(2Rδ−δ²) = 0.14107 m) |
| normal penalty εₙ | 1.0 × 10⁶ N/m³ |
| slave quadrature | 2 × 2 Gauss per QUAD4, reference-area measure |
| master band | |x| ≤ 0.35 m, y ∈ [0, 0.06] m (covers the slave) |
| tilt series | φ = 0.05 (asymmetric contact, non-zero moment) |

**Penetration convention.** The code measures the *perpendicular* (normal) gap,
`pen = −dot(p − q, n_facet)` = `(z_chord − z_slave)/√(1+m²)`. Classical 2-D
formulas use a *vertical* gap. The two differ at O((a/R)²); see §5, this is a
definition effect and must not be misread as discretisation error.

## 3 Method: three independent references

| oracle | what it is | why it is independent |
|---|---|---|
| `oracle_radial` | exact circle, radial normal gap, adaptive quadrature (`scipy.integrate.quad`, rtol 1e-13) | no mesh, no facets, no torch |
| `oracle_facet` | exact integral of the *faceted* description (piecewise-linear pen, piecewise-constant facet normals), 8-point Gauss per clipped facet | no torch, no mortar assembly, no quadrature rule of the code |
| paraboloid closed form | F = (4/3) εₙ W a δ | textbook limit, O(δ/R) |

Oracle self-validation (`convergence.json → oracle_selfcheck`): F = dE/dδ holds
to **1.3e-9** relative, and the closed form is reproduced to **0.10 %**.

Error budget used throughout:

* `faceting` = |oracle_facet − oracle_radial| → pure geometry, decays with h_m
* `quadrature` = |FE − oracle_facet| → single-pass Gauss sampling of the master kinks
* `total` = |FE − oracle_radial|

## 4 Results

### 4.1 Planar control: patch test is exact at every level

| n_slave / n_master | rel err F | rel err E |
|---|---|---|
| 10 / 8 | 0.0 | 4.2e-16 |
| 20 / 16 | 0.0 | 6.9e-16 |
| 40 / 32 | 0.0 | 4.2e-16 |

### 4.2 Curved master, master faceting refined (n_slave = 40)

| n_master | h_m [m] | F [N] | total err F | total err E | quadrature err F | faceting err F | peak pen [m] |
|---|---|---|---|---|---|---|---|
| 8 | 0.08750 | 33.7878 | 9.910 % | 15.447 % | 3.0e-4 | 9.937 % | 0.009896 |
| 16 | 0.04375 | 36.5915 | 2.434 % | 4.141 % | 6.4e-5 | 2.428 % | 0.009950 |
| 32 | 0.02187 | 37.2714 | 0.621 % | 1.037 % | 5.2e-5 | 0.616 % | 0.009976 |
| 64 | 0.01094 | 37.4435 | **0.162 %** | 0.259 % | 8.0e-5 | 0.154 % | 0.009988 |

Exact-circle reference: F = 37.5044 N, E = 0.150256 J·1e6, peak penetration = 0.010000 m.

### 4.3 Order of accuracy (pure geometry, numpy oracle, 7 levels)

| n_master | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| faceting err F | 9.94e-2 | 2.43e-2 | 6.16e-3 | 1.55e-3 | 3.57e-4 | 6.38e-5 | 1.00e-5 |

Least-squares observed order over all 7 levels: **2.18** (tail 256→512: 2.23); E:
2.12. This is the **second-order convergence statement** the repository was
missing. (Slightly above 2 because the leading one-sided sagitta bias partially
cancels against the next-order term; it is O(h²) behaviour, not superconvergence
in the usual sense.)

### 4.4 Joint refinement (slave kept 1.25× finer than the master)

| n_slave / n_master | total err F | total err E | observed order |
|---|---|---|---|
| 20 / 16 | 2.418 % | 4.090 % | — |
| 40 / 32 | 0.621 % | 1.037 % | — |
| 80 / 64 | **0.164 %** | 0.261 % | 1.94 (F) / 1.99 (E) |

### 4.5 Finding: slave quadrature must resolve the master kinks

Master fixed at n_master = 64 while the slave mesh is refined:

| n_slave | quadrature err F | total err F |
|---|---|---|
| 10 (coarser than master) | **4.60e-3** | 3.05e-3 |
| 20 | 1.41e-4 | 1.69e-3 |
| 40 | 8.00e-5 | 1.62e-3 |
| 160 | 2.71e-5 | 1.57e-3 |

Single-pass 2×2 Gauss is exact for a linear integrand, but the facet penetration
is only *piecewise* linear: at a master node the slope jumps, so a slave facet
straddling a master facet boundary is undersampled. The induced error is
**±0.46 % when the slave is coarser than the master** and decays once
h_slave ≲ h_master. Practical rule: keep the slave quadrature mesh at least as
fine as the master surface, or use 3×3 Gauss / subdivision. The error is
not one-signed (it partially cancelled in the total budget above), so it cannot
be removed by a safety factor.

### 4.6 Asymmetric contact (φ = 0.05): moment transfer

| n_master | F [N] | rel F | M [N] | rel M |
|---|---|---|---|---|
| 8 | 40.884 | 8.256 % | 2.0280 | 7.980 % |
| 16 | 43.602 | 2.065 % | 2.1571 | 2.118 % |
| 32 | 44.339 | 0.441 % | 2.1935 | 0.466 % |
| 64 | 44.527 | **0.0213 %** | 2.2029 | **0.0430 %** |

Exact: F = 44.5215 N, M = 2.20381 N·m. Observed order 2.80 (F) / 2.48 (M) — faster
than 2 for the same cancellation reason as §4.3. The moment is a genuine test:
with the earlier, incomplete nodal-moment formula (dropping the `z·F_x` term,
which is non-zero on a tilted slave) the same run reported a spurious 9 % moment
"error" — reported here so the trap is not repeated.

## 5 Shape of the error budget, and the one non-discretisation term

At n_master = 64 / n_slave = 40: total 0.162 % = faceting 0.154 % ⊕ quadrature
0.008 %. **Faceting dominates; refining only the slave cannot help.**

The normal-vs-vertical gap convention contributes
`rel_normal_vs_vertical = 0.402 %` at a/R = 0.141
(F_radial = 37.5044 N vs F_vertical = 37.6557 N; the classical paraboloid
F = (4/3)εₙWaδ = 37.6180 N sits in between). This is O((a/R)²) and **does not
vanish with mesh refinement** — it is the difference between "normal traction
integrated over the deformed slave area" (what the code does) and "vertical
pressure resultant" (what hand formulas give). Any comparison against a 2-D
closed form should expect a 0.4 %-level offset at this curvature, 0.04 % at
a/R = 0.045.

## 6 Error verdict (vs reference, explicit <3 % answer)

| quantity | reference | discrete value | error | < 3 %? |
|---|---|---|---|---|
| F, n_master = 16 / n_slave = 20 | 37.5044 N (exact circle) | 36.5980 N | 2.42 % | **yes** |
| F, n_master = 32 / n_slave = 40 | 37.5044 N | 37.2714 N | 0.621 % | yes |
| F, n_master = 64 / n_slave = 80 | 37.5044 N | 37.4430 N | 0.164 % | yes |
| E, n_master = 64 / n_slave = 80 | 0.150256 J×1e6 | 0.150647×1e6 | 0.261 % | yes |
| M, φ = 0.05, n_master = 64 | 2.20381 N·m | 2.20287 N·m | 0.043 % | yes |
| peak penetration, n_master = 64 | 0.010000 m | 0.009988 m | 0.12 % | yes |
| planar patch (any level) | εₙδA | identical | 0 (4e-16) | yes |
| **normal vs classical vertical gap** | 37.6557 N | 37.5044 N | **0.402 %** | yes, but it is a *definition* offset, not discretisation |

Answer: **yes, < 3 % from n_master ≈ 16 (h_m ≈ 0.044 m ≈ 0.31 a) on**; the
fitted O(h²) law predicts the 3 % threshold at h_m = 0.049 m, consistent with the
measured levels. At the finest level the error is 0.16 %, i.e. 18× inside budget.

## 7 Caveats / scope

* Single undeformed configuration, small sliding, one-pass penalty: this study
  quantifies *transfer* accuracy only. It makes no statement about Newton
  convergence with an evolving active set, about frictional history, or about
  large-sliding objectivity (covered by the existing consistent-tangent tests).
* The master was faceted **inscribed** (chords inside the circle). Over-
  approximated (tangent) facets change the sign of the bias and hence the
  constant of the O(h²) law, not the order.
* Only the normal direction is exercised; tangential/frictional transfer order
  is not addressed here.
* The peak-penetration column is a quadrature-point maximum, so it can only
  approach δ from below.

## 8 Reproduce

```bash
PYTHONPATH=src python3 benchmarks/mortar_curved_interface_convergence.py   # ~2 min, writes the JSON + figures
PYTHONPATH=src python3 -m pytest -q tests/test_mortar_curved_convergence.py # ~40 s
```