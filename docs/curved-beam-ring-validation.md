# Quarter-ring curved cantilever: closed-form verification

**Status:** verified against a closed-form curved-bar solution; added as a
regression gate. Scope is a linear elastic, thin, clamped quarter ring.

## 1. Case and physical background

The classical suite in `docs/classical-shell-suite-evidence.md` covers the
membrane-dominated Scordelis--Lo roof, the pinched cylinder, the hemisphere with
an 18 degree hole and large-rotation pure bending. It had no **bending-dominated
curved** case with a closed-form solution, so a curved element could only be
compared with a converged reference, never *verified*. A quarter ring closes
that gap: it couples hoop (membrane) action with bending, it is statically
determinate, and it has an exact thin-curved-bar solution.

Geometry: quarter of a circular ring of mid-surface radius `R = 10`, rectangular
section `width = 0.2` by `thickness = 0.1`, clamped over all six degrees of
freedom at one end, loaded at the free end along the free-end **tangent**
(`theta = 0` at the load, `theta = pi/2` at the clamp). Material
`E = 2.1e5`, `nu = 0.3`, unit load.

| parameter | value |
|---|---|
| radius `R` | 10 |
| section width / thickness | 0.2 / 0.1 |
| slenderness `R/t` | 100 |
| `E`, `nu` | 2.1e5, 0.3 |
| end load (tangent) | 1.0 |
| reference `delta_t` | 101.77756 |
| reference `delta_r` | 142.85714 |

## 2. Method

Castigliano on the thin curved bar with the correct lever arm
`R (1 - cos phi)` from the loaded end gives three coefficients of
`P R^3 / (E I) = 285.714`:

* tangent load, tangent displacement: `3 pi/4 - 2 = 0.356194`
* tangent load, radial (transverse) displacement: `1/2`
* radial load, radial displacement: `pi/4`

Axial and transverse-shear terms are retained
(`pi P R /(4 E A)` and `pi P R /(4 k G A)`, `k = 5/6`); they add `7.7e-5`
relative for this slenderness. Two element models are solved by
`tensorfem.curved_beam_benchmark.quarter_ring_shell`:
the flat-facet `shell4_stiffness` and the metric-consistent
`cylindrical_shell4_stiffness`. Each has an independent discretisation; the two
meshes are mirror images because the cylindrical element measures `theta` with
the opposite sense, so all gates compare magnitudes.

Reproduce:

```bash
PYTHONPATH=src python -m pytest -q tests/test_curved_beam_benchmark.py
OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/curved_beam_ring_evidence.py
```

Machine-readable evidence: `benchmarks/curved_beam_ring_evidence.json`;
figures: `docs/assets/curved-beam-ring/{convergence,tip-path,slenderness}.png`.

## 3. Results

Free-end displacements, flat-facet element (arc elements `n`, two elements
across the width, all numbers printed by `scripts/curved_beam_ring_evidence.py`):

| element | n | `delta_t` FE | exact | error | `delta_r` FE | exact | error |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 4 | 97.84098 | 101.77756 | 3.868% | 137.62654 | 142.85714 | 3.661% |
| flat | 8 | 100.10634 | 101.77756 | 1.642% | 140.86442 | 142.85714 | 1.395% |
| flat | 16 | 101.01723 | 101.77756 | 0.747% | 142.01632 | 142.85714 | 0.589% |
| flat | 32 | 101.40777 | 101.77756 | 0.363% | 142.46871 | 142.85714 | 0.272% |
| flat | 64 | 101.57999 | 101.77756 | 0.194% | 142.65667 | 142.85714 | 0.140% |
| cylindrical | 4 | 93.87242 | 101.77756 | 7.767% | 134.50642 | 142.85714 | 5.846% |
| cylindrical | 8 | 98.95462 | 101.77756 | 2.774% | 139.95841 | 142.85714 | 2.029% |
| cylindrical | 16 | 100.67999 | 101.77756 | 1.078% | 141.74472 | 142.85714 | 0.779% |
| cylindrical | 32 | 101.30518 | 101.77756 | 0.464% | 142.38197 | 142.85714 | 0.333% |
| cylindrical | 64 | 101.61146 | 101.77756 | 0.163% | 142.62971 | 142.85714 | 0.159% |

Thickness sweep at `n = 32` (normalised coefficient `delta_t E I / (P R^3)`,
closed form `0.356194`):

| element | R/t | coefficient FE | error |
|---|---:|---:|---:|
| flat | 500 | 0.354581 | 0.453% |
| flat | 200 | 0.354847 | 0.380% |
| flat | 100 | 0.354927 | 0.363% |
| flat | 50 | 0.355027 | 0.358% |
| cylindrical | 500 | 0.354539 | 0.465% |
| cylindrical | 200 | 0.354545 | 0.465% |
| cylindrical | 100 | 0.354568 | 0.464% |
| cylindrical | 50 | 0.354660 | 0.461% |

Root reactions reproduce the applied load and its moment
(`|R_t| / P = 1`, `|M| / (P R) = 1`, gates at `1e-6`).

## 4. Error judgement

| quantity | mesh | FE | closed form | relative error | < 3%? |
|---|---|---:|---:|---:|---|
| tangent displacement | flat n=32 | 101.40777 | 101.77756 | 0.363% | yes |
| tangent displacement | cylindrical n=32 | 101.30518 | 101.77756 | 0.464% | yes |
| radial displacement | flat n=32 | 142.46871 | 142.85714 | 0.272% | yes |
| radial displacement | cylindrical n=32 | 142.38197 | 142.85714 | 0.333% | yes |
| tangent displacement | flat n=4 | 97.84098 | 101.77756 | 3.868% | no (deliberate) |
| radial load response (reciprocity) | flat n=32 | 224.05 | 224.39948 | <1% (test gate) | yes |

Both elements reproduce the closed form to better than 1% once the arc is
resolved by 32 elements, and both coarse meshes exceed the 3% limit; the coarse
case is asserted to **fail** so the gate cannot be passed by re-tuning.

## 5. Notes, pitfalls and limits

* **The lever arm is the trap.** Deriving `M = P R cos phi` (measuring from the
  clamp instead of from the loaded end) yields `pi/4 P R^3/(E I) = 224.4` and a
  spurious "55-145% element error". Three independent computations (polygonal
  Euler--Bernoulli frame model, flat-facet shell, cylindrical shell) all converged
  to 101.77, which exposed the formula error rather than an element bug. Any new
  curved-shell benchmark should be checked against a second method before the
  element is blamed.
* **No membrane locking here.** The normalised coefficient is flat to 0.1% over
  `R/t` from 50 to 500, so this bending-dominated curved case is *not* sensitive
  to the thickness. That is the sharp contrast with the Scordelis--Lo roof
  (`docs/classic-shell-benchmarks.md`, 28-47% error), which is
  membrane-dominated: the flat-facet deficiency is specific to curved,
  membrane-dominated response, not to curvature itself.
* Only linear elasticity, clamped-edge, thin-section, dead-load behaviour is
  covered. The model carries no through-thickness stress resolution, so
  curved-beam fibre stresses (Winkler-type stress distribution) are outside the
  claim; the verified quantities are free-end displacements and reactions.
* The two meshes are mirror images (see section 2); the implementation reports
  magnitudes, so a sign error inside an element would not be caught by these
  gates alone. Sign consistency is instead covered by the existing element-level
  rigid-body and patch tests.
* Section distortion and warping are not modelled by either formulation; for
  `h/R` beyond about 0.2 the thin-curved-bar reference itself degrades and the
  case should not be used.