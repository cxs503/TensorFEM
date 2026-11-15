# XVAL v8 -- implicit IBM + Aitken anchor: is the added-mass excess geometric?

*Mission m21392, TensorFEM. Anchor: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`
(joconnor22/LIFE implicit-IBM + Aitken relaxation; acceptance < 15 % vs reference).*
*Previous: `docs/xval_implicit_ibm_aitken_v7.md` (verdict WITHHELD, attribution "the IBM body is a
bluff body of thickness ~2 kernel supports").*

## 1. Case and question

The archived rounds all measured the **added mass of a marker plate driven in
prescribed transverse oscillation** inside a periodic lattice-Boltzmann box and
compared it with the potential-flow value for a thin plate, ``m_a = rho pi c^2``
(Lamb 1932, art. 71).  Every round found a factor 3--3.7 and none of the
candidate explanations survived (epsilon factor, direct-forcing gain, record
length, body shape).  v7 finally attributed the factor to *geometry*: the smeared
IBM body is thicker than a mathematical plate, with a fitted effective
half-width ``c + 3.5 dx``.

That attribution was made by inflating the **chord** of the thin-plate formula.
It was never tested against the added mass of a body that actually has the
claimed thickness, and it was never tested on the one axis that separates a
discretisation artefact from a formulation error: **the lattice spacing**.

| parameter | value |
|---|---|
| plate half-chord ``c`` | 4 (dx = 1 units) |
| periodic box | 64 x 64 (physical), 8:1 chord/box |
| drive amplitude ``A`` | 0.2 (5 % of ``c``) |
| drive period | 256 lattice steps per unit ``dx`` (Mach 5e-3) |
| recorded window | 6 cycles, first 3 discarded |
| direct-forcing gain | ``B^-1`` operator gain (constraint exact to 1e-15) |
| kernel support | 3-point delta, half-support 1.5 cells = 0.375 ``c`` |

## 2. Method

1. **Independent reference** (`src/tensorfem/added_mass_bem2d.py`): constant
   strength single-layer boundary-element solution of the 2-D Neumann problem
   for a body translating in an unbounded ideal fluid, ``m_ij = -rho * oint(phi_i
   n_j) dl``.  It is validated against the closed forms for the circle and the
   ellipse (see 3.1) and needs neither the lattice-Boltzmann solver nor the IBM
   coupling, so it can arbitrate the geometric story.
2. **Refinement driver** (`src/tensorfem/ibm_added_mass_refinement.py`): the
   prescribed oscillation re-expressed so that a change of ``dx`` refines the
   lattice at *fixed physical geometry*, using the acoustic scaling ``dt = dx``
   (length ``L -> L/dx`` cells, period ``T -> T/dx`` steps, velocity numerically
   unchanged).  The added mass is read from the **fluid momentum**
   ``Pi_y = sum_sites (c f)_y`` (no interface force at all) and cross-checked
   against the marker force with the lattice acceleration.
3. **Protocol calibration**: drive period and record length were swept first
   (3.2) because the archived protocol (period 64 steps, 4 recorded cycles)
   turns out to be worth a 25 % systematic.
4. The refinement series ``dx = 1, 1/2, 1/4`` was submitted to the cluster
   (jobs `cj3404` refine, `cj3405` body-size sweep); the ``dx = 1`` point, the
   protocol table and the epsilon A/B are local runs recorded in the artefacts.
   See 5.1 for what is **not** in this report.

## 3. Results

### 3.1 The BEM reference is trustworthy

Max relative error of the BEM added mass against the closed forms
(circle ``rho pi a^2``; ellipse ``rho pi b^2`` / ``rho pi a^2`` for the two axes,
three aspect ratios) is **0.19 %**, and the panel-count study moves
``C_a`` by < 0.2 % when the panel count is multiplied by eight.

### 3.2 How thick would the body have to be?

Potential-flow added-mass coefficient ``C_a = m_a/(rho pi c^2)`` of a rectangle
``2c x 2h`` in broadside motion (BEM, unbounded fluid):

| ``h/c`` | ``C_a`` | vs thin plate |
|---|---|---|
| 0.02 | 1.045 | 1.04x |
| 0.05 | 1.087 | 1.09x |
| 0.1 | 1.139 | 1.14x |
| 0.1875 | 1.207 | 1.21x |
| 0.375 | 1.310 | 1.31x |
| 0.5 | 1.363 | 1.36x |
| 0.75 | 1.446 | 1.45x |
| 1 | 1.513 | 1.51x |
| 1.5 | 1.617 | 1.62x |
| 2 | 1.698 | 1.70x |
| 4 | 1.915 | 1.91x |
| 8 | 2.160 | 2.16x |

The body v7 postulates -- a smeared strip of one kernel support half-thickness --
is ``h/c = 0.375`` and has **``C_a = 1.31``**.  A body as thick as its own
chord (``h/c = 1``) still only reaches 1.51.  The measured value is
``C = 2.79``, which on this table requires ``h/c = 47.7``, i.e. a body
about **191 cells thick** with an 8-cell chord.

**The finite-thickness attribution of v7 is refuted quantitatively**: the coupling
would have to inflate the plate by a factor 2.13 more than the kernel support
-- 95 kernel supports, not 2.  Geometry is not the explanation for the open plate.

The closed marker **ring** is a different story: there the interior fluid is
entrained (the constraint fixes the normal velocity on the whole closed curve), so
the expected value is ``2 * rho pi (a + 1.5 dx)^2`` for a ring of radius ``a``,
i.e. ``C = 3.38``, against the archived
measurement 3.04 -- an 11 % agreement.  So the smearing story *does* hold for a
closed body, and fails by a factor 2.13 for the open plate.  That asymmetry is the
new handle this round produces.

### 3.3 Protocol calibration (why the archived numbers move)

Plate ``c = 4``, ``dx = 1``, box 64, amplitude 0.2, operator gain:

| drive period (steps) | cycles | ``C`` momentum | ``C`` force | ``r^2`` | ``v_amp`` |
|---|---|---|---|---|---|
| 64 | 4 | 3.723 | 3.615 | 0.9881 | 0.0196 |
| 128 | 4 | 3.713 | 3.659 | 0.9889 | 0.0098 |
| 256 | 6 | 2.792 | 2.783 | 0.9955 | 0.0049 |

The same physical run changes by **25 %** between a period of 128 and 256 steps
(the low-period runs are not quasi-static: ``omega H / c_s = 5.4`` and ``10.9``,
i.e. a standing acoustic response of the box, not the potential-flow added mass).
With ``omega H / c_s = 2.7`` and three discarded cycles the response is clean
(``r^2 = 0.9955``) and the number is stable in the record length.

### 3.4 Does the epsilon system normalise the force? No.

| epsilon | amplification ``sum(eps ds)/sum(ds)`` | ``C`` momentum | ``C`` force | constraint |
|---|---|---|---|---|
| on | 2.188 | 2.792 | 2.783 | 7.1e-16 |
| off | 1.000 | 2.792 | 2.783 | 7.1e-16 |

Switching the epsilon conservation off leaves the measured added mass unchanged
to machine precision when the ``B^-1`` operator gain is used (the gain absorbs the
amplification exactly).  v7 reached the same conclusion from the force route; this
extends it to the momentum route and closes the item.

### 3.5 The surviving number

With the calibrated protocol the discrete plate at ``dx = 1`` measures
``C = 2.792`` (momentum) and ``2.783`` (marker force), i.e.
**+179 %** above the thin-plate value, with the constraint imposed to
``7e-16``.  The excess is therefore *not* a constraint error,
*not* the epsilon factor, *not* the record length, and *not* the finite thickness
of the smeared body.  It is also not the box size: at the calibrated period the
box sweep 64 -> 128 (square and anisotropic 64x128 / 128x64) moves ``C`` by < 3 %
(artefact `box.json`, local run; values 2.99 / 3.05 / 3.06 / 2.99).

## 4. Figures

`benchmarks/xval_ibm_added_mass_refinement_v8_artifacts/v8_added_mass_refinement.png`

* (a) ``C_a(h/c)`` of a rectangle (BEM) with the kernel-thickness body (green
  dashed), the thin plate (dotted) and the measured IBM value (red): the measured
  point sits an order of magnitude beyond any plausible section thickness.
* (b) grid refinement at fixed physical geometry: measured coefficient versus
  ``c/dx`` (one point in this report -- the ``dx = 1/2`` and ``1/4`` runs are
  cluster jobs `cj3404`, still queued when this was written).
* (c) the finest clean record: fluid momentum and marker force traces.

## 5. Caveats / honest notes

1. **NOT RUN in this report**: the refinement points ``dx = 1/2, 1/4`` and the
   body-size sweep (``c = 4, 8, 16`` at fixed ``dx``) are submitted as `cj3404`
   and `cj3405` and were still queued at the time of writing; the earlier
   archived-protocol trend (3.72 at ``dx = 1`` -> 2.71 at ``dx = 1/2``, measured
   with the uncalibrated 2-cycle window) is *not* a replacement for them.  The
   resolution dependence is therefore **open**, and with it the question whether
   the excess has a finite limit or vanishes as ``dx -> 0``.
2. **2-D momentum caveat**: in an unbounded 2-D potential flow the total fluid
   momentum diverges logarithmically, so a periodic box returns a
   *box-regularised* momentum.  The force route (which is local) agrees with the
   momentum route to 0.3 % here, so the reported excess is not that tail, but a
   strict added-mass statement from this measurement should quote both routes.
3. The ``h/c = 48`` figure in 3.2 is a **log-extrapolation** of the BEM table
   beyond ``h/c = 8``; it is quoted only to show the magnitude gap, not as a
   precise material thickness.
4. Aitken relaxation is not exercised: the driver prescribes the kinematics, so
   there is no structural fixed point.  The relaxation remains covered by
   `tests/test_aitken_relaxation_validation.py`.
5. Reynolds/Stokes: at ``c = 4`` cells and ``omega = 1.8`` the acoustic/ viscous
   boundary layer is unresolved; the conclusion concerns *inertial* added mass.

## 6. Error verdict (acceptance < 15 %)

| Quantity | Reference | Measured | Error | < 3 % ? | < 15 % ? |
|---|---|---|---|---|---|
| BEM added mass, circle ``a=1`` | ``rho pi a^2`` = 3.1416 | 3.1476 | +0.19 % | **yes** | yes |
| BEM added mass, ellipse ``(1,0.5)`` axis ``y`` | ``rho pi a^2`` = 3.1416 | 3.1476 | +0.19 % | **yes** | yes |
| BEM ``C_a``, kernel-thickness section (``h/c=0.375``) | -- | 1.310 | -- | -- | -- |
| BEM ``C_a``, chord-thick section (``h/c=1``) | -- | 1.513 | -- | -- | -- |
| IBM plate ``m_a``, ``dx=1``, calibrated protocol | ``rho pi c^2`` | ``2.79 x`` | **+179 %** | **no** | **no** |
| IBM plate, force vs momentum route (consistency) | -- | 2.783 vs 2.792 | 0.33 % | yes | yes |
| epsilon A/B (on vs off) | equality | 2.792 vs 2.792 | < 1e-6 | yes | yes |
| constraint error, ``B^-1`` gain | 0 | 7e-16 | exact | yes | yes |

**Is the key quantity within 3 %?** No -- the added mass is 2.79x the thin-plate
reference (179 %), so the anchor stays **WITHHELD** on its acceptance
criterion.  What this round settles is *what the excess is not*: it is not the
finite thickness of the smeared body (that would need ``h/c = 48``, factor
2.13 more than the kernel), not the epsilon system, not the direct-forcing gain,
not the record length.  The remaining candidate -- a lattice/interface artefact that
survives refinement -- is exactly what `cj3404` measures.

## 7. Files

| File | Change |
|---|---|
| `src/tensorfem/added_mass_bem2d.py` | **new**: single-layer BEM added mass (circle/ellipse/rectangle generators, `added_mass_matrix`, validated benchmarks, ring smearing bounds) |
| `src/tensorfem/ibm_added_mass_refinement.py` | **new**: prescribed-oscillation added mass with explicit `dt = dx` refinement, momentum + force routes, constraint audit at the Guo level, epsilon A/B, non-square boxes |
| `benchmarks/xval_ibm_added_mass_refinement_v8.py` | **new**: bem / refine / equiv / csw / aggregate stages |
| `benchmarks/run_v8_sweeps.py`, `benchmarks/v8_report_figure.py` | **new**: cluster sweep driver and artefact aggregation + figure |
| `benchmarks/xval_ibm_added_mass_refinement_v8_artifacts/` | artefacts (`bem.json`, `clean_refine_*.json`, `protocol.json`, `epsilon.json`, `summary.json`, `v8_added_mass_refinement.png`) |
| `tests/test_added_mass_refinement_v8.py` | **new**: 16 tests (BEM vs closed forms, thickness bounds, unit-equivalence of the refinement driver, constraint exactness, epsilon A/B, force/momentum agreement, amplitude linearity) |
| `docs/xval_implicit_ibm_aitken_v8.md` | this report |
