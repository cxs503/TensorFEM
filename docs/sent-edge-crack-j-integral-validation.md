# SENT edge-crack J-integral verification (CASE-FEM-JINTEGRAL-EDGE-CRACK)

Anchor: `benchmarks/verified/fem_sent_j_integral/` (`result.json`, `gates_verified.json`, 4 figures).
Module: `src/tensorfem/edge_crack_sent.py`. Tests: `tests/test_sent_edge_crack_j_integral.py` (19 tests).

## 1. What was missing

`tensorfem.crack_tip_j_integral` integrates Rice's J on sampled fields and
`tensorfem.fatigue.crack_tip_quarter_point` builds a singular finite element
model, but the only configuration ever exercised numerically was a crack in a
**disc**, i.e. the infinite-plate idealisation, and its regression gate on the
recovered ``K_I`` was a factor-two band (``0.25 < K/K_ref < 0.75`` apart from a
15 % point check). No in-tree finite element model existed for an **edge crack
in a finite-width strip**, which is the configuration that carries the free
surface behind the crack mouth and the finite-width correction.

## 2. Model

* Upper-half SENT plate, ``W = 100 mm``, ``H/W = 2``, crack ``a = a/W * W`` from
  the free edge ``x = -a`` to the tip at the origin, ``sigma = 100 MPa`` uniform
  tension on ``y = H``, ``u_y = 0`` on the symmetry plane ahead of the crack,
  crack face / free edge / far edge natural (traction-free).
* Ray-fan six-node (quadratic) triangle mesh from the crack tip; rays are
  uniform in angle plus one ray per top corner so the rectangular boundary is
  reproduced exactly; radial stations are the *same normalised fractions* on
  every ray, so one station index is a closed tip-encircling mesh line.
* Barsoum quarter-point ring: the mid-side node of every edge that meets the
  tip sits at one quarter of the edge length.
* ``J`` from the **finite element** stress and displacement gradient on the fan
  mesh lines (three-point Gauss along quadratic edges, exact parent coordinates,
  no point-location search), upper arc doubled by mirror symmetry, then
  ``K_I = sqrt(J E')`` with ``E' = E`` (plane stress).

## 3. Reference (declared before measuring)

``K_I = sigma sqrt(pi a) F(a/W)``,
``F = 1.12 - 0.231 a/W + 10.55 (a/W)^2 - 21.72 (a/W)^3 + 30.39 (a/W)^4``
(Tada, Paris & Irwin, *Stress Analysis of Cracks Handbook*, 3rd ed.; single
edge notch tension, ``H/W >= 1``, ``a/W <= 0.6``, stated accuracy +-0.5 %).
``F(0.1) = 1.184``, ``F(0.3) = 1.660``, ``F(0.5) = 2.826``; the central-crack
factor is 1, i.e. the edge geometry raises ``F`` by +37 % at ``a/W = 0.2``, so
the comparison discriminates a model that misses the free surface.

## 4. Results (measured)

Three-level ladder, ``H/W = 2``, plane stress, contour at ``r = 0.15 R(theta)``:

| level | sectors / first ring / growth | T6 elements | a/W = 0.20 | a/W = 0.35 | a/W = 0.50 |
|---|---|---|---|---|---|
| coarse | 24 / 0.08 / 1.60 | 390 | -1.67 % | -1.79 % | -2.69 % |
| medium | 48 / 0.05 / 1.45 | 1050 | -1.48 % | -0.97 % | -1.55 % |
| fine | 96 / 0.03 / 1.32 | 2842 | **-0.92 %** | **-0.43 %** | **-0.82 %** |

(relative error of the recovered ``K_I`` against Tada; negative = the FE value
is below the handbook value.) Non-singular control (mid-side nodes at the
midpoint) at the medium level: -2.01 % (``a/W = 0.2``) and -2.02 %
(``a/W = 0.5``).

Global equilibrium of the discrete system: vertical reaction / applied load
= 1.000000000000 for every case (gate ``<= 1e-9``). Path independence over the
three canonical contours: spread 0.13 % / 0.06 % / 0.07 % of ``F``.

## 5. Notes and limits

* **The singular enrichment does not fail closed here.** A non-singular
  quadratic mesh reaches the same 3 % band, because the graded contours sit
  *outside* the innermost ring where the field is smooth enough for quadratic
  interpolation. This is recorded rather than hidden; what the quarter-point
  ring buys is the ``r^(-1/2)`` near-tip field, which is shown in
  ``sent_ligament_field.png`` (log-log slope against an ``r^(-1/2)`` guide).
* The residual error is one-signed (FE below the handbook) and decreases
  monotonically with ``h``; part of it is the finite strip height ``H/W = 2``
  rather than the element, which is why the anchor measures the ``H``
  sensitivity as its own gate instead of assuming it away.
* Corner-adjacent parent points of a few ray-fan elements can be mildly
  twisted; every quadrature point of every element is verified positive
  (``test_every_element_has_a_positive_jacobian_at_its_quadrature_points``).
* Scope: linear elastic plane stress, small strain, stationary mode-I, no
  plasticity, no crack closure, no cohesive traction on the faces, no 3-D
  crack front.

## 6. Error verdict

Key quantity ``K_I`` versus the handbook oracle: **0.43 % - 0.92 % on the finest
mesh, i.e. < 3 % (and < 1 %)**, at ``a/W`` = 0.20 / 0.35 / 0.50. Supporting
gates: equilibrium 1e-13, path spread < 0.15 %, ``H`` sensitivity and GCI
reported per case in `gates_verified.json`.
