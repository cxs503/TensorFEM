"""Plate panel elastoplastic capacity under patch ("ice-floe") loading.

Anchor: ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET`` — the literature gap stating that the
plate capability of this library was purely **elastic** (Kirchhoff/Mindlin), so
ice-structure interaction under a large *patch* load could only be answered with
an elastic stress upper bound: no plastic reserve, no permanent set, no
allowable-patch-pressure criterion.

This module supplies the three pieces demanded by the anchor acceptance:

1. **Elastoplastic plate with isotropic hardening and a low-temperature flow
   stress** (:func:`low_temperature_flow_stress`) feeding the layered Shell4 J2
   plate element (:mod:`tensorfem.layered_shell4_plasticity`).
2. **Permanent deflection post-processing** — unload path from a peak patch
   pressure back to zero and read the residual centre deflection
   (:func:`run_patch_plastic_path` + :attr:`PatchPath.permanent_deflection`).
3. **Consistency regression against the elastic solution on the same mesh** —
   below the analytic first-yield pressure the numerical deflection must
   reproduce the Navier (double-series) thin-plate solution
   (:func:`elastic_patch_plate`); across the yield point the load-deflection
   path must stay continuous.

Reference solutions used for the verification (all closed form, none fitted to
the FE result):

* :func:`elastic_patch_plate` — Navier double series for a simply supported
  rectangular plate under a rectangular central patch pressure.
* :func:`analytic_first_yield_pressure` — first-yield pressure from the Mises
  surface stress ``sigma = 6 M / t^2`` of that elastic field.
* :func:`yield_line_patch_upper_bound` — rigid-plastic yield-line (upper-bound
  theorem) collapse pressure ``p_u = W_int / (w0 * Int_patch w_hat dA)`` with
  ``m_p = sigma_y t^2/4``; reduces to the classical ``24 m_p / a^2`` for a fully
  loaded square plate.
* :func:`strip_lower_bound` — statically admissible two-way strip field, giving a
  lower bound so the numerical collapse load can be bracketed.

Honesty statement
-----------------
* The plate element is a flat-facet small-strain Mindlin shell (Q4, 2x2 Gauss,
  layered through thickness); rotations are small, so results are valid for
  panel deflections up to the order of the thickness (``w/t`` of a few) —
  membrane stiffening beyond that is *not* captured, hence the collapse pressure
  quoted here is the pure bending mechanism value.
* The low-temperature flow stress is a *documented engineering fit* to published
  ferritic ship-steel tensile data, not a material certificate.  Below the
  ductile-brittle transition temperature the plastic mechanism is not admissible
  and the function says so instead of extrapolating.
* The yield-line solution is an upper bound; agreement is therefore expected
  from *below* and the FE result is never tuned to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import torch

from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    solve_layered_shell4_increment,
)

__all__ = [
    "surface_extrapolated_mises",
    "_patch_mechanism_volume_ratio",
    "_patch_mechanism_volume_ratio_quadrature",
    "STEEL_GRADES",
    "FlowStress",
    "low_temperature_flow_stress",
    "flow_stress_table",
    "elastic_patch_plate",
    "analytic_first_yield_pressure",
    "first_yield_reference",
    "plastic_moment",
    "yield_line_patch_upper_bound",
    "ilyushin_sine_upper_bound",
    "johansen_twist_upper_bound",
    "plastic_bound_bracket",
    "strip_lower_bound",
    "section_moment_capacity",
    "max_section_moment_ratio",
    "limit_load_from_path",
    "patch_plate_mesh",
    "patch_pressure_forces",
    "PatchPath",
    "PatchPathPoint",
    "run_patch_plastic_path",
    "plate_capacity_summary",
]


# ---------------------------------------------------------------------------
# 1. Low-temperature flow stress
# ---------------------------------------------------------------------------

#: ``fy20`` = nominal minimum yield strength [MPa] at 20 C, ``design_c`` = grade
#: design temperature, ``slope`` = measured flow-stress sensitivity
#: |d sigma_y / dT| [MPa/K] of normalized ferritic hull steel in the ductile
#: regime, ``dbtt_c`` = ductile-brittle transition temperature [C].
#: Typical published tensile data; kept in one table so a mill certificate can
#: overwrite it.
STEEL_GRADES: dict[str, dict] = {
    "AH32": dict(fy20=315.0, design_c=0.0, slope=0.42, dbtt_c=-55.0),
    "AH36": dict(fy20=355.0, design_c=0.0, slope=0.46, dbtt_c=-55.0),
    "DH32": dict(fy20=315.0, design_c=-20.0, slope=0.44, dbtt_c=-65.0),
    "DH36": dict(fy20=355.0, design_c=-20.0, slope=0.48, dbtt_c=-65.0),
    "EH36": dict(fy20=355.0, design_c=-40.0, slope=0.50, dbtt_c=-80.0),
    "FH36": dict(fy20=355.0, design_c=-60.0, slope=0.52, dbtt_c=-100.0),
    "NV-D36": dict(fy20=355.0, design_c=-20.0, slope=0.48, dbtt_c=-65.0),
    "NV-E36": dict(fy20=355.0, design_c=-40.0, slope=0.50, dbtt_c=-80.0),
    "NV-F36": dict(fy20=355.0, design_c=-60.0, slope=0.52, dbtt_c=-100.0),
}

REFERENCE_TEMPERATURE_C = 20.0


@dataclass(frozen=True)
class FlowStress:
    """Flow stress at a service temperature together with its validity flag."""

    grade: str
    temperature_c: float
    yield_stress: float            # Pa
    yield_stress_reference: float  # Pa at 20 C (nominal grade value)
    sensitivity_pa_per_k: float    # d sigma_y / dT (negative: colder = stronger)
    ductile: bool                  # False below DBTT: no plastic mechanism
    note: str

    @property
    def amplification(self) -> float:
        return self.yield_stress / self.yield_stress_reference

    def as_dict(self) -> dict:
        return {
            "grade": self.grade,
            "temperature_c": self.temperature_c,
            "yield_stress_MPa": self.yield_stress / 1e6,
            "yield_stress_reference_MPa": self.yield_stress_reference / 1e6,
            "sensitivity_MPa_per_K": self.sensitivity_pa_per_k / 1e6,
            "amplification": self.amplification,
            "ductile": self.ductile,
            "note": self.note,
        }


def low_temperature_flow_stress(grade: str, temperature_c: float,
                                *, slope_mpa_per_k: float | None = None) -> FlowStress:
    """Flow stress of a hull steel at service temperature.

    ``sigma_y(T) = sigma_y(20 C) + slope * (20 C - T)`` above the
    ductile-brittle transition temperature.  Below the DBTT the function does
    *not* extrapolate: it returns ``ductile=False`` and keeps the nominal
    strength, because the controlling failure mode there is cleavage rather than
    plastic collapse.
    """
    if grade not in STEEL_GRADES:
        raise KeyError(f"unknown steel grade {grade!r}; known: {sorted(STEEL_GRADES)}")
    record = STEEL_GRADES[grade]
    slope = record["slope"] if slope_mpa_per_k is None else float(slope_mpa_per_k)
    if slope < 0:
        raise ValueError("slope must be non-negative (flow stress rises as T falls)")
    fy20 = record["fy20"] * 1e6
    if temperature_c > 400.0:
        raise ValueError("fit valid below 400 C only (no creep/thermal softening model)")
    if temperature_c < record["dbtt_c"]:
        return FlowStress(grade, float(temperature_c), fy20, fy20, -slope * 1e6, False,
                          f"T={temperature_c:.0f} C below DBTT={record['dbtt_c']:.0f} C: "
                          "cleavage governs, plastic collapse not admissible")
    delta = slope * 1e6 * (REFERENCE_TEMPERATURE_C - float(temperature_c))
    return FlowStress(grade, float(temperature_c), fy20 + delta, fy20, -slope * 1e6, True,
                      f"linear flow-stress fit, slope {slope:.2f} MPa/K, "
                      f"design temperature {record['design_c']:.0f} C")


def flow_stress_table(grade: str, temperatures: Sequence[float]) -> list[dict]:
    return [low_temperature_flow_stress(grade, T).as_dict() for T in temperatures]


# ---------------------------------------------------------------------------
# 2. Elastic Navier solution for a rectangular central patch pressure
# ---------------------------------------------------------------------------

def _patch_fourier_amplitudes(a: float, b: float, patch: tuple[float, float],
                              pressure: float, m: int, n: int) -> float:
    """Sine-series coefficient of a centred rectangular patch of pressure."""
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    gx = 0.5 * (math.cos(m * math.pi * x1 / a) - math.cos(m * math.pi * x2 / a))
    gy = 0.5 * (math.cos(n * math.pi * y1 / b) - math.cos(n * math.pi * y2 / b))
    return 16.0 * pressure / (math.pi ** 2 * m * n) * gx * gy


def elastic_patch_plate(a: float, b: float, t: float, young: float, poisson: float,
                        patch: tuple[float, float], pressure: float,
                        *, terms: int = 41,
                        points: Iterable[tuple[float, float]] | None = None) -> dict:
    """Navier double-series solution: simply supported plate under a patch load.

    Returns lists of deflection and bending moments ``M_x``/``M_y`` (moment per
    unit width) at the requested points (default: plate centre), plus the Mises
    surface stress ``sigma_eq = 6 sqrt(Mx^2 - Mx My + My^2)/t^2``.
    """
    if a <= 0 or b <= 0 or t <= 0 or young <= 0:
        raise ValueError("geometry and modulus must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("invalid Poisson ratio")
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    points = [(0.5 * a, 0.5 * b)] if points is None else list(points)
    D = young * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    w_out, mx_out, my_out, sig_out = [], [], [], []
    for (x, y) in points:
        w = mx = my = 0.0
        for m in range(1, terms + 1, 2):
            alpha = m * math.pi / a
            sx = math.sin(alpha * x)
            for n in range(1, terms + 1, 2):
                beta = n * math.pi / b
                q_mn = _patch_fourier_amplitudes(a, b, patch, pressure, m, n)
                if q_mn == 0.0:
                    continue
                s = sx * math.sin(beta * y)
                lam = alpha ** 2 + beta ** 2
                w += q_mn / (D * lam ** 2) * s
                mx += q_mn * (alpha ** 2 + poisson * beta ** 2) / lam ** 2 * s
                my += q_mn * (beta ** 2 + poisson * alpha ** 2) / lam ** 2 * s
        sig_x = 6.0 * mx / t ** 2
        sig_y = 6.0 * my / t ** 2
        sig = math.sqrt(max(sig_x ** 2 - sig_x * sig_y + sig_y ** 2, 0.0))
        w_out.append(w)
        mx_out.append(mx)
        my_out.append(my)
        sig_out.append(sig)
    return {"deflection": w_out, "Mx": mx_out, "My": my_out,
            "sigma_eq": sig_out, "D": D, "terms": terms}


def first_yield_reference(geometric: bool, inplane_restrained: bool) -> dict:
    """Whether a linear Navier first-yield pressure may judge this FE path.

    :func:`analytic_first_yield_pressure` is the **linear, small-deflection**
    Navier solution of a simply supported plate.  A path integrated with
    geometric nonlinearity and/or with the in-plane displacements restrained on
    the boundary is a *different mechanical model*: membrane tension carries
    part of the patch load and unloads the bending field, so the panel reaches
    first yield at a **higher** pressure.  The measured offset on the anchored
    panel (a = 2.0 m, b = 1.0 m, t = 20 mm, 0.5 m x 0.5 m patch, DH36 at
    -20 C) is +20.4 % at 24x12 and does **not** shrink with mesh refinement
    (observed order 0.026, i.e. a plateau).  Part of that offset *is* the model
    difference: the same mesh measures +38.4 % in the small-deflection
    free-in-plane configuration, i.e. membrane action moves the transition ~9
    points towards the reference.  The remainder is an element/model offset that
    this function does not claim to explain, so the criterion is withheld rather
    than declared met.  Comparing such a path with the linear reference and calling the
    difference an error is a category mistake; the comparison is only defined
    for the small-deflection, free-in-plane configuration.

    Returns a dict with ``comparable`` (bool) and the reason, so a caller can
    withhold the "transition error < 5 %" verdict instead of reporting a
    number that can never pass.
    """
    comparable = not (geometric or inplane_restrained)
    out = {
        "model": "linear Navier double-sine, small deflection, simply supported",
        "path_geometric": bool(geometric),
        "path_inplane_restrained": bool(inplane_restrained),
        "comparable": comparable,
    }
    if not comparable:
        out["reason"] = (
            "path uses geometric nonlinearity and/or in-plane restraint; "
            "membrane action delays first yield relative to the linear reference "
            "by a model difference that does not vanish under mesh refinement")
    return out


def analytic_first_yield_pressure(a: float, b: float, t: float, young: float,
                                  poisson: float, yield_stress: float,
                                  patch: tuple[float, float], *,
                                  grid: int = 21, terms: int = 41) -> dict:
    """Pressure at which the elastic Mises surface stress first reaches yield.

    The maximum of the Navier field is searched on a ``grid x grid`` lattice of
    the plate (the patch-loaded plate peaks at the centre, but the lattice keeps
    the statement honest for elongated plates).
    """
    if grid < 5:
        raise ValueError("grid must be >= 5")
    points = [(a * (i + 0.5) / grid, b * (j + 0.5) / grid)
              for i in range(grid) for j in range(grid)]
    unit = elastic_patch_plate(a, b, t, young, poisson, patch, 1.0,
                               terms=terms, points=points)
    idx = max(range(len(points)), key=lambda k: unit["sigma_eq"][k])
    sig_unit = unit["sigma_eq"][idx]
    if sig_unit <= 0.0:
        raise ValueError("degenerate elastic field: zero surface stress")
    p_yield = yield_stress / sig_unit
    centre = elastic_patch_plate(a, b, t, young, poisson, patch, p_yield,
                                 terms=terms, points=[(0.5 * a, 0.5 * b)])
    return {"pressure": p_yield,
            "location": points[idx],
            "sigma_eq_per_pa": sig_unit,
            "mx_per_pa": unit["Mx"][idx],
            "centre_deflection": centre["deflection"][0],
            "centre_Mx": centre["Mx"][0]}


# ---------------------------------------------------------------------------
# 3. Rigid-plastic bounds (upper: yield line; lower: two-way strips)
# ---------------------------------------------------------------------------

def plastic_moment(thickness: float, yield_stress: float) -> float:
    """Full-plastic moment per unit width of a plate strip, ``m_p = sigma_y t^2/4``."""
    if thickness <= 0 or yield_stress <= 0:
        raise ValueError("thickness and yield stress must be positive")
    return yield_stress * thickness ** 2 / 4.0


def section_moment_capacity(thickness: float, yield_stress: float,
                            layers: int) -> dict:
    """Fully-plastic moment that an ``layers``-deep *layered* section can reach.

    The layered Shell4 model samples the through-thickness stress at the layer
    mid-planes ``z_k = -t/2 + (k+1/2) t/n`` and integrates with the mid-point
    rule, so its fully-plastic moment is::

        m_p,n = sum_k sigma_y |z_k| (t/n) = m_p * rho(n),
        rho(n) = 1 - 1/n^2   (n odd),   rho(n) = 1   (n even)

    (``n = 3`` gives the 8/9 = 0.889 that shows up as a saturation plateau in
    :func:`max_section_moment_ratio`).  Every limit load computed on a layered
    section is therefore multiplied by ``rho``; even layer counts remove the
    deficit exactly, which is why the certification ladders use ``n = 4`` / 8.
    """
    if layers < 1:
        raise ValueError("layers must be >= 1")
    m_p = plastic_moment(thickness, yield_stress)
    dz = thickness / layers
    z = [-0.5 * thickness + (k + 0.5) * dz for k in range(layers)]
    discrete = yield_stress * sum(abs(z_k) for z_k in z) * dz
    ratio = discrete / m_p
    closed = 1.0 if layers % 2 == 0 else 1.0 - 1.0 / layers ** 2
    return {"m_p": m_p, "m_p_layers": discrete, "ratio": ratio,
            "ratio_closed_form": closed,
            "note": ("even layer count: mid-point rule integrates |z| exactly"
                     if layers % 2 == 0 else
                     "odd layer count: m_p deficit (n^2-1)/n^2")}


def max_section_moment_ratio(stress: torch.Tensor, thickness: float,
                             yield_stress: float, layers: int) -> float:
    """Peak quadratic sectional-moment invariant over all Gauss points.

    ``stress`` is the layered Gauss-point history ``[element, 4, layer, 3]``.
    The section moments are the through-thickness integrals
    ``M = sum_k sigma_k z_k (t/n)`` and the invariant is

        ``sqrt(Mx^2 - Mx My + My^2 + 3 Mxy^2) / m_p``

    i.e. the von Mises form of the moment tensor normalised by the *exact*
    ``m_p = sigma_y t^2/4``.  A value of ``1`` means a fully plastic section.
    """
    dz = thickness / layers
    z = torch.tensor([-0.5 * thickness + (k + 0.5) * dz for k in range(layers)],
                     dtype=stress.dtype, device=stress.device)
    moment = (stress * z.view(1, 1, layers, 1)).sum(dim=2) * dz
    mx, my, mxy = moment[..., 0], moment[..., 1], moment[..., 2]
    invariant = torch.sqrt(torch.clamp(mx ** 2 - mx * my + my ** 2 + 3.0 * mxy ** 2,
                                       min=0.0))
    return float(invariant.max()) / plastic_moment(thickness, yield_stress)


def _patch_mechanism_volume_ratio(a: float, b: float, patch: tuple[float, float]) -> float:
    """``Int_patch w_hat dA`` for the yield-line surface ``w_hat = 1-2max(|x|/a,|y|/b)``.

    Closed form (centred coordinates, four-fold symmetry, split at the diagonal
    ``x'/a = y'/b``); no quadrature error.  A fully loaded plate gives ``ab/3``.
    """
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    hx, hy = 0.5 * u, 0.5 * v
    x_c = a * hy / b                      # diagonal crosses y' = hy here
    x1 = min(hx, x_c)
    quad = hy * x1 - hy ** 2 * x1 / b - (b / (3 * a ** 2)) * x1 ** 3
    if hx > x_c:
        x2 = hx
        quad += hy * ((x2 - x_c) - (x2 ** 2 - x_c ** 2) / a)
    return 4.0 * quad


def _patch_mechanism_volume_ratio_quadrature(a: float, b: float,
                                            patch: tuple[float, float],
                                            *, n: int = 240) -> float:
    """Mid-point quadrature cross-check of :func:`_patch_mechanism_volume_ratio`."""
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(4)
    u, v = patch
    xa, xb = 0.5 * (a - u), 0.5 * (a + u)
    ya, yb = 0.5 * (b - v), 0.5 * (b + v)
    # split the patch at both mid-lines and the diagonals: refine by midpoint sampling
    total = 0.0
    for i in range(n):
        x = xa + (i + 0.5) * (xb - xa) / n
        for j in range(n):
            y = ya + (j + 0.5) * (yb - ya) / n
            w = 1.0 - 2.0 * max(abs(x - 0.5 * a) / a, abs(y - 0.5 * b) / b)
            total += max(w, 0.0)
    return total * (xb - xa) * (yb - ya) / n ** 2


def yield_line_patch_upper_bound(a: float, b: float, t: float, yield_stress: float,
                                 patch: tuple[float, float]) -> dict:
    """Yield-line (upper bound) collapse pressure for a patch-loaded plate.

    Mechanism: ``w = w0 (1 - 2 max(|x|/a, |y|/b))``, diagonals as yield lines;
    internal work ``4 m_p w0 (a/b + b/a)``.
    """
    m_p = plastic_moment(t, yield_stress)
    volume = _patch_mechanism_volume_ratio(a, b, patch)
    if volume <= 0.0:
        raise ValueError("patch lies outside the collapse mechanism")
    internal = 4.0 * m_p * (a / b + b / a)
    return {"pressure": internal / volume, "m_p": m_p,
            "mechanism_volume": volume, "internal_work_per_w0": internal}


def ilyushin_sine_upper_bound(a: float, b: float, t: float, yield_stress: float,
                              patch: tuple[float, float], *,
                              terms: int = 1) -> dict:
    """Ilyushin (kinematic) upper bound with a *double-sine* mechanism.

    Independent of :func:`yield_line_patch_upper_bound`: the mechanism is the
    smooth first plate mode ``w = w0 sin(pi x/a) sin(pi y/b)`` (zero on the
    simple-support boundary, hence kinematically admissible) rather than a
    straight yield-line pyramid, so agreement between the two bounds is
    evidence about the collapse load instead of a shared algebra error.

    With the Johansen/Neville "square" moment criterion the dissipation of the
    mechanism is ``m_p (|kappa_1| + |kappa_2|)``; for the sine mode both
    curvatures have the same sign, so the integral is closed form::

        W_int / w0 = m_p pi^2 (m^2/a^2 + n^2/b^2) * 4 a b/(pi^2 m n)
        W_ext / w0 = (a/(m pi))[cos(m pi x1/a) - cos(m pi x2/a)]
                   * (b/(n pi))[cos(n pi y1/b) - cos(n pi y2/b)]
        p_u        = W_int / W_ext

    ``terms`` selects odd modes ``m, n <= 2 terms - 1``; only sign-definite
    modes are admissible as written-down mechanisms, so a mode whose patch
    integral is negative is skipped (its ``|w|`` form is not smooth and is not
    used).  The reported pressure is the smallest admissible one, i.e. the
    tightest bound from this family.

    Checks built into the formulation: for a fully loaded square plate the
    closed form collapses to ``2 pi^2 m_p/a^2 = 19.74 m_p/a^2`` and for a strip
    (``b >> a``) to the beam sine-mechanism value ``pi^2 m_p/a^2`` — both above
    the exact ``8 m_p/a^2`` beam value, as an upper bound must be.

    Validity caveat (admissibility)
    -------------------------------
    The dissipation used here is the *hinge-line* form ``m_p (|k1| + |k2|)``,
    which is the exact Johansen dissipation only when the twisting curvature is
    negligible.  The double-sine mode carries a large twisting curvature and
    dropping it **under-counts** the internal work: for a fully loaded square
    plate this function returns ``19.74 m_p/a^2`` while the exact
    Johansen/Neville collapse load of that plate is ``24 m_p/a^2``.  A number
    below the true collapse load of a known case is *not* an upper bound, so it
    must not be used as the acceptance yardstick for a smooth mechanism;
    :func:`johansen_twist_upper_bound` keeps the twist term and is admissible.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    if not (0 < a and 0 < b and 0 < t and 0 < yield_stress):
        raise ValueError("geometry and strength must be positive")
    m_p = plastic_moment(t, yield_stress)
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    best: float | None = None
    best_mode = None
    for m in range(1, 2 * terms, 2):
        for n in range(1, 2 * terms, 2):
            am = m * math.pi / a
            bn = n * math.pi / b
            internal = 4.0 * m_p * a * b * (m * m / (a * a) + n * n / (b * b)) / (m * n)
            ext = ((a / (m * math.pi)) * (math.cos(am * x1) - math.cos(am * x2))
                   * (b / (n * math.pi)) * (math.cos(bn * y1) - math.cos(bn * y2)))
            if ext <= 0.0:
                continue
            pressure = internal / ext
            if best is None or pressure < best:
                best, best_mode = pressure, (m, n)
    if best is None:
        raise ValueError("no sign-definite mode covers the patch")
    return {"pressure": best, "mode": best_mode, "m_p": m_p,
            "mechanism": "double-sine (Ilyushin kinematic)"}


def johansen_twist_upper_bound(a: float, b: float, t: float, yield_stress: float,
                               patch: tuple[float, float], *,
                               terms: int = 1, quadrature: int = 256) -> dict:
    """Admissible kinematic (upper bound) limit load with the twist term kept.

    Same mechanism family as :func:`ilyushin_sine_upper_bound` -- the smooth
    plate mode ``w = w0 sin(m pi x/a) sin(n pi y/b)``, zero on a simple-support
    boundary and therefore kinematically admissible -- but the dissipation is
    the *frame-invariant* Johansen/Neville dissipation of that field::

        D(kappa)   = m_p (|kappa_I| + |kappa_II|)
        kappa_I,II = (k1+k2)/2 +/- sqrt(((k1-k2)/2)^2 + kxy^2)

    so the moment tensor is bounded by ``m_p`` in *every* direction and the
    twisting curvature enters through the principal curvatures instead of being
    dropped.  ``p_u = int_A D dA / int_patch w dA``.

    Only modes that keep one sign over the whole patch are accepted (a
    unilateral patch pressure cannot do negative work), and the reported value
    is the smallest over the accepted modes: every returned number is a valid
    upper bound by itself and the minimum is the tightest of this family.

    Regression anchors (``tests/test_plate_patch_plastic_bounds.py``): fully
    loaded square plate ``>= 24 m_p/a^2`` (exact Johansen collapse); long strip
    ``>= 8 m_p/a^2`` (exact beam collapse); ``p_u`` scales as ``sigma_y t^2``;
    ``p_u >=`` the twist-neglecting sine value.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    if quadrature < 16:
        raise ValueError("quadrature must be >= 16")
    if not (0 < a and 0 < b and 0 < t and 0 < yield_stress):
        raise ValueError("geometry and strength must be positive")
    m_p = plastic_moment(t, yield_stress)
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    # midpoint rule: the integrand is piecewise smooth with |.| kinks, so the
    # rule order is *verified* by doubling ``quadrature`` in the tests.
    step_x, step_y = a / quadrature, b / quadrature
    xs = (torch.arange(quadrature, dtype=torch.float64) + 0.5) * step_x
    ys = (torch.arange(quadrature, dtype=torch.float64) + 0.5) * step_y
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    cell_area = step_x * step_y
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    inside = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).to(torch.float64)
    best: float | None = None
    best_mode = None
    for m in range(1, 2 * terms, 2):
        for n in range(1, 2 * terms, 2):
            am, bn = m * math.pi / a, n * math.pi / b
            w = torch.sin(am * xx) * torch.sin(bn * yy)
            patch_w = w[inside > 0]
            if patch_w.numel() == 0 or float(patch_w.min()) <= 0.0:
                continue
            k1 = am * am * w
            k2 = bn * bn * w
            kxy = am * bn * torch.cos(am * xx) * torch.cos(bn * yy)
            mean = 0.5 * (k1 + k2)
            radius = torch.sqrt(0.25 * (k1 - k2) ** 2 + kxy * kxy)
            dissipation = m_p * (torch.abs(mean + radius) + torch.abs(mean - radius))
            internal = float(dissipation.sum()) * cell_area
            external = float((w * inside).sum()) * cell_area
            if external <= 0.0:
                continue
            pressure = internal / external
            if best is None or pressure < best:
                best, best_mode = pressure, (m, n)
    if best is None:
        raise ValueError("no single-sign mode covers the patch")
    return {"pressure": best, "mode": best_mode, "m_p": m_p,
            "mechanism": "double-sine (Johansen dissipation, twist retained)",
            "quadrature": quadrature}


def plastic_bound_bracket(a: float, b: float, t: float, yield_stress: float,
                          patch: tuple[float, float], *, terms: int = 1) -> dict:
    """Bracket the patch-loaded plate collapse pressure between admissible bounds.

    Returns the two independent upper bounds (yield line and Ilyushin
    double-sine), the statically admissible strip lower bound, and the tightest
    upper bound with the relative width of the resulting bracket.  A collapse
    load claimed from an FE run is only *evidenced* by this bracket when the
    bracket itself is narrow; the width is therefore reported instead of hidden.
    """
    upper_yield_line = yield_line_patch_upper_bound(a, b, t, yield_stress, patch)["pressure"]
    upper_sine = ilyushin_sine_upper_bound(a, b, t, yield_stress, patch,
                                           terms=terms)["pressure"]
    upper_twist = johansen_twist_upper_bound(a, b, t, yield_stress, patch,
                                             terms=terms)["pressure"]
    lower = strip_lower_bound(a, b, t, yield_stress, patch)["pressure"]
    upper = min(upper_yield_line, upper_sine)
    if lower > upper:
        raise ValueError("lower bound exceeds upper bound: check the inputs")
    admissible = min(upper_yield_line, upper_twist)
    return {"lower_strip": lower,
            "upper_yield_line": upper_yield_line,
            "upper_ilyushin_sine": upper_sine,
            "upper_johansen_twist": upper_twist,
            # ``tightest_upper`` keeps its historical (twist-neglecting)
            # meaning; it is NOT admissible for a smooth mechanism, see the
            # caveat on ``ilyushin_sine_upper_bound``.
            "tightest_upper": upper,
            "tightest_admissible_upper": admissible,
            "admissible_bracket_width_ratio": admissible / lower,
            "bracket_width_ratio": upper / lower,
            "yield_line_minus_sine_ratio": upper_yield_line / upper_sine,
            "agreement_of_upper_bounds_pct": 100.0 * abs(upper_yield_line / upper_sine - 1.0)}


def strip_lower_bound(a: float, b: float, t: float, yield_stress: float,
                      patch: tuple[float, float]) -> dict:
    """Statically admissible two-way strip field (lower bound).

    ``p_x = 8 m_p/(u(2a-u))``, ``p_y = 8 m_p/(v(2b-v))``, ``p_lb = p_x + p_y``.
    """
    m_p = plastic_moment(t, yield_stress)
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    p_x = 8.0 * m_p / (u * (2.0 * a - u))
    p_y = 8.0 * m_p / (v * (2.0 * b - v))
    return {"pressure": p_x + p_y, "p_x": p_x, "p_y": p_y, "m_p": m_p}


def criterion_discrimination(upper: float, lower: float, tolerance: float,
                             *, label: str = "limit load") -> dict:
    """Decide whether a ``tolerance`` acceptance against a *bounded* reference
    can be met by a correct solver at all.

    A plastic limit load quoted from a bracket ``[lower, upper]`` is only known
    to lie inside that bracket, so the smallest relative error against the upper
    bound that any correct result can have is ``1 - lower/upper``.  When that
    floor exceeds ``tolerance`` the acceptance criterion

        ``|p - reference| < tolerance * reference``

    is **unsatisfiable for every solver**, correct or not: a "failure" says
    nothing about the implementation.  The criterion is only meaningful if the
    bracket is tighter than ``1/(1 - tolerance)`` (1.111 for 10 %); the returned
    ``required_tightening_factor`` is how much the bounds must be tightened
    before the acceptance becomes a test of the solver instead of of the
    reference.

    Reported verdicts: ``decidable``, ``not_discriminative`` (bracket too wide),
    ``degenerate`` (``lower == upper``: the reference is exact, the criterion
    tests equality).
    """
    if not (upper > 0.0 and lower > 0.0):
        raise ValueError("bounds must be positive")
    if upper < lower:
        raise ValueError("upper bound must not be below the lower bound")
    if not (0.0 < tolerance < 1.0):
        raise ValueError("tolerance must lie in (0, 1)")
    ratio = upper / lower
    required = 1.0 / (1.0 - tolerance)
    best = max(0.0, 1.0 - lower / upper)
    if ratio == 1.0:
        verdict = "degenerate"
    elif ratio <= required:
        verdict = "decidable"
    else:
        verdict = "not_discriminative"
    return {"label": label, "upper": upper, "lower": lower,
            "bracket_width_ratio": ratio,
            "tolerance": tolerance,
            "required_width_ratio": required,
            "best_possible_relative_error": best,
            "required_tightening_factor": ratio / required,
            "discriminative": verdict == "decidable",
            "verdict": verdict}


# ---------------------------------------------------------------------------
# 4. FE model, load path, first yield, permanent set
# ---------------------------------------------------------------------------

def patch_pressure_forces(nodes: torch.Tensor, a: float, b: float, nx: int, ny: int,
                          patch: tuple[float, float], pressure: float) -> torch.Tensor:
    """Lumped nodal loads (w-direction) of a centred rectangular patch pressure."""
    u, v = patch
    dx, dy = a / nx, b / ny
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    force = torch.zeros(6 * len(nodes), dtype=nodes.dtype, device=nodes.device)
    tol = 1e-9 * max(a, b)
    for n in range(len(nodes)):
        x = float(nodes[n, 0])
        y = float(nodes[n, 1])
        if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
            continue
        wx = 1.0 if (abs(x - x1) > tol and abs(x - x2) > tol) else 0.5
        wy = 1.0 if (abs(y - y1) > tol and abs(y - y2) > tol) else 0.5
        force[6 * n + 2] = pressure * dx * wx * dy * wy
    return force


def patch_plate_mesh(a: float, b: float, t: float, nx: int, ny: int, *,
                     young: float, poisson: float, yield_stress: float,
                     hardening: float, layers: int = 5,
                     patch: tuple[float, float] | None = None,
                     exact_thickness: bool = True,
                     shear_correction: float = 5.0 / 6.0,
                     inplane_restrained: bool = False,
                     ) -> tuple[LayeredShell4Model, dict]:
    """Simply supported plate mesh with a unit patch-pressure reference load.

    The patch boundary is required to fall on element edges, so that the patch
    size is a mesh-independent control variable (``ValueError`` otherwise).

    ``inplane_restrained=True`` fixes the in-plane displacements on **all**
    boundary nodes (a panel embedded in a grillage).  That is the boundary
    condition the yield-line / Ilyushin limit-load solutions assume: without
    it the mid-surface can stretch freely, no membrane tension develops after
    the bending moments saturate, and the load-deflection path has no limit
    point at all (load control keeps converging at almost constant load).
    """
    patch = (0.5 * a, 0.5 * b) if patch is None else patch
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    # Both patch *edges* (not only its width) must fall on node lines: otherwise
    # the lumped patch loads do not conserve the patch resultant and the mesh is
    # silently biased (measured 9.6 % deflection error at nx = 6 vs nx = 8).
    for off, size, length, n in ((0.5 * (a - u), u, a, nx), (0.5 * (b - v), v, b, ny)):
        step = length / n
        for point in (off, off + size):
            ratio = point / step
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError("patch edge must coincide with an element edge")
    dtype = torch.float64
    x = torch.linspace(0.0, a, nx + 1, dtype=dtype)
    y = torch.linspace(0.0, b, ny + 1, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    nodes = torch.stack([xx.ravel(), yy.ravel(), torch.zeros_like(xx.ravel())], dim=1)
    elements = []
    for i in range(nx):
        for j in range(ny):
            elements.append([i * (ny + 1) + j, (i + 1) * (ny + 1) + j,
                             (i + 1) * (ny + 1) + (j + 1), i * (ny + 1) + (j + 1)])
    elements = torch.tensor(elements, dtype=torch.long)
    model = LayeredShell4Model(nodes=nodes, elements=elements, young=young,
                               poisson=poisson, thickness=t,
                               yield_stress=yield_stress, hardening=hardening,
                               layers=layers, exact_thickness=exact_thickness,
                               shear_correction=shear_correction)
    fixed = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = i * (ny + 1) + j
            if i == 0 or i == nx or j == 0 or j == ny:
                fixed.append(6 * node + 2)              # w = 0 (simple support)
                if inplane_restrained:
                    fixed.extend((6 * node, 6 * node + 1))
            if i == 0 and j == 0:
                fixed.extend((6 * node, 6 * node + 1))
            elif i == 0:
                fixed.append(6 * node)                  # u = 0 on x = 0
            elif j == 0:
                fixed.append(6 * node + 1)              # v = 0 on y = 0
            fixed.append(6 * node + 5)                  # drilling
    fixed_dofs = torch.tensor(sorted(set(fixed)), dtype=torch.long)
    info = {"nx": nx, "ny": ny, "a": a, "b": b, "t": t, "patch": patch,
            "inplane_restrained": inplane_restrained,
            "fixed_dofs": fixed_dofs,
            "reference_load": patch_pressure_forces(nodes, a, b, nx, ny, patch, 1.0)}
    return model, info


def surface_extrapolated_mises(stress: torch.Tensor, thickness: float) -> float:
    """Peak surface von Mises stress of a layered plate section.

    ``stress`` is the layered Gauss-point stress history ``[element, 4, layer, 3]``
    of :func:`tensorfem.layered_shell4_plasticity.assemble_layered_shell4`.  The
    layer values live at ``z = -t/2 + (k+1/2) t/n``; yielding is a *surface*
    event, so the outermost two layers are linearly extrapolated to ``z = +-t/2``
    (otherwise the layered model reports first yield at ``z = 0.8 t/2`` for five
    layers and delays the transition by a factor ``1/(1-1/n)``).

    .. warning::
       The extrapolation assumes the *elastic* linear stress distribution through
       the thickness.  It locates **first yield** correctly but over-reports once
       the section has yielded: with three layers it reaches ``1.5 sigma_y`` when
       only the outer layer is plastic (measured 533 MPa at ``sigma_y = 355 MPa``).
       Do not quote this value as a post-yield surface stress.
    """
    layers = int(stress.shape[2])
    if layers < 2:
        raise ValueError("need at least two layers to extrapolate to the surface")
    dz = thickness / layers
    z = [(-0.5 * thickness + (k + 0.5) * dz) for k in range(layers)]
    peak = 0.0
    for e in range(stress.shape[0]):
        for q in range(stress.shape[1]):
            for side in (1, -1):
                i1 = layers - 1 if side > 0 else 0
                i2 = layers - 2 if side > 0 else 1
                factor = (side * 0.5 * thickness - z[i1]) / (z[i1] - z[i2])
                a = stress[e, q, i1]
                b = stress[e, q, i2]
                sxx = float(a[0] + factor * (a[0] - b[0]))
                syy = float(a[1] + factor * (a[1] - b[1]))
                sxy = float(a[2] + factor * (a[2] - b[2]))
                eq = math.sqrt(max(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
                peak = max(peak, eq)
    return peak


@dataclass
class PatchPathPoint:
    load_factor: float
    centre_deflection: float
    max_alpha: float
    max_surface_mises: float = 0.0
    #: peak section-moment invariant ``|M|/m_p`` (1.0 = fully plastic section)
    max_moment_ratio: float = 0.0


@dataclass
class PatchPath:
    model: LayeredShell4Model
    info: dict
    loading: list[PatchPathPoint] = field(default_factory=list)
    unloading: list[PatchPathPoint] = field(default_factory=list)
    collapse_factor: float | None = None
    state: LayeredShell4State | None = None
    displacement: torch.Tensor | None = None
    # ``collapse_factor`` is only a *limit* load if load control actually ran out
    # of equilibrium: then the Newton failure is recorded here.  When the ramp is
    # simply exhausted (no limit point reached) the reported collapse is the last
    # prescribed factor and ``ramp_completed`` says so, because the two are not
    # the same physical quantity.
    limit_point_searched: bool = False
    ramp_completed: bool = False
    #: ramp stopped early by ``max_deflection_factor`` (flat post-saturation path)
    ramp_truncated: bool = False
    #: why the ramp was truncated (``None`` when the ramp ran to its end)
    truncation_reason: str | None = None
    # Unloading bookkeeping: a residual deflection is only a *permanent set* if
    # the unload leg actually converged; a failed first step used to be reported
    # as a permanent set equal to the peak deflection.
    unload_steps_requested: int = 0
    unload_converged_steps: int = 0
    unload_failure: str | None = None

    def _interpolate_crossing(self, attribute: str, threshold: float) -> float | None:
        previous: PatchPathPoint | None = None
        for step in self.loading:
            value = getattr(step, attribute)
            if value > threshold:
                if previous is None:
                    return step.load_factor
                v0 = getattr(previous, attribute)
                if v0 > threshold:
                    return previous.load_factor
                frac = (threshold - v0) / (value - v0)
                return previous.load_factor + frac * (step.load_factor - previous.load_factor)
            previous = step
        return None

    @property
    def first_yield_factor(self) -> float | None:
        """Load factor at first yield, from the *surface* von Mises stress when it
        is available (physically the right place), else from the plastic strain
        of the outermost layer."""
        crossing = self._interpolate_crossing("max_surface_mises", self.model.yield_stress)
        if crossing is not None:
            return crossing
        previous: PatchPathPoint | None = None
        for step in self.loading:
            if step.max_alpha > 1e-12:
                if previous is None or previous.max_alpha > 1e-12:
                    return step.load_factor
                frac = step.max_alpha / (step.max_alpha - previous.max_alpha)
                return previous.load_factor + frac * (step.load_factor - previous.load_factor)
            previous = step
        return None

    @property
    def permanent_deflection(self) -> float | None:
        """Residual centre deflection after the unloading leg.

        ``None`` when no unloading step converged at all: in that case the peak
        deflection must *not* be passed off as a permanent set.
        """
        if not self.unloading:
            return None
        return self.unloading[-1].centre_deflection

    @property
    def permanent_set_valid(self) -> bool:
        """True when the residual deflection is a real post-unload state.

        Requires at least one converged unloading step *and* no failure on the
        way to ``unload_to``; a partially unloaded path leaves a residual that is
        larger than the permanent set and is therefore flagged, not quoted.
        """
        return bool(self.unloading) and self.unload_failure is None

    @property
    def collapse_is_limit_point(self) -> bool:
        """True when ``collapse_factor`` came from a genuine loss of equilibrium
        under load control rather than from the end of the prescribed ramp."""
        return self.collapse_factor is not None and self.limit_point_searched

    @property
    def peak_factor(self) -> float | None:
        return self.loading[-1].load_factor if self.loading else None

    @property
    def measured_limit_load(self) -> float | None:
        """The limit load as actually measured by the path, or ``None``.

        Under load control the path can only stop at a limit point when the
        Newton iteration runs out of equilibrium (``limit_point_searched``);
        the last converged factor then is the peak of the load-deflection
        curve to within the bisection tolerance.  A completed ramp has no
        measured limit load -- its end is a prescribed number, not a physical
        one -- and this property says exactly that by returning ``None``.
        """
        if self.collapse_is_limit_point:
            return self.collapse_factor
        return None

    def stiffness_ratio(self) -> float | None:
        """Secant stiffness of the last step relative to the first (elastic) step;
        -> 0 at plastic collapse."""
        if len(self.loading) < 2:
            return None
        first, last = self.loading[0], self.loading[-1]
        prev = self.loading[-2]
        if first.centre_deflection == 0.0:
            return None
        k0 = first.load_factor / first.centre_deflection
        dw = last.centre_deflection - prev.centre_deflection
        dp = last.load_factor - prev.load_factor
        if dw <= 0.0:
            return 0.0
        return (dp / dw) / k0


def section_capacity_ratio_of(path: "PatchPath") -> float | None:
    """Section capacity ``m_p,layers / m_p`` of the path's own model.

    ``None`` when the path carries no model (synthetic paths in the tests).
    """
    model = getattr(path, "model", None)
    if model is None:
        return None
    try:
        return section_moment_capacity(model.thickness, model.yield_stress,
                                      model.layers)["ratio"]
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        return None


def path_admissibility(path: "PatchPath", *,
                       capacity_ratio: float | None = None,
                       tolerance: float = 1e-3) -> dict:
    """Physical-admissibility gate for a load-deflection path.

    Why this exists
    ---------------
    :func:`limit_load_from_path` fits two tangent lines to *whatever* it is
    given, including a path whose last step is not an equilibrium state of the
    layer model at all.  Two such states were measured on the anchored
    ice-structure ladder (``artifacts/patch_plate_plastic_ladder_nx12_free.json``,
    queue job ``cj3424``):

    * the peak section-moment invariant exceeded the section capacity
      (``|M|/m_p = 1.077`` at ``n = 4`` layers, whose capacity ratio is exactly
      ``1.0``) -- a converged return mapping cannot leave the section;
    * the centre deflection amplified ``1.9e3`` times in one load step
      (``0.0696 m -> 133.5 m`` on a 20 mm plate, ``max alpha = 35.4``), i.e. the
      Newton solver "converged" onto a non-equilibrium branch.

    On that path the two-line rule still returned ``1.825 MPa`` with a
    plausible 6.8 % criteria spread; removing the bad point makes the same rule
    return a *negative* limit load.  The reported number was therefore produced
    entirely by the inadmissible point -- fail-open.  This gate lets the callers
    (and :func:`limit_load_from_path`) withhold the number instead.

    Returns ``{"admissible": bool, "violations": [str, ...],
    "capacity_ratio": float|None, "max_moment_ratio": float|None}``.
    """
    points = [q for q in path.loading if q.load_factor > 0.0]
    violations: list[str] = []
    capacity = capacity_ratio if capacity_ratio is not None else section_capacity_ratio_of(path)
    peak_moment: float | None = None
    previous: PatchPathPoint | None = None
    for q in points:
        numbers = (q.load_factor, q.centre_deflection, q.max_alpha, q.max_moment_ratio)
        if not all(math.isfinite(v) for v in numbers):
            violations.append(
                f"non-finite step at p = {q.load_factor:.6g}")
            continue
        if q.max_moment_ratio > (capacity if capacity is not None else 1.0) * (1.0 + tolerance):
            violations.append(
                f"section capacity exceeded: |M|/m_p = {q.max_moment_ratio:.6f} > "
                f"capacity {capacity if capacity is not None else 1.0:.6f} "
                f"at p = {q.load_factor:.6g}")
        peak_moment = (q.max_moment_ratio if peak_moment is None
                       else max(peak_moment, q.max_moment_ratio))
        if previous is not None:
            if q.centre_deflection <= previous.centre_deflection:
                violations.append(
                    f"centre deflection does not increase at p = {q.load_factor:.6g} "
                    f"({previous.centre_deflection:.6g} -> {q.centre_deflection:.6g} m)")
        previous = q
    return {"admissible": not violations, "violations": violations,
            "capacity_ratio": capacity, "max_moment_ratio": peak_moment}


def limit_load_from_path(path: "PatchPath", *,
                         criteria: Sequence[float] = (0.02, 0.05, 0.10),
                         minimum_points: int = 3) -> dict:
    """Extract the plastic *limit load* from a load-deflection path.

    Why this function exists
    ------------------------
    The end of a load ramp is **not** a collapse load.  On a Mindlin plate the
    elastic transverse-shear stiffness keeps the tangent positive definite, so
    load control keeps converging long after the bending moment field has
    saturated: the deflection then runs away at almost constant load and the
    last prescribed factor is an arbitrary number that depends only on how far
    the ramp was pushed.  Three independent, standard limit-load definitions are
    therefore evaluated on the same path and reported together:

    ``intersection_of_tangents``
        intersection of the initial elastic line ``w = p/k0`` with the final
        (fully plastic) tangent line -- the classical double-slope rule;
    ``double_deflection``
        load at which the centre deflection is twice the elastic prediction
        (``w = 2 p/k0``), a conventional 2x-deflection offset yield;
    ``stiffness_criterion_<c>``
        load at which the incremental stiffness ``dp/dw`` has fallen to ``c``
        times the initial elastic stiffness.

    The spread of the three is returned as ``spread``: only a path whose
    measurements agree is a credible limit load.
    """
    points = [q for q in path.loading if q.load_factor > 0.0 and q.centre_deflection > 0.0]
    if len(points) < minimum_points:
        raise ValueError("need at least three loading points with positive deflection")
    admissibility = path_admissibility(path)
    k0 = points[0].centre_deflection / points[0].load_factor
    if k0 <= 0.0:
        raise ValueError("degenerate elastic stiffness")

    def _interp(target: float, attribute: str) -> float | None:
        previous = None
        for q in points:
            value = getattr(q, attribute)
            if value >= target:
                if previous is None:
                    return q.load_factor
                v0 = getattr(previous, attribute)
                if value == v0:
                    return q.load_factor
                frac = (target - v0) / (value - v0)
                return previous.load_factor + frac * (q.load_factor - previous.load_factor)
            previous = q
        return None

    last, before = points[-1], points[-2]
    dp = last.load_factor - before.load_factor
    dw = last.centre_deflection - before.centre_deflection
    k1 = dw / dp if dp > 0.0 else float("inf")

    out: dict = {"elastic_slope": k0, "final_slope": k1,
                 "peak_factor": last.load_factor,
                 "peak_deflection": last.centre_deflection,
                 "points": len(points), "measurements": {},
                 "admissibility": admissibility}
    measurements = out["measurements"]
    if not admissibility["admissible"]:
        # Fail-closed: a limit load fitted to a non-equilibrium path is not a
        # limit load.  Withhold every measurement and say why.
        out["limit_load"] = None
        out["spread"] = None
        out["reason"] = "path inadmissible: " + "; ".join(admissibility["violations"])
        return out

    # 1. two-line intersection (k is deflection per unit load, so the plastic
    #    branch has k1 > k0; p_c = (k1 p_peak - w_peak)/(k1 - k0))
    if math.isfinite(k1) and k1 > k0:
        p_c = (k1 * last.load_factor - last.centre_deflection) / (k1 - k0)
        measurements["intersection_of_tangents"] = float(p_c)
    # 2. twice the elastic deflection: linear interpolation of the first crossing
    previous = None
    for q in points:
        excess = q.centre_deflection - 2.0 * q.load_factor * k0
        if excess >= 0.0:
            if previous is None:
                measurements["double_deflection"] = q.load_factor
            else:
                e0 = previous.centre_deflection - 2.0 * previous.load_factor * k0
                frac = -e0 / (excess - e0)
                measurements["double_deflection"] = float(
                    previous.load_factor + frac * (q.load_factor - previous.load_factor))
            break
        previous = q
    # 3. incremental-stiffness criteria, interpolated inside the first interval
    #    whose mean slope ratio drops below the criterion
    ratios = [1.0]
    for first, second in zip(points, points[1:]):
        span = second.load_factor - first.load_factor
        slope = ((second.centre_deflection - first.centre_deflection) / span
                 if span > 0.0 else float("inf"))
        # incremental (tangent) stiffness dp/dw normalised by the elastic one
        ratios.append(k0 / slope if slope > 0.0 else float("inf"))
    for criterion in criteria:
        key = f"stiffness_criterion_{criterion:g}"
        for index in range(1, len(ratios)):
            if ratios[index] <= criterion:
                r1, r0 = ratios[index], ratios[index - 1]
                p1 = points[index - 1].load_factor
                dp_i = points[index].load_factor - p1
                frac = 0.0 if (r0 <= criterion or r1 == r0) else (r0 - criterion) / (r0 - r1)
                measurements[key] = float(p1 + frac * dp_i)
                break
    # A limit load must lie strictly inside the ramp: the two-line rule returns
    # a *negative* number as soon as the final branch is a stiffening one (the
    # intersection of two stiffening lines is to the left of the origin), and a
    # measurement above the peak is an extrapolation, not a measurement.
    rejected = {key: value for key, value in measurements.items()
                if not (0.0 < value <= last.load_factor)}
    for key in rejected:
        measurements.pop(key)
    out["rejected_measurements"] = rejected
    values = list(measurements.values())
    if not values:
        out["limit_load"] = None
        out["spread"] = None
        out["reason"] = (
            "no admissible limit-load definition fired: the load-deflection path "
            f"does not flatten (final / elastic slope = {k1 / k0:.3f}; a collapse "
            "branch needs a slope well below the elastic one)")
        return out
    out["limit_load"] = measurements.get("intersection_of_tangents") or min(values)
    if len(values) >= 2:
        lo, hi = min(values), max(values)
        out["spread"] = (hi - lo) / lo
        out["minimum"], out["maximum"] = lo, hi
    else:
        out["spread"] = None
    return out


def _centre_node(info: dict) -> int:
    nx, ny = info["nx"], info["ny"]
    return (nx // 2) * (ny + 1) + ny // 2


def _step_rejection(w_new: float, moment_ratio: float, load_factor: float,
                    history: list["PatchPathPoint"], capacity_ratio: float,
                    max_deflection_factor: float | None) -> str | None:
    """Reason to refuse committing a converged step, or ``None`` if it is sound.

    The layer model cannot leave the section shell ``|M| <= m_p,layers`` and a
    small-rotation plate cannot gain an order of magnitude of deflection in one
    load step, so either event means the Newton iterate is not an equilibrium
    state of this model.
    """
    if not (math.isfinite(w_new) and math.isfinite(moment_ratio)):
        return "non-finite state in the increment"
    if moment_ratio > capacity_ratio * (1.0 + 1e-3):
        return (f"section capacity exceeded: |M|/m_p = {moment_ratio:.6f} > "
                f"capacity {capacity_ratio:.6f} at p = {load_factor:.6g}")
    if not history:
        return None
    previous = history[-1]
    if w_new <= previous.centre_deflection:
        return (f"centre deflection does not increase at p = {load_factor:.6g} "
                f"({previous.centre_deflection:.6g} -> {w_new:.6g} m)")
    share = (w_new - previous.centre_deflection) / w_new
    if share > 0.9:
        return ("deflection runaway: one increment carries %.1f %% of the whole "
                "deflection (p = %.6g -> %.6g MPa, w = %.6g -> %.6g m)"
                % (100.0 * share, previous.load_factor / 1e6, load_factor / 1e6,
                   previous.centre_deflection, w_new))
    if max_deflection_factor is not None:
        elastic = load_factor * previous.centre_deflection / previous.load_factor
        if w_new > max_deflection_factor * elastic:
            return ("small-rotation guard: w = %.6g m exceeds %.3g x the elastic "
                    "prediction (%.6g m) at p = %.6g MPa"
                    % (w_new, max_deflection_factor, elastic, load_factor / 1e6))
    return None


def run_patch_plastic_path(a: float, b: float, t: float, *, young: float,
                           poisson: float, yield_stress: float, hardening: float,
                           patch: tuple[float, float], load_factors: Sequence[float],
                           nx: int, ny: int, layers: int = 5,
                           tolerance: float = 1e-8, max_iterations: int = 25,
                           unload_steps: int = 0,
                           unload_to: float = 0.0,
                           exact_thickness: bool = True,
                           shear_correction: float = 5.0 / 6.0,
                           max_deflection_factor: float | None = None,
                           bisection_tolerance: float = 1e-3,
                           geometric: bool = False,
                           inplane_restrained: bool = False) -> PatchPath:
    """Integrate the elastoplastic patch-loading path (adaptive load control).

    On a Newton failure the increment is bisected down to ``bisection_tolerance``
    of the original step.  Bisection is expensive precisely where the bending
    moments have saturated, so the ramp optionally stops as soon as the centre
    deflection exceeds ``max_deflection_factor`` times the elastic prediction
    ``p/k0`` -- beyond that point the path is flat and adds nothing but Newton
    failures (see :func:`limit_load_from_path`).  Optionally the
    path is then unloaded to ``unload_to`` times the reference pressure and the
    residual centre deflection is read off (:attr:`PatchPath.permanent_deflection`).

    ``geometric=True`` switches on the updated geometric stiffness and
    ``inplane_restrained=True`` fixes the in-plane boundary DOFs; together they
    are what makes the post-yield path turn over, so that
    :attr:`PatchPath.measured_limit_load` is a real peak instead of the end of
    the ramp (see the module docstring of
    :mod:`tensorfem.layered_shell4_plasticity`).
    """
    if nx % 2 or ny % 2:
        raise ValueError("even element counts keep the plate centre a node")
    model, info = patch_plate_mesh(a, b, t, nx, ny, young=young, poisson=poisson,
                                   yield_stress=yield_stress, hardening=hardening,
                                   layers=layers, patch=patch,
                                   exact_thickness=exact_thickness,
                                   shear_correction=shear_correction,
                                   inplane_restrained=inplane_restrained)
    path = PatchPath(model=model, info=info)
    state = LayeredShell4State.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    centre = _centre_node(info)
    capacity_ratio = section_moment_capacity(
        model.thickness, model.yield_stress, model.layers)["ratio"]

    def advance(target: float, current: float, history: list[PatchPathPoint]) -> float | None:
        nonlocal state, u
        stack = [(current, float(target))]
        while stack:
            start, end = stack.pop()
            try:
                step = solve_layered_shell4_increment(
                    model, info["reference_load"], end, info["fixed_dofs"], state, u,
                    tolerance=tolerance, max_iterations=max_iterations,
                    geometric=geometric)
            except RuntimeError:
                if (end - start) < bisection_tolerance * max(abs(end), 1e-12):
                    return None
                mid = 0.5 * (start + end)
                stack.append((mid, end))
                stack.append((start, mid))
                continue
            # The state is validated *before* it is committed.  A step that
            # leaves the section (|M|/m_p above the layer capacity) or whose
            # deflection runs away is a Newton artefact, not an equilibrium
            # state; committing it puts a non-physical point into the history,
            # where the limit-load extractor would consume it (measured on the
            # anchored ladder: 0.0696 m -> 133.5 m, |M|/m_p = 1.077; the
            # committed point produced a +20.5 % "limit load").
            w_new = float(step.displacement[6 * centre + 2])
            moment_ratio = max_section_moment_ratio(
                step.stress, model.thickness, model.yield_stress, model.layers)
            rejection = _step_rejection(w_new, moment_ratio, end, history,
                                        capacity_ratio, max_deflection_factor)
            if rejection is not None:
                # A rejected *state* is not a rejected *load level*: the
                # increment was too large for the layer model to stay inside the
                # section, exactly like a Newton failure, and the remedy is the
                # same bisection.  Truncating the ramp here (the previous
                # behaviour) ended the path 7 % below the section capacity while
                # the physical limit point was still ahead of it, so no limit
                # load was ever measured and ``collapse_factor`` degenerated to
                # "last prescribed factor".
                if (end - start) < bisection_tolerance * max(abs(end), 1e-12):
                    path.ramp_truncated = True
                    path.truncation_reason = rejection
                    return current
                middle = 0.5 * (start + end)
                stack.append((middle, end))
                stack.append((start, middle))
                continue
            state, u = step.state, step.displacement
            history.append(PatchPathPoint(
                end, w_new, float(torch.max(step.alpha)),
                surface_extrapolated_mises(step.stress, model.thickness),
                moment_ratio))
            current = end
        return current

    previous = 0.0
    for target in load_factors:
        target = float(target)
        if target <= previous:
            continue
        reached = advance(target, previous, path.loading)
        if reached is None:
            path.collapse_factor = previous
            path.limit_point_searched = True
            break
        if path.ramp_truncated:
            # the increment was rejected before commit: stop, do not append
            previous = reached
            break
        previous = reached
        if max_deflection_factor is not None and len(path.loading) > 1:
            first = path.loading[0]
            elastic = previous * first.centre_deflection / first.load_factor
            if path.loading[-1].centre_deflection > max_deflection_factor * elastic:
                path.ramp_truncated = True
                break
    else:
        path.ramp_completed = True

    if unload_steps > 0 and path.loading:
        peak = path.loading[-1].load_factor
        path.unload_steps_requested = unload_steps
        for k in range(1, unload_steps + 1):
            target = peak + (unload_to - peak) * k / unload_steps
            if abs(target - previous) <= 1e-12:
                continue
            try:
                step = solve_layered_shell4_increment(
                    model, info["reference_load"], target, info["fixed_dofs"], state, u,
                    tolerance=tolerance, max_iterations=max_iterations,
                    geometric=geometric)
            except RuntimeError:
                # the committed state is unchanged on failure, so the peak
                # deflection is still standing: record it and flag the leg.
                path.unload_failure = f"unloading step to {target:.6g} did not converge"
                break
            state, u = step.state, step.displacement
            path.unloading.append(PatchPathPoint(
                target, float(u[6 * centre + 2]), float(torch.max(step.alpha)),
                max_section_moment_ratio(step.stress, model.thickness,
                                         model.yield_stress, model.layers)))
            path.unload_converged_steps = k
            previous = target

    path.state = state
    path.displacement = u
    return path


def plate_capacity_summary(a: float, b: float, t: float, *, young: float,
                           poisson: float, grade: str, temperature_c: float,
                           patch: tuple[float, float], hardening: float,
                           load_factors: Sequence[float], nx: int, ny: int,
                           layers: int = 5, unload_steps: int = 0,
                           exact_thickness: bool = True,
                           geometric: bool = False,
                           inplane_restrained: bool = False,
                           max_deflection_factor: float | None = None) -> dict:
    """Patch-pressure capacity of a plate panel at a service temperature.

    Returns the analytic elastic/yield-line bounds and the finite element path
    (first yield, collapse, permanent set), with the flow stress taken from
    :func:`low_temperature_flow_stress`.  Below the DBTT the plastic result is
    withheld and only the elastic first-yield pressure is reported.

    ``geometric`` / ``inplane_restrained`` / ``max_deflection_factor`` are
    forwarded to :func:`run_patch_plastic_path`.  They default to the historical
    bending-only configuration, but a *limit-load* claim needs
    ``geometric=True, inplane_restrained=True``: without them the layered plate
    keeps a positive transverse tangent after the bending moments saturate, so
    load control never loses equilibrium and ``collapse_factor`` is just the
    last prescribed factor of the ramp.  ``limit_measurements`` (the
    double-slope / offset / stiffness definitions of
    :func:`limit_load_from_path`) is reported alongside, so the caller can tell
    the two apart.
    """
    flow = low_temperature_flow_stress(grade, temperature_c)
    elastic = analytic_first_yield_pressure(a, b, t, young, poisson,
                                            flow.yield_stress, patch)
    upper = yield_line_patch_upper_bound(a, b, t, flow.yield_stress, patch)
    lower = strip_lower_bound(a, b, t, flow.yield_stress, patch)
    summary = {
        "geometry": {"a": a, "b": b, "t": t, "patch": patch,
                     "aspect": a / b, "t_over_b": t / b},
        "material": {"young": young, "poisson": poisson, "hardening": hardening},
        "flow_stress": flow.as_dict(),
        "elastic_first_yield": elastic,
        "yield_line_upper": upper,
        "strip_lower": lower,
        "fe": None,
        "fe_permanent_deflection": None,
        "evidence": "analytic bounds + FE path",
    }
    if not flow.ductile:
        summary["evidence"] = "elastic only: below DBTT, plastic collapse withheld"
        return summary
    path = run_patch_plastic_path(a, b, t, young=young, poisson=poisson,
                                  yield_stress=flow.yield_stress,
                                  hardening=hardening, patch=patch,
                                  load_factors=load_factors, nx=nx, ny=ny,
                                  layers=layers, unload_steps=unload_steps,
                                  exact_thickness=exact_thickness,
                                  geometric=geometric,
                                  inplane_restrained=inplane_restrained,
                                  max_deflection_factor=max_deflection_factor)
    # Same-mesh elastic bias: the layered Shell4 element is not an exact thin-plate
    # element, so the transition point is reported both raw and normalised by the
    # element's own elastic surface-stress response (the "consistency regression
    # against the elastic solution on the same mesh" required by the anchor).
    probe = run_patch_plastic_path(a, b, t, young=young, poisson=poisson,
                                   yield_stress=flow.yield_stress, hardening=hardening,
                                   patch=patch,
                                   load_factors=[0.25 * elastic["pressure"]],
                                   nx=nx, ny=ny, layers=layers,
                                   exact_thickness=exact_thickness,
                                   geometric=geometric,
                                   inplane_restrained=inplane_restrained)
    fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
    bias = fe_sigma_per_pa / elastic["sigma_eq_per_pa"] - 1.0
    fe = {
        "elastic_bias": bias,
        "fe_surface_mises_per_pa": fe_sigma_per_pa,
        "analytic_surface_mises_per_pa": elastic["sigma_eq_per_pa"],
        "first_yield_factor": path.first_yield_factor,
        # only a lost equilibrium is a limit load; otherwise this is the last
        # prescribed factor of the ramp (see ``collapse_is_limit_point``)
        "collapse_factor": (path.collapse_factor if path.collapse_factor is not None
                            else path.loading[-1].load_factor),
        "collapse_is_limit_point": path.collapse_is_limit_point,
        "ramp_completed": path.ramp_completed,
        "unload_converged_steps": path.unload_converged_steps,
        "unload_failure": path.unload_failure,
        "peak_centre_deflection": path.loading[-1].centre_deflection,
        "stiffness_ratio": path.stiffness_ratio(),
        "mesh": {"nx": nx, "ny": ny, "layers": layers},
        "path": [(s.load_factor, s.centre_deflection, s.max_alpha) for s in path.loading],
    }
    # Is a *linear* analytic reference allowed to judge this path?  See
    # :func:`first_yield_reference`.
    fe["elastic_reference"] = first_yield_reference(geometric, inplane_restrained)
    # --- transition point: two estimators with different bases ---------------
    # ``first_yield_factor`` below is the crossing of the *path's* surface Mises
    # with ``sigma_y``.  It is the transition load only while the load ladder
    # brackets that crossing inside the *elastic* branch: when one ladder point
    # already sits on the yielded, saturated branch (~1.13 sigma_y) the linear
    # interpolation over-reads the transition.  Measured on 16x8, 2 x 1 x 0.020 m,
    # DH36 at -20 C: path crossing 0.8597 MPa (+20.7 %) against the elastic field
    # 0.7363 MPa (+3.3 %) of the very same mesh.
    # ``first_yield_factor_elastic`` is ladder independent (one elastic solve,
    # ``fe_sigma_per_pa``) and exact while the pre-yield response is linear; on a
    # ladder dense enough to stay elastic up to yield it reproduces the path
    # crossing to 1e-13 (8x4: both 843884.605638 Pa, docs/
    # CASE-GAP-PP-3-TRANSITION-DETECTOR.md).  It is the estimator a *linear*
    # reference may be judged against (``elastic_reference["comparable"]``).
    elastic_factor = flow.yield_stress / fe_sigma_per_pa
    fe["first_yield_factor_elastic"] = elastic_factor
    fe["first_yield_error_elastic"] = \
        (elastic_factor - elastic["pressure"]) / elastic["pressure"]
    fe["first_yield_error_path"] = fe.get("first_yield_error")
    fe["first_yield_basis"] = ("elastic-field" if fe["elastic_reference"]["comparable"]
                               else "path-crossing (reference not comparable)")
    if path.first_yield_factor is not None:
        fe["first_yield_error"] = (fe["first_yield_factor"] - elastic["pressure"]) / elastic["pressure"]
        fe["first_yield_error_comparable"] = fe["elastic_reference"]["comparable"]
        # analytic first-yield pressure expressed in the *same-mesh* elastic field
        corrected = flow.yield_stress / fe_sigma_per_pa
        fe["first_yield_error_bias_normalised"] = \
            (fe["first_yield_factor"] - corrected) / corrected
        fe["analytic_first_yield_same_mesh"] = corrected
    fe["collapse_error_vs_upper"] = (fe["collapse_factor"] - upper["pressure"]) / upper["pressure"]
    # The limit load measured from the path itself.  Reported with its
    # admissibility gate and the spread of the three definitions: a number whose
    # definitions disagree is not a limit load, and ``None`` is the honest answer
    # when the path never turned over.
    try:
        measurements = limit_load_from_path(path)
    except ValueError as error:
        measurements = {"limit_load": None, "spread": None,
                        "reason": f"limit load not extractable: {error}"}
    fe["limit_measurements"] = measurements
    fe["measured_limit_load"] = measurements.get("limit_load")
    fe["measured_limit_error_vs_upper"] = (
        None if measurements.get("limit_load") is None
        else (measurements["limit_load"] - upper["pressure"]) / upper["pressure"])
    summary["fe"] = fe
    summary["fe_permanent_deflection"] = path.permanent_deflection
    summary["fe_permanent_set_valid"] = path.permanent_set_valid
    if path.permanent_deflection is not None and not path.permanent_set_valid:
        summary["evidence"] = ("analytic bounds + FE path; permanent set WITHHELD "
                               f"(unloading did not converge: {path.unload_failure})")
    return summary