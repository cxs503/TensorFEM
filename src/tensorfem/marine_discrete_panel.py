"""Auditable discrete-stiffener and nonuniform-corrosion screening cases.

The functions use a simply-supported sinusoidal Ritz field.  Stiffeners are
individual Euler--Bernoulli line members sharing the plate displacement at
their attachment lines; they are not smeared into an orthotropic plate.
"""
from __future__ import annotations

import math


TOLERANCE = 0.03


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def discrete_stiffener_buckling(*, length: float = 2.0, width: float = 1.0,
                                thickness: float = 0.012,
                                young: float = 210.0e9, poisson: float = 0.3,
                                stiffener_ei: float = 1.2e5,
                                stiffener_count: int = 3,
                                longitudinal_half_waves: int = 1) -> dict[str, object]:
    """Return a plate-plus-discrete-line-beam Ritz buckling result.

    The beams lie at ``y=j*b/(n+1)`` and enforce compatible transverse
    displacement with the plate sine field.  The independent oracle uses
    ``sum(sin(j*pi/(n+1))**2)=(n+1)/2``.  Only flexural rigidity ``EI`` is
    represented, so torsion, weld flexibility and stiffener tripping are out
    of scope.
    """
    l, b, t = (_positive("length", length), _positive("width", width),
               _positive("thickness", thickness))
    e, ei = _positive("young", young), _positive("stiffener_ei", stiffener_ei)
    if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    if isinstance(stiffener_count, bool) or not isinstance(stiffener_count, int) or stiffener_count < 1:
        raise ValueError("stiffener_count must be a positive integer")
    if (isinstance(longitudinal_half_waves, bool)
            or not isinstance(longitudinal_half_waves, int)
            or longitudinal_half_waves < 1):
        raise ValueError("longitudinal_half_waves must be a positive integer")
    alpha = longitudinal_half_waves * math.pi / l
    beta = math.pi / b
    rigidity = e * t**3 / (12.0 * (1.0 - poisson**2))
    plate = rigidity * (alpha * alpha + beta * beta) ** 2 / (alpha * alpha)
    locations = [j * b / (stiffener_count + 1) for j in range(1, stiffener_count + 1)]
    sampled_sum = sum(math.sin(beta * y) ** 2 for y in locations)
    beam = 2.0 * ei * alpha * alpha * sampled_sum / b
    computed = plate + beam
    oracle_beam = ei * alpha * alpha * (stiffener_count + 1) / b
    oracle = plate + oracle_beam
    return {
        "critical_line_load": computed,
        "plate_contribution": plate,
        "discrete_beam_contribution": beam,
        "oracle_line_load": oracle,
        "relative_error": abs(computed / oracle - 1.0),
        "attachment_locations": locations,
        "compatibility": "shared sinusoidal transverse displacement at each beam line",
    }


def nonuniform_corrosion_buckling(*, cells: int, loss_intensity: float = 0.35,
                                  length: float = 2.0, width: float = 1.0,
                                  thickness: float = 0.012,
                                  young: float = 210.0e9,
                                  poisson: float = 0.3) -> dict[str, float]:
    """Midpoint Ritz estimate for a smooth, nonuniform thickness field.

    ``t(x,y)=t0*(1-eta*sin²(pi*x/a)*sin²(pi*y/b))**(1/3)`` makes local
    bending rigidity exactly ``D0*(1-eta*phi)``.  This provides a positive,
    traceable corrosion field and an exact integral oracle without treating a
    single point pit as a continuum feature.
    """
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2:
        raise ValueError("cells must be an integer of at least two")
    l, b, t = (_positive("length", length), _positive("width", width),
               _positive("thickness", thickness))
    e = _positive("young", young)
    eta = float(loss_intensity)
    if not math.isfinite(eta) or not 0.0 <= eta < 1.0:
        raise ValueError("loss_intensity must be finite in [0, 1)")
    if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    ax, by = math.pi / l, math.pi / b
    d0 = e * t**3 / (12.0 * (1.0 - poisson**2))
    acoef = (ax*ax + by*by)**2 - 2.0*(1.0-poisson)*ax*ax*by*by
    bcoef = 2.0*(1.0-poisson)*ax*ax*by*by
    area = l*b
    exact_integral = d0 * area * ((acoef+bcoef)/4.0 - eta*(9.0*acoef+bcoef)/64.0)
    denominator = ax*ax*area/4.0
    oracle = exact_integral / denominator
    dx, dy = l/cells, b/cells
    integral = 0.0
    minimum_thickness = t
    for i in range(cells):
        sx = math.sin(ax*(i+0.5)*dx)
        cx = math.cos(ax*(i+0.5)*dx)
        for j in range(cells):
            sy = math.sin(by*(j+0.5)*dy)
            cy = math.cos(by*(j+0.5)*dy)
            phi = sx*sx*sy*sy
            local_t = t*(1.0-eta*phi)**(1.0/3.0)
            minimum_thickness = min(minimum_thickness, local_t)
            density = d0*(1.0-eta*phi)*(acoef*sx*sx*sy*sy + bcoef*cx*cx*cy*cy)
            integral += density*dx*dy
    computed = integral/denominator
    return {"critical_line_load": computed, "oracle_line_load": oracle,
            "relative_error": abs(computed/oracle-1.0),
            "minimum_thickness": minimum_thickness,
            "pristine_line_load": d0*(ax*ax+by*by)**2/(ax*ax)}


def run_discrete_panel_qualification() -> dict[str, object]:
    """Execute the discrete coupling and nonuniform-corrosion gates."""
    stiffener = discrete_stiffener_buckling()
    # Two cells deliberately under-resolve the fourth-order trigonometric
    # product; three or more midpoint cells integrate this finite Fourier
    # content to roundoff.  Later refinements demonstrate preservation.
    meshes = (2, 3, 6, 12)
    corrosion = [nonuniform_corrosion_buckling(cells=n) for n in meshes]
    errors = [row["relative_error"] for row in corrosion]
    passed = (stiffener["relative_error"] < TOLERANCE
              and errors[0] > TOLERANCE and errors[1] < TOLERANCE
              and errors[-1] < TOLERANCE
              and max(errors[1:]) < 1.0e-12)
    if not passed:
        raise AssertionError("discrete marine panel qualification failed")
    return {
        "schema": "tensorfem.marine-discrete-panel-qualification/1.0",
        "tolerance": TOLERANCE,
        "stiffener": stiffener,
        "nonuniform_corrosion": {"cell_counts": list(meshes), "results": corrosion},
        "scope": ("linear sinusoidal plate bending with compatible discrete Euler-Bernoulli "
                  "line stiffeners and a smooth nonuniform thickness-sensitivity field; "
                  "not stiffener tripping, weld failure, local-pit resolution, postbuckling "
                  "or ultimate load"),
        "passed": True,
    }
