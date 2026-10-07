"""Marine-panel stiffening and degradation sensitivity qualification.

The cases in this module are deliberately closed-form or independently
converged screening models.  They qualify sensitivities used before a full
Shell4 collapse analysis; they do not qualify discrete stiffener tripping,
local pitting, welding simulation, or ultimate strength.
"""
from __future__ import annotations

import math

from .marine_plate_buckling import (
    finite_difference_uniaxial_buckling,
    isotropic_plate_rigidities,
    smeared_longitudinal_stiffener_rigidity,
)
from .marine_imperfect_strip import elastic_strip_oracle


TOLERANCE = 0.03


def _residual_yield_onset(*, residual_stress: float, fibres: int | None = None,
                          young: float = 100.0, yield_stress: float = 1.0,
                          imperfection: float = 0.02,
                          bending_stiffness: float = 2.0) -> float:
    """Return first local yield shortening for the elastic strip.

    ``fibres=None`` evaluates the continuum maximum at ``x=0`` and is the
    independent oracle.  A positive fibre count evaluates midpoint fibres,
    providing a spatial discretisation sequence.
    """
    values = (young, yield_stress, imperfection, bending_stiffness)
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError("material, yield, imperfection and bending inputs must be positive")
    if not math.isfinite(residual_stress) or not 0.0 <= residual_stress < yield_stress:
        raise ValueError("residual_stress must be finite, nonnegative and below yield")
    if fibres is not None and (isinstance(fibres, bool) or not isinstance(fibres, int) or fibres < 4):
        raise ValueError("fibres must be an integer of at least four")

    if fibres is None:
        locations = (0.0,)
    else:
        locations = tuple((i + 0.5) / fibres for i in range(fibres))

    def margin(shortening: float) -> float:
        q, _ = elastic_strip_oracle(
            shortening=shortening, young=young, imperfection=imperfection,
            bending_stiffness=bending_stiffness,
        )
        return max(abs(
            residual_stress * math.cos(2.0 * math.pi * x)
            + young * (shortening + 0.5 * math.pi**2
                       * math.cos(math.pi * x) ** 2
                       * (q*q - imperfection*imperfection))
        ) for x in locations) - yield_stress

    # The connected elastic branch exists through the first-yield event; the
    # deliberately tight upper bracket stays below its later limit point.
    lo, hi = 1.0e-12, 0.007
    if margin(lo) >= 0.0 or margin(hi) <= 0.0:
        raise RuntimeError("could not bracket first yield")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if margin(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run_marine_degradation_qualification() -> dict[str, object]:
    """Run traceable stiffener, uniform-corrosion and residual-stress gates."""
    e, nu, t0 = 210.0e9, 0.3, 0.012
    d0, dy0, h0 = isotropic_plate_rigidities(e, nu, t0)

    # Smeared longitudinal stiffener: FD eigenproblem versus continuous Navier.
    dx = smeared_longitudinal_stiffener_rigidity(
        plate_rigidity=d0, stiffener_young=e, stiffener_area=3.0e-4,
        eccentricity=0.04, spacing=0.5,
    )
    stiff = finite_difference_uniaxial_buckling(
        length=2.0, width=1.0, thickness=t0, rigidity_x=dx,
        rigidity_y=dy0, coupling_rigidity=h0, grid_x=32, grid_y=32,
    )

    # Uniform wastage has the exact classical sensitivity sigma_cr ~ t^2.
    corrosion_rows = []
    pristine_stress = None
    for loss in (0.0, 0.10, 0.20, 0.30):
        thickness = t0 * (1.0 - loss)
        d, dy, h = isotropic_plate_rigidities(e, nu, thickness)
        result = finite_difference_uniaxial_buckling(
            length=1.0, width=1.0, thickness=thickness, rigidity_x=d,
            rigidity_y=dy, coupling_rigidity=h, grid_x=32, grid_y=32,
        )
        if pristine_stress is None:
            pristine_stress = result.critical_stress
        computed_ratio = result.critical_stress / pristine_stress
        reference_ratio = (1.0 - loss) ** 2
        corrosion_rows.append({
            "thickness_loss_fraction": loss,
            "computed_stress_ratio": computed_ratio,
            "classical_t_squared_ratio": reference_ratio,
            "ratio_relative_error": abs(computed_ratio / reference_ratio - 1.0),
            "navier_relative_error": result.relative_error,
        })

    # Self-equilibrated weld-like residual field: midpoint fibres converge to
    # the continuum first-yield location.  The lower onset is a sensitivity,
    # not an ultimate-strength prediction.
    residual = 0.25
    oracle = _residual_yield_onset(residual_stress=residual)
    resolutions = (16, 32, 64, 128)
    onsets = [_residual_yield_onset(residual_stress=residual, fibres=n)
              for n in resolutions]
    errors = [abs(value / oracle - 1.0) for value in onsets]
    zero_oracle = _residual_yield_onset(residual_stress=0.0)

    passed = (
        stiff.relative_error < TOLERANCE
        and all(row["navier_relative_error"] < TOLERANCE for row in corrosion_rows)
        and all(row["ratio_relative_error"] < 1.0e-12 for row in corrosion_rows)
        and errors[-1] < errors[-2] < errors[-3] < errors[-4]
        and errors[-1] < TOLERANCE
        and oracle < zero_oracle
    )
    if not passed:
        raise AssertionError("marine degradation qualification failed")
    return {
        "schema": "tensorfem.marine-degradation-qualification/1.0",
        "tolerance": TOLERANCE,
        "scope": (
            "equivalent smeared stiffener, uniform thickness wastage and "
            "self-equilibrated residual-stress first-yield sensitivity; "
            "not discrete tripping, pitting, welding simulation or collapse"
        ),
        "stiffened_panel": {
            "computed_line_load": stiff.critical_line_load,
            "navier_line_load": stiff.reference_line_load,
            "relative_error": stiff.relative_error,
            "mode": [stiff.longitudinal_half_waves, stiff.transverse_half_waves],
        },
        "uniform_corrosion": corrosion_rows,
        "residual_stress": {
            "field": "sigma_r*cos(2*pi*x), exactly self-equilibrated",
            "ratio_to_yield": residual,
            "fibre_resolutions": list(resolutions),
            "computed_onsets": onsets,
            "continuum_onset": oracle,
            "zero_residual_onset": zero_oracle,
            "relative_errors": errors,
        },
        "passed": True,
    }
