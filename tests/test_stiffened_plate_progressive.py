"""Regression gates for the welded stiffened-plate progressive-collapse chain.

The module ``tensorfem.stiffened_plate_progressive`` shipped (mission m8351)
without any test; two of its entry points could not even be called.  These
tests pin the fixed behaviour to references derived here from plate/column
theory rather than to snapshots of the implementation.

Reference formulas (independent of the module under test):
* Navier simply-supported plate buckling, Timoshenko & Gere (1961) Eq. 9.6 /
  Paik et al. (2007) Eq. 2.1: ``sigma_cr = k*pi^2*D/(b^2*t)``.
* Elastic axial stiffness of the gross section: ``sigma_avg = E * epsilon``,
  because the reference load is one pascal over the gross area.
"""
import math

import pytest
import torch

from tensorfem.stiffened_plate_progressive import (
    WeldedPlateSpec,
    bay_imperfection_mode,
    build_welded_plate,
    path_summary,
    trace_welded_plate_path,
    welded_plate_buckling,
)

SPEC = WeldedPlateSpec(
    length=3.0, width=1.0, plate_t=0.01,
    web_h=0.15, web_t=0.01, flange_w=0.10, flange_t=0.01,
    spacing=0.5, young=206e9, poisson=0.3, sigma_y=315e6, E_t=2e9,
    nx=6, ny=4, layers=3,
)


def navier_stress(a, b, t, E, nu):
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    k, m = min(((m * b / a + a / (m * b)) ** 2, m) for m in range(1, 25))
    return k * math.pi ** 2 * D / (b ** 2 * t), k, m


def test_buckling_matches_independent_navier_closed_form():
    exact, k, m = navier_stress(SPEC.length, SPEC.spacing, SPEC.plate_t,
                                SPEC.young, SPEC.poisson)
    assert k == pytest.approx(4.0) and m == 6          # a/b = 6, long plate
    out = welded_plate_buckling(SPEC, nx=32, ny=8)
    assert out["sigma_cr_navier"] == pytest.approx(exact, rel=1e-12)
    assert out["sigma_cr_fem"] == pytest.approx(exact, rel=0.03)
    assert out["error"] == pytest.approx(out["sigma_cr_fem"] / exact - 1.0, rel=1e-12)


def test_buckling_error_decreases_under_refinement():
    exact, _, _ = navier_stress(SPEC.length, SPEC.spacing, SPEC.plate_t,
                                SPEC.young, SPEC.poisson)
    errors = [abs(welded_plate_buckling(SPEC, nx=nx, ny=ny)["sigma_cr_fem"] / exact - 1.0)
              for nx, ny in ((8, 2), (16, 4), (32, 8))]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.03
    # geometric (h-) refinement on a fixed domain: observed order must be positive
    p = math.log((errors[1] / errors[2])) / math.log(2.0)
    assert 0.0 < p < 6.0


def test_buckling_stress_is_resultant_over_plate_thickness():
    """Regression for the unit bug: the eigenvalue is a force/length, /t only."""
    out = welded_plate_buckling(SPEC, nx=16, ny=4)
    exact_stress, _, _ = navier_stress(SPEC.length, SPEC.spacing, SPEC.plate_t,
                                      SPEC.young, SPEC.poisson)
    assert out["sigma_cr_fem"] == pytest.approx(exact_stress, rel=0.10)
    assert out["sigma_cr_fem"] * SPEC.plate_t == pytest.approx(
        exact_stress * SPEC.plate_t, rel=0.10)
    # the former (resultant / (spacing * t)) bug inflated the stress by 1/spacing = 2
    assert out["sigma_cr_fem"] < 1.5 * exact_stress


def test_spec_validation_rejects_inconsistent_spacing():
    with pytest.raises(ValueError):
        WeldedPlateSpec(length=3.0, width=1.0, spacing=0.3).validate()


def test_bay_imperfection_mode_shape_and_bay_antisymmetry():
    mode = bay_imperfection_mode(SPEC, nx=18, ny=6)
    n_plate = (SPEC.nx + 1) * (SPEC.ny + 1)
    assert mode.shape == (n_plate,)
    assert float(mode.abs().max()) == pytest.approx(1.0)

    def value(x, y):
        i = int(round(x / SPEC.length * SPEC.nx))
        j = int(round(y / SPEC.width * SPEC.ny))
        return float(mode[j * (SPEC.nx + 1) + i])

    # same relative position inside bay 0 and bay 1 -> opposite signs
    assert value(1.5, 0.25) * value(1.5, 0.75) < 0.0


def test_build_rejects_wrong_imperfection_length():
    with pytest.raises(ValueError):
        build_welded_plate(SPEC, imperfection_mode=torch.ones(3, dtype=torch.float64))


def test_elastic_path_slope_returns_young_modulus():
    """sigma_avg / epsilon == E in the elastic range (no calibration factor)."""
    spec = WeldedPlateSpec(**{**SPEC.__dict__, "nx": 4, "ny": 2, "layers": 2})
    mode = bay_imperfection_mode(spec, nx=18, ny=6)
    case = build_welded_plate(spec, imperfection_mode=mode)
    trace = trace_welded_plate_path(case, steps=3)
    assert trace.converged and len(trace.points) == 3
    first = trace.points[0]
    epsilon = first.end_shortening / spec.length
    assert first.load_factor > 0.0 and epsilon > 0.0
    assert first.load_factor / epsilon == pytest.approx(spec.young, rel=0.05)
    # elastic: nothing has yielded and the load is monotone
    assert max(p.maximum_plastic_strain for p in trace.points) == 0.0
    assert all(a.load_factor < b.load_factor for a, b in zip(trace.points, trace.points[1:]))

    summary = path_summary(case, trace)
    assert summary["points_completed"] == 3
    assert summary["ultimate_stress"] == pytest.approx(trace.ultimate_load_factor)
    assert summary["peak_step"] == trace.ultimate_index + 1
    assert summary["ultimate_over_yield"] == pytest.approx(
        summary["ultimate_stress"] / spec.sigma_y)
    assert math.isfinite(summary["energy_absorbed"])
    assert len(summary["points"]) == 3


def test_path_rejects_nonpositive_step():
    spec = WeldedPlateSpec(**{**SPEC.__dict__, "nx": 4, "ny": 2, "layers": 2})
    case = build_welded_plate(spec, imperfection_mode=bay_imperfection_mode(spec, nx=18, ny=6))
    with pytest.raises(ValueError):
        trace_welded_plate_path(case, steps=2, step_size=0.0)