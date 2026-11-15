"""Limit-load extraction and layered-section capacity (anchor GAP-PP-3).

These tests pin the *measurement* side of the plate-panel plastic capacity: the
end of a load ramp is not a collapse load, the layered section has a closed-form
plastic moment ``m_p (1 - 1/n^2)`` for odd layer counts, and the tightest
available upper bound is the Ilyushin double-sine one.
"""
import math

import pytest
import torch

from tensorfem.plate_patch_plastic_limit import (
    PatchPath,
    PatchPathPoint,
    ilyushin_sine_upper_bound,
    limit_load_from_path,
    max_section_moment_ratio,
    patch_plate_mesh,
    plastic_bound_bracket,
    plastic_moment,
    section_moment_capacity,
    yield_line_patch_upper_bound,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State,
    solve_layered_shell4_increment,
)

A = B = 1.0
T = 0.02
E, NU, FY = 210e9, 0.3, 383.8e6
PATCH = (0.5 * A, 0.5 * B)


# ---------------------------------------------------------------------------
# Section capacity of a layered section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layers,expected", [
    (1, 0.0), (2, 1.0), (3, 8.0 / 9.0), (4, 1.0), (5, 24.0 / 25.0), (8, 1.0),
])
def test_section_moment_capacity_closed_form(layers, expected):
    """``m_p,n / m_p = 1`` (even n) or ``1 - 1/n^2`` (odd n)."""
    capacity = section_moment_capacity(T, FY, layers)
    assert capacity["ratio"] == pytest.approx(expected, rel=1e-12)
    assert capacity["ratio_closed_form"] == pytest.approx(expected, rel=1e-12)
    assert capacity["m_p"] == pytest.approx(plastic_moment(T, FY))


def test_section_moment_capacity_rejects_zero_layers():
    with pytest.raises(ValueError):
        section_moment_capacity(T, FY, 0)


@pytest.mark.parametrize("layers", [3, 4, 5, 8])
def test_max_section_moment_ratio_saturates_at_layer_capacity(layers):
    """A fully plastic section reproduces exactly ``m_p (1 - 1/n^2) / m_p``.

    The layer stresses are set to ``+-sigma_y`` according to the sign of ``z``,
    which is the limit state of a section under pure bending.  The measured
    invariant is therefore the *closed-form* capacity ratio, and this is the
    plateau a limit-load run must not exceed.
    """
    dz = T / layers
    stress = torch.zeros((1, 4, layers, 3), dtype=torch.float64)
    for k in range(layers):
        z = -0.5 * T + (k + 0.5) * dz
        stress[:, :, k, 0] = FY * (1.0 if z > 0 else -1.0)
    ratio = max_section_moment_ratio(stress, T, FY, layers)
    assert ratio == pytest.approx(section_moment_capacity(T, FY, layers)["ratio"],
                                  rel=1e-12)


def test_layer_capacity_never_exceeds_the_exact_plastic_moment():
    """Hard invariant: no discrete section may beat ``m_p = sigma_y t^2 / 4``."""
    for layers in range(1, 13):
        assert section_moment_capacity(T, FY, layers)["ratio"] <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Limit-load extraction from a load-deflection path
# ---------------------------------------------------------------------------

def _synthetic_path(elastic_slope: float, plateau_deflection: float,
                    factors=(0.4, 0.8, 1.2, 1.6, 2.0, 2.4)) -> PatchPath:
    """Elastic branch up to 1.2, then a perfectly flat (plastic) plateau."""
    points = []
    for factor in factors:
        deflection = max(elastic_slope * factor,
                         elastic_slope * 1.2 + (factor - 1.2) * 0.01)
        points.append(PatchPathPoint(factor, deflection, 0.0))
    return PatchPath(model=None, info={}, loading=points)


def test_limit_load_from_path_recovers_the_plateau():
    """A synthetic path with a known plateau has a known limit load."""
    elastic_slope = 1.0e-8
    path = _synthetic_path(elastic_slope, 0.0)
    estimate = limit_load_from_path(path, criteria=(0.1, 0.2, 0.3))
    measurements = estimate["measurements"]
    # the plateau starts at 1.2, so every definition must land near it
    for key in ("intersection_of_tangents", "double_deflection"):
        assert measurements[key] == pytest.approx(1.2, abs=0.05), key
    assert "stiffness_criterion_0.1" in measurements
    assert estimate["spread"] < 0.4
    assert estimate["limit_load"] < estimate["peak_factor"]


def test_limit_load_is_not_the_last_ramp_factor():
    """Regression: the end of the ramp must never be returned as the limit load."""
    path = _synthetic_path(1.0e-8, 0.0)
    estimate = limit_load_from_path(path)
    assert estimate["peak_factor"] == pytest.approx(2.4)
    assert estimate["limit_load"] < 0.6 * estimate["peak_factor"]


def test_limit_load_requires_enough_points():
    path = PatchPath(model=None, info={},
                     loading=[PatchPathPoint(1.0, 1e-3, 0.0),
                              PatchPathPoint(2.0, 1e-2, 0.0)])
    with pytest.raises(ValueError):
        limit_load_from_path(path)


def test_limit_load_ignores_zero_deflection_points():
    path = PatchPath(model=None, info={}, loading=[
        PatchPathPoint(0.0, 0.0, 0.0),
        PatchPathPoint(0.4, 4e-3, 0.0),
        PatchPathPoint(0.8, 8e-3, 0.0),
        PatchPathPoint(1.2, 1.6e-2, 0.0),
        PatchPathPoint(1.6, 5.0e-2, 0.0),
    ])
    estimate = limit_load_from_path(path, criteria=(0.2,))
    assert estimate["elastic_slope"] == pytest.approx(1e-2)
    assert estimate["limit_load"] is not None


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def test_ilyushin_sine_bound_is_the_tighter_upper_bound_for_a_patch():
    """The double-sine mechanism wastes much less work than the yield-line pyramid
    when the load covers only the central quarter, so it gives the tighter
    (and still admissible) upper bound."""
    sine = ilyushin_sine_upper_bound(A, B, T, FY, PATCH)["pressure"]
    line = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    assert sine < line
    # independent check of the sine bound from the closed form of this case:
    # W_int/w0 = 8 m_p, W_ext/w0 = 0.2027 m^2 -> 39.5 m_p/a^2
    assert sine == pytest.approx(8.0 * plastic_moment(T, FY) / 0.202668, rel=1e-3)


def test_bound_bracket_brackets_the_patch_case():
    bracket = plastic_bound_bracket(A, B, T, FY, PATCH)
    assert bracket["lower_strip"] < bracket["tightest_upper"]
    assert bracket["tightest_upper"] == pytest.approx(bracket["upper_ilyushin_sine"])
    assert bracket["bracket_width_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Element-level invariant
# ---------------------------------------------------------------------------

def test_elastic_section_capacity_is_never_exceeded_by_the_fe_section():
    """No Gauss point of an elastoplastic FE state may exceed the plastic moment."""
    model, info = patch_plate_mesh(A, B, T, 4, 4, young=E, poisson=NU,
                                   yield_stress=FY, hardening=1e6, layers=2,
                                   patch=PATCH, exact_thickness=True)
    state = LayeredShell4State.virgin(model)
    step = solve_layered_shell4_increment(
        model, info["reference_load"], 2.0 * 0.8692e6, info["fixed_dofs"], state,
        torch.zeros(model.n_dofs, dtype=torch.float64),
        tolerance=1e-8, max_iterations=25)
    ratio = max_section_moment_ratio(step.stress, T, FY, 2)
    capacity = section_moment_capacity(T, FY, 2)["ratio"]
    assert capacity == pytest.approx(1.0)
    assert ratio <= capacity + 1e-6


def test_patch_plate_mesh_accepts_a_shear_correction_and_rejects_a_bad_patch():
    model, _ = patch_plate_mesh(A, B, T, 8, 8, young=E, poisson=NU,
                                yield_stress=FY, hardening=1e6, layers=4,
                                patch=PATCH, shear_correction=5.0 / 6.0 * 4.0)
    assert model.shear_correction == pytest.approx(5.0 / 6.0 * 4.0)
    with pytest.raises(ValueError):
        patch_plate_mesh(A, B, T, 6, 6, young=E, poisson=NU, yield_stress=FY,
                         hardening=1e6, layers=4, patch=PATCH)