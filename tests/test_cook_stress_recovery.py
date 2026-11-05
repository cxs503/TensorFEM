"""Tests for the fail-closed Cook-membrane recovered-stress qualification gate.

Two halves:

* **positive/implementation** -- the recovery operator is exact for a uniform
  stress state, the recovered nodal field is single valued across element
  interfaces, the documented entry point reproduces the production payload, and
  the gate accepts the real refinement ladder;
* **negative controls** -- the gate is fed manufactured payloads (non-monotone,
  over-tolerance, non-finite, unit observed order, level mismatch) and
  non-physical tolerances, and must refuse every one of them.  A gate that only
  ever returns ``True`` is not evidence, so these controls are the point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.additional_benchmark_fields import cook_fields
from tensorfem.cook_stress_recovery import (
    MAX_TOLERANCE,
    load_reference,
    measure,
    qualify,
    recovered_nodal_stress,
)
from tensorfem.stress_recovery import interpolate_nodal_field, plane_stress_matrix

YOUNG = 1.0
POISSON = 1.0 / 3.0
# The Cook panel is 48 long; three rungs are enough for a real ladder check and keep
# the module test runtime near ten seconds.
LADDER = (8, 16, 32)


def _uniform_patch(divisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Structured unit-square Q4 patch, counter-clockwise per element."""
    axis = np.linspace(0.0, 1.0, divisions + 1)
    nodes = np.array([[x, y] for y in axis for x in axis], dtype=np.float64)
    elements = []
    for row in range(divisions):
        for column in range(divisions):
            base = row * (divisions + 1) + column
            elements.append([base, base + 1, base + divisions + 2, base + divisions + 1])
    return nodes, np.array(elements, dtype=np.int64)


def _uniform_displacement(nodes: np.ndarray, strain: tuple[float, float, float]) -> np.ndarray:
    """Displacement of a uniform infinitesimal strain state ``(exx, eyy, gxy)``."""
    exx, eyy, gxy = strain
    return np.stack(
        (exx * nodes[:, 0] + 0.5 * gxy * nodes[:, 1], eyy * nodes[:, 1] + 0.5 * gxy * nodes[:, 0]),
        axis=1,
    )


def test_recovery_is_exact_for_a_uniform_stress_state():
    """Barlow extrapolation + area-weighted averaging reproduces the exact constant field."""
    nodes, elements = _uniform_patch(3)
    strain = (2.0e-3, -1.0e-3, 4.0e-4)
    displacement = _uniform_displacement(nodes, strain)
    recovered = recovered_nodal_stress(nodes, elements, displacement)
    exact = (
        plane_stress_matrix(YOUNG, POISSON) @ torch.tensor(strain, dtype=torch.float64)
    ).numpy()
    field = recovered["field"].numpy()
    assert field.shape == (len(nodes), 3)
    assert np.abs(field - exact[None, :]).max() < 1e-13
    von_mises_exact = math.sqrt(
        exact[0] ** 2 - exact[0] * exact[1] + exact[1] ** 2 + 3.0 * exact[2] ** 2
    )
    assert np.abs(recovered["von_mises"].numpy() - von_mises_exact).max() < 1e-13
    # Interpolation of a constant field returns that constant, whatever the host element.
    sample = interpolate_nodal_field(
        torch.as_tensor(nodes), torch.as_tensor(elements), recovered["field"], [0.5, 0.5]
    )
    assert sample["spread"] < 1e-13
    assert sample["von_mises"] == pytest.approx(von_mises_exact, rel=1e-12)


def test_recovered_field_is_single_valued_on_interfaces():
    """Every containing element yields the same value at a shared point of the production field."""
    case = cook_fields(8)
    nodes = torch.as_tensor(np.asarray(case["nodes"]), dtype=torch.float64)
    elements = torch.as_tensor(np.asarray(case["elements"]), dtype=torch.long)
    field = torch.as_tensor(np.asarray(case["nodal_stress"]), dtype=torch.float64)
    for point in ([24.0, 37.0], [36.0, 33.0], [12.0, 24.0]):
        sample = interpolate_nodal_field(nodes, elements, field, point)
        assert sample["containing_elements"] >= 1
        assert sample["spread"] < 1e-13


def test_entry_point_reproduces_the_production_payload():
    """The documented wrapper must not introduce a second recovery implementation."""
    case = cook_fields(8)
    wrapped = recovered_nodal_stress(case["nodes"], case["elements"], case["displacement"])
    production = np.asarray(case["nodal_stress"], dtype=np.float64)
    assert np.abs(wrapped["field"].numpy() - production).max() == 0.0
    assert wrapped["protocol"].startswith("2x2 Gauss (Barlow)")


def test_production_ladder_qualifies_but_never_promotes_the_raw_field():
    """End-to-end: the real ladder passes the gate; the element-centre flag stays False."""
    verdict = qualify(LADDER)
    assert verdict["blocking_reasons"] == []
    assert verdict["stress_recovery_qualified"] is True
    assert verdict["stress_accuracy_qualified"] is False
    assert verdict["element_centre_field"]["qualified"] is False
    errors = verdict["error_sequence"]
    assert len(errors) == len(LADDER)
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < MAX_TOLERANCE
    assert verdict["observed_order"] > 1.0
    assert verdict["monotone_decreasing"] is True
    assert len(verdict["reference"]["sha256"]) == 64
    assert len(verdict["verdict_sha256"]) == 64
    assert verdict["tolerance"] == MAX_TOLERANCE


def _manufactured(errors, levels=LADDER) -> dict:
    return {
        "levels": [
            {"divisions": int(level), "worst_site_error": float(error)}
            for level, error in zip(levels, errors)
        ]
    }


@pytest.mark.parametrize(
    "errors, reason",
    [
        ((0.05, 0.06, 0.02), "non_monotone_error"),
        ((0.09, 0.05, 0.04), "finest_above_tolerance"),
        ((0.09, 0.045, 0.0201), "observed_order_not_above_one"),
        ((0.09, float("nan"), 0.02), "non_finite_error"),
        ((0.09, float("inf"), 0.02), "non_finite_error"),
    ],
)
def test_gate_refuses_manufactured_ladders(errors, reason):
    verdict = qualify(LADDER, measured=_manufactured(errors))
    assert verdict["stress_recovery_qualified"] is False
    assert reason in verdict["blocking_reasons"]
    assert verdict["stress_accuracy_qualified"] is False


def test_gate_requires_at_least_two_rungs():
    with pytest.raises(ValueError, match="at least two mesh levels"):
        qualify((8,), measured=_manufactured((0.01,), levels=(8,)))


def test_gate_rejects_levels_out_of_order():
    with pytest.raises(ValueError, match="strictly increasing"):
        qualify((16, 8, 4), measured=_manufactured((0.01, 0.02, 0.03), levels=(16, 8, 4)))


def test_gate_rejects_measured_level_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        qualify(LADDER, measured=_manufactured((0.01, 0.004, 0.001), levels=(8, 16, 64)))


@pytest.mark.parametrize("tolerance", [0.05, 0.031, 0.0, -0.01, float("nan")])
def test_gate_refuses_to_widen_or_degenerate_the_acceptance_limit(tolerance):
    """The 3 % acceptance limit is a fixed gate, not a knob."""
    with pytest.raises(ValueError, match="tolerance must be finite"):
        qualify(LADDER, tolerance=tolerance, measured=_manufactured((0.01, 0.004, 0.001)))


def test_verdict_digest_tracks_the_judged_numbers():
    first = qualify(LADDER, measured=_manufactured((0.09, 0.02, 0.001)))
    second = qualify(LADDER, measured=_manufactured((0.09, 0.0201, 0.001)))
    assert first["verdict_sha256"] != second["verdict_sha256"]


def test_measure_records_every_site_and_the_raw_flag():
    """The measured ladder reports the production site block, not only the worst value."""
    measured = measure((8,))
    reference = load_reference()
    assert measured["reference"]["sha256"] == reference["sha256"]
    row = measured["levels"][0]
    assert row["divisions"] == 8
    assert len(row["sites"]) == len(reference["points"])
    assert row["worst_site_error"] == max(site["error"] for site in row["sites"])
    assert row["raw_field_qualified"] is False
    assert row["worst_boundary_spread"] < 1e-9
