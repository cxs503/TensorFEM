"""Tests for the production Q4 stress-recovery path and its Cook certification.

Three layers are checked:

* algebraic exactness of the recovery operator (constant/linear stress patches,
  extrapolation weights, single valuedness at element boundaries);
* parity between the production (torch) recovery and the independent audit (numpy)
  recovery that the earlier verification used, so the productionisation cannot
  silently change the numbers the audit certified;
* the fail-closed gates: the recovered field meets the 3 % Cook gate at the frozen
  reference, the stored element-centre field still does not, and a defective ladder
  can never be reported as qualified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfem.additional_benchmark_fields import cook_fields
from tensorfem.continuum import elasticity_matrix
from tensorfem.cook_probe_protocol import nodal_recovered_stress as audit_recovered_stress
from tensorfem.stress_recovery import (
    GAUSS_POINTS,
    extrapolation_matrix,
    interpolate_nodal_stress,
    locate_point,
    q4_gauss_stresses,
    q4_shape,
    recover_nodal_stress,
    recovery_verdict,
    relative_error,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "cook_q4_pointwise_protocol.json"
EVIDENCE = ROOT / "benchmarks" / "cook_stress_recovery_evidence.json"

# Three deliberately distorted quadrilaterals (skewed, non-affine) so the tests
# cannot pass by accidentally assuming a rectangular structured grid.
SKEWED_NODES = torch.tensor(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.2],
        [0.1, 1.0],
        [2.2, 1.1],
        [4.1, 1.4],
        [0.3, 2.2],
        [2.3, 2.4],
        [4.4, 2.7],
    ],
    dtype=torch.float64,
)
SKEWED_ELEMENTS = torch.tensor(
    [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=torch.long
)


def _plane_stress_dmat(elements: int) -> torch.Tensor:
    return elasticity_matrix(
        torch.ones(elements, dtype=torch.float64),
        torch.full((elements,), 1.0 / 3.0, dtype=torch.float64),
        "stress",
    )


def _analytic_von_mises(sxx: float, syy: float, sxy: float) -> float:
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def _cook_reference() -> dict:
    data = json.loads(PROTOCOL.read_text())
    return {
        "probes": [
            (entry["point"], entry["von_mises"]["extrapolated"])
            for entry in data["reference"]["probes"]
        ],
        "scale": data["reference"]["scale_von_mises"],
    }


# --------------------------------------------------------------------------- algebra


def test_extrapolation_matrix_is_linear_extrapolation_from_gauss_points():
    matrix = extrapolation_matrix(torch.zeros(1, dtype=torch.float64))
    # Constant fields must pass through unchanged: every row sums to one.
    assert torch.allclose(matrix.sum(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-14)
    # A field linear in (xi, eta) must be reproduced at the nodes from its Gauss values.
    for node, (nx, ny) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        expected = torch.tensor([1.0 + 0.5 * nx, 1.0 - 0.25 * ny, 0.0], dtype=torch.float64)
        gauss = torch.tensor(
            [[1.0 + 0.5 * gx, 1.0 - 0.25 * gy, 0.0] for gx, gy in GAUSS_POINTS],
            dtype=torch.float64,
        )
        assert torch.allclose(matrix[node] @ gauss, expected, atol=1e-12)


def test_uniform_stress_patch_is_reproduced_exactly_on_skewed_elements():
    """u = (eps_x * x, eps_y * y) is a constant-strain state: recovery must be exact."""
    eps_x, eps_y = 3.0e-3, -1.2e-3
    displacement = torch.stack(
        (eps_x * SKEWED_NODES[:, 0], eps_y * SKEWED_NODES[:, 1]), dim=1
    )
    dmat = _plane_stress_dmat(len(SKEWED_ELEMENTS))
    nodal = recover_nodal_stress(SKEWED_NODES, SKEWED_ELEMENTS, dmat, displacement)
    gauss = q4_gauss_stresses(
        SKEWED_NODES[SKEWED_ELEMENTS],
        dmat,
        displacement[SKEWED_ELEMENTS].reshape(len(SKEWED_ELEMENTS), 8),
    )
    # Gauss-point stresses are constant (exact strain), the nodal field equals it.
    assert torch.allclose(gauss, gauss[:, :1].expand_as(gauss), atol=1e-12)
    expected = gauss[0, 0]
    assert torch.allclose(nodal, expected.unsqueeze(0).expand_as(nodal), atol=1e-12)
    # Independent check of the constitutive result: plane stress with the same strain.
    young, poisson = 1.0, 1.0 / 3.0
    sxx = young / (1 - poisson**2) * (eps_x + poisson * eps_y)
    syy = young / (1 - poisson**2) * (eps_y + poisson * eps_x)
    assert abs(float(expected[0]) - sxx) < 1e-10
    assert abs(float(expected[1]) - syy) < 1e-10
    assert abs(float(expected[2])) < 1e-12


def test_production_recovery_matches_the_audit_implementation():
    """The certified numbers came from the numpy audit path; production must agree."""
    case = cook_fields(8)
    nodes = torch.tensor(np.asarray(case["nodes"]), dtype=torch.float64)
    elements = torch.tensor(np.asarray(case["elements"]), dtype=torch.long)
    displacement = torch.tensor(np.asarray(case["displacement"]), dtype=torch.float64)
    dmat = _plane_stress_dmat(len(elements))
    production = recover_nodal_stress(nodes, elements, dmat, displacement).numpy()
    audit = audit_recovered_stress(
        np.asarray(nodes), np.asarray(elements), np.asarray(displacement)
    )
    assert np.abs(production - audit).max() < 1e-12


# ---------------------------------------------------------------------- interpolation


def test_nodal_field_is_single_valued_on_an_element_edge():
    case = cook_fields(8)
    nodes = torch.tensor(np.asarray(case["nodes"]), dtype=torch.float64)
    elements = torch.tensor(np.asarray(case["elements"]), dtype=torch.long)
    nodal = torch.tensor(np.asarray(case["stress_nodal"]), dtype=torch.float64)
    # Interior node shared by four elements: the discontinuous raw field is ambiguous
    # here, the recovered field must not be.
    probe = nodes[len(nodes) // 2]
    located = locate_point(nodes, elements, probe)
    assert len(located["hits"]) >= 2, "the probe should sit on a shared mesh entity"
    result = interpolate_nodal_stress(nodes, elements, nodal, probe)
    assert result["candidate_count"] >= 2
    assert result["element_spread"] < 1e-9
    assert result["single_valued"] is True


def test_locate_point_round_trip_on_distorted_elements():
    dmat = _plane_stress_dmat(len(SKEWED_ELEMENTS))
    elements = SKEWED_ELEMENTS
    for element in range(len(elements)):
        corners = SKEWED_NODES[elements[element]]
        for xi, eta in ((0.3, -0.6), (-0.75, 0.4), (0.55, 0.65)):
            point = q4_shape(xi, eta, corners) @ corners
            located = locate_point(SKEWED_NODES, elements, point)
            match = next(hit for hit in located["hits"] if hit["element"] == element)
            assert abs(match["local"][0] - xi) < 1e-9
            assert abs(match["local"][1] - eta) < 1e-9
    assert dmat.shape[0] == len(elements)


def test_locate_point_rejects_a_point_outside_the_mesh():
    with pytest.raises(ValueError):
        locate_point(SKEWED_NODES, SKEWED_ELEMENTS, torch.tensor([-1.0, -1.0]))


# --------------------------------------------------------------------------- verdict


def test_recovery_verdict_is_fail_closed():
    assert recovery_verdict([0.08, 0.02, 0.004, 0.001], gate=0.03)["qualified"] is True
    # A positive but too large finest error fails.
    assert recovery_verdict([0.5, 0.2, 0.05], gate=0.03)["qualified"] is False
    # Non-monotone ladders fail even when the finest value is small.
    verdict = recovery_verdict([0.001, 0.05, 0.0001], gate=0.03)
    assert verdict["monotone"] is False and verdict["qualified"] is False
    assert recovery_verdict([0.001, 0.05, 0.0001], gate=0.03, require_monotone=False)[
        "qualified"
    ] is True
    for bad in ([0.01, float("nan")], [0.01, -0.01], []):
        with pytest.raises(ValueError):
            recovery_verdict(bad, gate=0.03)


def test_relative_error_uses_the_documented_scale_floor():
    assert relative_error(0.0, 1.0, 1.0) == pytest.approx(1.0)
    assert relative_error(0.0, 0.0, 1.0) == pytest.approx(0.0)
    # The 5 %-of-scale floor keeps a near-zero reference from exploding: without the
    # floor this case would report 1.0 instead of 2e-8.
    assert relative_error(0.05, 1e-12, 1.0) == pytest.approx(1.0)
    assert relative_error(0.0, 1e-9, 1.0) == pytest.approx(2e-8, rel=1e-6)


# -------------------------------------------------------------- production q4 fields


def test_cook_fields_exposes_the_recovered_nodal_field_without_touching_the_raw_one():
    case = cook_fields(8)
    nodes = np.asarray(case["nodes"])
    assert "stress_nodal" in case and "von_mises_nodal" in case and "recovery" in case
    assert np.asarray(case["stress_nodal"]).shape == (len(nodes), 3)
    assert np.asarray(case["von_mises_nodal"]).shape == (len(nodes),)
    assert case["recovery"]["method"] == "q4_gauss_extrapolation_area_weighted_nodal"
    assert case["recovery"]["stored_element_centre_field_unchanged"] is True
    # The element-centre field keeps its own (unqualified) status.
    assert case["stress_accuracy_qualified"] is False
    assert np.asarray(case["stress"]).shape == (len(case["elements"]), 3)


def test_cook_recovered_field_meets_the_3pct_gate():
    reference = _cook_reference()
    for divisions, bound in ((16, 0.025), (32, 0.006), (64, 0.0015)):
        case = cook_fields(divisions)
        nodes = torch.tensor(np.asarray(case["nodes"]), dtype=torch.float64)
        elements = torch.tensor(np.asarray(case["elements"]), dtype=torch.long)
        nodal = torch.tensor(np.asarray(case["stress_nodal"]), dtype=torch.float64)
        worst = 0.0
        for point, target in reference["probes"]:
            value = interpolate_nodal_stress(nodes, elements, nodal, point)["von_mises"]
            assert value > 0.0
            worst = max(worst, relative_error(value, target, reference["scale"]))
        assert worst < bound, (divisions, worst)
        assert worst < 0.03, (divisions, worst)


def test_cook_element_centre_field_still_fails_the_gate():
    """Regression guard: the recovery must not be used to launder the raw field."""
    reference = _cook_reference()
    case = cook_fields(16)
    centres = np.asarray(case["centers"])
    stress = np.asarray(case["stress"])
    worst = 0.0
    for point, target in reference["probes"]:
        index = int(np.argmin(np.linalg.norm(centres - np.asarray(point), axis=1)))
        sxx, syy, sxy = stress[index]
        worst = max(
            worst,
            relative_error(_analytic_von_mises(sxx, syy, sxy), target, reference["scale"]),
        )
    assert worst > 0.03


# --------------------------------------------------------------------- frozen evidence


def test_frozen_recovery_evidence_is_a_passing_ladder():
    evidence = json.loads(EVIDENCE.read_text())
    ladder = evidence["ladder"]
    errors = [level["max_recovered_relative_error"] for level in ladder]
    assert len(errors) == 4
    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 0.03
    assert evidence["verdict"]["recovered"]["qualified"] is True
    assert evidence["verdict"]["raw_element_centre"]["qualified"] is False
    # Every probe must have been single valued once recovered.
    assert max(level["max_element_spread"] for level in ladder) < 1e-9