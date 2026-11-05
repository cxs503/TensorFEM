"""Tests for the production Gauss-point (Barlow) stress recovery.

Anchors
-------
* **Element consistency**: the Torch strain matrix must reproduce the NumPy audit formula
  bit-for-bit on a *tapered* element (where applying ``inv(J)`` without the transpose is
  wrong — this is the defect the oracle itself carried until 2026-11-03).
* **Patch exactness**: a linear displacement field produces a constant stress, which the
  recovery must return exactly (machine precision), otherwise the recovery injects energy.
* **Single-valuedness**: on an element interface the recovered field must not depend on the
  containing element, which is exactly what the raw element-centre field cannot offer.
* **Accuracy**: the Cook membrane recovered field must be below the 3 % gate on the finest
  committed level and must decrease monotonically over the ladder.
* **Fail-closed**: the raw element-centre protocol flag must stay ``False``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfem.additional_benchmark_fields import cook_fields
from tensorfem.cook_probe_protocol import nodal_recovered_stress
from tensorfem.cook_stress_reference import q4_strain_matrix as numpy_q4_strain
from tensorfem.stress_recovery import (
    GAUSS_ORDER,
    interpolate_nodal_field,
    locate_point,
    q4_gauss_stresses,
    q4_strain_matrix,
    recover_nodal_stress,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "benchmarks" / "cook_membrane_stress_reference.json").read_text())[
    "reference"
]

TAPERED_ELEMENT = torch.tensor(
    ((0.0, 0.0), (2.0, 0.0), (1.4, 1.2), (0.3, 1.0)), dtype=torch.float64
)


def test_strain_matrix_matches_numpy_audit_formula():
    for xi, eta in GAUSS_ORDER + ((0.0, 0.0), (0.37, -0.62)):
        torch_matrix = q4_strain_matrix(xi, eta, TAPERED_ELEMENT)
        numpy_matrix, measure = numpy_q4_strain(xi, eta, TAPERED_ELEMENT.numpy())
        assert torch_matrix.shape == (3, 8)
        assert float(np.max(np.abs(torch_matrix.numpy() - numpy_matrix))) < 1e-14
        assert measure > 0.0


def patch_case():
    """Linear displacement field on a distorted two-element patch: constant known stress.

    The two Q4 elements share the edge 1-4, so the patch also gives an interface point where
    the raw element-centre field is double valued and the recovered field must be single.
    """
    nodes = torch.tensor(
        ((0.0, 0.0), (1.0, 0.05), (2.0, 0.0), (2.1, 1.0), (1.0, 0.95), (-0.1, 1.0)),
        dtype=torch.float64,
    )
    elements = torch.tensor(((0, 1, 4, 5), (1, 2, 3, 4)), dtype=torch.int64)
    young, poisson = 1.0, 0.3
    gradient = torch.tensor(((1e-3, 4e-4), (-2e-4, 5e-4)), dtype=torch.float64)
    displacement = nodes @ gradient  # linear -> constant strain
    matrix = torch.tensor(
        (
            (young / (1 - poisson**2), young * poisson / (1 - poisson**2), 0.0),
            (young * poisson / (1 - poisson**2), young / (1 - poisson**2), 0.0),
            (0.0, 0.0, young / (2 * (1 + poisson))),
        ),
        dtype=torch.float64,
    )
    strain = torch.tensor(
        (gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]), dtype=torch.float64
    )
    expected = matrix @ strain
    return nodes, elements, displacement, young, poisson, expected


def test_patch_recovery_is_exact():
    nodes, elements, displacement, young, poisson, expected = patch_case()
    gauss = q4_gauss_stresses(nodes, elements, displacement, young, poisson)
    assert gauss.shape == (2, 4, 3)
    # Every Gauss point already carries the exact constant stress.
    assert float(torch.max(torch.abs(gauss - expected))) < 1e-15
    nodal = recover_nodal_stress(nodes, elements, gauss)
    assert float(torch.max(torch.abs(nodal - expected))) < 1e-15
    for node in range(len(nodes)):
        sample = interpolate_nodal_field(nodes, elements, nodal, nodes[node].tolist())
        assert max(abs(a - b) for a, b in zip(sample["stress"], expected.tolist())) < 1e-15


def test_recovered_field_is_single_valued_on_element_interfaces():
    nodes, elements, displacement, young, poisson, expected = patch_case()
    gauss = q4_gauss_stresses(nodes, elements, displacement, young, poisson)
    nodal = recover_nodal_stress(nodes, elements, gauss)
    # Midpoint of the shared edge 1-4: contained by both elements, on the interface.
    midpoint = (nodes[1] + nodes[4]).div(2.0)
    hits = locate_point(nodes, elements, midpoint.tolist())
    assert len(hits) == 2, hits
    assert all(hit["on_boundary"] for hit in hits)
    sample = interpolate_nodal_field(nodes, elements, nodal, midpoint.tolist())
    assert sample["on_element_boundary"] is True
    assert sample["containing_elements"] == 2
    assert sample["spread"] < 1e-14


def cook_reference_points():
    return [
        {"point": point["point"], "von_mises_reference": point["von_mises"]["extrapolated"]}
        for point in REFERENCE["points"]
    ]


def worst_error(case):
    return case["stress_recovery"]["worst_site_error"]


def test_cook_recovered_ladder_is_accurate_and_monotone():
    coarse = cook_fields(8, reference=REFERENCE)
    fine = cook_fields(16, reference=REFERENCE)
    # Two independent implementations must agree (Torch production vs NumPy audit recovery).
    audit = nodal_recovered_stress(
        np.asarray(coarse["nodes"]),
        np.asarray(coarse["elements"]),
        np.asarray(coarse["displacement"]),
    )
    production = np.asarray(coarse["nodal_stress"])
    scale = float(np.abs(audit).max())
    assert float(np.abs(production - audit).max()) / scale < 1e-10
    assert worst_error(fine) < worst_error(coarse)
    assert worst_error(fine) < 0.03, worst_error(fine)
    assert fine["stress_recovery"]["passed"] is True
    assert fine["stress_recovery"]["worst_boundary_spread"] < 1e-12
    # The recovered order on this rung pair is between first and second order, comfortably
    # above the O(h) of the raw element-centre field.
    order = math.log(worst_error(coarse) / worst_error(fine)) / math.log(2.0)
    assert order > 1.2, order


def test_raw_protocol_stays_fail_closed():
    case = cook_fields(8)
    assert case["stress_accuracy_qualified"] is False
    # Without an independent reference the recovery block reports the field but makes no claim.
    assert case["stress_recovery"]["passed"] is None
    with_reference = cook_fields(8, reference=REFERENCE)
    assert with_reference["stress_accuracy_qualified"] is False
    assert len(with_reference["stress_recovery"]["sites"]) == len(REFERENCE["points"])
    assert len(with_reference["nodal_stress"]) == len(with_reference["nodes"])


def test_probe_sites_include_interface_coincident_points():
    """The five reference probes must stay on the grid nodes: that is why recovery is needed."""
    case = cook_fields(8, reference=REFERENCE)
    sites = case["stress_recovery"]["sites"]
    assert sum(site["on_element_boundary"] for site in sites) >= 3