"""Acceptance gates for the three-level analytic verification ladder.

Every test here is a *gate*: it compares a production solver path against an
independent closed-form/tabulated reference and fails when the relative error
exceeds the threshold published in ``tensorfem.verification_ladder.CRITERIA``.
The thresholds are imported, not repeated, so a change of the published
criterion cannot silently drift away from the test.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.modal import (
    simply_supported_beam_modes,
    simply_supported_exact_angular_frequency,
)
from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation
from tensorfem.plate_uniform_load import clamped_uniform_ritz
from tensorfem.verification_ladder import (
    CRITERIA,
    TIMOSHENKO_CLAMPED_COEFFICIENT,
    CheckResult,
    LadderReport,
    euler_bernoulli_static_solve,
    l1_beam_rungs,
    l2_plate_rungs,
    l3_modal_rungs,
    modal_mac,
    navier_ss_deflection,
    plate_rigidity,
    ss_udl_deflection,
    ss_udl_max_deflection,
)

E = 200.0e9
NU = 0.3
H = 0.01
SIDE = 1.0
PRESSURE = 1.0e3


# ---------------------------------------------------------------------------
# L1 -- beam statics
# ---------------------------------------------------------------------------

def test_l1_beam_sections_and_cantilever_meet_criteria():
    checks, details = l1_beam_rungs(elements=8)
    by_id = {check.criterion: check for check in checks}
    assert by_id["L1-SS-UDL"].passed
    assert abs(by_id["L1-SS-UDL"].relative_error) < CRITERIA["L1-SS-UDL"]
    assert by_id["L1-CANTILEVER-TIP"].passed
    assert abs(by_id["L1-CANTILEVER-TIP"].relative_error) < CRITERIA["L1-CANTILEVER-TIP"]
    # every one of the five published sections, not just the worst
    sections = details["L1"]["sections"]
    assert len(sections) == 5
    assert [row["x_over_L"] for row in sections] == [0.10, 0.20, 0.30, 0.40, 0.50]
    assert all(abs(row["relative_error"]) < CRITERIA["L1-SS-UDL"] for row in sections)


def test_l1_beam_interior_error_converges_quartic():
    """The cubic Hermite element is *nodally* exact; only interior points converge.

    The measured quantity is the maximum relative error over 24 interior
    sections that are deliberately *not* nodes of any of the meshes used.  Its
    rate must approach the O(h^4) interpolation rate of the cubic; a rate below
    3 means the consistent load vector or the interpolation is wrong, so the
    rate itself is a gate.
    """
    length, EI, q = 2.0, 1.0e5, 1.0e3
    samples = [length * (0.05 + 0.9 * (k + 0.37) / 24.0) for k in range(24)]
    errors = []
    for elements in (2, 4, 8):
        ss = euler_bernoulli_static_solve(
            length, elements, EI, uniform_load=q, support="pinned-pinned"
        )
        errors.append(max(
            abs(ss.deflection(x) / ss_udl_deflection(x, length, q, EI) - 1.0)
            for x in samples
        ))
    rate_1 = math.log(errors[0] / errors[1], 2.0)
    rate_2 = math.log(errors[1] / errors[2], 2.0)
    assert errors[2] < errors[1] < errors[0]
    assert rate_1 > 3.0 and rate_2 > 3.0


def test_l1_nodal_values_are_machine_exact():
    """Nodal exactness is a property of the cubic element, asserted as such.

    It is *not* evidence of general accuracy: the manifest documents that the
    closed-form fields are cubic/quartic and the element reproduces them at the
    nodes.  This test pins that behaviour so a regression (wrong load vector)
    shows up immediately.
    """
    EI = 1.0e5
    L = 3.0
    q = 2.0e3
    ss = euler_bernoulli_static_solve(L, 5, EI, uniform_load=q, support="pinned-pinned")
    for node in ss.nodes.tolist():  # nodes are exact; the supports have w = 0
        reference = ss_udl_deflection(node, L, q, EI)
        if reference == 0.0:
            assert abs(float(ss.deflection(node))) < 1e-15
            continue
        assert abs(ss.deflection(node) / reference - 1.0) < 1e-12
    # a=0.5 L is *not* a node of a 5-element mesh, so this value is interpolation
    assert abs(ss.deflection(0.5 * L) / ss_udl_max_deflection(L, q, EI) - 1.0) < 1e-3


def test_l1_cantilever_field_matches_closed_form_at_interior_points():
    EI = 2.4e5
    L = 1.5
    P = 500.0
    cantilever = euler_bernoulli_static_solve(
        L, 8, EI, tip_load=P, support="clamped-free"
    )
    for x in (0.1 * L, 0.4 * L, 0.75 * L, L):
        reference = P * x**2 * (3.0 * L - x) / (6.0 * EI)
        assert abs(cantilever.deflection(x) / reference - 1.0) < 1e-10


def test_beam_static_rejects_bad_input():
    with pytest.raises(ValueError):
        euler_bernoulli_static_solve(0.0, 4, 1.0e5)
    with pytest.raises(ValueError):
        euler_bernoulli_static_solve(1.0, 4, -1.0)
    with pytest.raises(ValueError):
        euler_bernoulli_static_solve(1.0, 4, 1.0e5, support="free-free")


# ---------------------------------------------------------------------------
# L2 -- plate statics
# ---------------------------------------------------------------------------

def test_l2_plate_rungs_meet_criteria_at_published_mesh():
    checks, details = l2_plate_rungs(elements=12)
    by_id = {check.criterion: check for check in checks}
    assert by_id["L2-SS-NAVIER"].passed
    assert abs(by_id["L2-SS-NAVIER"].relative_error) < CRITERIA["L2-SS-NAVIER"]
    assert by_id["L2-CLAMPED"].passed
    assert abs(by_id["L2-CLAMPED"].relative_error) < CRITERIA["L2-CLAMPED"]
    # the second Ritz anchor must agree with the tabulated coefficient
    assert details["L2"]["references"]["clamped_vs_ritz_relative_error"] < 0.01


def test_l2_navier_truncation_bias_is_small_but_reported():
    """The 11x11 reference is itself truncated; the bias is quantified, not hidden."""
    D = plate_rigidity(E, NU, H)
    eleven = navier_ss_deflection(0.5, 0.5, SIDE, D, PRESSURE, terms=11)
    converged = navier_ss_deflection(0.5, 0.5, SIDE, D, PRESSURE, terms=401)
    coefficient = converged / (PRESSURE * SIDE**4 / D)
    assert abs(coefficient / 0.00406 - 1.0) < 2e-3  # Timoshenko's tabulated value
    # the 11x11 truncation is an *under*-estimate, and the bias is below 0.5 %
    bias = eleven / converged - 1.0
    assert -5e-3 < bias < 0.0
    # the double-sine sum is *alternating* at the plate centre (sin(m pi/2) flips
    # sign for odd m), so successive truncations oscillate; what must shrink is
    # the distance to the 401-term value
    series = [
        navier_ss_deflection(0.5, 0.5, SIDE, D, PRESSURE, terms=terms)
        for terms in (5, 11, 41, 401)
    ]
    distances = [abs(value - series[-1]) for value in series[:-1]]
    assert all(b < a for a, b in zip(distances, distances[1:]))
    assert series[0] != series[1] != series[2]


def test_l2_plate_mesh_convergence_is_monotone():
    """Coarse-mesh errors must shrink; a wrong element would not behave like this."""
    D = plate_rigidity(E, NU, H)
    reference = navier_ss_deflection(0.5, 0.5, SIDE, D, PRESSURE, terms=401)
    errors = []
    for nx in (8, 12):
        result = solve_plate_on_winkler_foundation(
            nx, SIDE, E, NU, H, 0.0, PRESSURE, boundary="simply_supported"
        )
        w = result.displacement.view(-1, 3)[:, 0]
        centre = int(torch.argmin(
            (result.nodes[:, 0] - 0.5).abs() + (result.nodes[:, 1] - 0.5).abs()
        ))
        errors.append(abs(float(w[centre]) / reference - 1.0))
    assert 0.0 < errors[1] < errors[0]


def test_l2_clamped_reference_agrees_with_timoshenko_table():
    ritz = clamped_uniform_ritz(
        side=SIDE, young=E, poisson=NU, thickness=H, pressure=PRESSURE
    )
    coefficient = ritz.deflection(0.5, 0.5) / (PRESSURE * SIDE**4 / plate_rigidity(E, NU, H))
    assert abs(coefficient / TIMOSHENKO_CLAMPED_COEFFICIENT - 1.0) < 0.01


# ---------------------------------------------------------------------------
# L3 -- modal
# ---------------------------------------------------------------------------

def test_l3_beam_modal_meets_criterion_and_shapes_align():
    checks, details = l3_modal_rungs(beam_elements=16, plate_elements=8, plate_modes=1)
    by_id = {check.criterion: check for check in checks}
    assert by_id["L3-BEAM-MODAL"].passed
    rows = details["L3"]["beam"]["modes"]
    assert len(rows) == 5
    assert all(abs(row["relative_error"]) < CRITERIA["L3-BEAM-MODAL"] for row in rows)
    assert all(row["mac"] > 0.999 for row in rows)


def test_l3_plate_modal_meets_criterion():
    checks, details = l3_modal_rungs(beam_elements=8, plate_elements=16, plate_modes=3)
    by_id = {check.criterion: check for check in checks}
    assert by_id["L3-PLATE-MODAL"].passed
    rows = details["L3"]["plate"]["modes"]
    assert [(row["m"], row["n"]) for row in rows] == [(1, 1), (1, 2), (2, 1)]
    assert all(abs(row["relative_error"]) < CRITERIA["L3-PLATE-MODAL"] for row in rows)
    # the degenerate pair must be degenerate in the FE solution too
    pair = rows[1]["computed_hz"], rows[2]["computed_hz"]
    assert abs(pair[0] / pair[1] - 1.0) < 1e-9
    # and its shape must lie inside the {(1,2),(2,1)} eigenspace
    assert all(row["subspace_capture"] > 0.999 for row in rows[1:])


def test_beam_modal_frequencies_have_expected_ratios():
    EI, rho_a, L = 1.0e5, 12.0, 2.5
    result = simply_supported_beam_modes(L, 16, EI, rho_a, modes=5)
    for mode in range(1, 6):
        reference = simply_supported_exact_angular_frequency(L, EI, rho_a, mode=mode)
        assert abs(float(result.angular_frequencies[mode - 1]) / reference - 1.0) < 0.01


def test_mac_rejects_orthogonal_shapes():
    assert modal_mac([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert modal_mac([1.0, 0.0], [0.0, 1.0]) < 1e-12


# ---------------------------------------------------------------------------
# Report plumbing / fail-closed behaviour
# ---------------------------------------------------------------------------

def test_check_result_is_fail_closed():
    good = CheckResult("X", "d", 1.0, 1.0, 3.0e-3, 5.0e-3)
    bad = CheckResult("Y", "d", 1.0, 1.0, 0.02, 5.0e-3)
    report = LadderReport(checks=(good, bad), details={})
    assert good.passed and not bad.passed
    assert report.all_passed is False
    assert [check.criterion for check in report.failed()] == ["Y"]
    payload = report.as_dict()
    assert payload["all_passed"] is False
    assert payload["failed_criteria"] == ["Y"]
    assert payload["criteria"] == CRITERIA


def test_ladder_report_serialises_all_six_criteria():
    l1_checks, l1_details = l1_beam_rungs(elements=4)
    l2_checks, l2_details = l2_plate_rungs(elements=2)
    l3_checks, l3_details = l3_modal_rungs(
        beam_elements=4, plate_elements=4, plate_modes=1
    )
    details = {}
    details.update(l1_details)
    details.update(l2_details)
    details.update(l3_details)
    report = LadderReport(checks=l1_checks + l2_checks + l3_checks, details=details)
    payload = report.as_dict()
    assert set(payload["criteria"]) == {
        "L1-SS-UDL", "L1-CANTILEVER-TIP", "L2-SS-NAVIER",
        "L2-CLAMPED", "L3-BEAM-MODAL", "L3-PLATE-MODAL",
    }
    assert {check["criterion"] for check in payload["checks"]} == set(payload["criteria"])
    for check in payload["checks"]:
        assert 0.0 <= check["tolerance"] < 0.1
        assert math.isfinite(check["relative_error"])
    # coarse meshes must not be able to fake a pass of the plate rungs
    coarse = {check["criterion"]: check for check in payload["checks"]}
    assert coarse["L2-SS-NAVIER"]["tolerance"] == CRITERIA["L2-SS-NAVIER"]