"""Tests for the anchor-free plate/shell buckling closure operators.

Every assertion here is about the *operator* layer: two routes to the same
governing equation, two discretisations of the same clamped plate, or a
structural clause that any reduction curve must obey.  No test compares a value
with a handbook/code number, because those anchors are withheld (see
``tensorfem.plate_buckling_closure.anchor_entry`` and
``papers/artifacts/w26/plate-shell-buckling-closure/anchor_register.json``).
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from tensorfem.plate_buckling_closure import (
    ANCHOR_REQUESTS,
    anchor_entry,
    bfs_uniaxial_coefficient,
    check_reduction_curve,
    closed_form_uniaxial_coefficient,
    coefficient_closure,
    coefficient_from_critical_stress,
    critical_stress_from_coefficient,
    elastic_ceiling_violations,
    navier_biaxial_coefficient,
    navier_uniaxial_coefficient,
    register_digest,
    restraint_ladder_uniaxial,
    ritz_uniaxial_coefficient,
    slenderness,
    synthetic_reduction_specimen,
)

YOUNG = 2.1e11
POISSON = 0.3
THICKNESS = 0.012


# ---------------------------------------------------------------------------
# the derived coefficient envelope
# ---------------------------------------------------------------------------
def test_closed_form_coefficient_is_four_at_unit_aspect_ratio():
    coefficient, mode, transverse = closed_form_uniaxial_coefficient(aspect_ratio=1.0)
    assert coefficient == pytest.approx(4.0, rel=1e-15)
    assert (mode, transverse) == (1, 1)


@pytest.mark.parametrize("aspect_ratio", [1.0, 2.0, 3.0, 4.0])
def test_closed_form_coefficient_is_four_at_integer_aspect_ratios(aspect_ratio):
    coefficient, mode, _ = closed_form_uniaxial_coefficient(aspect_ratio=aspect_ratio)
    assert coefficient == pytest.approx(4.0, rel=1e-14)
    assert mode == int(aspect_ratio)


@pytest.mark.parametrize("aspect_ratio", [math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
def test_envelope_maximum_is_four_and_a_half_at_sqrt_two(aspect_ratio):
    coefficient, _, _ = closed_form_uniaxial_coefficient(aspect_ratio=aspect_ratio)
    assert coefficient == pytest.approx(4.5, rel=1e-14)


def test_envelope_is_bounded_by_four_and_four_point_five_above_unit_aspect_ratio():
    for step in range(0, 400):
        aspect_ratio = 1.0 + step * 0.01
        coefficient, _, _ = closed_form_uniaxial_coefficient(aspect_ratio=aspect_ratio)
        assert 4.0 - 1e-12 <= coefficient <= 4.5 + 1e-12


def test_short_plate_envelope_rises_above_four_point_five():
    # a plate shorter than it is wide has no integer mode below u = 1, so as
    # alpha falls the envelope climbs past 4.5 (a/b = 0.5 gives 6.25 = (2 + 0.5)^2)
    values = [closed_form_uniaxial_coefficient(aspect_ratio=a)[0] for a in (0.9, 0.7, 0.5, 0.25)]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))
    assert values[0] > 4.0
    assert values[1] > 4.5
    assert values[2] == pytest.approx(6.25, rel=1e-14)


def test_envelope_is_non_increasing_along_integer_multiples():
    for integer in (2, 3, 5):
        for aspect_ratio in (0.7, 1.0, 1.4, 2.2, 3.0):
            shifted, _, _ = closed_form_uniaxial_coefficient(aspect_ratio=integer * aspect_ratio)
            base, _, _ = closed_form_uniaxial_coefficient(aspect_ratio=aspect_ratio)
            assert shifted <= base + 1e-12 * base


def test_closed_form_rejects_bad_input():
    with pytest.raises(ValueError):
        closed_form_uniaxial_coefficient(aspect_ratio=0.0)
    with pytest.raises(ValueError):
        closed_form_uniaxial_coefficient(aspect_ratio=1.0, max_half_waves=0)
    with pytest.raises(TypeError):
        closed_form_uniaxial_coefficient(aspect_ratio="1")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# two independent routes: closure residual
# ---------------------------------------------------------------------------
def test_analytic_recompute_residual_is_below_one_e_minus_ten():
    cases = [
        {"id": "C1", "length": 1.0, "width": 1.0},
        {"id": "C2", "length": 2.0, "width": 1.0},
        {"id": "C3", "length": math.sqrt(2.0), "width": 1.0},
    ]
    rows = coefficient_closure(
        cases=cases, young=YOUNG, poisson=POISSON, thickness=THICKNESS, tolerance=1e-10
    )
    assert [row["case"] for row in rows] == ["C1", "C2", "C3"]
    for row in rows:
        assert row["pass"] is True
        assert row["relative_residual"] <= 1e-10


def test_navier_route_agrees_with_the_closed_form_at_several_aspect_ratios():
    for aspect_ratio in (0.5, 1.0, 1.25, 2.0, 2.5, 3.5):
        width = 0.8
        length = aspect_ratio * width
        analytic, analytic_mode, _ = closed_form_uniaxial_coefficient(aspect_ratio=aspect_ratio)
        numeric, numeric_mode, _ = navier_uniaxial_coefficient(
            length=length, width=width, young=YOUNG, poisson=POISSON, thickness=THICKNESS
        )
        assert numeric == pytest.approx(analytic, rel=1e-12)
        assert numeric_mode == analytic_mode


def test_biaxial_operator_reduces_to_the_uniaxial_one():
    uniaxial, _, _ = navier_biaxial_coefficient(aspect_ratio=1.3, load_ratio=0.0)
    reference, _, _ = closed_form_uniaxial_coefficient(aspect_ratio=1.3)
    assert uniaxial == pytest.approx(reference, rel=1e-13)


def test_equal_biaxial_compression_halves_the_square_plate_coefficient():
    coefficient, m, n = navier_biaxial_coefficient(aspect_ratio=1.0, load_ratio=1.0)
    assert (m, n) == (1, 1)
    assert coefficient == pytest.approx(2.0, rel=1e-14)


# ---------------------------------------------------------------------------
# coefficient <-> stress operator: dimensions and identities
# ---------------------------------------------------------------------------
def test_coefficient_stress_round_trip_is_exact():
    stress = critical_stress_from_coefficient(
        coefficient=4.5, young=YOUNG, poisson=POISSON, thickness=THICKNESS, width=1.0
    )
    back = coefficient_from_critical_stress(
        critical_stress=stress, young=YOUNG, poisson=POISSON, thickness=THICKNESS, width=1.0
    )
    assert back == pytest.approx(4.5, rel=1e-15)


def test_coefficient_is_invariant_under_uniform_geometric_scaling():
    for scale in (0.25, 1.0, 7.0):
        stress = critical_stress_from_coefficient(
            coefficient=4.0,
            young=YOUNG,
            poisson=POISSON,
            thickness=scale * THICKNESS,
            width=scale * 1.0,
        )
        recovered = coefficient_from_critical_stress(
            critical_stress=stress,
            young=YOUNG,
            poisson=POISSON,
            thickness=scale * THICKNESS,
            width=scale * 1.0,
        )
        assert recovered == pytest.approx(4.0, rel=1e-14)


def test_stress_is_linear_in_young_and_poisson_enters_only_through_rigidity():
    base = critical_stress_from_coefficient(
        coefficient=4.0, young=YOUNG, poisson=POISSON, thickness=THICKNESS, width=1.0
    )
    doubled = critical_stress_from_coefficient(
        coefficient=4.0, young=2.0 * YOUNG, poisson=POISSON, thickness=THICKNESS, width=1.0
    )
    assert doubled == pytest.approx(2.0 * base, rel=1e-15)
    other = critical_stress_from_coefficient(
        coefficient=4.0, young=YOUNG, poisson=0.25, thickness=THICKNESS, width=1.0
    )
    expected_ratio = (1.0 - POISSON**2) / (1.0 - 0.25**2)
    assert other / base == pytest.approx(expected_ratio, rel=1e-15)


def test_slenderness_is_the_square_root_of_the_stress_ratio():
    assert slenderness(yield_stress=4.0e8, critical_stress=1.0e8) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        slenderness(yield_stress=-1.0, critical_stress=1.0e8)


def test_critical_stress_rejects_bad_material():
    with pytest.raises(ValueError):
        critical_stress_from_coefficient(
            coefficient=4.0, young=YOUNG, poisson=0.5, thickness=THICKNESS, width=1.0
        )
    with pytest.raises(ValueError):
        critical_stress_from_coefficient(
            coefficient=0.0, young=YOUNG, poisson=POISSON, thickness=THICKNESS, width=1.0
        )


# ---------------------------------------------------------------------------
# discretisation: BFS Hermite element vs polynomial Rayleigh-Ritz
# ---------------------------------------------------------------------------
def test_bfs_simple_support_reproduces_the_closed_form_coefficient():
    point = bfs_uniaxial_coefficient(aspect_ratio=1.0, n=6, restraint=0.0)
    assert point.coefficient == pytest.approx(4.0, rel=5e-4)
    assert (point.n_x, point.n_y) == (6, 6)


def test_bfs_converges_from_above_on_the_simple_support_case():
    coarse = bfs_uniaxial_coefficient(aspect_ratio=1.0, n=4, restraint=0.0)
    fine = bfs_uniaxial_coefficient(aspect_ratio=1.0, n=6, restraint=0.0)
    assert fine.coefficient <= coarse.coefficient
    assert fine.coefficient - 4.0 < coarse.coefficient - 4.0


def test_clamped_coefficient_is_bracketed_by_two_discretisations():
    fe = bfs_uniaxial_coefficient(aspect_ratio=1.0, n=6, restraint=None)
    ritz = ritz_uniaxial_coefficient(degree=4)
    # the clamped coefficient must be well above the simple-support value and the
    # two routes must agree to better than 1% at this resolution
    assert fe.coefficient > 4.0
    assert ritz > fe.coefficient
    assert abs(ritz - fe.coefficient) / ritz < 1.0e-2


def test_restraint_ladder_is_monotone_from_simple_support_to_clamped():
    ladder = restraint_ladder_uniaxial(restraints=(0.0, 10.0, 1.0e3), aspect_ratio=1.0, n=4)
    values = [point.coefficient for point in ladder]
    assert values[0] == pytest.approx(4.0, rel=5e-3)
    assert values[0] < values[1] < values[2]
    assert values[1] > 4.0


def test_bfs_rejects_bad_restraint_and_mesh():
    with pytest.raises(ValueError):
        bfs_uniaxial_coefficient(n=4, restraint=-1.0)
    with pytest.raises(TypeError):
        bfs_uniaxial_coefficient(n=4, restraint="clamped")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bfs_uniaxial_coefficient(n=0, restraint=0.0)


def test_ritz_coefficient_is_an_upper_bound_that_settles_under_refinement():
    values = [ritz_uniaxial_coefficient(degree=degree) for degree in (2, 3, 4, 5)]
    assert all(later < earlier + 5.0e-2 for earlier, later in zip(values, values[1:]))
    assert abs(values[-1] - values[-2]) / values[-1] < 5.0e-3


# ---------------------------------------------------------------------------
# reduction-curve contract
# ---------------------------------------------------------------------------
def test_synthetic_specimen_passes_the_structural_contract():
    report = check_reduction_curve(synthetic_reduction_specimen)
    assert report["pass"] is True
    assert report["clauses"]["stocky_plateau_length"]["status"] == "SKIPPED_PARAMETER_WITHHELD"
    assert report["clauses"]["stocky_plateau_length"]["pass"] is None


def test_plateau_clause_is_checked_when_a_parameter_is_supplied():
    report = check_reduction_curve(synthetic_reduction_specimen, flat_limit=0.0)
    assert report["clauses"]["stocky_plateau_length"]["status"] == "CHECKED"
    assert report["clauses"]["stocky_plateau_length"]["pass"] is True


@pytest.mark.parametrize(
    "curve, expected_clause",
    [
        (lambda lam: 1.0 + 0.1 * lam, "range"),
        (lambda lam: synthetic_reduction_specimen(lam, decay_exponent=2.0), "elastic_ceiling"),
        (lambda lam: 1.0 - 0.5 * math.sin(2.0 * lam), "monotone"),
        (lambda lam: 1.0 if lam < 1.0 else 0.2, "continuity"),
    ],
)
def test_mutated_curves_are_rejected_by_the_expected_clause(curve, expected_clause):
    report = check_reduction_curve(curve, tolerance=1e-6)
    assert report["pass"] is False
    assert report["clauses"][expected_clause]["pass"] is False


def test_elastic_ceiling_violations_are_located():
    violations = elastic_ceiling_violations(lambdas=(0.5, 2.0), chi_values=(0.9, 0.9))
    assert len(violations) == 1
    lam, chi, ceiling = violations[0]
    assert lam == 2.0
    assert chi == pytest.approx(0.9)
    assert ceiling == pytest.approx(0.25)
    with pytest.raises(ValueError):
        elastic_ceiling_violations(lambdas=(0.0,), chi_values=(1.0,))


def test_specimen_rejects_an_exponent_that_cannot_meet_the_ceiling():
    with pytest.raises(ValueError):
        synthetic_reduction_specimen(1.0, decay_exponent=0.0)


# ---------------------------------------------------------------------------
# anchor register: withholding is a behaviour, not an omission
# ---------------------------------------------------------------------------
def test_anchor_without_source_is_withheld_and_records_no_number():
    entry = anchor_entry(ANCHOR_REQUESTS[0])
    assert entry["status"] == "WITHHELD"
    assert entry["verbatim_quote"] is None
    assert entry["quote_location"] is None
    assert entry["source_sha256"] is None
    assert entry["numeric_value_recorded"] is False
    assert "reason" in entry and "closure_path" in entry


def test_anchor_with_source_is_verbatim_and_digested():
    payload = b"quoted text from the standard"
    entry = anchor_entry(
        ANCHOR_REQUESTS[1],
        verbatim_quote="k = 4.0 for a simply supported plate",
        quote_location="clause 7.2, page 14, eq. (7.3)",
        source_bytes=payload,
        source_name="standard.pdf",
    )
    assert entry["status"] == "VERBATIM"
    assert entry["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["quote_location"].startswith("clause 7.2")


def test_anchor_is_withheld_when_any_piece_of_evidence_is_missing():
    entry = anchor_entry(
        ANCHOR_REQUESTS[2], verbatim_quote="chi = 1", quote_location="p. 1", source_bytes=None
    )
    assert entry["status"] == "WITHHELD"
    assert entry["verbatim_quote"] is None


def test_anchor_requests_cover_the_three_named_sources():
    joined = " ".join(item["anchor"] for item in ANCHOR_REQUESTS)
    assert "Timoshenko" in joined
    assert "DNV-RP-C201" in joined
    assert "EN 1993-1-6" in joined


def test_register_digest_is_key_order_independent():
    first = register_digest({"a": 1, "b": [1.0, 2.0]})
    second = register_digest({"b": [1.0, 2.0], "a": 1})
    assert first == second
    assert len(first) == 64
    # the digest must be sensitive to the payload, otherwise it certifies nothing
    assert register_digest({"a": 2, "b": [1.0, 2.0]}) != first


# ---------------------------------------------------------------------------
# published artifact
# ---------------------------------------------------------------------------
def test_published_register_artifact_matches_its_own_digest():
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "papers"
        / "artifacts"
        / "w26"
        / "plate-shell-buckling-closure"
        / "anchor_register.json"
    )
    if not path.exists():  # pragma: no cover - artifact not built in this checkout
        pytest.skip("anchor register artifact not built")
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("content_sha256")
    assert register_digest(payload) == digest
    assert payload["analytic_recompute"]["pass"] is True
    assert payload["analytic_recompute"]["max_relative_residual"] <= 1e-10
    assert payload["anchor_summary"]["verbatim"] == 0
    assert payload["anchor_summary"]["criterion_1"] == "HONEST_WITHHELD"
    for entry in payload["anchors"]:
        assert entry["status"] == "WITHHELD"
        assert entry["numeric_value_recorded"] is False
    assert payload["reduction_curve_operator"]["checker_has_teeth"] is True
    assert payload["boundary_conditions"]["restraint_ladder_monotone"] is True
