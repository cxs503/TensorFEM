"""Tests for the level ice sheet on a Pasternak bed and its failure modes.

The acceptance anchors of the campaign node are exercised here in cheap form
(the full-resolution sweeps live in ``benchmarks/pasternak_ice_sheet_bending.py``):

* (a) the strip FE peak deflection must reproduce the analytic penetration;
* (b) the FE-extracted shear layer length must scale as ``G_s^{-1/2}``;
* (c) the flexural capacity of a long sheet must be independent of the sheet
  length (plateau), the foundering branch must scale linearly with length, and
  the local-crushing mechanism must only govern for very small contacts.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.ice_sheet_pasternak import (
    IceSheetProperties,
    branch_fit,
    buoyancy_stiffness,
    failure_mode,
    flexural_failure_load,
    foundering_load,
    ice_plate_rigidity,
    ice_sheet_characteristic_lengths,
    local_crushing_load,
    plate_peak_deflection,
    strip_patch_fem,
    transition_scan,
)
from tensorfem.pasternak_beam import (
    decay_modes_from_nodes,
    pasternak_characteristic_lengths,
    pasternak_infinite_beam_point_load,
    pasternak_penetration,
    power_law_exponent,
)


@pytest.fixture(scope="module")
def sheet() -> IceSheetProperties:
    return IceSheetProperties()


# ---------------------------------------------------------------------------
# constitutive / geometric quantities
# ---------------------------------------------------------------------------
def test_rigidity_buoyancy_and_freeboard(sheet: IceSheetProperties):
    expected_d = sheet.young * sheet.thickness**3 / (12.0 * (1.0 - sheet.poisson**2))
    assert sheet.rigidity == pytest.approx(expected_d, rel=1e-14)
    assert sheet.buoyancy == pytest.approx(1025.0 * 9.81, rel=1e-14)
    assert sheet.buoyancy == pytest.approx(buoyancy_stiffness(1025.0, 9.81), rel=1e-14)
    assert sheet.freeboard == pytest.approx(
        (1.0 - 917.0 / 1025.0) * sheet.thickness, rel=1e-14)
    assert sheet.moment_capacity == pytest.approx(
        sheet.flexural_strength * sheet.thickness**2 / 6.0, rel=1e-14)


def test_rigidity_scales_with_thickness_cubed():
    thin = IceSheetProperties(thickness=0.5)
    thick = IceSheetProperties(thickness=1.0)
    assert thick.rigidity / thin.rigidity == pytest.approx(8.0, rel=1e-12)
    assert thick.moment_capacity / thin.moment_capacity == pytest.approx(4.0, rel=1e-12)


def test_classical_characteristic_length_of_sea_ice(sheet: IceSheetProperties):
    """1 m of sea ice on water: ``(D/rho g)^{1/4}`` is about 21 m."""
    lengths = ice_sheet_characteristic_lengths(sheet)
    # classical ice-sheet length (D/k)^(1/4) and the Hetenyi length (4D/k)^(1/4)
    assert lengths["l_ice"] == pytest.approx(
        (sheet.rigidity / sheet.buoyancy) ** 0.25, rel=1e-12)
    assert 14.0 < lengths["l_ice"] < 15.5
    assert lengths["l_winkler"] == pytest.approx(math.sqrt(2.0) * lengths["l_ice"],
                                                 rel=1e-12)
    assert lengths["regime"] == "oscillatory"  # Winkler limit is Hetenyi-like


def test_shear_and_soft_lengths_bracket_the_winkler_length(sheet: IceSheetProperties):
    threshold = 2.0 * math.sqrt(sheet.rigidity * sheet.buoyancy)
    lengths = ice_sheet_characteristic_lengths(sheet, 100.0 * threshold)
    assert lengths["regime"] == "monotone"
    assert lengths["l_shear"] < lengths["l_ice"] < lengths["l_winkler"]
    assert lengths["l_winkler"] < lengths["l_soft"]
    assert lengths["l_dominant"] == pytest.approx(lengths["l_soft"], rel=1e-12)
    assert lengths["l_shear"] == pytest.approx(
        math.sqrt(sheet.rigidity / (100.0 * threshold)), rel=1e-2)


# ---------------------------------------------------------------------------
# (a) strip peak deflection against the point-load closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pasternak_fraction", [0.0, 2.0, 20.0])
def test_strip_peak_matches_point_load_limit(sheet: IceSheetProperties,
                                             pasternak_fraction: float):
    """A small patch load must reproduce the point-load analytic penetration."""
    threshold = 2.0 * math.sqrt(sheet.rigidity * sheet.buoyancy)
    pasternak = pasternak_fraction * threshold
    lengths = ice_sheet_characteristic_lengths(sheet, pasternak)
    half = 0.05
    solid = strip_patch_fem(sheet, length=8.0 * lengths["l_dominant"],
                            pasternak=pasternak, contact_half_length=half,
                            load=1.0e4)
    peak = float(solid["w"].abs().max())
    analytic = pasternak_penetration(1.0e4, sheet.rigidity, sheet.buoyancy,
                                     pasternak)
    # a 10 cm patch still spreads the load, so allow up to 3 %
    assert peak == pytest.approx(analytic, rel=3e-2)
    assert solid["reaction"]["total"] == pytest.approx(1.0e4, rel=1e-4)


def test_strip_patch_deflection_decreases_with_bed_shear(sheet: IceSheetProperties):
    peak = [strip_patch_fem(sheet, length=300.0, pasternak=gs,
                            contact_half_length=1.0, load=1.0e4)["w"].abs().max()
            for gs in (0.0, 1.0e7, 1.0e8, 1.0e9)]
    assert all(float(a) > float(b) for a, b in zip(peak[:-1], peak[1:]))


# ---------------------------------------------------------------------------
# (b) characteristic-length scaling from the FE field
# ---------------------------------------------------------------------------
def test_fe_shear_length_scales_as_inverse_sqrt(sheet: IceSheetProperties):
    threshold = 2.0 * math.sqrt(sheet.rigidity * sheet.buoyancy)
    shears = [10.0 * threshold, 20.0 * threshold, 40.0 * threshold]
    extracted = []
    for pasternak in shears:
        lengths = pasternak_characteristic_lengths(sheet.rigidity,
                                                   sheet.buoyancy, pasternak)
        h_target = lengths["l_shear"] / 10.0
        n = int(6.0 * (lengths["l_soft"] + lengths["l_shear"]) / h_target)
        model_length = h_target * n
        solid = strip_patch_fem(sheet, length=model_length, n_elements=n,
                                pasternak=pasternak, contact_half_length=h_target,
                                load=1.0e4)
        window = int(4.0 * lengths["l_shear"] / solid["mesh"]) + 2
        half = solid["w"].numel() // 2
        fast, _ = decay_modes_from_nodes(solid["w"].abs()[half:][:window],
                                         solid["mesh"], start=1)
        extracted.append(1.0 / fast.real)
    fit = power_law_exponent(shears, extracted)
    assert abs(fit["exponent"] + 0.5) < 0.02
    assert fit["r2"] > 0.999


# ---------------------------------------------------------------------------
# (c) failure modes and their transitions
# ---------------------------------------------------------------------------
def test_flexural_capacity_plateaus_for_long_sheets(sheet: IceSheetProperties):
    """Classical result: a long sheet's flexural capacity is length independent."""
    long_a = flexural_failure_load(sheet, sheet_length=400.0,
                                   contact_half_length=0.5)
    long_b = flexural_failure_load(sheet, sheet_length=800.0,
                                   contact_half_length=0.5)
    assert long_a["load"] == pytest.approx(long_b["load"], rel=2e-2)
    # and it matches the infinite-beam Hetenyi limit 4 M_cap / l_c
    lc = ice_sheet_characteristic_lengths(sheet)["l_winkler"]  # Hetenyi P/(2kw0)
    classical = 4.0 * sheet.moment_capacity / lc
    assert long_b["load"] / classical == pytest.approx(1.0, abs=5e-2)


def test_flexural_capacity_increases_with_bed_shear(sheet: IceSheetProperties):
    threshold = 2.0 * math.sqrt(sheet.rigidity * sheet.buoyancy)
    capacities = [flexural_failure_load(sheet, sheet_length=300.0,
                                        pasternak=f * threshold,
                                        contact_half_length=0.5)["load"]
                  for f in (0.0, 10.0, 100.0)]
    assert capacities[0] < capacities[1] < capacities[2]


def test_foundering_load_is_linear_in_length(sheet: IceSheetProperties):
    assert foundering_load(sheet, sheet_length=100.0) == pytest.approx(
        2.0 * foundering_load(sheet, sheet_length=50.0), rel=1e-14)
    assert foundering_load(sheet, sheet_length=100.0) == pytest.approx(
        sheet.water_density * sheet.gravity * sheet.freeboard * 100.0, rel=1e-14)


def test_local_crushing_load_scales_with_contact_length(sheet: IceSheetProperties):
    assert local_crushing_load(sheet, contact_half_length=0.2) == pytest.approx(
        5.0 * local_crushing_load(sheet, contact_half_length=0.04), rel=1e-14)


def test_failure_mode_ranks_the_smallest_mechanism(sheet: IceSheetProperties):
    record = failure_mode(sheet, sheet_length=400.0, contact_half_length=0.5)
    assert record["mode"] == min(record["loads"], key=record["loads"].get)
    assert record["critical_load"] == min(record["loads"].values())
    assert record["l_ratio"] == pytest.approx(record["l_c"] / 400.0, rel=1e-14)


def test_short_sheets_founder_and_long_sheets_fail_in_bending(sheet: IceSheetProperties):
    short = failure_mode(sheet, sheet_length=5.0, contact_half_length=0.5)
    long = failure_mode(sheet, sheet_length=500.0, contact_half_length=0.5)
    assert short["mode"] == "foundering"
    assert long["mode"] == "global_flexure"


def test_tiny_contacts_crush_locally(sheet: IceSheetProperties):
    """Local crushing governs only below a contact half-length of a few mm."""
    tiny = failure_mode(sheet, sheet_length=400.0, contact_half_length=0.001)
    typical = failure_mode(sheet, sheet_length=400.0, contact_half_length=0.5)
    assert tiny["mode"] == "local_crushing"
    assert typical["mode"] == "global_flexure"
    assert tiny["contact_ratio"] < 1e-3


def test_transition_scan_is_sorted_and_covers_both_branches(sheet: IceSheetProperties):
    threshold = 2.0 * math.sqrt(sheet.rigidity * sheet.buoyancy)
    records = transition_scan(sheet, sheet_lengths=[5.0, 10.0, 20.0, 100.0, 400.0],
                              pasternak_values=[0.0, 20.0 * threshold],
                              contact_half_length=0.5)
    ratios = [rec["l_ratio"] for rec in records]
    assert ratios == sorted(ratios)
    modes = {rec["mode"] for rec in records}
    assert "foundering" in modes and "global_flexure" in modes
    # within one foundation family the foundering branch is linear in L, i.e.
    # P ~ (l_c/L)^{-1}
    family = [rec for rec in records if rec["pasternak"] == 0.0]
    fit = branch_fit(family, "foundering")
    assert fit["exponent"] == pytest.approx(-1.0, abs=1e-9)
    with pytest.raises(ValueError):
        branch_fit(family, "local_crushing")


def test_plate_cross_check_uses_the_existing_plate_solution(sheet: IceSheetProperties):
    """2-D plate Pasternak response must be monotone in the bed shear."""
    values = [plate_peak_deflection(sheet, load=1.0e5, pasternak=gs)
              for gs in (0.0, 1.0e7, 1.0e8)]
    assert all(a > b for a, b in zip(values[:-1], values[1:]))
    assert values[0] > 0.0


def test_strip_field_is_consistent_with_the_beam_closed_form(sheet: IceSheetProperties):
    """The strip deflection away from the patch follows the analytic decay."""
    solid = strip_patch_fem(sheet, length=300.0, pasternak=0.0,
                            contact_half_length=1.0, load=1.0e5)
    nodes = solid["nodes"]
    w = solid["w"]
    peak = int(w.abs().argmax())
    analytic = pasternak_infinite_beam_point_load(
        nodes - nodes[peak], 0.0, 1.0e5, sheet.rigidity, sheet.buoyancy, 0.0)
    scale = float(w[peak]) / float(analytic[peak])
    mask = (nodes > 40.0) & (nodes < 260.0)
    error = (w[mask] - scale * analytic[mask]).abs() / float(w[peak])
    assert float(error.max()) < 0.15  # patch-vs-point-load difference, 1 m patch


# ---------------------------------------------------------------------------
# validation paths
# ---------------------------------------------------------------------------
def test_invalid_sheet_properties_raise():
    with pytest.raises(ValueError):
        IceSheetProperties(thickness=0.0)
    with pytest.raises(ValueError):
        IceSheetProperties(young=-1.0)
    with pytest.raises(ValueError):
        IceSheetProperties(poisson=0.6)
    with pytest.raises(ValueError):
        IceSheetProperties(ice_density=1200.0)  # denser than water
    with pytest.raises(ValueError):
        IceSheetProperties(gravity=0.0)
    with pytest.raises(ValueError):
        IceSheetProperties(flexural_strength=0.0)
    with pytest.raises(ValueError):
        IceSheetProperties(crushing_strength=-1.0)
    with pytest.raises(ValueError):
        ice_plate_rigidity(0.0, 0.3, 1.0)
    with pytest.raises(ValueError):
        ice_plate_rigidity(1.0e9, 0.9, 1.0)
    with pytest.raises(ValueError):
        buoyancy_stiffness(0.0, 9.81)
    with pytest.raises(ValueError):
        buoyancy_stiffness(1025.0, -1.0)


def test_invalid_loads_and_contacts_raise(sheet: IceSheetProperties):
    with pytest.raises(ValueError):
        strip_patch_fem(sheet, length=100.0, contact_half_length=0.0)
    with pytest.raises(ValueError):
        strip_patch_fem(sheet, length=100.0, contact_half_length=60.0)
    with pytest.raises(ValueError):
        local_crushing_load(sheet, contact_half_length=0.0)
    with pytest.raises(ValueError):
        foundering_load(sheet, sheet_length=0.0)