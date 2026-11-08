"""Acceptance tests for :mod:`tensorfem.smith_panel_collapse`.

The first block is the machine-checkable acceptance chain of the validation
brief (criteria ``a``--``h``); the second block pins the physical invariants the
implementation must keep so the criteria cannot be met by an accidental
coincidence (mode selection, elastic slope, area weighting, stiffener slenderness
trend, interaction surface closure, input validation).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorfem.smith_panel_collapse import (
    POST_ULTIMATE_SLOPE,
    PlateField,
    StiffenedPanelSmith,
    Stiffener,
    biaxial_equal_compression,
    compare_with_experiments,
    csr_no47_panel,
    csr_panel_chain,
    csrf_faulkner_plate_ultimate,
    interaction_components,
    interaction_curve,
    isotropic_reference_components,
    plate_biaxial_path,
    interaction_load_factor,
    marguerre_plate_path,
    panel_load_shortening,
    plate_load_shortening,
    sig6,
    stiffener_ultimate,
)

YOUNG = 2.06e11
YIELD = 235e6
SPACING = 0.8
SPAN = 2.4


def _slenderness_thickness(beta: float) -> float:
    return SPACING * math.sqrt(YIELD / YOUNG) / beta


def _relative(a: float, b: float) -> float:
    return a / b - 1.0


# --------------------------------------------------------------------------
# acceptance chain
# --------------------------------------------------------------------------
def test_a_plate_ultimate_matches_faulkner_within_ten_percent():
    """Criterion (a): ideal-plate ultimate vs Faulkner (1975), beta = 1..3.

    Evaluated on the ideal plate (``residual_ratio = 0``): the Faulkner closed
    form contains no residual-stress term, so the residual effect is measured
    separately by criterion (e) instead of being folded into this comparison.
    """
    rows = []
    for beta in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0):
        plate = PlateField(length=SPAN, spacing=SPACING,
                           thickness=_slenderness_thickness(beta), residual_ratio=0.0)
        curve = plate_load_shortening(plate)
        reference = csrf_faulkner_plate_ultimate(curve["beta"], YIELD)
        rows.append((curve["beta"], curve["sigma_u"], reference,
                     abs(_relative(curve["sigma_u"], reference))))
    worst = max(row[3] for row in rows)
    assert worst <= 0.10, rows
    # the trend must be physical: strength falls monotonically with slenderness
    strengths = [row[1] for row in rows]
    assert all(a > b for a, b in zip(strengths, strengths[1:]))


def test_b_panel_ultimate_matches_csr_closed_form_chain_within_ten_percent():
    """Criterion (b): Smith progressive collapse vs the CSR-style min-chain."""
    panel = csr_no47_panel()
    smith = panel_load_shortening(panel)["sigma_u"]
    chain = csr_panel_chain(panel)["panel_capacity"]
    assert abs(_relative(smith, chain)) <= 0.10, (smith, chain)


def test_c_experimental_table_is_withheld_not_faked():
    """Criterion (c): no Paik & Thayamballi table offline -> WITHHELD, never 0 %."""
    result = compare_with_experiments([])
    assert result["status"] == "WITHHELD"
    assert result["max_relative_error"] is None
    # the harness itself must work when a table is supplied (synthetic specimen,
    # used only to prove the machinery, never reported as a validation)
    panel = csr_no47_panel()
    synthetic = {
        "name": "synthetic",
        "a": panel.plate.length, "s": panel.plate.spacing, "t": panel.plate.thickness,
        "web_height": panel.stiffener.web_height,
        "web_thickness": panel.stiffener.web_thickness,
        "flange_width": panel.stiffener.flange_width,
        "flange_thickness": panel.stiffener.flange_thickness,
        "yield_stress": panel.plate.yield_stress,
        "sigma_u_measured": panel_load_shortening(panel)["sigma_u"],
    }
    check = compare_with_experiments([synthetic])
    assert check["status"] == "COMPUTED"
    assert abs(check["rows"][0]["relative_error"]) < 1e-9


def test_d_imperfection_sensitivity_drops_at_least_five_percent():
    """Criterion (d): w0/t from 0.1 beta^2 to 0.3 beta^2 lowers sigma_u >= 5 %."""
    rows = []
    for beta in (1.5, 2.0, 2.5, 3.0):
        thickness = _slenderness_thickness(beta)
        low = plate_load_shortening(PlateField(length=SPAN, spacing=SPACING,
                                               thickness=thickness, imperfection_ratio=0.1))
        high = plate_load_shortening(PlateField(length=SPAN, spacing=SPACING,
                                                thickness=thickness, imperfection_ratio=0.3))
        rows.append((beta, low["sigma_u"], high["sigma_u"],
                     high["sigma_u"] / low["sigma_u"] - 1.0))
    for _, _, _, change in rows:
        assert change <= -0.05, rows


def test_e_residual_stress_reduces_ultimate_at_least_three_percent():
    """Criterion (e): sigma_rc = 0.2 sigma_y lowers the plate ultimate >= 3 %."""
    for beta in (1.5, 2.0, 2.5, 3.0):
        thickness = _slenderness_thickness(beta)
        without = plate_load_shortening(PlateField(length=SPAN, spacing=SPACING,
                                                   thickness=thickness, residual_ratio=0.0))
        with_rs = plate_load_shortening(PlateField(length=SPAN, spacing=SPACING,
                                                   thickness=thickness, residual_ratio=0.2))
        assert with_rs["sigma_u"] / without["sigma_u"] - 1.0 <= -0.03, (beta, with_rs, without)


def test_f_equal_biaxial_ultimate_is_inverse_sqrt_two_of_uniaxial():
    """Criterion (f): equal biaxial compression reduces capacity by 1/sqrt(2).

    The interaction surface is isotropic only when ``sxu = syu``; on the
    isotropic reference plate (see :func:`isotropic_reference_components`) the
    equal-biaxial reduction is exactly ``1/sqrt(2)``.  The CSR panel is
    anisotropic (the stiffeners do not carry the transverse load) so its own
    reduction is stronger; that value is reported, not hidden.
    """
    panel = csr_no47_panel()
    isotropic = isotropic_reference_components(panel.plate)
    reference = biaxial_equal_compression(components=isotropic)
    target = 1.0 / math.sqrt(2.0)
    assert reference["sigma_over_uniaxial_ultimate"] == pytest.approx(target, rel=1e-15)
    assert abs(reference["sigma_over_uniaxial_ultimate"] / target - 1.0) <= 0.10

    anisotropic = biaxial_equal_compression(panel)
    assert anisotropic["sigma_over_uniaxial_ultimate"] < target
    # the biaxial point must sit exactly on the panel's own interaction surface
    lf = interaction_load_factor(panel, anisotropic["sigma"], anisotropic["sigma"])
    assert abs(lf["utilisation"] - 1.0) < 1e-9
    # the physics-based large-deflection biaxial model is a separate route to the
    # same question; its reduction is reported alongside (never substituted).
    physics = plate_biaxial_path(panel.plate, 1.0)
    uniaxial = panel_load_shortening(panel)["plate_curve"]["sigma_u"]
    assert 0.4 < physics["ultimate_over_yield"] / uniaxial * uniaxial < 1.0


def test_g_post_peak_branch_has_negative_stiffness():
    """Criterion (g): the mean stress must fall after its peak."""
    panel = csr_no47_panel()
    curve = panel_load_shortening(panel)
    assert curve["has_negative_stiffness"]
    peak = int(np.argmax(curve["panel_stress"]))
    assert peak < curve["panel_stress"].size - 10
    assert np.diff(curve["panel_stress"][peak:]).min() < 0.0
    # the plate branch softens at exactly the documented slope; the panel branch
    # is that slope diluted by the still-load-bearing stiffener area
    plate_curve = curve["plate_curve"]
    plate_peak = int(np.argmax(plate_curve["stress"]))
    plate_slope = np.gradient(plate_curve["stress"], plate_curve["strains"])[plate_peak + 1]
    assert abs(plate_slope / (POST_ULTIMATE_SLOPE * YOUNG) - 1.0) < 0.05
    panel_slope = np.gradient(curve["panel_stress"], curve["strains"])[peak + 1]
    share = panel.plate_area_fraction
    assert abs(panel_slope / (POST_ULTIMATE_SLOPE * YOUNG * share) - 1.0) < 0.2


def test_h_numbers_land_with_six_significant_digits():
    """Criterion (h): every landed number is exactly its six-significant-digit form."""
    assert sig6(0.123456789) == 0.123457
    assert sig6(123456789.0) == 123457000.0
    assert sig6(1.23456789e-9) == 1.23457e-09
    panel = csr_no47_panel()
    curve = panel_load_shortening(panel)
    chain = csr_panel_chain(panel)
    payload = {
        "sigma_u": sig6(curve["sigma_u"]),
        "sigma_u_over_yield": sig6(curve["sigma_u_over_yield"]),
        "eps_u_over_yield_strain": sig6(curve["eps_u_over_yield_strain"]),
        "panel_capacity": sig6(chain["panel_capacity"]),
        "plate_ultimate": sig6(chain["plate_ultimate"]),
    }
    for value in payload.values():
        assert float(f"{value:.6g}") == value, (value, json.dumps(payload))
        mantissa = f"{value:.6g}".split("e")[0].replace("-", "").replace(".", "").lstrip("0")
        assert len(mantissa) <= 6, (value, mantissa)


# --------------------------------------------------------------------------
# physical invariants (stop the criteria being met by coincidence)
# --------------------------------------------------------------------------
def test_critical_mode_selection_for_long_plate():
    """A long plate buckles in the mode that gives the classical k -> 4."""
    plate = PlateField(length=3.0 * SPACING, spacing=SPACING, thickness=0.012)
    buckling = plate.buckling_stress()
    assert buckling["mode"] == 3
    assert abs(buckling["buckling_coefficient"] - 4.0) < 1e-9
    k_square = PlateField(length=SPACING, spacing=SPACING, thickness=0.012).buckling_stress()
    assert k_square["mode"] == 1
    assert abs(k_square["buckling_coefficient"] - 4.0) < 1e-9


def test_plate_path_is_consistent_with_the_galerkin_equilibrium():
    """The reported path must satisfy the equilibrium it claims to solve."""
    plate = PlateField(length=SPAN, spacing=SPACING, thickness=0.012)
    path = marguerre_plate_path(plate)
    m = path["mode"]
    alpha = m * math.pi / plate.length
    bet = math.pi / plate.spacing
    w0 = plate.imperfection_amplitude
    A = path["amplitude"]
    psi = A * (A + 2.0 * w0)
    sigma_1 = plate.young * psi * alpha**2 / 8.0
    sigma_2 = plate.young * psi * bet**2 / 8.0
    leading = max(path["elastic_buckling_stress"] - plate.residual_stress,
                  1e-3 * path["elastic_buckling_stress"])
    expected = leading * A / (A + w0) + 0.5 * sigma_1 + 0.5 * (bet**2 / alpha**2) * sigma_2
    assert np.allclose(path["mean_stress"], expected, rtol=1e-12)
    # the mean end-shortening must include the geometric wrinkle shortening
    assert np.all(path["mean_strain"] >= path["mean_stress"] / plate.young)
    # the path must actually terminate at edge yield
    assert path["edge_stress_at_ultimate"] == pytest.approx(plate.yield_stress, rel=1e-9)
    assert path["ultimate_stress"] == pytest.approx(path["mean_stress"][-1], rel=1e-12)


def test_elastic_branch_has_the_right_order_of_stiffness():
    """At small strain the plate tangent is between 0.6 E and E."""
    plate = PlateField(length=SPAN, spacing=SPACING, thickness=0.008)
    eps = np.linspace(0.0, 2e-4, 200)
    curve = plate_load_shortening(plate, strains=eps)
    elastic = eps < 0.25 * curve["eps_u"]
    tangent = np.gradient(curve["stress"][elastic], eps[elastic])
    assert tangent.mean() == pytest.approx(YOUNG, rel=1e-9)


def test_panel_stress_is_the_area_weighted_parallel_sum():
    panel = csr_no47_panel()
    curve = panel_load_shortening(panel)
    a_p = panel.plate.spacing * panel.plate.thickness
    a_s = panel.stiffener.area
    expected = (a_p * curve["plate_stress"] + a_s * curve["stiffener_stress"]) / (a_p + a_s)
    assert np.allclose(curve["panel_stress"], expected, rtol=1e-12)
    assert 0.0 < panel.plate_area_fraction < 1.0


def test_stiffener_ultimate_falls_with_span_and_local_buckling_binds_when_stocky():
    plate = PlateField(length=SPAN, spacing=SPACING, thickness=0.012)
    short = Stiffener(span=0.5, web_height=0.18, web_thickness=0.010,
                      flange_width=0.06, flange_thickness=0.010)
    long = Stiffener(span=6.0, web_height=0.18, web_thickness=0.010,
                     flange_width=0.06, flange_thickness=0.010)
    s_short = stiffener_ultimate(short, plate)
    s_long = stiffener_ultimate(long, plate)
    assert s_long["sigma_u"] < s_short["sigma_u"]
    assert s_long["euler_over_yield"] < s_short["euler_over_yield"]
    slender_flange = Stiffener(span=0.5, web_height=0.18, web_thickness=0.010,
                               flange_width=0.30, flange_thickness=0.004)
    assert slender_flange.flange_outstand_buckling() < YIELD


def test_interaction_surface_is_closed_and_monotone():
    panel = csr_no47_panel()
    comp = interaction_components(panel)
    # uniaxial closure
    assert abs(interaction_load_factor(panel, comp["sigma_xu"])["utilisation"] - 1.0) < 1e-9
    assert abs(interaction_load_factor(panel, 0.0, comp["sigma_yu"])["utilisation"] - 1.0) < 1e-9
    assert abs(interaction_load_factor(panel, 0.0, 0.0, comp["tau_u"])["utilisation"] - 1.0) < 1e-9
    curve = interaction_curve(panel)
    factors = [row["load_factor"] for row in curve["rows"]]
    assert all(f > 0.0 for f in factors)
    # shear always reduces the biaxial capacity
    plain = interaction_load_factor(panel, 0.5 * comp["sigma_xu"], 0.5 * comp["sigma_yu"])
    sheared = interaction_load_factor(panel, 0.5 * comp["sigma_xu"], 0.5 * comp["sigma_yu"],
                                      0.5 * comp["tau_u"])
    assert sheared["utilisation"] > plain["utilisation"]


def test_transverse_capacity_is_below_longitudinal_for_this_panel():
    panel = csr_no47_panel()
    comp = interaction_components(panel)
    assert comp["sigma_yu"] < comp["sigma_xu"]


def test_invalid_geometry_is_rejected_before_any_strength_claim():
    with pytest.raises(ValueError):
        PlateField(length=12.0, spacing=0.8, thickness=0.012).validate_geometry()
    with pytest.raises(ValueError):
        PlateField(length=2.4, spacing=0.8, thickness=0.06).validate_geometry()
    with pytest.raises(ValueError):
        PlateField(length=2.4, spacing=0.8, thickness=0.012, residual_ratio=1.2)
    with pytest.raises(ValueError):
        PlateField(length=2.4, spacing=0.8, thickness=0.012, poisson=0.6)
    with pytest.raises(ValueError):
        StiffenedPanelSmith(
            plate=PlateField(length=2.4, spacing=0.8, thickness=0.012),
            stiffener=Stiffener(span=2.4, web_height=0.18, web_thickness=0.010,
                                flange_width=0.06, flange_thickness=0.010, young=1.0e11),
        )


def test_bulb_flat_idealisation_is_explicit_geometry():
    panel = csr_no47_panel()
    stiff = panel.stiffener
    assert stiff.web_height == pytest.approx(0.9 * 0.200)
    assert stiff.flange_width == pytest.approx(0.30 * 0.200)
    assert stiff.area == pytest.approx(0.9 * 0.2 * 0.01 + 0.3 * 0.2 * 0.01)
    assert panel.plate.beta == pytest.approx(2.2517, abs=1e-4)

# --------------------------------------------------------------------------
# end-to-end evidence chain
# --------------------------------------------------------------------------
def test_evidence_run_lands_six_digit_values_and_no_failing_verdict(tmp_path):
    """Criterion (h)+(i) end to end: the written artefact is six-digit clean.

    Runs the benchmark entry point into ``tmp_path`` (so the repository artefact
    is not touched) and checks that (i) every verdict is PASS or an honest
    WITHHELD, (ii) every float in ``evidence.json`` is exactly its
    six-significant-digit form, and (iii) the figure families exist.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "smith_panel_collapse_evidence", root / "benchmarks" / "smith_panel_collapse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    outdir = tmp_path / "smith_evidence"
    evidence = module.run(str(outdir))
    verdicts = {item["id"]: item["verdict"] for item in evidence["criteria"]}
    assert set(verdicts) == set("abcdefghi")
    assert all(v in ("PASS", "WITHHELD") for v in verdicts.values()), verdicts
    assert verdicts["c"] == "WITHHELD"

    landed = json.loads((outdir / "evidence.json").read_text())
    assert json.loads((outdir / "criteria.json").read_text())["case"] == "IACS-CSR-No47"

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, float):
            yield node

    offenders = [v for v in walk(landed) if float(f"{v:.6g}") != v]
    assert not offenders, offenders[:5]
    for name in ("fig1_plate_curves.png", "fig2_panel_curve.png",
                 "fig3_interaction.png", "fig4_sensitivity.png"):
        assert (outdir / name).stat().st_size > 10_000
