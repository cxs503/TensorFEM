"""Acceptance tests for the hull-girder CSR bending chain (mission IN50-N07).

Criteria covered:
    C1  midship section inertia consistency (parallel axis vs strip integration)
    C2  total hogging/sagging moment vs the DNV GL 2019 / UR S11 composed value
    C3  stress vs the analytic ``M y / I`` value (FE curvature recovery)
    C4  CSR safety factor ``SF = sigma_allow / sigma`` is reported and >= 1
    C5  ``M_total / M_ultimate <= 1``
    C6  this file (one test per criterion)
    C7  the four chain figures are produced and non-trivial
"""

import math

import numpy as np
import pytest

from tensorfem.hull_girder_csr_chain import (
    GRAVITY,
    MILD_STEEL_YIELD,
    WATER_DENSITY,
    HullForm,
    WeightItem,
    beam_fem,
    beam_moment_from_load,
    close_deck_scantling,
    csr_allowable_stress,
    csr_check,
    equivalent_wave_coefficient,
    holtrop_wave_bending_moment,
    homogeneous_loaded_tanker_weights,
    hull_girder_ultimate_moment,
    johnson_ostenfeld,
    kvlcc2_hull_form,
    kvlcc2_midship_section,
    member_compression_strength,
    plot_hull_girder_chain,
    rule_wave_bending_moment,
    run_kvlcc2_csr_chain,
    section_inertia_profile,
    solve_equilibrium_draft_trim,
    ur_s11_minimum_section_modulus,
    wave_coefficient,
    wave_load_distribution,
    weight_centroid,
    weight_per_length,
)


@pytest.fixture(scope="module")
def chain():
    return run_kvlcc2_csr_chain()


# ---------------------------------------------------------------------------
# C1 - midship section inertia
# ---------------------------------------------------------------------------

def test_c1_midship_inertia_consistency(chain):
    section = chain["section"]
    assert section["I_midship"] == pytest.approx(section["I_strip"], rel=0.05)
    assert chain["criteria"]["C1_pass"]
    assert chain["criteria"]["C1_I_midship_error"] <= 0.05
    # physically meaningful magnitude for a 320 m / 58 m tanker
    assert 1000.0 < section["I_midship"] < 2600.0
    # neutral axis inside the hull, deck modulus at the UR S11 minimum
    assert 0.35 * 30.0 < section["y_neutral_axis"] < 0.55 * 30.0


def test_c1_section_properties_are_consistent():
    section = kvlcc2_midship_section()
    y_na = section.neutral_axis()
    inertia = section.moment_of_inertia()
    w_deck, w_bottom = section.section_moduli()
    y_top, y_bottom = section.extreme_fibres()
    assert w_deck == pytest.approx(inertia / y_top)
    assert w_bottom == pytest.approx(inertia / y_bottom)
    assert y_na + y_top == pytest.approx(30.0, abs=0.05)   # deck at D = 30 m
    assert y_na - y_bottom == pytest.approx(0.0, abs=0.05)  # keel at y = 0
    assert section.axial_residual() < 1e-12
    assert section.strip_inertia(strips=400) == pytest.approx(inertia, rel=1e-3)


def test_c1_deck_scantling_closes_against_ur_s11():
    w_min = ur_s11_minimum_section_modulus(L=320.0, B=58.0, Cb=0.8095)
    assert w_min == pytest.approx(96.38, rel=1e-3)  # m3, deck modulus
    section, audit = close_deck_scantling(target_modulus=w_min)
    w_deck, _ = section.section_moduli()
    assert w_deck >= w_min
    assert audit["converged"] is True
    assert audit["W_deck_final"] == pytest.approx(w_deck)


# ---------------------------------------------------------------------------
# C2 - total moment vs DNV GL 2019 / IACS UR S11
# ---------------------------------------------------------------------------

def test_c2_total_moment_matches_dnv_rule(chain):
    moments = chain["moments"]
    rule = chain["rule"]
    assert rule["C"] == pytest.approx(10.75)  # 300 m <= L <= 350 m
    assert moments["M_wh_rule"] == pytest.approx(9.82e9, rel=0.01)
    assert moments["M_ws_rule"] == pytest.approx(10.60e9, rel=0.01)
    assert chain["criteria"]["C2_pass"]
    assert chain["criteria"]["C2_M_total_hog_vs_DNV_error"] <= 0.10
    assert chain["criteria"]["C2_M_total_sag_vs_DNV_error"] <= 0.10
    expected_hog = abs(moments["M_SW_mid"] - moments["M_wh_rule"])
    assert moments["M_total_hog"] == pytest.approx(expected_hog, rel=0.10)


def test_c2_rule_formula_limits_and_wave_coefficient():
    assert wave_coefficient(100.0) == pytest.approx(10.75 - 2.0 ** 1.5, rel=1e-12)
    assert wave_coefficient(320.0) == pytest.approx(10.75)
    assert wave_coefficient(400.0) == pytest.approx(10.75 - (50.0 / 150.0) ** 1.5)
    with pytest.raises(ValueError):
        wave_coefficient(50.0)
    with pytest.raises(ValueError):
        wave_coefficient(600.0)
    rule = rule_wave_bending_moment(L=320.0, B=58.0, Cb=0.8095)
    # L^2 scaling of the rule moment
    big = rule_wave_bending_moment(L=350.0, B=58.0, Cb=0.8095)
    assert big["M_wh"] == pytest.approx(9.82e9 * (350.0 / 320.0) ** 2, rel=0.01)


def test_c2_holtrop_form_reproduces_rule_value():
    kwargs = dict(L=320.0, B=58.0, Cb=0.8095, h_w=13.5)
    coeff = equivalent_wave_coefficient(**kwargs)
    rule = rule_wave_bending_moment(L=320.0, B=58.0, Cb=0.8095)
    direct = holtrop_wave_bending_moment(**kwargs, k_w=coeff["k_w_hog"])
    assert direct == pytest.approx(rule["M_wh"], rel=1e-12)
    # the published k_w = 11.6 over-predicts by roughly one order of magnitude
    wrong = holtrop_wave_bending_moment(**kwargs, k_w=11.6)
    assert wrong / rule["M_wh"] > 5.0


def test_c2_geometric_closure_before_loads():
    form = kvlcc2_hull_form()
    checks = form.verify()
    assert checks["Cb_error"] < 1e-3
    assert checks["displacement_t"] == pytest.approx(320_000.0, rel=0.01)
    assert checks["lcb_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert 5.0 < checks["L_over_B"] < 6.0
    assert 2.5 < checks["B_over_T"] < 3.0


# ---------------------------------------------------------------------------
# C3 - stress vs analytic M y / I
# ---------------------------------------------------------------------------

def test_c3_stress_matches_analytic(chain):
    stresses = chain["stresses"]
    assert chain["criteria"]["C3_pass"]
    assert chain["criteria"]["C3_stress_vs_analytic_error"] <= 0.01
    assert stresses["deck_hog"] == pytest.approx(
        stresses["sigma_analytic_deck_hog"], rel=0.01
    )
    assert stresses["bottom_sag"] == pytest.approx(
        stresses["sigma_analytic_bottom_sag"], rel=0.01
    )
    # hogging puts the deck in tension and the bottom in compression
    assert stresses["deck_hog"] > 0.0 > stresses["bottom_hog"]
    assert stresses["deck_sag"] < 0.0 < stresses["bottom_sag"]
    assert chain["beam"]["max_moment_error_vs_analytic"] <= 0.01


def test_c3_beam_fe_reproduces_closed_form():
    x = np.linspace(0.0, 100.0, 120)
    q = np.full_like(x, -1.0e5)  # uniform downward load
    ei = np.full_like(x, 2.0e11 * 0.5)
    fem = beam_fem(x, q, ei)
    analytic = beam_moment_from_load(x, q)["moment"]
    assert np.max(np.abs(fem["moment"] - analytic)) / np.max(np.abs(analytic)) < 5e-3
    # reaction total balances the applied load
    assert fem["reaction"].sum() == pytest.approx(1.0e5 * 100.0, rel=1e-6)


def test_c3_stress_sign_and_neutral_axis_direct():
    section = kvlcc2_midship_section()
    inertia = section.moment_of_inertia()
    y_top, y_bottom = section.extreme_fibres()
    moment = -5.0e9  # hogging (sagging positive convention)
    sigma_deck = -(moment * y_top) / inertia      # sigma = -M y / I
    sigma_bottom = -(moment * (-y_bottom)) / inertia
    assert sigma_deck > 0.0 > sigma_bottom
    # deck tension equals bottom compression only for a symmetric section;
    # here the bottom modulus is larger, so the deck stress is higher
    assert abs(sigma_deck) > abs(sigma_bottom)


# ---------------------------------------------------------------------------
# C4 - CSR safety factor
# ---------------------------------------------------------------------------

def test_c4_csr_safety_factors(chain):
    csr = chain["csr"]
    allowable = csr["allowable"]
    assert allowable["sigma_allow"] == pytest.approx(
        min(MILD_STEEL_YIELD / 1.43, 220e6)
    )
    for case in ("hogging", "sagging"):
        record = csr[case]
        assert record["safety_factor"] == pytest.approx(
            record["sigma_allow"] / record["sigma"]
        )
        assert record["sigma"] == pytest.approx(
            record["moment"] / record["section_modulus"]
        )
        assert record["safety_factor"] >= 1.0
    assert chain["criteria"]["C4_pass"]
    assert chain["criteria"]["C4_safety_factor_hog"] >= 1.0


def test_c4_csr_check_governing_criterion():
    res = csr_check(moment=1.0e10, section_modulus=100.0, ultimate_moment=2.0e10)
    assert res["sigma"] == pytest.approx(1.0e8)
    assert res["safety_factor"] == pytest.approx(164335664.34 / 1.0e8, rel=1e-6)
    assert res["utilisation"] == pytest.approx(0.5)
    assert res["ultimate_passes"] is True
    assert res["passes"] is True
    heavy = csr_check(moment=2.0e10, section_modulus=100.0, ultimate_moment=1.5e10)
    assert heavy["passes"] is False
    assert heavy["ultimate_passes"] is False


# ---------------------------------------------------------------------------
# C5 - ultimate strength utilisation
# ---------------------------------------------------------------------------

def test_c5_ultimate_utilisation(chain):
    ultimate = chain["ultimate"]
    for direction in ("hogging", "sagging"):
        record = ultimate[direction]
        assert record["M_ultimate"] > record["M_elastic_yield"]
        assert 1.0 <= record["shape_factor"] <= 1.4
        assert record["axial_residual_rel"] < 1e-9
        moments = np.asarray(record["moment_curve"])
        assert np.all(np.diff(moments[: len(moments) // 2]) >= -1e-6)
    assert chain["criteria"]["C5_utilisation_hog"] <= 1.0
    assert chain["criteria"]["C5_utilisation_sag"] <= 1.0
    assert chain["criteria"]["C5_pass"]


def test_c5_johnson_ostenfeld_limits():
    assert johnson_ostenfeld(235e6, 1.0e9) == pytest.approx(235e6 * (1 - 235 / 4000))
    assert johnson_ostenfeld(235e6, 1.0e8) == pytest.approx(1.0e8)
    section = kvlcc2_midship_section()
    member = section.members[0]  # deck plating, 30 mm at 0.9 m spacing
    strength = member_compression_strength(member)
    assert 0.5 * MILD_STEEL_YIELD < strength["sigma_compression"] < MILD_STEEL_YIELD
    assert strength["governing"] in ("plate", "column")


# ---------------------------------------------------------------------------
# Loads and equilibrium (inputs to C2/C3)
# ---------------------------------------------------------------------------

def test_weight_and_buoyancy_equilibrium():
    form = kvlcc2_hull_form()
    items, total_mass_t = homogeneous_loaded_tanker_weights(form)
    lcg = weight_centroid(form, items)
    assert 0.4 * form.Lpp < lcg < 0.6 * form.Lpp
    w = weight_per_length(form, items)
    x = form.x()
    assert float(np.trapezoid(w, x)) == pytest.approx(total_mass_t * 1000.0 * GRAVITY, rel=1e-9)
    equilibrium = solve_equilibrium_draft_trim(form, total_mass_t, lcg)
    assert equilibrium["force_residual_rel"] < 1e-10
    assert equilibrium["moment_residual_rel"] < 1e-9
    assert equilibrium["draft_m"] == pytest.approx(form.T, rel=0.01)


def test_wave_load_has_zero_net_force_and_moment():
    form = kvlcc2_hull_form()
    x = form.x()
    q_wave = wave_load_distribution(form, h_w=13.5)
    assert abs(float(np.trapezoid(q_wave, x))) / np.max(np.abs(q_wave)) < 1e-9
    moment_about_mid = float(np.trapezoid(q_wave * (x - form.Lpp / 2.0), x))
    assert abs(moment_about_mid) / (np.max(np.abs(q_wave)) * form.Lpp) < 1e-12
    # crest amidships must produce a hogging wave moment
    wave_moment = beam_moment_from_load(x, q_wave)["moment"]
    mid = int(np.argmin(np.abs(x - form.Lpp / 2.0)))
    assert wave_moment[mid] < 0.0


def test_holtrop_input_validation_and_units():
    M = holtrop_wave_bending_moment(L=320.0, B=58.0, Cb=0.8095, h_w=13.5)
    assert M > 0.0
    doubled = holtrop_wave_bending_moment(
        L=320.0, B=58.0, Cb=0.8095, h_w=27.0
    )
    assert doubled == pytest.approx(2.0 * M, rel=1e-12)
    with pytest.raises(TypeError):
        holtrop_wave_bending_moment(L="320", B=58.0, Cb=0.8095, h_w=13.5)
    with pytest.raises(ValueError):
        holtrop_wave_bending_moment(L=320.0, B=58.0, Cb=0.8095, h_w=-1.0)


def test_inertia_profile_is_positive_and_midship_peak():
    form = kvlcc2_hull_form()
    profile = section_inertia_profile(form, 1600.0)
    assert np.all(profile > 0.0)
    assert profile.max() == pytest.approx(1600.0, rel=1e-9)
    mid = int(np.argmin(np.abs(form.x() - form.Lpp / 2.0)))
    assert profile[mid] == pytest.approx(profile.max(), rel=1e-6)


def test_hull_form_validation():
    with pytest.raises(ValueError):
        HullForm(Lpp=320.0, B=58.0, D=30.0, T=20.8, Cb=1.2)
    with pytest.raises(ValueError):
        HullForm(Lpp=320.0, B=58.0, D=20.0, T=20.8, Cb=0.8)
    with pytest.raises(ValueError):
        WeightItem("bad", 100.0, 5.0, 5.0)


# ---------------------------------------------------------------------------
# C6 / full chain and C7 - figures
# ---------------------------------------------------------------------------

def test_c6_full_chain_all_criteria_pass(chain):
    criteria = chain["criteria"]
    for key in ("C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass"):
        assert criteria[key], f"{key} failed"
    assert criteria["all_pass"] is True
    assert len(chain["distributions"]["nodes"]) == len(chain["distributions"]["M_hog"])
    assert chain["schema"] == "tensorfem.hull-girder-csr-chain/1.0"


def test_c7_figures_are_generated(chain, tmp_path):
    figures = plot_hull_girder_chain(chain, outdir=str(tmp_path))
    assert len(figures) == 4
    for figure in figures:
        assert figure["caption"]
        path = figure["path"]
        assert path.endswith(".png")
        size = __import__("os").path.getsize(path)
        assert size > 20_000, f"{path} is suspiciously small ({size} bytes)"
    names = {figure["name"] for figure in figures}
    assert names == {
        "moment_distribution",
        "stress_distribution",
        "section_stress",
        "ultimate_moment_curvature",
    }


def test_allowable_stress_governing_switch():
    mild = csr_allowable_stress(235e6)
    assert mild["sigma_allow"] == pytest.approx(235e6 / 1.43)
    assert mild["governing"] == "sigma_y/gamma_m"
    high = csr_allowable_stress(355e6)
    assert high["sigma_allow"] == pytest.approx(min(355e6 / 1.43, 220e6))
    assert high["governing"] == "CSR 220 MPa"


def test_ultimate_moment_is_positive_and_exported():
    section = kvlcc2_midship_section()
    hog = hull_girder_ultimate_moment(section, direction="hog")
    assert hog["M_ultimate"] > 0.0
    assert math.isfinite(hog["curvature_ultimate"])
    with pytest.raises(ValueError):
        hull_girder_ultimate_moment(section, direction="roll")