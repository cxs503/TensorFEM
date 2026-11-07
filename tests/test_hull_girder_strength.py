"""Tests for hull_girder_strength.py — full chain validation against KVLCC2."""

import math

import pytest

from tensorfem.hull_girder_strength import (
    HullSection,
    csr_check,
    design_wave_bending_moment,
    run_kvlcc2_benchmark,
    still_water_bending_moment,
)


def test_design_wave_bending_moment_is_positive_and_scales():
    M = design_wave_bending_moment(L=320, B=58, Cb=0.708, h_w=13.5)
    assert M > 0
    # Double L → M should quadruple (L² dependence)
    M2 = design_wave_bending_moment(L=640, B=58, Cb=0.708, h_w=13.5)
    assert M2 == pytest.approx(4 * M, rel=1e-10)


def test_still_water_bending_moment_sign():
    # LCB aft of midships → positive (sagging)
    assert still_water_bending_moment(displacement=1e6, LCB=10.0, L=320) > 0
    # LCB forward → negative (hogging)
    assert still_water_bending_moment(displacement=1e6, LCB=-10.0, L=320) < 0


def test_hull_section_properties():
    sec = HullSection()
    sec.add_component("top", 1.0, 2.0, 210e9, 235e6)
    sec.add_component("bottom", 1.0, -2.0, 210e9, 235e6)
    assert abs(sec.neutral_axis()) < 1e-12
    I = sec.moment_of_inertia()
    assert I == pytest.approx(1.0 * 2.0**2 + 1.0 * (-2.0)**2)  # 8.0
    W_top, W_bottom = sec.section_modulus()
    assert W_top == pytest.approx(I / 2.0)
    assert W_bottom == pytest.approx(I / 2.0)


def test_csr_check_safety_factors():
    res = csr_check(M_total=1e9, W_top=1e6, W_bottom=1e6, yield_stress=235e6, gamma_m=1.43)
    assert res.sigma_allow == pytest.approx(235e6 / 1.43)
    assert res.SF_top == pytest.approx(res.sigma_allow / 1000.0)
    # When M_ultimate is not provided, utilisation should be 0.0
    assert res.utilisation == 0.0
    assert res.M_ultimate == 0.0

    # With M_ultimate provided
    res2 = csr_check(M_total=1e9, W_top=1e6, W_bottom=1e6, yield_stress=235e6,
                     gamma_m=1.43, M_ultimate=2e9)
    assert res2.utilisation == pytest.approx(1e9 / 2e9)


def test_kvlcc2_benchmark_passes_all_gates():
    report = run_kvlcc2_benchmark()
    assert report["passed"]
    assert report["validation"]["I_error_pass"]
    assert report["validation"]["stress_error_pass"]
    # Safety factors should be reasonable (> 0)
    assert report["hogging"]["SF_top"] > 0
    assert report["sagging"]["SF_bottom"] > 0
    # Ultimate utilisation should be < 1.0 for this simplified case
    assert report["ultimate"]["utilisation_hog"] < 1.0


def test_strict_inputs_wave():
    with pytest.raises(TypeError):
        design_wave_bending_moment(L="320", B=58, Cb=0.708, h_w=13.5)
    with pytest.raises(ValueError):
        design_wave_bending_moment(L=-1, B=58, Cb=0.708, h_w=13.5)


def test_strict_inputs_sw():
    with pytest.raises(TypeError):
        still_water_bending_moment(displacement="1e6", LCB=0, L=320)
    with pytest.raises(ValueError):
        still_water_bending_moment(displacement=1e6, LCB=float("nan"), L=320)
