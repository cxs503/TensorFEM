"""Gates for the stiffener local-instability limit states (web buckling + tripping).

Anchor: GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE (declared limitation "stiffener web
local buckling / tripping and frame tripping are not modelled").

The independent oracle used here is the Hermite finite-element eigenvalue solve
written in ``benchmarks/grillage_local_instability.py`` -- a different
discretisation from the module's closed-form sine Ritz, so the comparison is not
the solver checking itself.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from tensorfem.stiffener_tripping import (  # noqa: E402
    TeeTorsionSection,
    plate_buckling_coefficient,
    plate_rotational_restraint,
    stiffener_instability_check,
    tripping_stress,
    web_local_buckling_stress,
)
from grillage_local_instability import (  # noqa: E402
    beam_rotational_spring_fem,
    tripping_eigen_fem,
)

TEE = TeeTorsionSection(0.25, 0.010, 0.080, 0.012)


def test_section_properties_closed_form():
    assert TEE.area == pytest.approx(0.0025 + 0.00096, rel=1e-12)
    # St Venant constant of the two open rectangles
    j = (0.25 * 0.010 ** 3 + 0.080 * 0.012 ** 3) / 3.0
    assert TEE.saint_venant_constant == pytest.approx(j, rel=1e-12)
    # warping constant: numerical integral vs the monosymmetric closed form
    assert TEE.warping_constant == pytest.approx(TEE.warping_constant_closed_form,
                                                 rel=1e-3)
    # polar radius about the attachment line = I_p/A + d^2
    assert TEE.polar_radius_squared == pytest.approx(
        TEE.polar_inertia / TEE.area + TEE.centroid_z ** 2, rel=1e-12)


def test_tripping_matches_free_strut_fe():
    got = tripping_stress(TEE, span=2.1, lateral_translation="free")["sigma_cr_Pa"]
    ref = tripping_eigen_fem(TEE, span=2.1, lateral_translation="free", elements=16)
    assert got == pytest.approx(ref, rel=1e-4)


def test_tripping_matches_restrained_fe_for_weak_spring():
    k = 0.01 * plate_rotational_restraint(plate_thickness=0.012,
                                          stiffener_spacing=0.60)
    got = tripping_stress(TEE, span=2.1, rotational_restraint=k)["sigma_cr_Pa"]
    ref = tripping_eigen_fem(TEE, span=2.1, rotational_restraint=k, elements=24)
    assert got == pytest.approx(ref, rel=0.02)


def test_sine_ritz_is_upper_bound_for_strong_restraint():
    """Documented bias: the sine Ritz overestimates the restrained tripping stress."""
    k = plate_rotational_restraint(plate_thickness=0.020, stiffener_spacing=0.35)
    ritz = tripping_stress(TEE, span=2.1, rotational_restraint=k)["sigma_cr_Pa"]
    fe = tripping_eigen_fem(TEE, span=2.1, rotational_restraint=k, elements=24)
    assert ritz > fe
    assert ritz / fe > 2.0  # quantified in gate T14b3 of the evidence run


def test_tripping_mode_summation_and_restraint_suppression():
    vals = [tripping_stress(TEE, span=2.1, modes=m)["sigma_cr_Pa"] for m in (1, 2, 4)]
    assert max(vals) - min(vals) <= 1e-9 * vals[0]
    suppressed = tripping_stress(TEE, span=2.1, rotational_restraint=1.0e14)
    assert suppressed["sigma_cr_Pa"] > 1e3 * TEE.young


def test_plate_rotational_restraint_matches_beam_fem():
    d = TEE.young * 0.020 ** 3 / (12.0 * (1.0 - 0.3 ** 2))
    fem = beam_rotational_spring_fem(span=0.175, rigidity=d, far_edge="clamped")
    mod = plate_rotational_restraint(plate_thickness=0.020, stiffener_spacing=0.35,
                                     young=TEE.young, poisson=0.3, far_edge="clamped")
    assert mod == pytest.approx(fem, rel=1e-9)
    assert mod == pytest.approx(4.0 * d / 0.175, rel=1e-12)


def test_web_local_buckling_scalings_and_coefficient():
    w1 = web_local_buckling_stress(web_height=0.25, web_thickness=0.010)
    w2 = web_local_buckling_stress(web_height=0.25, web_thickness=0.020)
    assert w2 / w1 == pytest.approx(4.0, rel=1e-12)
    w3 = web_local_buckling_stress(web_height=0.50, web_thickness=0.010)
    assert w1 / w3 == pytest.approx(4.0, rel=1e-12)
    assert plate_buckling_coefficient(web_height=0.55, panel_length=2.75) == pytest.approx(
        4.0, rel=1e-3)
    # k from the exact aspect ratio is used when the panel length is known
    k_eff = (web_local_buckling_stress(web_height=0.55, web_thickness=0.010,
                                       panel_length=2.8)
             / web_local_buckling_stress(web_height=0.55, web_thickness=0.010))
    assert k_eff == pytest.approx(
        plate_buckling_coefficient(web_height=0.55, panel_length=2.8) / 4.0, rel=1e-12)


def test_monotonicity():
    spans = [tripping_stress(TEE, span=L)["sigma_cr_Pa"] for L in (5.0, 2.1, 1.0)]
    assert spans[0] < spans[1] < spans[2]
    wide = TeeTorsionSection(0.25, 0.010, 0.160, 0.012)
    assert tripping_stress(wide, span=2.1)["sigma_cr_Pa"] > tripping_stress(
        TEE, span=2.1)["sigma_cr_Pa"]
    thin = TeeTorsionSection(0.25, 0.006, 0.080, 0.012)
    assert tripping_stress(thin, span=2.1)["sigma_cr_Pa"] < tripping_stress(
        TEE, span=2.1)["sigma_cr_Pa"]


def test_combined_check_bracket_and_governing():
    chk = stiffener_instability_check(TEE, span=2.1, stiffener_spacing=0.35,
                                      plate_thickness=0.020, yield_stress=3.55e8)
    assert chk["tripping_sigma_low_Pa"] <= chk["tripping_sigma_high_Pa"]
    assert chk["limiting_stress_Pa"] == min(chk["tripping_sigma_low_Pa"],
                                            chk["web_local_buckling"]["min_Pa"])
    assert chk["governing"] in ("tripping", "web_local_buckling")
    # the ice-belt case-study stiffener: local state is below the column state
    assert 0.5 < chk["limiting_over_yield"] < 0.8
    # no plate -> no restraint -> the bracket collapses onto the free value
    bare = stiffener_instability_check(TEE, span=2.1, stiffener_spacing=0.35,
                                       plate_thickness=0.0)
    assert bare["tripping_sigma_pinned_Pa"] == bare["tripping_sigma_free_Pa"]
    assert bare["plate_rotational_restraint_clamped_N"] == 0.0


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        TeeTorsionSection(-0.1, 0.010, 0.080, 0.012)
    with pytest.raises(ValueError):
        TeeTorsionSection(0.25, 0.010, 0.080, 0.012, poisson=0.6)
    with pytest.raises(ValueError):
        tripping_stress(TEE, span=0.0)
    with pytest.raises(ValueError):
        tripping_stress(TEE, span=2.1, rotational_restraint=-1.0)
    with pytest.raises(ValueError):
        tripping_stress(TEE, span=2.1, lateral_translation="sideways")
    with pytest.raises(ValueError):
        web_local_buckling_stress(web_height=0.25, web_thickness=0.010,
                                 edge_condition="free")
    with pytest.raises(ValueError):
        plate_rotational_restraint(plate_thickness=0.02, stiffener_spacing=0.35,
                                   far_edge="free")
