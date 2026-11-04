"""Reference-formula and cross-discretisation audit of shear buckling coefficients.

The pre-audit ``classical_shear_coefficient`` applied the ``a/b >= 1`` branch of
the Timoshenko--Gere fit to every aspect ratio.  Because the coefficient is
normalised by the panel width ``b``, rotating the same panel by 90 degrees maps a
uniform shear state onto a uniform shear state, so the *exact* coefficient obeys
``k(a) * a^2 = k(1/a)``.  These tests pin that identity, pin the two independent
discretisations against each other, and record where the classical fit stops
being a 3 % reference.
"""
import math

import numpy as np
import pytest

from tensorfem.marine_plate_buckling import isotropic_plate_rigidities
from tensorfem.plate_buckling_hermite import (
    POISSON,
    THICKNESS,
    YOUNG,
    hermite_shear_coefficient,
)
from tensorfem.plate_shear_buckling import (
    LITERATURE_KS_LONG_PANEL,
    LITERATURE_KS_SQUARE,
    classical_shear_coefficient,
    ritz_inplane_buckling_load,
    run_shear_buckling_qualification,
    shear_buckling_coefficient,
)

GATE = 0.03


def _ritz(aspect_ratio, basis=14):
    rigidity, _, coupling = isotropic_plate_rigidities(YOUNG, POISSON, THICKNESS)
    mode = ritz_inplane_buckling_load(
        length=aspect_ratio, width=1.0, rigidity_x=rigidity, rigidity_y=rigidity,
        coupling_rigidity=coupling, basis_x=basis, basis_y=basis)
    return shear_buckling_coefficient(
        shear_load=mode.multiplier, rigidity=rigidity, width=1.0)


def test_classical_coefficient_obeys_the_rotation_duality():
    """``k(a) * a^2 == k(1/a)``: a 90 degree rotation of the same panel."""
    for aspect in (0.4, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0, 6.0):
        assert classical_shear_coefficient(aspect) * aspect**2 == pytest.approx(
            classical_shear_coefficient(1.0 / aspect), rel=1e-12)


def test_pre_audit_single_branch_violated_the_duality():
    """The retired formula is kept here as a negative control, not as a reference."""
    def legacy(aspect):
        return 5.34 + 4.0 / aspect**2

    assert legacy(1.0) == pytest.approx(9.34)
    # ...but below a/b = 1 it is 18.8 % below the duality-consistent value
    assert abs(legacy(0.5) / classical_shear_coefficient(0.5) - 1.0) == pytest.approx(
        0.1585, abs=1e-3)
    violation = abs(legacy(0.5) * 0.25 / legacy(2.0) - 1.0)
    assert violation > 0.1  # the identity the audit pins is violated by >10 %


def test_branches_join_continuously_at_the_square_panel():
    assert classical_shear_coefficient(1.0) == pytest.approx(LITERATURE_KS_SQUARE)
    assert classical_shear_coefficient(1.0 + 1e-9) == pytest.approx(
        classical_shear_coefficient(1.0 - 1e-9), rel=1e-8)
    assert classical_shear_coefficient(1.0e6) == pytest.approx(
        LITERATURE_KS_LONG_PANEL, abs=1e-8)


def test_two_independent_discretisations_agree_at_the_square_panel():
    """Sine-product Ritz and C1 Hermite FE must agree far inside the 3 % gate."""
    ritz = _ritz(1.0)
    fe = hermite_shear_coefficient(aspect_ratio=1.0, n_y=8)
    assert abs(ritz - fe) / ritz < 0.005
    assert abs(ritz / LITERATURE_KS_SQUARE - 1.0) < GATE
    assert abs(fe / LITERATURE_KS_SQUARE - 1.0) < GATE


def test_hermite_fe_obeys_the_rotation_duality():
    direct = hermite_shear_coefficient(aspect_ratio=2.0, n_y=8)
    rotated = hermite_shear_coefficient(aspect_ratio=0.5, n_y=12)
    assert direct * 4.0 == pytest.approx(rotated, rel=5e-3)


def test_classical_fit_is_not_a_three_percent_reference_at_two_to_one():
    """The fit itself leaves the 3 % band near a/b = 2; both discretisations agree.

    This is why the qualification grades the finite-aspect closed form *and* the
    tabulated anchor instead of trusting the fit alone.
    """
    ritz = _ritz(2.0)
    fe = hermite_shear_coefficient(aspect_ratio=2.0, n_y=10)
    assert abs(ritz - fe) / ritz < 0.005          # solver pair agrees
    fit = classical_shear_coefficient(2.0)
    assert abs(fit / ritz - 1.0) > GATE           # the fit is outside the band
    assert abs(fit / ritz - 1.0) < 0.05           # but only by a few percent


def test_shear_qualification_reports_both_references():
    report = run_shear_buckling_qualification(basis=12)
    assert report["passed"] is True
    for row in report["cases"]:
        if row["literature_k_s"] is None:
            continue
        assert row["relative_error"] < GATE
        assert row["relative_error_vs_closed_form"] < GATE
        assert row["closed_form_k_s"] == pytest.approx(
            classical_shear_coefficient(row["aspect_ratio"]))
        # the long panel is graded against the asymptote *and* the finite aspect fit
        if row["aspect_ratio"] > 1.0:
            assert row["closed_form_k_s"] > row["literature_k_s"]


def test_coarse_span_mesh_is_refused_instead_of_returning_noise():
    with pytest.raises(ValueError):
        hermite_shear_coefficient(aspect_ratio=1.0, n_y=2)
