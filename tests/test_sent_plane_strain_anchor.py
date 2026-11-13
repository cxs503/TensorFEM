"""Plane-strain leg of the SENT J-integral case (``CASE-FEM-JINTEGRAL-EDGE-CRACK``).

The graded driver lives in ``benchmarks/verified/fem_sent_plane_strain/`` and
runs the three-level ladder, the ``nu`` invariance probe and the global-energy
oracle.  This module keeps the *fast* invariants under ``pytest``: the two
handbook fits, the plane-strain effective modulus, and the one defect the case
tolerance cannot see -- using ``E' = E`` under plane strain biases ``K_I`` by
``1/sqrt(1 - 0.3**2) - 1 = +4.8 %``, which is still inside the 5 % tolerance.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.edge_crack_sent import (
    DEFAULT_SENT_LEVELS,
    j_integral_subring,
    sent_k_from_j,
    sent_k_reference,
    sent_t6_mesh,
    solve_sent,
    tada_sent_shape_factor,
    tada_sent_shape_factor_closed_form,
)

WIDTH = 0.10
ALPHA = 0.3
HEIGHT = 0.20
SIGMA = 100e6
YOUNG = 210e9
NU = 0.3
COARSE = DEFAULT_SENT_LEVELS[0]
CANONICAL = 0.15


def _recovered(plane: str, nu: float = NU, alpha: float = ALPHA) -> dict:
    """Coarse-mesh SENT solve; return ``F`` from the canonical fan line."""
    mesh = sent_t6_mesh(
        WIDTH,
        alpha * WIDTH,
        HEIGHT,
        n_sectors=COARSE["n_sectors"],
        first_ring=COARSE["first_ring"],
        growth=COARSE["growth"],
    )
    solved = solve_sent(
        mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=nu, plane=plane
    )
    stations = list(mesh.stations)
    index = min(range(1, len(stations) - 1), key=lambda i: abs(stations[i] - CANONICAL))
    record = j_integral_subring(
        mesh, solved.displacement, index, young_pa=YOUNG, poisson=nu, plane=plane
    )
    k = sent_k_from_j(record["j_j_m2"], YOUNG, nu, plane=plane)
    return {
        "j": record["j_j_m2"],
        "k": k,
        "shape_factor": k / (SIGMA * math.sqrt(math.pi * alpha * WIDTH)),
        "reaction_ratio": solved.reaction_ratio,
        "n_elements": mesh.n_elements,
    }


@pytest.fixture(scope="module")
def strain_coarse() -> dict:
    return _recovered("strain")


@pytest.fixture(scope="module")
def stress_coarse() -> dict:
    return _recovered("stress")


# --------------------------------------------------------------------------- #
# reference formulas
# --------------------------------------------------------------------------- #
def test_handbook_factor_at_anchor_ratio():
    """F(a/W = 0.3) of the Tada polynomial fit, to the printed precision."""
    assert tada_sent_shape_factor(ALPHA) == pytest.approx(1.659919, abs=5e-7)


def test_closed_form_reference_is_independent_but_consistent():
    """The secant form is a different approximation of the same solution.

    It must (a) differ from the polynomial fit -- otherwise it adds no
    information about the reference uncertainty -- and (b) still sit inside the
    +-0.5 % band the handbook quotes for the fit.
    """
    poly = tada_sent_shape_factor(ALPHA)
    closed = tada_sent_shape_factor_closed_form(ALPHA)
    spread = (closed - poly) / poly
    assert closed == pytest.approx(1.655113, abs=5e-7)
    assert 1e-4 < abs(spread) < 5e-3


def test_closed_form_rejects_out_of_range_ratio():
    with pytest.raises(ValueError):
        tada_sent_shape_factor_closed_form(0.0)
    with pytest.raises(ValueError):
        tada_sent_shape_factor_closed_form(1.2)


def test_handbook_sif_at_anchor_configuration():
    """K_I = sigma sqrt(pi a) F: 100 MPa, 30 mm crack, F(0.3) = 1.659919."""
    k = sent_k_reference(SIGMA, ALPHA * WIDTH, WIDTH)
    assert k / 1e6 == pytest.approx(50.957, abs=5e-3)


# --------------------------------------------------------------------------- #
# plane-strain effective modulus
# --------------------------------------------------------------------------- #
def test_plane_strain_effective_modulus_ratio():
    """K = sqrt(J E') differs only by sqrt(E'_strain / E'_stress)."""
    k_stress = sent_k_from_j(1.0, YOUNG, NU, plane="stress")
    k_strain = sent_k_from_j(1.0, YOUNG, NU, plane="strain")
    assert k_strain / k_stress == pytest.approx(
        1.0 / math.sqrt(1.0 - NU ** 2), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# the defect the 5 % case tolerance cannot see
# --------------------------------------------------------------------------- #
def test_plane_strain_matches_plane_stress_shape_factor(strain_coarse, stress_coarse):
    """K_I is a material-independent LEFM quantity.

    Michel's theorem makes the in-plane stress field of a traction-loaded body
    independent of ``E``/``nu``; J and E' must therefore cancel exactly in
    ``K = sqrt(J E')``.  A slip in that pairing shows up here as a ~4.8 % gap
    between the two constitutive branches, i.e. *inside* the case tolerance.
    """
    ratio = strain_coarse["shape_factor"] / stress_coarse["shape_factor"]
    assert ratio == pytest.approx(1.0, rel=5e-3)
    # the J values themselves must differ (w depends on the constitutive branch)
    assert strain_coarse["j"] != pytest.approx(stress_coarse["j"], rel=5e-3)


def test_naive_plane_stress_modulus_would_bias_by_four_point_eight_percent(
    strain_coarse,
):
    """Quantify the slip the test above detects, so the band has a scale."""
    biased = strain_coarse["k"] * math.sqrt(1.0 - NU ** 2)
    k_ref = sent_k_reference(SIGMA, ALPHA * WIDTH, WIDTH)
    err_correct = strain_coarse["k"] / k_ref - 1.0
    err_biased = biased / k_ref - 1.0
    # sqrt(1 - nu^2) = 0.9539: the slip moves K down by 4.61 %, i.e. inside the 5 % case tolerance
    assert err_correct - err_biased == pytest.approx(0.0461, abs=2e-3)


# --------------------------------------------------------------------------- #
# accuracy at the anchor configuration
# --------------------------------------------------------------------------- #
def test_coarse_plane_strain_inside_case_tolerance(strain_coarse):
    """Coarse mesh already meets the 5 % case tolerance at a/W = 0.3 (plane strain)."""
    k_ref = sent_k_reference(SIGMA, ALPHA * WIDTH, WIDTH)
    assert abs(strain_coarse["k"] / k_ref - 1.0) < 0.05
    assert strain_coarse["reaction_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert strain_coarse["n_elements"] > 0