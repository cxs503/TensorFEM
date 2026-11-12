"""Bending-moment verification of the foundation-coupled Mindlin Q4 plate.

``GAP-FEM-WINKLER`` acceptance: *deflection **and bending-moment** distribution vs
the series solution < 5 %*.  The deflection half is covered by
``tests/test_foundation_plate_bending.py``; this file closes the moment half.

Four independent layers:

1. **literature anchor of the series** -- for ``k = G = 0`` the centre moment of
   a simply supported square plate under uniform load is ``Mx = My = 0.0479 q a^2``
   (Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
   Table 8, ``nu = 0.3``).  The series is checked against that value, so the
   reference is not self-referential.
2. **recovery operator exactness** -- a patch test with an imposed rotation field
   whose curvature is known analytically, plus a quadratic field for the
   Gauss-to-node extrapolation.
3. **variational audit** -- the bending energy of the recovered moment field must
   reproduce ``(1/2) u^T K u`` in the thin-plate limit (the residual is exactly
   the MITC transverse-shear energy, which scales as ``t^2``).
4. **FE vs series for the moment field** on Winkler and Pasternak foundations,
   including the marine buoyancy case ``k = rho g`` of a steel plate in sea water.
"""
import math

import pytest
import torch

from tensorfem.foundation_plate_bending import (
    assemble_plate_foundation_system,
    bending_energy_from_moments,
    compare_finite_plate_moments,
    element_bending_moments,
    finite_plate_foundation_fem,
    navier_foundation_deflection,
    navier_foundation_moments,
    navier_mindlin_foundation_deflection,
    navier_mindlin_foundation_moments,
    nodal_bending_moments,
    plate_rigidity,
)

YOUNG, POISSON, THICK = 3.0e10, 0.3, 0.1
SIDE, PRESSURE = 10.0, 1.0e4
D_PLATE = plate_rigidity(YOUNG, POISSON, THICK)
TIMOSHENKO_CENTER_MOMENT = 0.0479            # table coefficient, nu = 0.3


def _center():
    return torch.tensor([[SIDE / 2, SIDE / 2]], dtype=torch.float64)


# ---------------------------------------------------------------------------
# 1. literature anchor of the analytical series
# ---------------------------------------------------------------------------

def test_series_matches_timoshenko_uniform_load():
    """``Mx(centre) = 0.0479 q a^2`` for ``nu = 0.3`` (Timoshenko table)."""
    moments = navier_foundation_moments(
        _center(), side_x=SIDE, side_y=SIDE,
        young=YOUNG, poisson=POISSON, thickness=THICK, pressure=PRESSURE,
        n_modes=99,
    )
    reference = TIMOSHENKO_CENTER_MOMENT * PRESSURE * SIDE**2
    assert float(moments[0, 0]) == pytest.approx(reference, rel=5e-3)
    assert float(moments[0, 1]) == pytest.approx(reference, rel=5e-3)   # square
    assert float(moments[0, 2]) == pytest.approx(0.0, abs=1e-6 * reference)


def test_series_mode_convergence():
    """The moment series is converged long before ``n_modes = 99``."""
    coarse = navier_foundation_moments(
        _center(), side_x=SIDE, side_y=SIDE, pressure=PRESSURE, n_modes=21,
    )
    fine = navier_foundation_moments(
        _center(), side_x=SIDE, side_y=SIDE, pressure=PRESSURE, n_modes=99,
    )
    assert float(abs(coarse[0, 0] - fine[0, 0]) / fine[0, 0]) <= 1e-3


def test_series_moment_sign_and_symmetry():
    """Sagging under downward load, and ``Mxy`` vanishes on both centre lines."""
    pts = torch.tensor([
        [SIDE / 2, SIDE / 2],
        [0.25 * SIDE, SIDE / 2],
        [SIDE / 2, 0.25 * SIDE],
    ], dtype=torch.float64)
    moments = navier_foundation_moments(
        pts, side_x=SIDE, side_y=SIDE, pressure=PRESSURE,
    )
    assert float(moments[:, 0].min()) > 0.0            # sagging everywhere
    assert float(moments[1, 2]) == pytest.approx(0.0, abs=1e-6 * PRESSURE * SIDE**2)
    assert float(moments[2, 2]) == pytest.approx(0.0, abs=1e-6 * PRESSURE * SIDE**2)
    # Quarter point, (Mxy = -D(1-nu) w,xy) convention: w,xy > 0 there, so Mxy < 0.
    # The FE recovery uses the same convention, which is what the rel-error tests
    # below assert.
    quarter = navier_foundation_moments(
        torch.tensor([[0.25 * SIDE, 0.25 * SIDE]], dtype=torch.float64),
        side_x=SIDE, side_y=SIDE, pressure=PRESSURE,
    )
    assert float(quarter[0, 2]) < 0.0


def test_foundation_reduces_center_moment():
    """A Winkler support carries part of the load: ``Mx`` must drop monotonically."""
    values = [
        float(navier_foundation_moments(
            _center(), side_x=SIDE, side_y=SIDE, pressure=PRESSURE, winkler=k,
        )[0, 0])
        for k in (0.0, 1.0e4, 1.0e5, 1.0e6)
    ]
    assert all(b < a for a, b in zip(values, values[1:]))
    assert values[0] > 4.0 * values[-1]


# ---------------------------------------------------------------------------
# 2. recovery operator exactness
# ---------------------------------------------------------------------------

def test_patch_test_uniform_curvature():
    """Imposed ``theta_x = x`` on one element -> ``kappa_x = 1``, ``Mx = D``."""
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    elements = torch.tensor([[0, 1, 2, 3]])
    dofs = torch.zeros(12, dtype=torch.float64)
    dofs[1::3] = nodes[:, 0]
    moments, coords = element_bending_moments(
        nodes, elements, dofs, young=YOUNG, poisson=POISSON, thickness=THICK,
    )
    assert moments.shape == (1, 4, 3) and coords.shape == (1, 4, 2)
    assert torch.allclose(moments[0, :, 0], torch.full((4,), D_PLATE, dtype=torch.float64),
                          rtol=1e-12)
    assert torch.allclose(moments[0, :, 1], torch.full((4,), POISSON * D_PLATE,
                                                       dtype=torch.float64), rtol=1e-12)
    assert float(moments[0, :, 2].abs().max()) <= 1e-12 * D_PLATE


def test_nodal_extrapolation_exact_for_linear_moment_field():
    """A linear curvature field is reproduced exactly by Gauss->node extrapolation."""
    sol = finite_plate_foundation_fem(
        side_x=SIDE, side_y=SIDE, nx=8, ny=8, pressure=PRESSURE,
        young=YOUNG, poisson=POISSON, thickness=THICK,
    )
    nodes, elements = sol.nodes, sol.elements
    nx = ny = 8
    xs = torch.linspace(0.0, SIDE, nx + 1, dtype=torch.float64)
    dofs = torch.zeros(nodes.shape[0] * 3, dtype=torch.float64)
    dofs[1::3] = 0.5 * nodes[:, 0] ** 2          # theta_x = x^2/2 -> kappa_x = x
    nodal = nodal_bending_moments(
        nodes, elements, dofs, young=YOUNG, poisson=POISSON, thickness=THICK,
    )
    # interior nodes only (boundary nodes see only one/two elements)
    interior = [
        j * (nx + 1) + i for i in range(1, nx) for j in range(1, ny)
    ]
    idx = torch.tensor(interior)
    expected = D_PLATE * nodes[idx, 0]
    assert torch.allclose(nodal[idx, 0], expected, rtol=1e-10, atol=1e-6 * D_PLATE)
    assert float(nodal[idx, 0].min()) == pytest.approx(float(xs[1]) * D_PLATE, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. variational audit (moment energy = elastic energy in the thin limit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thickness,tol", [(0.02, 1e-4), (0.002, 1e-6)])
def test_moment_energy_reproduces_elastic_energy(thickness, tol):
    """The MITC shear energy is the only residual and it scales as ``t^2``."""
    sol = finite_plate_foundation_fem(
        side_x=SIDE, side_y=SIDE, nx=12, ny=12, pressure=PRESSURE,
        young=YOUNG, poisson=POISSON, thickness=thickness,
    )
    from_moments = bending_energy_from_moments(
        sol.nodes, sol.elements, sol.deflection,
        young=YOUNG, poisson=POISSON, thickness=thickness,
    )
    rows, cols, vals, _, ndof = assemble_plate_foundation_system(
        sol.nodes, sol.elements,
        young=YOUNG, poisson=POISSON, thickness=thickness,
    )
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    K.index_put_((rows, cols), vals, accumulate=True)
    reference = 0.5 * float(sol.deflection @ K @ sol.deflection)
    assert from_moments > 0.0
    assert abs(from_moments - reference) / reference <= tol


# ---------------------------------------------------------------------------
# 4. FE moment field vs the Navier series
# ---------------------------------------------------------------------------

def test_fe_moments_match_series_uniform_load():
    report = compare_finite_plate_moments(nx=32, ny=32)
    assert report["max_mx_rel_error"] <= 0.02
    assert report["max_my_rel_error"] <= 0.02
    assert report["center_mx_rel_error"] <= 0.01
    assert report["mxy_rel_error"] <= 0.05
    assert report["max_w_rel_error"] <= 0.01
    assert report["center_mx"] > 0.0                    # sagging, correct sign


def test_fe_moments_match_series_winkler():
    report = compare_finite_plate_moments(nx=32, ny=32, winkler=1.0e5)
    assert report["max_mx_rel_error"] <= 0.02
    assert report["max_my_rel_error"] <= 0.02
    assert report["center_mx_rel_error"] <= 0.01
    assert report["mxy_rel_error"] <= 0.05


def test_fe_moments_match_series_pasternak():
    report = compare_finite_plate_moments(
        nx=32, ny=32, winkler=1.0e5, pasternak=1.0e6,
    )
    assert report["max_mx_rel_error"] <= 0.02
    assert report["max_my_rel_error"] <= 0.02
    assert report["mxy_rel_error"] <= 0.05


def test_fe_moments_match_series_marine_buoyancy():
    """Marine case: steel plate ``t = 0.1 m`` floating, ``k = rho g = 10055 N/m3``."""
    k_buoy = 1025.0 * 9.81
    report = compare_finite_plate_moments(nx=32, ny=32, winkler=k_buoy)
    assert report["max_mx_rel_error"] <= 0.02
    assert report["center_mx_rel_error"] <= 0.01
    # The foundation carries part of the load, so Mx must drop, and the series
    # itself must show the same drop (the FE/series agreement is the check above).
    unreinforced = compare_finite_plate_moments(nx=32, ny=32)
    assert report["center_mx"] < unreinforced["center_mx"]
    assert 0.5 < report["center_mx"] / unreinforced["center_mx"] < 0.99


def test_moment_error_is_second_order_in_h():
    errors = [
        compare_finite_plate_moments(nx=n, ny=n)["center_mx_rel_error"]
        for n in (8, 16, 32)
    ]
    assert errors[0] > errors[1] > errors[2]
    for coarse, fine in zip(errors, errors[1:]):
        assert 2.0 <= coarse / fine <= 6.0


def test_deflection_and_moment_consistency_at_centre():
    """Same solve: ``w`` and ``Mx`` are compared against the same series."""
    report = compare_finite_plate_moments(nx=32, ny=32, winkler=1.0e5)
    center = torch.tensor([[SIDE / 2, SIDE / 2]], dtype=torch.float64)
    w_series = navier_foundation_deflection(
        center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
        thickness=THICK, winkler=1.0e5, pressure=PRESSURE,
    )
    row = report["rows"][-1]
    assert row["w_analytic"] == pytest.approx(float(w_series[0]), rel=1e-12)
    assert row["w_rel_error"] <= 0.01
    assert math.isfinite(report["foundation_reaction"])

# ---------------------------------------------------------------------------
# 5. Mindlin (thick-plate) moment series: thickness-independence of the moment
# ---------------------------------------------------------------------------
# ``GAP-FEM-WINKLER`` doc flagged "厚板弯矩级数" as the next step: for a thick
# plate (t/a > 0.05) the Kirchhoff series is only accurate to O((t/a)^2) on the
# *deflection* side.  This section closes that verification debt with the
# decisive finding that the *bending-moment* field of a uniformly-loaded,
# simply-supported plate is **thickness-independent**: the Mindlin and
# Kirchhoff series give the same Mx/My/Mxy (a classical result -- the moment is
# fixed by the load distribution, not by transverse shear), while the
# deflection does need the Mindlin series (up to ~17 % at t/a = 0.2).

def test_mindlin_moment_series_collapses_onto_kirchhoff_in_thin_limit():
    """In the thin-plate limit the Mindlin moment series == Kirchhoff series.

    ``navier_mindlin_foundation_moments`` is documented to collapse onto
    ``navier_foundation_moments`` as ``S -> inf`` (``X -> -a W``, ``Y -> -b W``).
    This is the analytic-consistency anchor for the thick-plate reference.
    """
    center = _center()
    for t in (0.1, 0.3, 0.5):
        mk = navier_foundation_moments(
            center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
            thickness=t, pressure=PRESSURE, n_modes=99,
        )
        mm = navier_mindlin_foundation_moments(
            center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
            thickness=t, pressure=PRESSURE, n_modes=99,
        )
        assert torch.allclose(mk, mm, rtol=1e-6, atol=1e-6 * D_PLATE)


def test_mindlin_moment_is_thickness_independent_for_uniform_load():
    """The SS-plate bending moment under uniform load does not depend on t.

    Classical result: for a uniformly-loaded simply-supported plate the moment
    field is fixed by the load distribution (the Kirchhoff and Mindlin
    solutions give the same Mx/My/Mxy at any t/a).  This falsifies the doc's
    "厚板需改用 Mindlin 级数弯矩" as a *moment*-side requirement -- the
    deflection side is the one that needs the Mindlin series.
    """
    center = _center()
    m_thin = navier_mindlin_foundation_moments(
        center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
        thickness=0.1, pressure=PRESSURE, n_modes=99,
    )
    for t in (0.5, 1.0, 2.0):                      # t/a = 0.05 .. 0.20
        m_thick = navier_mindlin_foundation_moments(
            center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
            thickness=t, pressure=PRESSURE, n_modes=99,
        )
        assert torch.allclose(m_thick, m_thin, rtol=1e-6, atol=1e-6 * D_PLATE)


def test_mindlin_deflection_does_depend_on_thickness():
    """Contrast: the *deflection* does need the Mindlin series for thick plates.

    At t/a = 0.2 the Kirchhoff deflection is ~17 % below the Mindlin one, so the
    doc's "厚板" caveat is real on the deflection side -- only the moment side
    is thickness-independent.
    """
    center = _center()
    dk = navier_foundation_deflection(
        center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
        thickness=2.0, pressure=PRESSURE, n_modes=99,
    )
    dm = navier_mindlin_foundation_deflection(
        center, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
        thickness=2.0, pressure=PRESSURE, n_modes=99,
    )
    rel = abs(float(dk[0]) - float(dm[0])) / float(dm[0])
    assert rel > 0.10                              # >10 % at t/a = 0.2


def test_fe_moments_match_mindlin_series_thick_plate():
    """FE vs Mindlin moment series for a thick plate (t/a = 0.1) < 3 %.

    Closes the ``GAP-FEM-WINKLER`` "厚板弯矩" verification debt: the FE
    (Mindlin Q4) moment field matches the Mindlin series to < 3 % even at
    t/a = 0.1, where the Kirchhoff series is only a O((t/a)^2) approximation
    of the FE kinematics on the deflection side.
    """
    report = compare_finite_plate_moments(
        nx=32, ny=32, thickness=1.0, pressure=PRESSURE, reference="mindlin",
    )
    assert report["max_mx_rel_error"] <= 0.03
    assert report["max_my_rel_error"] <= 0.03
    assert report["center_mx_rel_error"] <= 0.01
    assert report["mxy_rel_error"] <= 0.05
    assert report["max_w_rel_error"] <= 0.03
