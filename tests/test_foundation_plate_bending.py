"""Analytic-numerical closure for plate bending on Winkler / Pasternak / Kerr foundations.

Three independent verification layers are exercised here:

* **closed forms** (Hetényi ``kei`` solution, two-parameter ``K0`` solution) are
  checked against a Hankel quadrature oracle that does not use them, and
  against ``mpmath``/``scipy`` Kelvin-function values,
* **element matrices** are checked against the scalar reference implementations
  of ``tensorfem.plate`` / ``tensorfem.winkler_foundation`` to machine
  precision, plus analytic energy identities,
* **the FE model** is checked against the closed forms (5 radii), against the
  exact Mindlin Navier series, against the Archimedes draft for buoyancy, and
  through an h-refinement GCI study.

The physical test bed is the marine case unless stated otherwise: a steel plate
``E = 210 GPa``, ``nu = 0.3``, ``t = 0.05 m`` in sea water ``rho = 1025 kg/m3``,
whose buoyancy coefficient is ``k = rho g = 10055 N/m3``.
"""
import math

import pytest
import torch

from tensorfem.foundation_plate_bending import (
    batched_plate_foundation_element_stiffness,
    compare_infinite_plate_point_load,
    deflection_on_grid,
    finite_plate_foundation_fem,
    graded_axis,
    graded_square_mesh,
    infinite_plate_point_load_fem,
    infinite_plate_two_parameter_point_load,
    infinite_plate_winkler_point_load,
    k0_complex,
    kerr_equivalent_pasternak,
    navier_foundation_deflection,
    navier_mindlin_foundation_deflection,
    pasternak_foundation_roots,
    plate_foundation_element_stiffness,
    plate_rigidity,
    q4_pasternak_shear_stiffness,
    solve_assembled_system,
    winkler_characteristic_length,
)
from tensorfem.convergence_verification import ConvergenceLevel, evaluate_convergence_study
from tensorfem.plate import q4_mindlin_stiffness
from tensorfem.winkler_foundation import q4_winkler_foundation_stiffness

D_T = torch.float64

# Marine test bed
RHO_W, G_ACC, RHO_S = 1025.0, 9.81, 7850.0
YOUNG, POISSON, THICK = 2.1e11, 0.3, 0.05
K_BUOY = RHO_W * G_ACC
D_PLATE = plate_rigidity(YOUNG, POISSON, THICK)
L_CHAR = winkler_characteristic_length(D_PLATE, K_BUOY)


# ---------------------------------------------------------------------------
# Independent Hankel-quadrature oracle (does not use the closed forms)
# ---------------------------------------------------------------------------

def hankel_oracle(r: float, load: float, rigidity: float, winkler: float, pasternak: float) -> float:
    """``w(r) = P/(2 pi) int_0^inf J0(a r) a /(D a^4 + G a^2 + k) da``.

    The integral is taken half-oscillation by half-oscillation of ``J0`` (the
    interval length is ``pi/r``), which keeps every ``quad`` call on a smooth,
    sign-definite integrand, and is truncated where the ``1/(D a^4)`` tail is
    provably negligible.
    """
    scipy_integrate = pytest.importorskip("scipy.integrate")
    scipy_special = pytest.importorskip("scipy.special")

    def integrand(alpha: float) -> float:
        return (scipy_special.j0(alpha * r) * alpha
                / (rigidity * alpha**4 + pasternak * alpha**2 + winkler))

    a2, b2 = pasternak_foundation_roots(rigidity, winkler, pasternak)
    root_scale = max(abs(a2), abs(b2))
    # r = 0 is the least oscillatory case: one interval is enough
    cutoff = 80.0 * root_scale if r == 0.0 else max(80.0 * root_scale, 80.0 / r)
    step = math.pi / r if r > 0.0 else cutoff
    total = 0.0
    edges = [0.0]
    while edges[-1] < cutoff:
        edges.append(min(edges[-1] + step, cutoff))
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += scipy_integrate.quad(integrand, lo, hi, limit=200)[0]
    return load / (2.0 * math.pi) * total


def hankel_tail_bound(r: float, load: float, rigidity: float, winkler: float, pasternak: float) -> float:
    """Bound on the neglected tail ``int_cutoff^inf |integrand| da <= 1/(2 D cutoff^2)``."""
    a2, b2 = pasternak_foundation_roots(rigidity, winkler, pasternak)
    root_scale = max(abs(a2), abs(b2))
    cutoff = 80.0 * root_scale if r == 0.0 else max(80.0 * root_scale, 80.0 / r)
    return load / (2.0 * math.pi) / (2.0 * rigidity * cutoff**2)


# ---------------------------------------------------------------------------
# 1. Rigidity, characteristic length, foundation algebra
# ---------------------------------------------------------------------------

def test_plate_rigidity_and_characteristic_length():
    t = 0.05
    expected_d = YOUNG * t**3 / (12.0 * (1.0 - POISSON**2))
    assert plate_rigidity(YOUNG, POISSON, t) == pytest.approx(expected_d, rel=1e-14)
    expected_l = (expected_d / K_BUOY) ** 0.25
    assert winkler_characteristic_length(expected_d, K_BUOY) == pytest.approx(expected_l, rel=1e-14)
    # A 50 mm steel plate in water: the Hetényi length is a few metres.
    assert 3.0 < L_CHAR < 5.0
    with pytest.raises(ValueError):
        plate_rigidity(-1.0, 0.3, 0.05)
    with pytest.raises(ValueError):
        winkler_characteristic_length(1.0, 0.0)


def test_kerr_reduction_limits():
    k_u, k_l, g_p = 4.0e7, 2.0e7, 3.0e7
    k_eq, g_eq = kerr_equivalent_pasternak(k_u, k_l, g_p)
    assert k_eq == pytest.approx(k_u * k_l / (k_u + k_l), rel=1e-15)
    assert g_eq == pytest.approx(k_u * g_p / (k_u + k_l), rel=1e-15)
    assert k_eq < min(k_u, k_l)                      # springs in series soften
    # G_p = 0 -> plain Winkler with the series coefficient
    assert kerr_equivalent_pasternak(k_u, k_l, 0.0) == pytest.approx((k_eq, 0.0))
    # plate tied to the shear layer (k_u -> inf) -> Pasternak (k_l, G_p)
    k_rigid, g_rigid = kerr_equivalent_pasternak(1.0e14, k_l, g_p)
    assert k_rigid == pytest.approx(k_l, rel=1e-6)
    assert g_rigid == pytest.approx(g_p, rel=1e-6)
    # rigid base (k_l -> inf) pins the shear layer: no spreading, G_eq -> 0
    k_base, g_base = kerr_equivalent_pasternak(k_u, 1.0e14, g_p)
    assert k_base == pytest.approx(k_u, rel=1e-6)
    assert g_base < 1e-6 * g_p
    with pytest.raises(ValueError):
        kerr_equivalent_pasternak(0.0, k_l, g_p)
    with pytest.raises(ValueError):
        kerr_equivalent_pasternak(k_u, k_l, -1.0)


# ---------------------------------------------------------------------------
# 2. Closed-form references vs independent oracles
# ---------------------------------------------------------------------------

def test_winkler_centre_deflection_closed_form():
    """``w(0) = P/(8 sqrt(D k))`` is the Hetényi centre deflection."""
    load = 2.0e5
    w0 = infinite_plate_winkler_point_load(torch.zeros(1), load, D_PLATE, K_BUOY)
    assert float(w0[0]) == pytest.approx(load / (8.0 * math.sqrt(D_PLATE * K_BUOY)), rel=1e-12)


@pytest.mark.parametrize("r_over_l", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_two_parameter_matches_hankel_quadrature(r_over_l):
    """The closed form must equal the Hankel integral to far below 1e-5."""
    load = 2.0e5
    pstd = 3.0e7                               # above 2 sqrt(k D): real roots branch
    r = r_over_l * L_CHAR
    closed = float(
        infinite_plate_two_parameter_point_load(
            torch.tensor([r]), load, D_PLATE, K_BUOY, pstd
        )[0]
    )
    oracle = hankel_oracle(r, load, D_PLATE, K_BUOY, pstd)
    bound = hankel_tail_bound(r, load, D_PLATE, K_BUOY, pstd)
    assert abs(closed - oracle) <= 1e-5 * abs(oracle) + bound


def test_two_parameter_zero_shear_reduces_to_hetenyi():
    """``G -> 0`` must reproduce the ``kei`` solution, including the uplift ring."""
    load, r = 2.0e5, torch.tensor([0.0, 0.3, 1.0, 2.0, 3.0]) * L_CHAR
    winkler_only = infinite_plate_winkler_point_load(r, load, D_PLATE, K_BUOY)
    two_parameter = infinite_plate_two_parameter_point_load(
        r, load, D_PLATE, K_BUOY, 1.0e-6
    )
    assert torch.allclose(two_parameter, winkler_only, rtol=1e-8, atol=0.0)
    # Hetényi's solution has a shallow uplift ring at some radius; the
    # two-parameter solution does not once the shear layer is active (the
    # reaction is spread, so the plate never lifts off).
    scan = torch.linspace(0.0, 12.0, 4001) * L_CHAR
    winkler_scan = infinite_plate_winkler_point_load(scan, load, D_PLATE, K_BUOY)
    assert float(winkler_scan.min()) < 0.0
    two_parameter_scan = infinite_plate_two_parameter_point_load(
        scan, load, D_PLATE, K_BUOY, 3.0e7
    )
    assert float(two_parameter_scan.min()) > 0.0


def test_two_parameter_roots_and_degenerate_guard():
    rigidity, winkler = 2.7e6, 1.0e7
    # G = 2 sqrt(k D) is the double root
    critical = 2.0 * math.sqrt(winkler * rigidity)
    a, b = pasternak_foundation_roots(rigidity, winkler, critical * (1.0 + 1e-3))
    assert a.imag == 0.0 and b.imag == 0.0 and a.real > b.real > 0.0
    a, b = pasternak_foundation_roots(rigidity, winkler, 0.5 * critical)
    assert abs(a.imag) > 0.0 and abs(b.imag) > 0.0
    assert b.conjugate() == a
    with pytest.raises(ValueError):  # exactly degenerate: closed form is useless
        infinite_plate_two_parameter_point_load(
            torch.tensor([1.0]), 1.0, rigidity, winkler, critical
        )


def test_k0_complex_accuracy():
    """Complex ``K0`` against SciPy's real-argument values (both series branches)."""
    special = pytest.importorskip("scipy.special")
    for x in (0.5, 2.0, 8.0, 9.4, 9.6, 15.0, 25.0):
        got = k0_complex(torch.tensor([complex(x, 0.0)], dtype=torch.complex128))[0].real
        ref = special.k0(x)
        assert abs(float(got) - float(ref)) <= 1e-6 * abs(float(ref))


# ---------------------------------------------------------------------------
# 3. Element matrices
# ---------------------------------------------------------------------------

def _quad_element(side: float = 0.4) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]], dtype=D_T
    )


@pytest.mark.parametrize("shear_treatment", ["mitc", "sri"])
@pytest.mark.parametrize("winkler,pasternak", [(0.0, 0.0), (1.0e7, 0.0), (0.0, 3.0e6), (1.0e7, 3.0e6)])
def test_batched_element_stiffness_matches_scalar(shear_treatment, winkler, pasternak):
    """The vectorised element must reproduce the scalar reference assemblies."""
    nodes, elements = graded_square_mesh(4, 3.0, 2.0)
    xy = nodes[elements]
    batch = batched_plate_foundation_element_stiffness(
        xy, YOUNG, POISSON, THICK,
        winkler=winkler, pasternak=pasternak, shear_treatment=shear_treatment,
    )
    for e in range(xy.shape[0]):
        scalar = q4_mindlin_stiffness(
            xy[e], YOUNG, POISSON, THICK, shear_treatment=shear_treatment
        )
        if winkler:
            scalar = scalar + q4_winkler_foundation_stiffness(xy[e], winkler)
        if pasternak:
            scalar = scalar + q4_pasternak_shear_stiffness(xy[e], pasternak)
        assert float((batch[e] - scalar).abs().max()) <= 1e-12 * float(scalar.abs().max())


def test_pasternak_element_energy_identity():
    """A linear field ``w = c x`` stores exactly ``G c^2 A`` of shear-layer energy."""
    xy = _quad_element(0.4)
    g, c = 3.0e6, 2.5e-3
    kg = q4_pasternak_shear_stiffness(xy, g)
    assert float(kg.abs().max()) > 0.0
    u = torch.zeros(12, dtype=D_T)
    u[0::3] = c * xy[:, 0]                       # w = c*x, rotations zero
    energy = 0.5 * float(u @ kg @ u)
    area = 0.4**2
    assert energy == pytest.approx(0.5 * g * c**2 * area, rel=1e-12)
    # constant w carries no energy, and G = 0 gives the zero matrix
    u_const = torch.zeros(12, dtype=D_T)
    u_const[0::3] = 1.0
    # constant w is a zero-energy mode: residual is round-off on the g*A scale
    assert abs(float(u_const @ kg @ u_const)) < 1e-13 * g * area
    assert float(q4_pasternak_shear_stiffness(xy, 0.0).abs().max()) == 0.0
    with pytest.raises(ValueError):
        q4_pasternak_shear_stiffness(xy, -1.0)


def test_element_stiffness_is_symmetric():
    xy = _quad_element(0.25)
    ke = plate_foundation_element_stiffness(
        xy, YOUNG, POISSON, THICK, winkler=1.0e7, pasternak=3.0e6
    )
    assert float((ke - ke.T).abs().max()) <= 1e-10 * float(ke.abs().max())


# ---------------------------------------------------------------------------
# 4. Infinite plate: FE vs Hetényi / Kerr / buoyancy
# ---------------------------------------------------------------------------

def test_five_radii_vs_hetenyi_at_2pct():
    """Acceptance (a): FE vs Hetényi at 5 radii must be within 2 %."""
    report = compare_infinite_plate_point_load(
        (0.5, 1.0, 1.5, 2.0, 3.0),
        load=2.0e5, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=1.0e7, n=40,
    )
    assert len(report["rows"]) == 5
    for row in report["rows"]:
        assert row["rel_error"] <= 0.02, row
    assert report["max_rel_error"] <= 0.02
    # load path: the foundation reaction reproduces the applied load
    assert report["equilibrium_rel_error"] <= 1e-8


def test_buoyancy_archimedes_draft():
    """Own weight of a floating plate gives the Archimedes draft ``rho_s t / rho_w``."""
    own_weight = RHO_S * G_ACC * THICK
    solution = infinite_plate_point_load_fem(
        0.0, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=K_BUOY, pressure=own_weight, n=30,
    )
    draft = float(solution.deflection_at(torch.zeros(1, 2))[0])
    assert draft == pytest.approx(own_weight / K_BUOY, rel=1e-9)
    assert draft == pytest.approx(RHO_S * THICK / RHO_W, rel=1e-9)
    # every point of the plate floats at the same draft (uniform particular solution)
    far = float(solution.deflection_at(torch.tensor([[10.0 * L_CHAR, 0.0]]))[0])
    assert far == pytest.approx(draft, rel=1e-8)


def test_buoyancy_point_load_within_3pct():
    """Acceptance (c): buoyancy coefficient k = rho g, FE vs closed form <= 3 %."""
    load = 2.0e5
    solution = infinite_plate_point_load_fem(
        load, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=K_BUOY, pressure=RHO_S * G_ACC * THICK, n=40,
    )
    radii = torch.tensor([0.25, 0.5, 1.0, 2.0, 3.0]) * L_CHAR
    points = torch.stack((radii, torch.zeros_like(radii)), dim=1)
    fem = solution.deflection_at(points)
    analytic = (
        infinite_plate_winkler_point_load(radii, load, D_PLATE, K_BUOY)
        + RHO_S * G_ACC * THICK / K_BUOY
    )
    rel = (fem - analytic).abs() / analytic
    assert float(rel.max()) <= 0.03, [float(v) for v in rel]
    assert abs(solution.foundation_reaction / solution.applied_load - 1.0) <= 1e-8


def test_kerr_beam_on_foundation_within_2pct():
    """Acceptance (b): Kerr three-parameter foundation reduced to Pasternak <= 2 %."""
    load = 2.0e5
    k_upper, k_lower, g_shear = 4.0e7, 2.0e7, 6.0e7
    k_eq, g_eq = kerr_equivalent_pasternak(k_upper, k_lower, g_shear)
    report = compare_infinite_plate_point_load(
        (0.5, 1.0, 1.5, 2.0, 3.0),
        load=load, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=k_eq, pasternak=g_eq, n=40,
    )
    for row in report["rows"]:
        assert row["rel_error"] <= 0.02, row
    assert report["equilibrium_rel_error"] <= 1e-8


def test_fe_solution_insensitive_to_truncation_and_mesh():
    """The 0.4 % agreement is a discretisation, not a truncation, artefact."""
    base = compare_infinite_plate_point_load((0.5, 1.0, 2.0), load=2.0e5,
                                             young=YOUNG, poisson=POISSON,
                                             thickness=THICK, winkler=1.0e7, n=40)
    wide = compare_infinite_plate_point_load((0.5, 1.0, 2.0), load=2.0e5,
                                             young=YOUNG, poisson=POISSON, thickness=THICK,
                                             winkler=1.0e7, n=40,
                                             half_width_in_lengths=30.0)
    fine = compare_infinite_plate_point_load((0.5, 1.0, 2.0), load=2.0e5,
                                             young=YOUNG, poisson=POISSON, thickness=THICK,
                                             winkler=1.0e7, n=60)
    for r_base, r_wide, r_fine in zip(base["rows"], wide["rows"], fine["rows"]):
        assert abs(r_wide["fem"] - r_base["fem"]) / r_base["fem"] <= 0.01
        assert abs(r_fine["fem"] - r_base["fem"]) / r_base["fem"] <= 0.01


def test_graded_mesh_is_valid():
    axis = graded_axis(20, 10.0, 4.0)
    assert float(axis[0]) == 0.0
    assert float(axis[-1]) == pytest.approx(10.0, rel=1e-14)
    steps = (axis[1:] - axis[:-1]).diff()
    assert bool((axis[1:] - axis[:-1] > 0).all())
    assert bool((steps > 0).all())              # strictly growing element size
    nodes, elements = graded_square_mesh(12, 5.0, 3.0)
    assert nodes.shape == (169, 2) and elements.shape == (144, 4)
    # every element keeps a positive Jacobian in the assembly
    solve = infinite_plate_point_load_fem(
        1.0e5, young=YOUNG, poisson=POISSON, thickness=THICK, winkler=1.0e7, n=12
    )
    assert torch.isfinite(solve.deflection).all()


# ---------------------------------------------------------------------------
# 5. Finite plate: Navier series vs FE
# ---------------------------------------------------------------------------

def test_kirchhoff_navier_reproduces_classical_coefficient():
    """SS square, uniform load: ``w_max = 0.00406 q a^4 / D`` (Kirchhoff)."""
    q, a = 1.0e5, 1.0
    w = float(navier_foundation_deflection(
        torch.tensor([[0.5 * a, 0.5 * a]]), side_x=a, side_y=a, pressure=q,
        young=YOUNG, poisson=POISSON, thickness=THICK, n_modes=199,
    )[0])
    classical = 0.00406 * q * a**4 / D_PLATE
    assert w == pytest.approx(classical, rel=2e-3)


def test_navier_series_convergence():
    q, a = 1.0e5, 1.0
    centre = torch.tensor([[0.5 * a, 0.5 * a]])
    coarse = float(navier_foundation_deflection(centre, side_x=a, side_y=a, pressure=q,
                                                winkler=2.0e7, n_modes=99)[0])
    fine = float(navier_foundation_deflection(centre, side_x=a, side_y=a, pressure=q,
                                              winkler=2.0e7, n_modes=399)[0])
    assert abs(fine - coarse) <= 1e-8 * abs(fine)


def test_finite_plate_fem_vs_mindlin_navier_at_2pct():
    """Acceptance (d): FEM vs Navier series <= 2 % (thick plate, hard SS)."""
    q, a, k = 1.0e5, 1.0, 2.0e7
    centre = torch.tensor([[0.5 * a, 0.5 * a]])
    reference = float(navier_mindlin_foundation_deflection(
        centre, side_x=a, side_y=a, young=YOUNG, poisson=POISSON, thickness=THICK,
        pressure=q, winkler=k, n_modes=199,
    )[0])
    fem = float(finite_plate_foundation_fem(
        side_x=a, side_y=a, nx=24, ny=24, young=YOUNG, poisson=POISSON,
        thickness=THICK, pressure=q, winkler=k,
    ).deflection_at(centre)[0])
    assert abs(fem - reference) / reference <= 0.02, (fem, reference)


def test_thin_plate_limit_gives_kirchhoff():
    """``t -> 0``: Mindlin Navier and FEM must both collapse onto Kirchhoff."""
    q, a = 1.0e5, 1.0
    thin = 0.005
    centre = torch.tensor([[0.5 * a, 0.5 * a]])
    kirchhoff = float(navier_foundation_deflection(
        centre, side_x=a, side_y=a, young=YOUNG, poisson=POISSON, thickness=thin,
        pressure=q, n_modes=199,
    )[0])
    mindlin = float(navier_mindlin_foundation_deflection(
        centre, side_x=a, side_y=a, young=YOUNG, poisson=POISSON, thickness=thin,
        pressure=q, n_modes=199,
    )[0])
    fem = float(finite_plate_foundation_fem(
        side_x=a, side_y=a, nx=20, ny=20, young=YOUNG, poisson=POISSON,
        thickness=thin, pressure=q,
    ).deflection_at(centre)[0])
    assert abs(mindlin - kirchhoff) / kirchhoff <= 0.01
    assert abs(fem - kirchhoff) / kirchhoff <= 0.02


def test_finite_plate_point_load_matches_series():
    """A central point load: FE vs the Mindlin Navier series (slow but convergent)."""
    a, p = 1.0, 5.0e4
    centre = torch.tensor([[0.5 * a, 0.5 * a]])
    reference = float(navier_mindlin_foundation_deflection(
        centre, side_x=a, side_y=a, young=YOUNG, poisson=POISSON, thickness=0.05,
        point_load=p, n_modes=399,
    )[0])
    fem = float(finite_plate_foundation_fem(
        side_x=a, side_y=a, nx=24, ny=24, young=YOUNG, poisson=POISSON,
        thickness=0.05, point_load=p,
    ).deflection_at(centre)[0])
    assert abs(fem - reference) / abs(reference) <= 0.03, (fem, reference)


def test_grid_convergence_index_under_2pct():
    """Acceptance (e): h-refinement GCI of the foundation plate centre deflection."""
    q, a, k = 1.0e5, 1.0, 2.0e7
    centre = torch.tensor([[0.5 * a, 0.5 * a]])
    reference = float(navier_mindlin_foundation_deflection(
        centre, side_x=a, side_y=a, young=YOUNG, poisson=POISSON, thickness=THICK,
        pressure=q, winkler=k, n_modes=299,
    )[0])
    levels = []
    for n in (8, 16, 32):
        value = float(finite_plate_foundation_fem(
            side_x=a, side_y=a, nx=n, ny=n, young=YOUNG, poisson=POISSON,
            thickness=THICK, pressure=q, winkler=k,
        ).deflection_at(centre)[0])
        levels.append(ConvergenceLevel(1.0 / n, value, f"n={n}"))
    report = evaluate_convergence_study(
        levels, qoi="centre deflection", expected_order=2.0, reference_value=reference
    )
    assert report["passed"] is True, report
    assert report["gci"]["relative_gci"] <= 0.02
    assert abs(report["finest_order"] - 2.0) <= 0.2
    # the Richardson value must agree with the exact series to well under 1 %
    assert abs(report["gci"]["richardson_value"] - reference) / reference <= 0.01


def test_dense_and_sparse_solvers_agree():
    """The scipy path (large mesh) and the torch path (small mesh) must coincide."""
    nodes, elements = graded_square_mesh(6, 3.0, 3.0)
    from tensorfem.foundation_plate_bending import assemble_plate_foundation_system
    system = assemble_plate_foundation_system(
        nodes, elements, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=1.0e7, pressure=1.0e4,
    )
    fixed = torch.tensor([1, 2, 4], dtype=torch.long)
    dense = solve_assembled_system(system, fixed, dense_limit=10**6)
    sparse = solve_assembled_system(system, fixed, dense_limit=1)
    assert torch.allclose(dense, sparse, rtol=1e-10, atol=1e-14)


def test_interpolation_helpers():
    nodes, elements = graded_square_mesh(4, 2.0, 2.0)
    w = torch.zeros(3 * nodes.shape[0], dtype=D_T)
    w[0::3] = nodes[:, 0] + 2.0 * nodes[:, 1]   # exact bilinear field
    points = torch.tensor([[0.3, 0.7], [1.1, 0.2], [0.0, 0.0]], dtype=D_T)
    values = deflection_on_grid(nodes, w, 4, 4, points)
    expected = points[:, 0] + 2.0 * points[:, 1]
    assert torch.allclose(values, expected, rtol=1e-12, atol=1e-15)