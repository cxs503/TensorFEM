"""P9263-N01: Mindlin plate tangential contact stiffness + foundation-model switching.

Acceptance criteria exercised here (numbers from the campaign node):

    C1  simply supported square plate, uniform load: centre deflection within 5%
        of the exact first-order-shear-deformation (Mindlin-Timoshenko) Navier
        solution, for a Winkler *and* for a Pasternak foundation;
    C2  Winkler -> Pasternak switch: the Pasternak deflection is smaller and the
        FE ratio agrees with the exact ratio within 2%;
    C3  mesh refinement: error decreases monotonically with observed order >= 1.5;
    C4  one interface (:func:`make_foundation` + one solver) reaches Winkler,
        Pasternak and elastic half-space, with monotone stiffening limits;
    C5  negative controls fail as designed (wrong Gp sign, wrong reference in the
        thick-plate regime, legacy ad-hoc thick-plate estimate).

The reference oracle ``fsdt_navier_ss_deflection`` is an independent modal series;
the FE side is a 3-DOF-per-node Q4 Mitc assembly, so agreement is a real check and
not a re-evaluation of the same code path.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.tensorfem.elastic_halfspace import settlement as half_space_settlement
from src.tensorfem.plate import structured_square_mesh
from src.tensorfem.plate_foundation_analytical import (
    fsdt_ss_center_deflection,
    kirchhoff_ss_center,
    mindlin_ss_center_correction,
)
from src.tensorfem.plate_foundation_interface import (
    ElasticHalfSpaceFoundation,
    PasternakFoundation,
    WinklerFoundation,
    element_bending_moments,
    fsdt_navier_ss_deflection,
    half_space_patch_compliance,
    make_foundation,
    q4_mindlin_tangential_contact_stiffness,
    q4_normal_foundation_stiffness,
    q4_shear_layer_stiffness,
    solve_mindlin_plate_foundation,
)

# reference case: 1 m x 1 m steel panel, h/a = 0.05 (moderately thick)
E = 2.0e11
NU = 0.3
H = 0.05
A = 1.0
Q = 1.0e4
K_NORMAL = 1.0e9
G_SHEAR = 2.0e7
KAPPA = 5.0 / 6.0

EXACT_FLEXURAL = E * H ** 3 / (12.0 * (1.0 - NU ** 2))
EXACT_SHEAR_MODULUS = E / (2.0 * (1.0 + NU))
EXACT_SHEAR_RIGIDITY = KAPPA * EXACT_SHEAR_MODULUS * H


def solve(n: int, foundation=None, k_tangential: float = 0.0, thickness: float = H):
    return solve_mindlin_plate_foundation(
        n, side=A, young=E, poisson=NU, thickness=thickness, pressure=Q,
        foundation=foundation, tangential_contact=k_tangential, kappa=KAPPA)


def exact(k_normal: float = 0.0, g_shear: float = 0.0, k_tangential: float = 0.0,
          thickness: float = H) -> float:
    return fsdt_navier_ss_deflection(
        q=Q, a=A, b=A, young=E, poisson=NU, thickness=thickness, k_normal=k_normal,
        g_shear=g_shear, k_tangential=k_tangential, kappa=KAPPA, terms=161)["deflection"]


# ---------------------------------------------------------------------------
# element-level checks
# ---------------------------------------------------------------------------

def test_element_matrices_are_symmetric_and_physically_consistent():
    nodes, elems = structured_square_mesh(1, A)
    xy = nodes[elems[0]]
    k_w = q4_normal_foundation_stiffness(xy, K_NORMAL)
    k_s = q4_shear_layer_stiffness(xy, G_SHEAR)
    k_t = q4_mindlin_tangential_contact_stiffness(xy, 1.0e10, H)

    for ke in (k_w, k_s, k_t):
        assert ke.shape == (12, 12)
        assert torch.allclose(ke, ke.T, atol=1e-12)

    # Winkler block acts on w only and is positive definite there
    w_idx = torch.arange(4) * 3
    assert torch.allclose(k_w[1::3], torch.zeros_like(k_w[1::3]))
    assert float(torch.linalg.eigvalsh(k_w[w_idx[:, None], w_idx]).min()) > 0.0

    # the shear layer ignores a constant w field (rigid-body translation)
    constant_w = torch.zeros(12, dtype=torch.float64)
    constant_w[w_idx] = 1.0
    assert abs(float(constant_w @ k_s @ constant_w)) < 1e-12 * float(k_s.abs().max())

    # the tangential contact term lives on the rotation DOFs and scales with h^2
    assert torch.allclose(k_t[w_idx], torch.zeros_like(k_t[w_idx]))
    k_t_thin = q4_mindlin_tangential_contact_stiffness(xy, 1.0e10, H / 2.0)
    assert float(k_t.norm()) / float(k_t_thin.norm()) == pytest.approx(4.0, rel=1e-12)
    assert float(q4_mindlin_tangential_contact_stiffness(xy, 0.0, H).norm()) == 0.0


# ---------------------------------------------------------------------------
# C1: centre deflection against the exact FSDT (Timoshenko-Mindlin) Navier series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["winkler", "pasternak"])
def test_centre_deflection_matches_fsdt_navier(kind: str):
    model = (make_foundation("winkler", k_normal=K_NORMAL) if kind == "winkler"
             else make_foundation("pasternak", k_normal=K_NORMAL, g_shear=G_SHEAR))
    reference = exact(**({"k_normal": K_NORMAL} if kind == "winkler"
                         else {"k_normal": K_NORMAL, "g_shear": G_SHEAR}))
    w = solve(12, model).centre_deflection()
    rel = abs(w / reference - 1.0) * 100.0
    assert rel <= 5.0, f"{kind}: rel error {rel:.2f}% > 5%"


def test_thin_plate_limit_recovers_kirchhoff():
    """h/a = 0.005: the FSDT series must collapse onto the classical Navier value."""
    thin = 0.005
    D = E * thin ** 3 / (12.0 * (1.0 - NU ** 2))
    w_kirchhoff = kirchhoff_ss_center(Q, A, A, D, M=80, N=80)
    w_fsdt = exact(thickness=thin)
    assert abs(w_fsdt / w_kirchhoff - 1.0) < 2.0e-3


# ---------------------------------------------------------------------------
# C2: foundation-model switch Winkler -> Pasternak
# ---------------------------------------------------------------------------

def test_pasternak_is_stiffer_than_winkler_like_theory():
    w_w = solve(12, WinklerFoundation(K_NORMAL)).centre_deflection()
    w_p = solve(12, PasternakFoundation(K_NORMAL, G_SHEAR)).centre_deflection()
    ref_w, ref_p = exact(k_normal=K_NORMAL), exact(k_normal=K_NORMAL, g_shear=G_SHEAR)
    assert w_p < w_w
    assert (w_p / w_w) / (ref_p / ref_w) == pytest.approx(1.0, rel=2.0e-2)


def test_pasternak_zero_shear_reproduces_winkler_exactly():
    w_w = solve(12, WinklerFoundation(K_NORMAL)).centre_deflection()
    w_p = solve(12, PasternakFoundation(K_NORMAL, 0.0)).centre_deflection()
    assert w_p == pytest.approx(w_w, rel=1e-14)


# ---------------------------------------------------------------------------
# Mindlin tangential contact stiffness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k_t", [1.0e10, 1.0e11])
def test_tangential_contact_stiffness_matches_exact_series(k_t: float):
    """The rotation-DOF contact matrix is graded by the same exact solution."""
    w = solve(8, WinklerFoundation(K_NORMAL), k_tangential=k_t).centre_deflection()
    reference = exact(k_normal=K_NORMAL, k_tangential=k_t)
    rel = abs(w / reference - 1.0) * 100.0
    assert rel <= 5.0, f"k_t={k_t:g}: rel error {rel:.2f}%"


def test_tangential_contact_stiffens_monotonically():
    w = [solve(8, WinklerFoundation(K_NORMAL), k_tangential=k_t).centre_deflection()
         for k_t in (0.0, 1.0e9, 1.0e10, 1.0e11, 1.0e12)]
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1))


def test_bonded_limit_is_the_poisson_square_problem():
    """k_t -> infinity rigidly ties the section: only transverse shear survives.

    The governing equation collapses to ``S lap(w) + q = 0`` with ``w = 0`` on the
    edges, whose centre value is the classical membrane/torsion constant
    ``0.0736714 q a^2 / S`` (Timoshenko & Woinarski-Krieger; Saint-Venant torsion
    of a square bar).  That is an oracle independent of the plate series.
    """
    w_bonded = fsdt_navier_ss_deflection(
        q=Q, a=A, b=A, young=E, poisson=NU, thickness=H, k_tangential=math.inf,
        kappa=KAPPA, terms=161)["deflection"]
    assert w_bonded / (Q * A ** 2 / EXACT_SHEAR_RIGIDITY) == pytest.approx(0.0736714, rel=1e-4)


# ---------------------------------------------------------------------------
# C3: mesh convergence
# ---------------------------------------------------------------------------

def test_mesh_convergence_order_at_least_1_5():
    model = WinklerFoundation(K_NORMAL)
    reference = exact(k_normal=K_NORMAL)
    errors, sizes = [], []
    for n in (6, 8, 12):
        w = solve(n, model).centre_deflection()
        errors.append(abs(w / reference - 1.0))
        sizes.append(A / n)
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), errors
    order = math.log(errors[1] / errors[2]) / math.log(sizes[1] / sizes[2])
    assert order >= 1.5, f"observed order {order:.2f}"


# ---------------------------------------------------------------------------
# C4: unified interface, three foundation kinds
# ---------------------------------------------------------------------------

def test_factory_reaches_three_models_and_rejects_unknown_kinds():
    winkler = make_foundation("winkler", k_normal=K_NORMAL)
    pasternak = make_foundation("two_parameter", k_normal=K_NORMAL, g_shear=G_SHEAR)
    half_space = make_foundation("half-space", young=8.6e8, poisson=NU)
    assert (winkler.kind, pasternak.kind, half_space.kind) == ("winkler", "pasternak", "half_space")
    for model in (winkler, pasternak, half_space):
        sol = solve(6, model)
        assert sol.centre_deflection() > 0.0
    with pytest.raises(ValueError):
        make_foundation("kerr-ish")


def test_stiffening_limits_are_monotone_for_every_model():
    """k -> inf, Gp -> inf and Es -> inf must each drive the deflection to zero."""
    sequences = (
        [solve(6, WinklerFoundation(k)).centre_deflection() for k in (1e9, 1e11, 1e13)],
        [solve(6, PasternakFoundation(K_NORMAL, g)).centre_deflection()
         for g in (2e7, 2e9, 2e11)],
        [solve(6, ElasticHalfSpaceFoundation(e, NU)).centre_deflection()
         for e in (1e9, 1e11, 1e13)],
    )
    for seq in sequences:
        assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), seq
        assert seq[-1] / seq[0] < 0.1, seq


def test_half_space_model_reaches_the_boussinesq_settlement_factors():
    """The patch compliance must integrate the closed-form Boussinesq kernel.

    Reference values are obtained by an independent 2-D Gauss quadrature of
    ``settlement(p, E, nu, B, B, x, y)`` over the loaded square, not from the
    patch model itself: average factor 0.946402, centre factor 1.122200
    (the textbook flexible-square factors 0.95 / 1.12 are the rounded forms).
    """
    e_s, q, nu = 1.0e7, 1.0e5, NU
    import numpy as np
    gx, gw = np.polynomial.legendre.leggauss(20)
    total = 0.0
    for a_i, w_i in zip(gx, gw):
        for b_i, w_j in zip(gx, gw):
            total += w_i * w_j * half_space_settlement(
                q, e_s, nu, A, A, 0.5 * (a_i + 1.0), 0.5 * (b_i + 1.0))
    scale = q * A * (1.0 - nu ** 2) / e_s
    reference_average = total / 4.0 / scale
    reference_centre = half_space_settlement(q, e_s, nu, A, A, 0.5, 0.5) / scale
    assert reference_average == pytest.approx(0.9464, abs=2e-3)
    assert reference_centre == pytest.approx(1.1222, abs=2e-3)

    centre_errors = []
    for n in (8, 12):
        nodes, elems = structured_square_mesh(n, A)
        c_p = half_space_patch_compliance(nodes, elems, e_s, nu)
        xy = nodes[elems]
        area = ((xy[:, :, 0].max(1).values - xy[:, :, 0].min(1).values)
                * (xy[:, :, 1].max(1).values - xy[:, :, 1].min(1).values))
        # Betti reciprocity: A_i C_ij is symmetric
        weighted = area[:, None] * c_p
        assert float((weighted - weighted.T).abs().max() / weighted.abs().max()) < 1e-12
        w_patch = c_p @ torch.full((elems.shape[0],), q, dtype=torch.float64)
        average = float((w_patch * area).sum() / area.sum()) / scale
        assert average == pytest.approx(reference_average, rel=2e-3)
        centres = xy.mean(dim=1)
        idx = int(torch.argmin(torch.linalg.norm(
            centres - torch.tensor([0.5, 0.5], dtype=torch.float64), dim=1)))
        centre_value = float(w_patch[idx]) / scale
        centre_errors.append(abs(centre_value / reference_centre - 1.0))
        # a patch average is not a point value: 2% is the resolution of the
        # coarsest patch, and it must shrink with refinement
        assert centre_value == pytest.approx(reference_centre, rel=2e-2)
    assert centre_errors[1] < centre_errors[0]


def test_bending_moments_recover_the_patch_field():
    sol = solve(8, WinklerFoundation(K_NORMAL))
    moments = element_bending_moments(sol)
    assert moments["Mx"].shape == (64,)
    # Mx must be extreme where the curvature is: at the plate centre for this case
    centres = sol.nodes[sol.elements].mean(dim=1)
    idx = int(torch.argmin(torch.linalg.norm(
        centres - torch.tensor([0.5, 0.5], dtype=torch.float64), dim=1)))
    assert abs(float(moments["Mx"][idx])) == pytest.approx(
        float(moments["Mx"].abs().max()), rel=1e-6)


# ---------------------------------------------------------------------------
# C5: negative controls
# ---------------------------------------------------------------------------

class _WrongSignShear(PasternakFoundation):
    """Negative control: Pasternak element matrix with the shear sign flipped."""

    def element_stiffness(self, xy, *, thickness: float = 0.0):
        return (q4_normal_foundation_stiffness(xy, self.k_normal)
                - q4_shear_layer_stiffness(xy, self.g_shear))


def test_negative_control_wrong_shear_sign_must_fail_the_gate():
    w_winkler = solve(12, WinklerFoundation(K_NORMAL)).centre_deflection()
    w_wrong = solve(12, _WrongSignShear(K_NORMAL, G_SHEAR)).centre_deflection()
    assert w_wrong > w_winkler, "a wrong-sign shear layer must soften the plate"
    assert abs(w_wrong / exact(k_normal=K_NORMAL) - 1.0) * 100.0 > 5.0


def test_negative_control_kirchhoff_reference_is_wrong_for_thick_plates():
    """h/a = 0.2: the FE matches FSDT but is off by > 15% against Kirchhoff."""
    thick = 0.2
    w_fem = solve(12, WinklerFoundation(K_NORMAL), thickness=thick).centre_deflection()
    ref_fsdt = exact(k_normal=K_NORMAL, thickness=thick)
    D = E * thick ** 3 / (12.0 * (1.0 - NU ** 2))
    ref_kirchhoff = kirchhoff_ss_center(Q, A, A, D, M=80, N=80)
    assert abs(w_fem / ref_fsdt - 1.0) * 100.0 <= 5.0
    assert abs(w_fem / ref_kirchhoff - 1.0) * 100.0 > 15.0


def test_legacy_thick_plate_estimate_is_now_exact():
    """``mindlin_ss_center_correction`` used to hard-code nu = 0.3 and C_s = 0.384.

    At h/a = 0.2 that ad-hoc estimate is 14% high; the function now returns the
    exact FSDT double series (equal to the module-level oracle), so the two
    independent implementations agree and the old value is demonstrably wrong.
    """
    thick = 0.2
    D = E * thick ** 3 / (12.0 * (1.0 - NU ** 2))
    shear_modulus = E / (2.0 * (1.0 + NU))
    S = KAPPA * shear_modulus * thick
    exact_value = exact(thickness=thick)
    legacy_now = mindlin_ss_center_correction(Q, A, A, D, thick, shear_modulus)
    legacy_old_style = (kirchhoff_ss_center(Q, A, A, D, M=80, N=80)
                        + Q * A ** 2 / S * 0.384 * (1.0 - 0.3) / 2.0)
    assert legacy_now == pytest.approx(exact_value, rel=1e-6)
    assert abs(legacy_old_style / exact_value - 1.0) > 0.10
    # the shear deflection really is ~20% of the total at h/a = 0.2
    assert exact_value / kirchhoff_ss_center(Q, A, A, D, M=80, N=80) == pytest.approx(1.207, abs=5e-3)


def test_navier_series_limits_are_consistent():
    """k_t = 0 must reproduce the classical Mindlin (bending + shear) series."""
    terms = 41
    w_formula = fsdt_navier_ss_deflection(
        q=Q, a=A, b=A, young=E, poisson=NU, thickness=H, k_normal=K_NORMAL,
        kappa=KAPPA, terms=terms)["deflection"]
    D = EXACT_FLEXURAL
    S = EXACT_SHEAR_RIGIDITY
    w_classical = 0.0
    for m in range(1, 2 * terms, 2):
        am = m * math.pi / A
        sm = 1.0 if m % 4 == 1 else -1.0
        for n in range(1, 2 * terms, 2):
            lam = am ** 2 + (n * math.pi / A) ** 2
            qmn = 16.0 * Q / (math.pi ** 2 * m * n)
            sn = 1.0 if n % 4 == 1 else -1.0
            w_classical += qmn / (K_NORMAL + D * lam ** 2 / (1.0 + D * lam / S)) * sm * sn
    assert w_formula == pytest.approx(w_classical, rel=1e-12)
    # the pure-python oracle in plate_foundation_analytical agrees as well
    assert fsdt_ss_center_deflection(Q, A, A, D, H, EXACT_SHEAR_MODULUS,
                                    k_normal=K_NORMAL, M=terms,
                                    N=terms) == pytest.approx(w_formula, rel=1e-12)