"""Plate bending on Winkler / Pasternak / Kerr foundations: analytic anchors + FE closure.

The module closes the analytic-numerical loop for foundation-coupled plate
bending, i.e. it provides, side by side,

* **closed-form reference solutions** for an *infinite* Kirchhoff plate resting
  on a foundation and loaded by a point load ``P``,
* a **Mindlin-Reissner Q4 finite-element reference model** of the *same*
  boundary-value problem (quarter model, graded mesh, free far field), and
* a **Navier double-sine series** for a *finite*, simply supported plate on the
  same foundation (the finite-domain anchor used to qualify the FE mesh).

Foundation models
-----------------
Winkler (one parameter)::

    p = k w                        D grad^4 w + k w = q

Pasternak / Vlasov (two parameters, shear layer ``G``)::

    p = k w - G lap w              D grad^4 w - G lap w + k w = q

Kerr (three parameters; two spring layers ``k_u`` (in contact with the plate)
and ``k_l`` (on the rigid base) separated by a shear layer ``G_p``) reduces to
the Pasternak class through :func:`kerr_equivalent_pasternak`; the reduction is
an algebraic identity of the Kerr idealisation and its limits are asserted in
the test suite (``G_p = 0`` -> Winkler, ``k_l -> inf`` -> Pasternak ``(k_u, G_p)``).

Closed-form references
----------------------
Winkler, point load at the origin (Hetényi 1946), with ``l = (D/k)^(1/4)``::

    w(r) = (P l^2 / (2 pi D)) * (-kei(r/l)) = -(P/(2 pi k l^2)) kei(r/l)

so ``w(0) = P/(8 sqrt(D k))``.  This is exactly
:func:`tensorfem.hetenyi_reference.infinite_plate_point_load` written with the
flexural rigidity.

Two-parameter (Pasternak), point load at the origin.  With ``a^2 + b^2 = G/D``
and ``a^2 b^2 = k/D`` the Hankel integral
``w = P/(2 pi) int J0(ar) xi /(D xi^4 + G xi^2 + k) dxi`` evaluates in closed
form to

    w(r) = (P / (2 pi D)) * [K0(a r) - K0(b r)] / (b^2 - a^2).

Both roots are real once ``G >= 2 sqrt(k D)``; otherwise ``a = conj(b)`` and the
expression is evaluated with the principal-branch complex Kelvin function, which
for ``G -> 0`` collapses *identically* to the Hetényi ``kei`` form above (the
``k``-only limit is checked in the tests).  Unlike the Winkler solution the
two-parameter solution has no uplift ring when the roots are real: the shear
layer spreads the reaction, so the plate stays non-negative.

Buoyancy coupling
-----------------
A floating plate is a Winkler plate with ``k = rho g`` (hydrostatic restoring
pressure per unit deflection) and, as a body load, its own submerged weight per
unit area ``q_w = rho_s g t``.  For a uniform pressure the infinite-plate
solution is the constant ``w = q_w/k = rho_s t / rho`` -- i.e. the Archimedes
draft -- which is used as the buoyancy self-consistency anchor.

Conventions
-----------
* ``w > 0`` downward, ``P > 0`` downward; the plate rigidities are Kirchhoff
  values ``D = E t^3 / (12 (1 - nu^2))``.
* The foundation energy is ``1/2 int (k w^2 + G |grad w|^2) dA`` so both
  foundation element matrices are positive semi-definite.
* FE dofs per node are ``(w, theta_x, theta_y)`` with ``gamma_xz = dw/dx +
  theta_x`` and ``gamma_yz = dw/dy + theta_y`` (the convention of
  :mod:`tensorfem.plate`).

Everything here is float64, torch-native; the only optional dependency is SciPy
for the sparse direct solve of large meshes (a clear ``RuntimeError`` is raised
when it is missing and the mesh is too large to solve densely).
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import torch

from .hetenyi_reference import _K0_ASYMPTOTIC_COEFFS, kelvin_kei
from .plate import _shape, q4_mindlin_stiffness
from .winkler_foundation import q4_winkler_foundation_stiffness

__all__ = [
    "plate_rigidity",
    "winkler_characteristic_length",
    "kerr_equivalent_pasternak",
    "k0_complex",
    "infinite_plate_winkler_point_load",
    "pasternak_foundation_roots",
    "infinite_plate_two_parameter_point_load",
    "q4_pasternak_shear_stiffness",
    "plate_foundation_element_stiffness",
    "batched_plate_foundation_element_stiffness",
    "batched_uniform_pressure_load",
    "graded_square_mesh",
    "assemble_plate_foundation_system",
    "solve_assembled_system",
    "FoundationPlateSolution",
    "infinite_plate_point_load_fem",
    "finite_plate_foundation_fem",
    "uniform_rect_mesh",
    "product_mesh",
    "deflection_on_grid",
    "navier_foundation_deflection",
    "navier_mindlin_foundation_deflection",
    "compare_infinite_plate_point_load",
]

_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Basic plate / foundation algebra
# ---------------------------------------------------------------------------

def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Kirchhoff flexural rigidity ``D = E t^3 / (12 (1 - nu^2))``."""
    if young <= 0.0 or thickness <= 0.0:
        raise ValueError("young and thickness must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must satisfy -1 < nu < 0.5")
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def winkler_characteristic_length(rigidity: float, winkler: float) -> float:
    """Hetényi characteristic length ``l = (D/k)^(1/4)`` (metres)."""
    if rigidity <= 0.0 or winkler <= 0.0:
        raise ValueError("rigidity and winkler coefficient must be positive")
    return (rigidity / winkler) ** 0.25


def kerr_equivalent_pasternak(
    k_upper: float,
    k_lower: float,
    shear_modulus: float,
) -> tuple[float, float]:
    """Reduce the Kerr three-parameter foundation to ``(k_eq, G_eq)``.

    Kerr's model (a shear layer carrying springs ``k_u`` towards the plate and
    springs ``k_l`` towards the rigid base) is equivalent to a Pasternak
    foundation with

        k_eq = k_u k_l / (k_u + k_l),      G_eq = k_u G_p / (k_u + k_l).

    Derivation: with ``w_1`` the shear-layer deflection, interface equilibrium
    ``k_u (w - w_1) = k_l w_1 - G_p lap w_1`` gives, after eliminating ``w_1``
    to first order in ``G_p``, ``p = k_u (w - w_1) = k_eq w - G_eq lap w``.

    Limits: ``G_p = 0`` -> Winkler ``k_eq``; ``k_u -> inf`` (plate tied to the
    shear layer) -> Pasternak ``(k_l, G_p)``; ``k_l -> inf`` (rigid base pins
    the shear layer, so no spreading) -> Winkler ``k_u`` with ``G_eq -> 0``.
    """
    if k_upper <= 0.0 or k_lower <= 0.0:
        raise ValueError("both spring layers must be positive")
    if shear_modulus < 0.0:
        raise ValueError("shear modulus must be non-negative")
    total = k_upper + k_lower
    return k_upper * k_lower / total, k_upper * shear_modulus / total


# ---------------------------------------------------------------------------
# Complex Kelvin function K0 (principal branch, |arg z| < pi)
# ---------------------------------------------------------------------------

def k0_complex(z: torch.Tensor, *, series_limit: float = 9.5, n_terms: int = 200) -> torch.Tensor:
    """Complex ``K0(z)`` to ~1e-9 relative accuracy for ``|arg z| < pi/2``.

    Ascending series ``K0 = -(ln(z/2)+gamma) I0(z) + sum_k (z^2/4)^k/(k!)^2 H_k``
    for ``|z| <= series_limit`` and the six-term Hankel expansion

        K0(z) ~ sqrt(pi/(2z)) exp(-z) sum_k c_k / z^k

    beyond it.  The switch is set where the two error curves cross (measured
    against ``mpmath.besselk`` at 40 digits): the series loses digits like
    ``exp(|Re z|)`` and reaches 2e-4 at ``|z| = 14``, the asymptotic form
    reaches 8e-7 at ``|z| = 9``, so the crossover is at ``|z| ~ 9.5-10`` where
    both forms are good to ~6e-7; ``series_limit = 9.5`` therefore keeps the
    whole range below 1e-6 (the pre-existing
    :func:`tensorfem.hetenyi_reference.kelvin_kei` switches at 14 and is
    therefore ~1e-7 near ``|z| = 14``).
    """
    z = z.to(torch.complex128)
    out = torch.empty_like(z)
    magnitude = z.abs()
    small = magnitude <= series_limit
    if bool(small.any()):
        zs = z[small]
        q = zs * zs / 4.0
        term = torch.ones_like(q)
        i0 = torch.ones_like(q)
        acc = torch.zeros_like(q)
        harmonic = 0.0
        for k in range(1, n_terms + 1):
            term = term * q / (k * k)
            harmonic += 1.0 / k
            i0 = i0 + term
            acc = acc + term * harmonic
        out[small] = -(torch.log(zs / 2.0) + _GAMMA) * i0 + acc
    large = ~small
    if bool(large.any()):
        zl = z[large]
        series = torch.zeros_like(zl)
        for k, coeff in enumerate(_K0_ASYMPTOTIC_COEFFS):
            series = series + coeff / zl**k
        out[large] = torch.sqrt(math.pi / (2.0 * zl)) * torch.exp(-zl) * series
    return out


# ---------------------------------------------------------------------------
# Closed-form infinite-plate references
# ---------------------------------------------------------------------------

def infinite_plate_winkler_point_load(
    r: torch.Tensor,
    load: float,
    rigidity: float,
    winkler: float,
) -> torch.Tensor:
    """Hetényi deflection ``w(r) = -(P/(2 pi k l^2)) kei(r/l)`` (downward positive)."""
    if load <= 0.0:
        raise ValueError("load must be positive (downward)")
    ell = winkler_characteristic_length(rigidity, winkler)
    r = torch.as_tensor(r, dtype=torch.float64)
    return -(load / (2.0 * math.pi * winkler * ell * ell)) * kelvin_kei(r / ell)


def pasternak_foundation_roots(
    rigidity: float,
    winkler: float,
    pasternak: float,
) -> tuple[complex, complex]:
    """Roots ``(a, b)`` of ``D xi^4 + G xi^2 + k = 0`` in ``xi^2``, principal branch.

    ``a, b`` always have non-negative real part, so ``K0(a r)`` / ``K0(b r)``
    decay with ``r``.
    """
    if rigidity <= 0.0 or winkler <= 0.0:
        raise ValueError("rigidity and winkler coefficient must be positive")
    if pasternak < 0.0:
        raise ValueError("pasternak shear modulus must be non-negative")
    total = pasternak / rigidity
    product = winkler / rigidity
    discriminant = total * total - 4.0 * product
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        a2 = 0.5 * (total + root)
        b2 = 0.5 * (total - root)
        return complex(math.sqrt(max(a2, 0.0)), 0.0), complex(math.sqrt(max(b2, 0.0)), 0.0)
    imag = 0.5 * math.sqrt(-discriminant)
    a = _complex_sqrt(complex(0.5 * total, imag))
    return a, a.conjugate()


def _complex_sqrt(z: complex) -> complex:
    """Principal square root (branch cut on the negative real axis)."""
    return complex(math.sqrt(0.5 * (abs(z) + z.real)), math.copysign(
        math.sqrt(max(0.5 * (abs(z) - z.real), 0.0)), z.imag))


def infinite_plate_two_parameter_point_load(
    r: torch.Tensor,
    load: float,
    rigidity: float,
    winkler: float,
    pasternak: float = 0.0,
) -> torch.Tensor:
    """Closed-form deflection of an infinite plate on a two-parameter foundation.

    ``w(r) = (P/(2 pi D)) [K0(a r) - K0(b r)] / (b^2 - a^2)`` with
    ``a^2 + b^2 = G/D``, ``a^2 b^2 = k/D``; ``pasternak = 0`` reproduces the
    Hetényi ``kei`` solution bit-for-bit to round-off.
    """
    if load <= 0.0:
        raise ValueError("load must be positive (downward)")
    if pasternak < 0.0:
        raise ValueError("pasternak shear modulus must be non-negative")
    r = torch.as_tensor(r, dtype=torch.float64)
    if pasternak == 0.0:
        return infinite_plate_winkler_point_load(r, load, rigidity, winkler)
    a, b = pasternak_foundation_roots(rigidity, winkler, pasternak)
    a2, b2 = a * a, b * b
    scale = max(abs(a2), abs(b2))
    if abs(b2 - a2) < 1e-7 * scale:
        raise ValueError(
            "degenerate two-parameter foundation (a == b); the closed form loses "
            "all significant digits, use the quadrature reference instead"
        )
    za = torch.full_like(r, a, dtype=torch.complex128) * r.to(torch.complex128)
    zb = torch.full_like(r, b, dtype=torch.complex128) * r.to(torch.complex128)
    value = (k0_complex(za) - k0_complex(zb)) / (b2 - a2)
    # K0 has a logarithmic singularity at the origin; the *difference* has the
    # finite limit -ln(a/b)/(b^2-a^2), which is what the r = 0 entry needs
    # (inf - inf is not representable in floating point).
    at_origin = r == 0.0
    if bool(at_origin.any()):
        limit = -cmath.log(a / b) / (b2 - a2)
        value = value.clone()
        value[at_origin] = torch.tensor(complex(limit), dtype=torch.complex128)
    return (load / (2.0 * math.pi * rigidity)) * value.real.to(r.dtype)


# ---------------------------------------------------------------------------
# Element matrices for the foundation
# ---------------------------------------------------------------------------

def q4_pasternak_shear_stiffness(xy: torch.Tensor, shear_modulus: float) -> torch.Tensor:
    """12x12 shear-layer stiffness ``G int grad(N_w) . grad(N_w) dA``.

    Only the transverse-displacement entries (0, 3, 6, 9) are populated;
    constant ``w`` fields carry no energy, so the matrix is positive
    semi-definite (rank 8 on a Q4 element).
    """
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    dtype, device = xy.dtype, xy.device
    g = float(shear_modulus)
    kg = torch.zeros((12, 12), dtype=dtype, device=device)
    if g == 0.0:
        return kg
    if g < 0.0:
        raise ValueError("shear modulus must be non-negative")
    gp = 1.0 / 3.0**0.5
    block = torch.zeros((4, 4), dtype=dtype, device=device)
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
            _, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            dN = dN_nat @ torch.linalg.inv(J).T
            block = block + g * (dN @ dN.T) * torch.linalg.det(J)
    idx = torch.arange(4, device=device) * 3
    kg[idx[:, None], idx] += block
    return kg


def plate_foundation_element_stiffness(
    xy: torch.Tensor,
    young: float,
    poisson: float,
    thickness: float,
    *,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    shear_treatment: str = "mitc",
) -> torch.Tensor:
    """Plate + foundation 12x12 element stiffness."""
    ke = q4_mindlin_stiffness(
        xy, young, poisson, thickness, shear_treatment=shear_treatment
    )
    if winkler:
        ke = ke + q4_winkler_foundation_stiffness(xy, winkler)
    if pasternak:
        ke = ke + q4_pasternak_shear_stiffness(xy, pasternak)
    return ke


# ---------------------------------------------------------------------------
# Meshes and assembly
# ---------------------------------------------------------------------------

def graded_axis(n: int, length: float, beta: float = 4.0, *, dtype=torch.float64) -> torch.Tensor:
    """``sinh``-graded 1-D coordinates on ``[0, length]`` clustering at 0."""
    if n < 1:
        raise ValueError("n must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    t = torch.arange(n + 1, dtype=dtype) / n
    return length * torch.sinh(beta * t) / math.sinh(beta)


def product_mesh(
    xs: torch.Tensor, ys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(nodes, elements)`` of the tensor-product Q4 mesh ``xs x ys``.

    Node ``(i, j)`` (``i`` along x, ``j`` along y) has index ``j*(nx+1)+i`` so
    that the deflection reshapes as ``(ny+1, nx+1)``.
    """
    nx, ny = xs.numel() - 1, ys.numel() - 1
    if nx < 1 or ny < 1:
        raise ValueError("mesh needs at least one element per direction")
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    j, i = torch.meshgrid(
        torch.arange(ny, dtype=torch.long), torch.arange(nx, dtype=torch.long), indexing="ij"
    )
    n0 = j.reshape(-1) * (nx + 1) + i.reshape(-1)
    elements = torch.stack((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1), dim=1)
    return nodes, elements


def graded_square_mesh(
    n: int, length: float, beta: float = 4.0, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(nodes, elements)`` of a graded Q4 mesh on ``[0, length]^2``.

    The grading clusters elements at the origin (the point-load / symmetry
    corner) so that the local load singularity is resolved without inflating
    the far-field element count.
    """
    x = graded_axis(n, length, beta, dtype=dtype)
    return product_mesh(x, x)


def uniform_rect_mesh(
    nx: int, ny: int, side_x: float, side_y: float, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform Q4 mesh on ``[0, side_x] x [0, side_y]``."""
    if side_x <= 0.0 or side_y <= 0.0:
        raise ValueError("plate sides must be positive")
    xs = torch.linspace(0.0, side_x, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, side_y, ny + 1, dtype=dtype)
    return product_mesh(xs, ys)


def _gauss_kinematics(
    xy: torch.Tensor, xif: float, etaf: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched ``(N, grad N, det J)`` at one Gauss point for ``(ne, 4, 2)`` coords."""
    N, g_nat = _shape(xy.new_tensor(xif), xy.new_tensor(etaf))
    J = torch.einsum("ia,nic->nac", g_nat, xy)          # rows: natural, cols: physical
    det_j = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    if bool((det_j <= 0.0).any()):
        raise ValueError("plate element has non-positive Jacobian")
    # Explicit 2x2 inverse: J^{-1} = [[J11, -J01], [-J10, J00]] / det
    inv = torch.stack((
        torch.stack((J[:, 1, 1], -J[:, 0, 1]), dim=-1),
        torch.stack((-J[:, 1, 0], J[:, 0, 0]), dim=-1),
    ), dim=-2) / det_j[:, None, None]
    dN = torch.einsum("ia,nca->nic", g_nat, inv)
    return N, dN, det_j


def _shear_rows_batch(xy: torch.Tensor, xif: float, etaf: float) -> torch.Tensor:
    """Batched ``(ne, 2, 12)`` transverse-shear strain-displacement rows."""
    N, dN, _ = _gauss_kinematics(xy, xif, etaf)
    ne = xy.shape[0]
    rows = xy.new_zeros((ne, 2, 12))
    rows[:, 0, 0::3] = dN[:, :, 0]
    rows[:, 0, 1::3] = N
    rows[:, 1, 0::3] = dN[:, :, 1]
    rows[:, 1, 2::3] = N
    return rows


def batched_plate_foundation_element_stiffness(
    xy: torch.Tensor,
    young: float,
    poisson: float,
    thickness: float,
    *,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    shear_treatment: str = "mitc",
) -> torch.Tensor:
    """Vectorised plate + foundation element stiffness, ``(ne, 4, 2) -> (ne, 12, 12)``.

    Identical formulation to :func:`tensorfem.plate.q4_mindlin_stiffness`
    (selective reduced integration ``"sri"`` or the Dvorkin-Bathe assumed
    natural shear strain ``"mitc"``) plus the Winkler and Pasternak foundation
    terms; the per-element agreement with the scalar implementation is asserted
    in ``tests/test_foundation_plate_bending.py``.  Batching removes the Python
    loop over elements, which dominates the run time on a loaded machine.
    """
    if xy.dim() != 3 or xy.shape[1:] != (4, 2):
        raise ValueError("xy must have shape (ne, 4, 2)")
    if shear_treatment not in ("sri", "mitc"):
        raise ValueError("shear_treatment must be 'sri' or 'mitc'")
    dtype = xy.dtype
    ne = xy.shape[0]
    if thickness <= 0.0 or not -1.0 < poisson < 0.5:
        raise ValueError("invalid plate material or thickness")
    Db = young * thickness**3 / (12.0 * (1.0 - poisson**2)) * xy.new_tensor([
        [1.0, poisson, 0.0],
        [poisson, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - poisson)],
    ])
    Ds = (5.0 / 6.0) * young / (2.0 * (1.0 + poisson)) * thickness * torch.eye(
        2, dtype=dtype, device=xy.device
    )
    gp = 1.0 / 3.0**0.5
    ke = xy.new_zeros((ne, 12, 12))
    winkler_block = xy.new_zeros((ne, 4, 4))
    pasternak_block = xy.new_zeros((ne, 4, 4))
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            N, dN, det_j = _gauss_kinematics(xy, xif, etaf)
            Bb = xy.new_zeros((ne, 3, 12))
            Bb[:, 0, 1::3] = dN[:, :, 0]
            Bb[:, 1, 2::3] = dN[:, :, 1]
            Bb[:, 2, 1::3] = dN[:, :, 1]
            Bb[:, 2, 2::3] = dN[:, :, 0]
            ke = ke + torch.einsum("nki,kl,nlj->nij", Bb, Db, Bb) * det_j[:, None, None]
            if winkler:
                winkler_block = winkler_block + winkler * torch.einsum(
                    "i,j,n->nij", N, N, det_j
                )
            if pasternak:
                pasternak_block = pasternak_block + pasternak * torch.einsum(
                    "nic,njc,n->nij", dN, dN, det_j
                )
    if shear_treatment == "sri":
        rows = _shear_rows_batch(xy, 0.0, 0.0)
        _, _, det_j = _gauss_kinematics(xy, 0.0, 0.0)
        ke = ke + torch.einsum(
            "nki,kl,nlj->nij", rows, Ds, rows
        ) * (4.0 * det_j)[:, None, None]
    else:
        bottom = _shear_rows_batch(xy, 0.0, -1.0)
        top = _shear_rows_batch(xy, 0.0, 1.0)
        left = _shear_rows_batch(xy, -1.0, 0.0)
        right = _shear_rows_batch(xy, 1.0, 0.0)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                _, _, det_j = _gauss_kinematics(xy, xif, etaf)
                Bs = xy.new_zeros((ne, 2, 12))
                Bs[:, 0, :] = 0.5 * (1.0 - etaf) * bottom[:, 0, :] + 0.5 * (1.0 + etaf) * top[:, 0, :]
                Bs[:, 1, :] = 0.5 * (1.0 - xif) * left[:, 1, :] + 0.5 * (1.0 + xif) * right[:, 1, :]
                ke = ke + torch.einsum("nki,kl,nlj->nij", Bs, Ds, Bs) * det_j[:, None, None]
    ke = ke.clone()
    ke[:, 0::3, 0::3] += winkler_block + pasternak_block
    return ke


def batched_uniform_pressure_load(xy: torch.Tensor, pressure: float) -> torch.Tensor:
    """Vectorised consistent nodal loads ``(ne, 12)`` for a uniform pressure."""
    gp = 1.0 / 3.0**0.5
    fe = xy.new_zeros((xy.shape[0], 12))
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            N, _, det_j = _gauss_kinematics(xy, xif, etaf)
            fe[:, 0::3] += pressure * torch.einsum("i,n->ni", N, det_j)
    return fe


def assemble_plate_foundation_system(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    *,
    young: float,
    poisson: float,
    thickness: float,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    shear_treatment: str = "mitc",
    pressure: float = 0.0,
    nodal_loads: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Assemble ``K`` as COO triplets plus the consistent load vector.

    ``pressure`` is a uniform pressure over the whole plate (positive
    downward), ``nodal_loads`` maps node indices to concentrated vertical
    forces.  Returns ``(rows, cols, values, force, ndof)``.
    """
    nnodes = nodes.shape[0]
    ndof = 3 * nnodes
    conn = elements.reshape(-1, 4)
    xy = nodes[conn]                       # (ne, 4, 2)
    ke = batched_plate_foundation_element_stiffness(
        xy, young, poisson, thickness,
        winkler=winkler, pasternak=pasternak, shear_treatment=shear_treatment,
    )
    dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=2).reshape(-1, 12)
    rows = dofs.repeat_interleave(12, dim=1).reshape(-1)
    cols = dofs.repeat(1, 12).reshape(-1)
    vals = ke.reshape(-1)
    force = torch.zeros(ndof, dtype=torch.float64)
    if pressure:
        fe = batched_uniform_pressure_load(xy, pressure)
        force.index_add_(0, dofs.reshape(-1), fe.reshape(-1))
    if nodal_loads:
        for node, value in nodal_loads.items():
            force[3 * int(node)] += float(value)
    return rows, cols, vals, force, ndof


def solve_assembled_system(
    system: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    fixed_dofs: torch.Tensor,
    *,
    dense_limit: int = 4000,
) -> torch.Tensor:
    """Solve the assembled system with ``fixed_dofs`` eliminated (zero RHS there)."""
    rows, cols, vals, force, ndof = system
    fixed = torch.as_tensor(fixed_dofs, dtype=torch.long)
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    u = torch.zeros(ndof, dtype=torch.float64)
    if free.numel() == 0:
        return u
    if ndof <= dense_limit:
        K = torch.zeros((ndof, ndof), dtype=torch.float64)
        # Duplicate COO entries must accumulate (fancy-index assignment would
        # silently keep only the last one and destroy the element coupling).
        K.index_put_((rows, cols), vals, accumulate=True)
        K_ff = K[free[:, None], free]
        u[free] = torch.linalg.solve(K_ff, force[free])
        return u
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "solving a mesh with more than dense_limit dofs needs SciPy "
            "(scipy.sparse.linalg); install scipy or reduce the mesh"
        ) from exc
    K = sp.coo_matrix(
        (vals.numpy(), (rows.numpy(), cols.numpy())), shape=(ndof, ndof)
    ).tocsr()
    K_ff = K[free.numpy()[:, None], free.numpy()]
    sol = spla.spsolve(K_ff.tocsc(), force[free].numpy())
    u[free] = torch.from_numpy(sol)
    return u


# ---------------------------------------------------------------------------
# FE driver: infinite plate (quarter model, graded mesh, free far field)
# ---------------------------------------------------------------------------

def deflection_on_grid(
    nodes: torch.Tensor,
    deflection: torch.Tensor,
    nx: int,
    ny: int,
    points: torch.Tensor,
) -> torch.Tensor:
    """Bilinear interpolation of a Q4 ``w`` field on a structured mesh.

    ``nodes`` follows the :func:`product_mesh` ordering (``j*(nx+1)+i``).
    """
    pts = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 2).contiguous()
    x = nodes[: nx + 1, 0].contiguous()
    y = nodes[:: nx + 1, 1].contiguous()
    px = pts[:, 0].contiguous()
    py = pts[:, 1].contiguous()
    grid = deflection[0::3].reshape(ny + 1, nx + 1)
    i = torch.clamp(torch.searchsorted(x, px, right=True) - 1, 0, nx - 1)
    j = torch.clamp(torch.searchsorted(y, py, right=True) - 1, 0, ny - 1)
    xi = ((px - x[i]) / (x[i + 1] - x[i])).clamp(0.0, 1.0)
    eta = ((py - y[j]) / (y[j + 1] - y[j])).clamp(0.0, 1.0)
    w00 = grid[j, i]
    w10 = grid[j, i + 1]
    w01 = grid[j + 1, i]
    w11 = grid[j + 1, i + 1]
    return (
        w00 * (1 - xi) * (1 - eta)
        + w10 * xi * (1 - eta)
        + w01 * (1 - xi) * eta
        + w11 * xi * eta
    )


@dataclass(frozen=True)
class FoundationPlateSolution:
    """Result of a foundation-coupled FE solve on a structured graded mesh."""

    nodes: torch.Tensor
    elements: torch.Tensor
    nx: int
    ny: int
    deflection: torch.Tensor
    ndof: int
    applied_load: float
    foundation_reaction: float
    pressure: float = 0.0
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def axis_coordinates(self) -> torch.Tensor:
        """Coordinates of the bottom (``y = 0``) node row."""
        return self.nodes[: self.nx + 1, 0]

    def deflection_at(self, points: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of ``w`` at arbitrary points of the mesh."""
        return deflection_on_grid(self.nodes, self.deflection, self.nx, self.ny, points)


def infinite_plate_point_load_fem(
    load: float,
    *,
    young: float = 3.0e10,
    poisson: float = 0.3,
    thickness: float = 0.1,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    pressure: float = 0.0,
    half_width_in_lengths: float = 20.0,
    n: int = 60,
    beta: float = 4.0,
    shear_treatment: str = "mitc",
    dense_limit: int = 4000,
) -> FoundationPlateSolution:
    """Quarter-model FE solution of an infinite plate with a central point load.

    The domain is the quarter square ``[0, L]^2`` with ``L =
    half_width_in_lengths * l`` and the symmetry conditions ``theta_x = 0`` on
    ``x = 0``, ``theta_y = 0`` on ``y = 0`` (so ``dw/dx = dw/dy = 0`` there).
    A quarter of the load ``P/4`` acts at the symmetry corner and the outer
    edges are free, which is the natural truncation of the Winkler/Pasternak
    far field.  ``pressure`` adds a uniform load (e.g. the plate's own weight).

    ``foundation_reaction`` is the discrete foundation force
    ``sum_i (K_foundation u)_i`` over the transverse dofs; for a free outer edge
    it must reproduce the applied load, which is an equilibrium audit of the
    assembly.
    """
    ell = winkler_characteristic_length(
        plate_rigidity(young, poisson, thickness), winkler
    )
    length = half_width_in_lengths * ell
    nodes, elements = graded_square_mesh(n, length, beta)
    system = assemble_plate_foundation_system(
        nodes, elements,
        young=young, poisson=poisson, thickness=thickness,
        winkler=winkler, pasternak=pasternak,
        shear_treatment=shear_treatment,
        pressure=pressure,
        nodal_loads={0: 0.25 * load},
    )
    tol = 1e-9 * length
    on_x0 = nodes[:, 0].abs() <= tol
    on_y0 = nodes[:, 1].abs() <= tol
    fixed = torch.cat((
        3 * torch.nonzero(on_x0, as_tuple=False).flatten() + 1,
        3 * torch.nonzero(on_y0, as_tuple=False).flatten() + 2,
    )).unique()
    u = solve_assembled_system(system, fixed, dense_limit=dense_limit)
    reaction = _foundation_reaction(nodes, elements, u, winkler, pasternak)
    axis = graded_axis(n, length, beta)
    metadata = {
        "characteristic_length": ell,
        "domain_half_width": length,
        "first_element_size": float(axis[1] - axis[0]),
        "ndof": float(system[4]),
        "n_elements": float(elements.shape[0]),
        "n": float(n),
        "beta": float(beta),
    }
    return FoundationPlateSolution(
        nodes=nodes,
        elements=elements,
        nx=n,
        ny=n,
        deflection=u,
        ndof=system[4],
        applied_load=0.25 * load + pressure * length * length,
        foundation_reaction=reaction,
        pressure=pressure,
        metadata=metadata,
    )


def finite_plate_foundation_fem(
    *,
    side_x: float,
    side_y: float,
    nx: int,
    ny: int,
    young: float = 3.0e10,
    poisson: float = 0.3,
    thickness: float = 0.1,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    pressure: float = 0.0,
    point_load: float = 0.0,
    shear_treatment: str = "mitc",
    dense_limit: int = 4000,
) -> FoundationPlateSolution:
    """FE solution of a simply supported plate on a two-parameter foundation.

    Uniform Q4 mesh, hard simple supports (``w = 0`` on all four edges plus the
    tangential rotation, i.e. ``theta_x = 0`` on the ``y`` edges and ``theta_y =
    0`` on the ``x`` edges -- the same convention as
    :func:`tensorfem.plate.solve_simply_supported_sine_square`).  A uniform
    pressure ``pressure`` and/or a central point load ``point_load`` may be
    applied; the closed-form counterpart is
    :func:`navier_foundation_deflection`.
    """
    nodes, elements = uniform_rect_mesh(nx, ny, side_x, side_y)
    nodal_loads: dict[int, float] = {}
    if point_load:
        if nx % 2 or ny % 2:
            raise ValueError("a central point load needs an even nx and ny")
        nodal_loads[(ny // 2) * (nx + 1) + nx // 2] = float(point_load)
    system = assemble_plate_foundation_system(
        nodes, elements,
        young=young, poisson=poisson, thickness=thickness,
        winkler=winkler, pasternak=pasternak,
        shear_treatment=shear_treatment,
        pressure=pressure,
        nodal_loads=nodal_loads,
    )
    tol_x = 1e-9 * side_x
    tol_y = 1e-9 * side_y
    on_x = (nodes[:, 0].abs() <= tol_x) | ((nodes[:, 0] - side_x).abs() <= tol_x)
    on_y = (nodes[:, 1].abs() <= tol_y) | ((nodes[:, 1] - side_y).abs() <= tol_y)
    edge = on_x | on_y
    fixed = torch.cat((
        3 * torch.nonzero(edge, as_tuple=False).flatten(),        # w = 0
        3 * torch.nonzero(on_y, as_tuple=False).flatten() + 1,    # theta_x = 0
        3 * torch.nonzero(on_x, as_tuple=False).flatten() + 2,    # theta_y = 0
    )).unique()
    u = solve_assembled_system(system, fixed, dense_limit=dense_limit)
    applied = pressure * side_x * side_y + point_load
    reaction = _foundation_reaction(nodes, elements, u, winkler, pasternak)
    return FoundationPlateSolution(
        nodes=nodes,
        elements=elements,
        nx=nx,
        ny=ny,
        deflection=u,
        ndof=system[4],
        applied_load=applied,
        foundation_reaction=reaction,
        pressure=pressure,
        metadata={
            "side_x": side_x,
            "side_y": side_y,
            "n_elements": float(elements.shape[0]),
            "ndof": float(system[4]),
        },
    )


def _foundation_reaction(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    u: torch.Tensor,
    winkler: float,
    pasternak: float,
) -> float:
    """Discrete foundation force ``sum_i (K_foundation u)_i`` over ``w`` dofs.

    For a free outer edge the plate carries no vertical edge force, so this
    number must equal the applied load: a load-path audit of the assembly.
    """
    if not winkler and not pasternak:
        return 0.0
    conn = elements.reshape(-1, 4)
    xy = nodes[conn]
    ne = xy.shape[0]
    gp = 1.0 / 3.0**0.5
    block = xy.new_zeros((ne, 4, 4))
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            N, dN, det_j = _gauss_kinematics(xy, xif, etaf)
            if winkler:
                block = block + winkler * torch.einsum("i,j,n->nij", N, N, det_j)
            if pasternak:
                block = block + pasternak * torch.einsum("nic,njc,n->nij", dN, dN, det_j)
    w_local = u.reshape(-1, 3)[conn][:, :, 0]                    # (ne, 4)
    local_force = torch.einsum("nij,nj->ni", block, w_local)     # (ne, 4)
    nodal_force = torch.zeros(u.shape[0] // 3, dtype=u.dtype)
    nodal_force.index_add_(0, conn.reshape(-1), local_force.reshape(-1))
    return float(nodal_force.sum())


# ---------------------------------------------------------------------------
# Finite simply supported plate on a foundation: Navier double-sine series
# ---------------------------------------------------------------------------

def navier_mindlin_foundation_deflection(
    points: torch.Tensor,
    *,
    side_x: float,
    side_y: float,
    young: float = 3.0e10,
    poisson: float = 0.3,
    thickness: float = 0.1,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    pressure: float = 0.0,
    point_load: float = 0.0,
    point: tuple[float, float] | None = None,
    n_modes: int = 99,
    shear_correction: float = 5.0 / 6.0,
) -> torch.Tensor:
    """Exact Navier solution of the *Mindlin-Reissner* plate on a foundation.

    Mode shapes ``w = W sin(ax) sin(by)``, ``theta_x = X cos(ax) sin(by)``,
    ``theta_y = Y sin(ax) cos(by)`` satisfy ``w = 0`` and zero tangential
    rotation on all four edges -- the *hard* simple support enforced by
    :func:`finite_plate_foundation_fem` -- so this series is the exact reference
    for the thick-plate FE model (the Kirchhoff series
    :func:`navier_foundation_deflection` is its ``G h -> inf`` limit).

    Per mode the three equilibrium equations are

        [D(a^2 + (1-nu) b^2/2) + S] X + [D(1+nu) a b/2] Y + S a W = 0
        [D(1+nu) a b/2] X + [D(b^2 + (1-nu) a^2/2) + S] Y + S b W = 0
        S a X + S b Y + [(S + G)(a^2 + b^2) + k] W = q_mn

    with ``S = kappa G_s h`` the transverse shear stiffness, ``D`` the flexural
    rigidity and ``G`` the foundation shear modulus (which adds to the shear
    stiffness acting on ``w``).
    """
    if side_x <= 0.0 or side_y <= 0.0:
        raise ValueError("plate sides must be positive")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    rigidity = plate_rigidity(young, poisson, thickness)
    shear = shear_correction * young / (2.0 * (1.0 + poisson)) * thickness
    pts = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 2)
    odd = torch.arange(1, n_modes + 1, 2, dtype=torch.float64)
    mm = odd.repeat_interleave(odd.numel())
    nn = odd.repeat(odd.numel())
    a = mm * math.pi / side_x
    b = nn * math.pi / side_y
    q_mn = torch.zeros_like(mm)
    if pressure:
        q_mn = q_mn + 16.0 * pressure / (math.pi**2 * mm * nn)
    if point_load:
        x0, y0 = (0.5 * side_x, 0.5 * side_y) if point is None else point
        q_mn = q_mn + (
            4.0 * point_load / (side_x * side_y)
            * torch.sin(a * x0) * torch.sin(b * y0)
        )
    m = mm.numel()
    K = torch.zeros((m, 3, 3), dtype=torch.float64)
    K[:, 0, 0] = rigidity * (a**2 + 0.5 * (1 - poisson) * b**2) + shear
    K[:, 1, 1] = rigidity * (b**2 + 0.5 * (1 - poisson) * a**2) + shear
    K[:, 0, 1] = K[:, 1, 0] = rigidity * 0.5 * (1 + poisson) * a * b
    K[:, 0, 2] = K[:, 2, 0] = shear * a
    K[:, 1, 2] = K[:, 2, 1] = shear * b
    K[:, 2, 2] = (shear + pasternak) * (a**2 + b**2) + winkler
    rhs = torch.zeros((m, 3), dtype=torch.float64)
    rhs[:, 2] = q_mn
    unknowns = torch.linalg.solve(K, rhs)
    W = unknowns[:, 2]
    sx = torch.sin(pts[:, None, 0] * a[None, :])
    sy = torch.sin(pts[:, None, 1] * b[None, :])
    return (W[None, :] * sx * sy).sum(dim=1)


def navier_foundation_deflection(
    points: torch.Tensor,
    *,
    side_x: float,
    side_y: float,
    young: float = 3.0e10,
    poisson: float = 0.3,
    thickness: float = 0.1,
    winkler: float = 0.0,
    pasternak: float = 0.0,
    pressure: float = 0.0,
    point_load: float = 0.0,
    point: tuple[float, float] | None = None,
    n_modes: int = 99,
) -> torch.Tensor:
    """Navier series for a simply supported Kirchhoff plate on a foundation.

    The foundation enters the eigen-coefficient of each sinusoidal mode::

        w_mn = q_mn / (D pi^4 (m^2/a^2 + n^2/b^2)^2
                       + G pi^2 (m^2/a^2 + n^2/b^2) + k)

    with ``q_mn = 16 q0/(pi^2 m n)`` for a uniform pressure ``q0`` and
    ``q_mn = 4 P/(a b) sin(m pi x0/a) sin(n pi y0/b)`` for a point load at
    ``(x0, y0)``.  Only odd ``m, n`` are retained (symmetric loading); the
    series is truncated at ``m, n <= n_modes``.
    """
    if side_x <= 0.0 or side_y <= 0.0:
        raise ValueError("plate sides must be positive")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    rigidity = plate_rigidity(young, poisson, thickness)
    pts = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 2)
    odd = torch.arange(1, n_modes + 1, 2, dtype=torch.float64)
    mm = odd.repeat_interleave(odd.numel())
    nn = odd.repeat(odd.numel())
    lam = mm**2 / side_x**2 + nn**2 / side_y**2
    denom = rigidity * math.pi**4 * lam**2 + pasternak * math.pi**2 * lam + winkler
    coeff = torch.zeros_like(mm)
    if pressure:
        coeff = coeff + 16.0 * pressure / (math.pi**2 * mm * nn)
    if point_load:
        x0, y0 = (0.5 * side_x, 0.5 * side_y) if point is None else point
        coeff = coeff + (
            4.0 * point_load / (side_x * side_y)
            * torch.sin(mm * math.pi * x0 / side_x)
            * torch.sin(nn * math.pi * y0 / side_y)
        )
    coeff = coeff / denom
    sx = torch.sin(math.pi * pts[:, None, 0] * mm[None, :] / side_x)
    sy = torch.sin(math.pi * pts[:, None, 1] * nn[None, :] / side_y)
    return (coeff[None, :] * sx * sy).sum(dim=1)


# ---------------------------------------------------------------------------
# Convenience: analytic vs FE comparison for the infinite-plate point load
# ---------------------------------------------------------------------------

def compare_infinite_plate_point_load(
    radii_in_lengths: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    *,
    load: float = 1.0e5,
    young: float = 3.0e10,
    poisson: float = 0.3,
    thickness: float = 0.1,
    winkler: float = 1.0e7,
    pasternak: float = 0.0,
    half_width_in_lengths: float = 20.0,
    n: int = 60,
    beta: float = 4.0,
    shear_treatment: str = "mitc",
    dense_limit: int = 4000,
) -> dict[str, object]:
    """Compare FE vs closed form at ``radii_in_lengths * l`` along ``y = 0``.

    Returns a report dict with per-radius analytic / FE values and the relative
    error, plus the equilibrium and characteristic-length diagnostics.
    """
    solution = infinite_plate_point_load_fem(
        load, young=young, poisson=poisson, thickness=thickness,
        winkler=winkler, pasternak=pasternak,
        half_width_in_lengths=half_width_in_lengths, n=n, beta=beta,
        shear_treatment=shear_treatment, dense_limit=dense_limit,
    )
    ell = solution.metadata["characteristic_length"]
    radii = torch.tensor(radii_in_lengths, dtype=torch.float64) * ell
    points = torch.stack((radii, torch.zeros_like(radii)), dim=1)
    fem = solution.deflection_at(points)
    analytic = infinite_plate_two_parameter_point_load(
        radii, load, plate_rigidity(young, poisson, thickness), winkler, pasternak
    )
    errors = (fem - analytic).abs() / analytic.abs()
    rows = [
        {
            "r_over_l": float(r_i),
            "r": float(rad),
            "analytic": float(a),
            "fem": float(f),
            "rel_error": float(e),
        }
        for r_i, rad, a, f, e in zip(radii_in_lengths, radii, analytic, fem, errors)
    ]
    return {
        "characteristic_length": ell,
        "domain_half_width": solution.metadata["domain_half_width"],
        "n": n,
        "ndof": solution.ndof,
        "rows": rows,
        "max_rel_error": float(errors.max()),
        "applied_load": solution.applied_load,
        "foundation_reaction": solution.foundation_reaction,
        "equilibrium_rel_error": float(
            abs(solution.foundation_reaction - solution.applied_load)
            / solution.applied_load
        ),
        "solution": solution,
    }