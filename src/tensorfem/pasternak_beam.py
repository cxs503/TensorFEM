"""Beam on a Pasternak (two-parameter) foundation -- the beam-level counterpart of
:mod:`tensorfem.foundation_plate_bending`.

Why this module exists
----------------------
The repository already carries the *plate* Pasternak model (closed-form Kelvin
``K0`` solution, Q4 shear-layer stiffness, Navier reference) in
:mod:`tensorfem.foundation_plate_bending` and the *beam* Winkler model in
:mod:`tensorfem.beam_foundation`.  The beam-level shear-coupling model was the
missing rung of that ladder, so this module adds it without duplicating the
plate machinery.

Strong form
-----------
A Euler-Bernoulli beam of rigidity ``EI`` on a Pasternak foundation reacts with

    p(x) = k w(x) - G_s w''(x),          [Pasternak 1954; Vlasov & Leont'ev 1960]

i.e. a Winkler spring bed (``k``, N/m^2 for a beam strip) plus a shear layer of
modulus ``G_s`` (N) that couples neighbouring springs.  Equilibrium under a
transverse load ``q`` gives

    EI w'''' - G_s w'' + k w = q.                       (1)

``G_s = 0`` recovers the Winkler problem ``EI w'''' + k w = q``, whose infinite
beam point-load solution is Hetenyi (1946)

    w(x) = P beta A(beta|x|) / (2k),  A = e^{-beta|x|}(cos beta x + sin beta|x|),
    beta = (k / 4EI)^{1/4}.

Weak form (the form implemented here)
-------------------------------------
Test function ``v`` with ``v, v'`` admissible:

    int EI w'' v'' dx  +  int k w v dx  +  int G_s w' v' dx  =  int q v dx.   (2)

The third term is the new one: ``G_s int N'^T N' dx`` on a C1 Hermite element.
Constant ``w`` carries no shear energy, so the shear-layer matrix is positive
*semi*-definite and the total stiffness stays positive definite for ``k > 0``.

Exact point-load solution of (1)
--------------------------------
With ``mu = lambda^2`` the characteristic equation is ``EI mu^2 - G_s mu + k = 0``:

    mu_{1,2} = (G_s +/- sqrt(G_s^2 - 4 E I k)) / (2 E I).

For ``G_s < 2 sqrt(EI k)`` the roots are a complex pair, the decay is
oscillatory (Winkler-like); for ``G_s > 2 sqrt(EI k)`` they are real and the
response is the sum of two monotone exponential layers:

* a fast *shear* boundary layer ``l_shear = 1/lambda_fast -> sqrt(EI/G_s)``,
* a slow *soft* layer ``l_soft = 1/lambda_slow -> sqrt(G_s/k)``.

For a unit point load ``P`` at ``x = 0`` the even solution is

    w(x) = A e^{-lambda_fast |x|} + B e^{-lambda_slow |x|},
    A = P / (2 EI lambda_f (lambda_s^2 - lambda_f^2)),
    B = -P / (2 EI lambda_s (lambda_s^2 - lambda_f^2)),

which reduces algebraically to the Hetenyi ``A(beta x)`` form when ``G_s = 0``
(see ``tests/test_pasternak_beam.py::test_closed_form_reduces_to_hetenyi``), and
the peak penetration is

    w(0) = P / (2 EI lambda_f lambda_s (lambda_f + lambda_s)).

Limitations
-----------
* Euler-Bernoulli kinematics only (no transverse-shear deformation of the beam
  itself); for stocky ice strips use :func:`tensorfem.beam_foundation.
  timoshenko_beam_stiffness`-based assembly.
* Bilateral foundation: no tensionless (unilateral) contact here; that lives in
  :mod:`tensorfem.beam_foundation` / :mod:`tensorfem.contact.unilateral`.
* ``k`` and ``G_s`` are element-wise constant.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import torch

from .beam_foundation import (
    eb_beam_stiffness,
    hermite_consistent_load,
    hermite_foundation_stiffness,
    hermite_shapes,
)

FLOAT = torch.float64

__all__ = [
    "hermite_pasternak_shear_stiffness",
    "pasternak_characteristic_roots",
    "pasternak_infinite_beam_point_load",
    "pasternak_penetration",
    "pasternak_characteristic_lengths",
    "pasternak_peak_moment",
    "pasternak_beam_element_stiffness",
    "PasternakBeamModel",
    "assemble_pasternak_beam",
    "solve_pasternak_beam",
    "decay_modes_from_nodes",
    "power_law_exponent",
]


# ---------------------------------------------------------------------------
# element matrices
# ---------------------------------------------------------------------------
def hermite_pasternak_shear_stiffness(L: float, shear: float, *, order: int = 4,
                                      dtype=FLOAT) -> torch.Tensor:
    """Shear-layer matrix ``G_s int_0^L N'^T N' dx`` (4x4, C1 Hermite).

    ``N'`` are quadratics, so the integrand is a quartic and 3-point Gauss is
    already exact; the default 4-point rule keeps the same rule as
    :func:`tensorfem.beam_foundation.hermite_foundation_stiffness_quadrature`.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")
    L = float(L)
    g = float(shear)
    K = torch.zeros((4, 4), dtype=dtype)
    if g == 0.0:
        return K
    if g < 0.0:
        raise ValueError("pasternak shear modulus must be non-negative")
    if order == 1:
        points, weights = [0.5], [1.0]
    elif order == 2:
        s = 1.0 / math.sqrt(3.0)
        points, weights = [0.5 - 0.5 * s, 0.5 + 0.5 * s], [0.5, 0.5]
    elif order == 3:
        s = math.sqrt(0.6)
        points = [0.5 - 0.5 * s, 0.5, 0.5 + 0.5 * s]
        weights = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]
    elif order == 4:
        a = math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        b = math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        wa = (18.0 + math.sqrt(30.0)) / 36.0
        wb = (18.0 - math.sqrt(30.0)) / 36.0
        points = [0.5 - 0.5 * b, 0.5 - 0.5 * a, 0.5 + 0.5 * a, 0.5 + 0.5 * b]
        weights = [0.5 * wb, 0.5 * wa, 0.5 * wa, 0.5 * wb]
    else:
        raise ValueError("supported Gauss orders: 1, 2, 3, 4")
    for xi, wt in zip(points, weights):
        _, dN = hermite_shapes(torch.tensor(xi, dtype=dtype), L)
        K = K + (g * wt * L) * torch.outer(dN, dN)
    return K


def pasternak_beam_element_stiffness(E: float, Iz: float, k: float, shear: float,
                                     L: float, *, dtype=FLOAT) -> torch.Tensor:
    """Total 4x4 element matrix: bending + Winkler + shear layer."""
    K = eb_beam_stiffness(E, Iz, L, dtype=dtype)
    if k:
        K = K + torch.as_tensor(hermite_foundation_stiffness(L, k, dtype=dtype),
                                dtype=dtype)
    if shear:
        K = K + hermite_pasternak_shear_stiffness(L, shear, dtype=dtype)
    return K


# ---------------------------------------------------------------------------
# closed form for the infinite beam
# ---------------------------------------------------------------------------
def _principal_sqrt(value: complex) -> complex:
    """Square root with non-negative real part (decaying branch)."""
    root = cmath.sqrt(value)
    if root.real < 0.0:
        root = -root
    return root


def pasternak_characteristic_roots(EI: float, k: float, shear: float):
    """Roots ``(lambda_fast, lambda_slow)`` of ``EI l^4 - G_s l^2 + k = 0``.

    Both roots have non-negative real part (decaying branches); they are
    returned as ``(fast, slow)`` ordered by decreasing real part.  ``G_s = 0``
    gives the Hetenyi pair ``beta (1 +/- i)`` with ``beta = (k/4EI)^{1/4}``.
    """
    if EI <= 0.0 or k <= 0.0:
        raise ValueError("EI and k must be positive")
    if shear < 0.0:
        raise ValueError("pasternak shear modulus must be non-negative")
    a = float(EI)
    b = float(shear)
    c = float(k)
    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        r = math.sqrt(disc)
        mu_fast = (b + r) / (2.0 * a)
        mu_slow = (b - r) / (2.0 * a) if b - r > 0.0 else 0.0
        lam_f = _principal_sqrt(complex(max(mu_fast, 0.0), 0.0))
        lam_s = _principal_sqrt(complex(max(mu_slow, 0.0), 0.0))
        return lam_f, lam_s
    mu = complex(b / (2.0 * a), math.sqrt(-disc) / (2.0 * a))
    lam = _principal_sqrt(mu)
    return lam, lam.conjugate()


def _tied(lam_a: complex, lam_b: complex, rel: float = 1e-6) -> bool:
    """True when the two roots are numerically indistinguishable (double root)."""
    scale = max(abs(lam_a), abs(lam_b), 1e-300)
    return abs(lam_a - lam_b) <= rel * scale


def pasternak_infinite_beam_point_load(x_eval, x_load: float, P: float, EI: float,
                                       k: float, shear: float) -> torch.Tensor:
    """Deflection ``w(x)`` of an infinite beam on a Pasternak foundation.

    ``w = A exp(-lam_f |x-x0|) + B exp(-lam_s |x-x0|)`` with the amplitudes of
    the module docstring; downward positive.  ``shear = 0`` reproduces
    :func:`tensorfem.beam_foundation.hetenyi_point_load_deflection`.
    """
    if P == 0.0:
        return torch.zeros_like(torch.as_tensor(x_eval, dtype=FLOAT))
    lam_f, lam_s = pasternak_characteristic_roots(EI, k, shear)
    if _tied(lam_f, lam_s):
        # double root: w = P (1 + lambda|x|) exp(-lambda|x|) / (4 EI lambda^3)
        lam = 0.5 * (lam_f + lam_s)
        r = (torch.as_tensor(x_eval, dtype=FLOAT)
             - torch.as_tensor(x_load, dtype=FLOAT)).abs()
        coef = complex(P / (4.0 * EI * lam**3))
        return (coef * (1.0 + lam * r.to(torch.complex128))
                * torch.exp(-lam * r.to(torch.complex128))).real.to(FLOAT)
    denom = 2.0 * EI * (lam_s * lam_s - lam_f * lam_f)
    A = complex(P) / (denom * lam_f)
    B = -complex(P) / (denom * lam_s)
    r = (torch.as_tensor(x_eval, dtype=FLOAT)
         - torch.as_tensor(x_load, dtype=FLOAT)).abs().to(torch.complex128)
    value = A * torch.exp(-lam_f * r) + B * torch.exp(-lam_s * r)
    return value.real.to(FLOAT)


def pasternak_penetration(P: float, EI: float, k: float, shear: float) -> float:
    """Peak deflection under a point load, ``P / (2 EI l_f l_s (l_f + l_s))``.

    ``shear = 0`` returns the Hetenyi penetration ``P beta / (2k)``; the
    large-shear limit is ``P / (2 sqrt(k G_s))`` (the shear layer spreads the
    load, so the penetration *decreases* with ``G_s``).
    """
    lam_f, lam_s = pasternak_characteristic_roots(EI, k, shear)
    value = complex(P) / (2.0 * EI * lam_f * lam_s * (lam_f + lam_s))
    return float(value.real)


def pasternak_peak_moment(P: float, EI: float, k: float, shear: float) -> float:
    """Bending moment at the point load, ``M(0) = -EI (A l_f^2 + B l_s^2)``.

    Sagging positive (``w`` downward positive).  ``shear = 0`` gives the
    Hetenyi value ``P l_c / 4`` with ``l_c = (4EI/k)^{1/4}``; the shear layer
    spreads the load and therefore *reduces* ``M(0)``.
    """
    lam_f, lam_s = pasternak_characteristic_roots(EI, k, shear)
    if _tied(lam_f, lam_s):
        # double root: M(0) = -EI w''(0) = P / (4 lambda)
        lam = 0.5 * (lam_f + lam_s)
        return float((complex(P) / (4.0 * lam)).real)
    denom = 2.0 * EI * (lam_s * lam_s - lam_f * lam_f)
    A = complex(P) / (denom * lam_f)
    B = -complex(P) / (denom * lam_s)
    return float((-EI * (A * lam_f ** 2 + B * lam_s ** 2)).real)


def pasternak_characteristic_lengths(EI: float, k: float, shear: float) -> dict:
    """The three length scales of the Pasternak beam, all in metres.

    ``l_winkler = (4EI/k)^{1/4}``  -- Hetenyi length, independent of ``G_s``;
    ``l_shear = 1/Re lambda_fast -> sqrt(EI/G_s)``  -- shear boundary layer;
    ``l_soft  = 1/Re lambda_slow -> sqrt(G_s/k)``   -- soft (load-spreading) layer;
    ``l_effective = P / (2 k w(0))`` -- the length that would give the same
    penetration on a Winkler bed, ``-> sqrt(G_s/k)`` for large ``G_s``.
    """
    lam_f, lam_s = pasternak_characteristic_roots(EI, k, shear)
    winkler = (4.0 * EI / k) ** 0.25
    l_shear = 1.0 / max(lam_f.real, 1e-300)
    l_soft = 1.0 / max(lam_s.real, 1e-300)
    pen = pasternak_penetration(1.0, EI, k, shear)
    l_eff = 1.0 / (2.0 * k * pen)
    return {
        "l_winkler": winkler,
        "l_shear": l_shear,
        "l_soft": l_soft,
        "l_effective": l_eff,
        "lambda_fast": lam_f,
        "lambda_slow": lam_s,
        "regime": "oscillatory" if abs(lam_f.imag) > 1e-12 * abs(lam_f) else "monotone",
    }


# ---------------------------------------------------------------------------
# finite element assembly
# ---------------------------------------------------------------------------
@dataclass
class PasternakBeamModel:
    """Uniform beam strip of ``n_elements`` Hermite elements on a Pasternak bed."""

    length: float
    n_elements: int
    E: float
    Iz: float
    k: float = 0.0
    shear: float = 0.0
    width: float = 1.0

    def __post_init__(self):
        if self.length <= 0.0 or self.E <= 0.0 or self.Iz <= 0.0:
            raise ValueError("length, E and Iz must be positive")
        if self.n_elements < 1:
            raise ValueError("n_elements must be at least 1")
        if self.k < 0.0 or self.shear < 0.0:
            raise ValueError("k and shear must be non-negative")
        if self.k == 0.0 and self.shear == 0.0:
            raise ValueError("a foundation is required (k > 0 or shear > 0)")
        if self.width <= 0.0:
            raise ValueError("width must be positive")

    @property
    def h(self) -> float:
        return self.length / self.n_elements

    @property
    def nodes(self) -> torch.Tensor:
        return torch.linspace(0.0, self.length, self.n_elements + 1, dtype=FLOAT)

    @property
    def ndof(self) -> int:
        return 2 * (self.n_elements + 1)


def assemble_pasternak_beam(model: PasternakBeamModel, nodal_loads=None,
                            uniform_load: float = 0.0, *,
                            dense: bool = False) -> dict:
    """Global system of the beam on a Pasternak bed.

    The grid is a chain of two-node elements, so the global matrix is exactly
    *block tridiagonal* with 2x2 nodal blocks; it is returned in that compact
    form (``D`` diagonal blocks, ``U`` upper coupling blocks, ``f`` load) which
    keeps the cost linear in the number of elements.  ``dense=True`` also
    materialises the full matrix (small systems, cross-checks, unit tests).

    Stiffness contributions: bending (Hermite cubic, exact), Winkler
    ``k int N^T N dx`` (closed form) and the Pasternak shear layer
    ``G_s int N'^T N' dx`` (Gauss quadrature).
    """
    n = model.n_elements
    L = model.h
    ke = pasternak_beam_element_stiffness(model.E, model.Iz, model.k, model.shear, L)
    D = torch.zeros((n + 1, 2, 2), dtype=FLOAT)
    U = torch.zeros((n, 2, 2), dtype=FLOAT)
    for e in range(n):
        D[e] += ke[0:2, 0:2]
        D[e + 1] += ke[2:4, 2:4]
        U[e] = ke[0:2, 2:4]
    f = torch.zeros((n + 1, 2), dtype=FLOAT)
    if uniform_load:
        fe = hermite_consistent_load(L, uniform_load * model.width)
        for e in range(n):
            f[e] += fe[0:2]
            f[e + 1] += fe[2:4]
    if nodal_loads:
        for node, value in nodal_loads.items():
            f[int(node) % (n + 1), 0] += float(value)
    out = {"D": D, "U": U, "f": f, "ke": ke}
    if dense:
        ndof = model.ndof
        K = torch.zeros((ndof, ndof), dtype=FLOAT)
        for e in range(n):
            idx = torch.arange(2 * e, 2 * e + 4)
            K[idx[:, None], idx] += ke
        out["K"] = K
    return out


def _expand_blocks(asm: dict, ndof: int) -> torch.Tensor:
    D, U = asm["D"], asm["U"]
    K = torch.zeros((ndof, ndof), dtype=FLOAT)
    n = U.shape[0]
    for i in range(n + 1):
        sl = slice(2 * i, 2 * i + 2)
        K[sl, sl] += D[i]
    for e in range(n):
        a = slice(2 * e, 2 * e + 2)
        b = slice(2 * e + 2, 2 * e + 4)
        K[a, b] += U[e]
        K[b, a] += U[e].transpose(0, 1)
    return K


def _inv2(a, b, c, d):
    det = a * d - b * c
    return (d / det, -b / det, -c / det, a / det)


def _block_thomas(D: torch.Tensor, U: torch.Tensor, f: torch.Tensor,
                  fixed: torch.Tensor) -> torch.Tensor:
    """Solve a symmetric block-tridiagonal system with zero-prescribed dofs.

    The chain of two-node beam elements makes the global matrix exactly block
    tridiagonal (2x2 nodal blocks), so a block Thomas sweep solves it in linear
    time and memory -- needed because the characteristic-layer studies run
    meshes of >10^4 elements.  Prescribed zero dofs are eliminated by zeroing
    their row/column and placing a unit pivot, which leaves the right-hand side
    unchanged (the prescribed value is zero).  The scalar arithmetic is plain
    Python: per-node torch dispatch would dominate the runtime otherwise.
    """
    m = D.shape[0]                     # number of nodes
    n = U.shape[0]                     # number of element couplings = m - 1
    Dl = D.tolist()
    Ul = U.tolist()
    gl = f.tolist()
    for dof in fixed.tolist():
        node, local = divmod(int(dof), 2)
        for k in range(2):
            Dl[node][local][k] = 0.0
            Dl[node][k][local] = 0.0
        Dl[node][local][local] = 1.0
        gl[node][local] = 0.0
        if node < n:
            Ul[node][local][0] = 0.0
            Ul[node][local][1] = 0.0
        if node > 0:
            Ul[node - 1][0][local] = 0.0
            Ul[node - 1][1][local] = 0.0
    inv = [_inv2(Dl[i][0][0], Dl[i][0][1], Dl[i][1][0], Dl[i][1][1])
           for i in range(m)]
    # forward sweep: D'_{i+1} = D_{i+1} - U_i^T D_i'^{-1} U_i,
    #                f'_{i+1} = f_{i+1} - U_i^T D_i'^{-1} f'_i
    for i in range(n):
        ia, ib, ic, id_ = inv[i]
        u00, u01 = Ul[i][0]
        u10, u11 = Ul[i][1]
        g0, g1 = gl[i]
        v0 = ia * g0 + ib * g1                      # D'^{-1} f'_i
        v1 = ic * g0 + id_ * g1
        w00 = ia * u00 + ib * u10                   # D'^{-1} U_i
        w01 = ia * u01 + ib * u11
        w10 = ic * u00 + id_ * u10
        w11 = ic * u01 + id_ * u11
        Dl[i + 1][0][0] -= u00 * w00 + u10 * w10    # U^T (D'^{-1} U)
        Dl[i + 1][0][1] -= u00 * w01 + u10 * w11
        Dl[i + 1][1][0] -= u01 * w00 + u11 * w10
        Dl[i + 1][1][1] -= u01 * w01 + u11 * w11
        gl[i + 1][0] -= u00 * v0 + u10 * v1
        gl[i + 1][1] -= u01 * v0 + u11 * v1
        inv[i + 1] = _inv2(Dl[i + 1][0][0], Dl[i + 1][0][1],
                           Dl[i + 1][1][0], Dl[i + 1][1][1])
    x = [[0.0, 0.0] for _ in range(m)]
    ia, ib, ic, id_ = inv[m - 1]
    g0, g1 = gl[m - 1]
    x[m - 1][0] = ia * g0 + ib * g1
    x[m - 1][1] = ic * g0 + id_ * g1
    for i in range(m - 2, -1, -1):
        u00, u01 = Ul[i][0]
        u10, u11 = Ul[i][1]
        r0 = gl[i][0] - (u00 * x[i + 1][0] + u01 * x[i + 1][1])
        r1 = gl[i][1] - (u10 * x[i + 1][0] + u11 * x[i + 1][1])
        ia, ib, ic, id_ = inv[i]
        x[i][0] = ia * r0 + ib * r1
        x[i][1] = ic * r0 + id_ * r1
    return torch.tensor(x, dtype=FLOAT)


# 3-point Gauss rule on the unit interval (exact for the quartic integrands)
_GL3 = ((0.5 - 0.5 * math.sqrt(0.6), 5.0 / 18.0),
        (0.5, 8.0 / 18.0),
        (0.5 + 0.5 * math.sqrt(0.6), 5.0 / 18.0))
# 4-point rule: exact to degree 7, needed for the N^T N products (degree 6)
_GL4 = ((0.5 - 0.5 * math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 - math.sqrt(30.0)) / 72.0),
        (0.5 - 0.5 * math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 + math.sqrt(30.0)) / 72.0),
        (0.5 + 0.5 * math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 + math.sqrt(30.0)) / 72.0),
        (0.5 + 0.5 * math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 - math.sqrt(30.0)) / 72.0))


def _basis_at(points, L: float) -> dict:
    """Hermite basis ``N``, ``dN``, ``d2N`` stacked at the given quadrature points."""
    L = float(L)
    Ns, dNs, d2Ns = [], [], []
    for xi, _wt in points:
        Ns.append(torch.tensor([1 - 3 * xi**2 + 2 * xi**3,
                                L * (xi - 2 * xi**2 + xi**3),
                                3 * xi**2 - 2 * xi**3,
                                L * (-xi**2 + xi**3)], dtype=FLOAT))
        dNs.append(torch.tensor([(-6 * xi + 6 * xi**2) / L,
                                 1 - 4 * xi + 3 * xi**2,
                                 (6 * xi - 6 * xi**2) / L,
                                 -2 * xi + 3 * xi**2], dtype=FLOAT))
        d2Ns.append(torch.tensor([(-6 + 12 * xi) / L**2,
                                  (-4 + 6 * xi) / L,
                                  (6 - 12 * xi) / L**2,
                                  (-2 + 6 * xi) / L], dtype=FLOAT))
    return {"N": torch.stack(Ns), "dN": torch.stack(dNs), "d2N": torch.stack(d2Ns)}
# 4-point rule: exact to degree 7, needed for the N^T N products (degree 6)
_GL4 = ((0.5 - 0.5 * math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 - math.sqrt(30.0)) / 72.0),
        (0.5 - 0.5 * math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 + math.sqrt(30.0)) / 72.0),
        (0.5 + 0.5 * math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 + math.sqrt(30.0)) / 72.0),
        (0.5 + 0.5 * math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
         (18.0 - math.sqrt(30.0)) / 72.0))


def _basis_at(points, L: float) -> dict:
    """Hermite basis values ``N``, ``dN``, ``d2N`` stacked at the given points."""
    L = float(L)
    Ns, dNs, d2Ns = [], [], []
    for xi, _wt in points:
        Ns.append(torch.tensor([1 - 3 * xi**2 + 2 * xi**3,
                                L * (xi - 2 * xi**2 + xi**3),
                                3 * xi**2 - 2 * xi**3,
                                L * (-xi**2 + xi**3)], dtype=FLOAT))
        dNs.append(torch.tensor([(-6 * xi + 6 * xi**2) / L,
                                 1 - 4 * xi + 3 * xi**2,
                                 (6 * xi - 6 * xi**2) / L,
                                 -2 * xi + 3 * xi**2], dtype=FLOAT))
        d2Ns.append(torch.tensor([(-6 + 12 * xi) / L**2,
                                  (-4 + 6 * xi) / L,
                                  (6 - 12 * xi) / L**2,
                                  (-2 + 6 * xi) / L], dtype=FLOAT))
    return {"N": torch.stack(Ns), "dN": torch.stack(dNs), "d2N": torch.stack(d2Ns)}


def _element_matrices(model: PasternakBeamModel, L: float) -> dict:
    """Basis values at the Gauss points of the element (``_GL3`` by default)."""
    del model
    return _basis_at(_GL3, L)


def _element_dofs(model: PasternakBeamModel, u: torch.Tensor) -> torch.Tensor:
    """Full dof vector -> ``(n_elements, 4)`` element dof blocks (vectorised)."""
    u = torch.as_tensor(u, dtype=FLOAT).flatten()
    if u.numel() != model.ndof:
        raise ValueError("expected the full 2*(n_elements+1) dof vector")
    pairs = u.reshape(-1, 2)
    return torch.cat((pairs[:-1], pairs[1:]), dim=1)


def bending_moments_pasternak(model: PasternakBeamModel,
                              u: torch.Tensor) -> torch.Tensor:
    """Nodal bending moment ``M = -EI w''`` (sagging positive, ``w`` downward +)."""
    basis = _element_matrices(model, model.h)
    wl = _element_dofs(model, u)
    d2 = basis["d2N"]                      # (3, 4)
    curv = wl @ d2.transpose(0, 1)         # (n_elements, 3)
    left = -model.E * model.Iz * curv[:, 0]
    right = -model.E * model.Iz * curv[:, -1]
    M = torch.empty(model.n_elements + 1, dtype=FLOAT)
    M[:-1] = left
    M[-1] = right[-1]
    if model.n_elements > 1:
        M[1:-1] = 0.5 * (left[1:] + right[:-1])
    return M


def foundation_reaction(model: PasternakBeamModel, u: torch.Tensor) -> dict:
    """Foundation force resultants, split by layer.

    ``winkler``
        ``int k w dx`` -- the *net* vertical reaction of the bed, by Gauss
        quadrature on the Hermite field.  It balances the applied load on a free
        beam (minus any support reactions), because a rigid vertical translation
        of the beam only stretches the Winkler springs: the shear term of the
        weak form contributes nothing to that mode, so the shear layer spreads
        the reaction without carrying net vertical force.
    ``shear_boundary``
        ``-G_s [w']_0^L`` -- the shear traction the layer exerts at the two ends
        of the model, i.e. the boundary term of ``int -G_s w'' dx``.  At a free
        end the natural condition ``EI w3 - G_s w' = 0`` (``w3`` = third
        derivative) makes this traction balance the beam's own internal shear, so
        it is *not* an external force.  It is reported because it is what makes a
        finite model differ from an infinite one; for long, well-resolved models
        it is small.
    ``total``
        Equal to ``winkler`` (the net reaction).  Do **not** add
        ``shear_boundary`` to it when checking global equilibrium.
    """
    basis = _basis_at(_GL3, model.h)
    wl = _element_dofs(model, u)
    wt = torch.tensor([w for _, w in _GL3], dtype=FLOAT)
    value = wl @ basis["N"].transpose(0, 1)
    winkler = float((wt * model.h * model.k * value).sum()) if model.k else 0.0
    theta = torch.as_tensor(u, dtype=FLOAT).flatten()[1::2]
    shear_boundary = (-model.shear * float(theta[-1] - theta[0])
                      if model.shear else 0.0)
    return {"winkler": winkler, "shear_boundary": shear_boundary,
            "total": winkler}


def beam_energy(model: PasternakBeamModel, u: torch.Tensor) -> dict:
    """Strain-energy split ``U = U_bending + U_winkler + U_shear``."""
    basis = _basis_at(_GL4, model.h)
    wl = _element_dofs(model, u)
    wt = torch.tensor([w for _, w in _GL4], dtype=FLOAT)
    scale = 0.5 * model.h * wt
    curv = wl @ basis["d2N"].transpose(0, 1)
    slope = wl @ basis["dN"].transpose(0, 1)
    value = wl @ basis["N"].transpose(0, 1)
    u_b = float((scale * model.E * model.Iz * curv**2).sum())
    u_s = float((scale * model.shear * slope**2).sum()) if model.shear else 0.0
    u_k = float((scale * model.k * value**2).sum()) if model.k else 0.0
    return {"bending": u_b, "winkler": u_k, "shear": u_s,
            "total": u_b + u_k + u_s}


def solve_pasternak_beam(model: PasternakBeamModel, nodal_loads=None,
                         uniform_load: float = 0.0, fixed_dofs=None,
                         supports=None, *, diagnostics: bool = True,
                         dense: bool = False) -> dict:
    """Solve the beam on a Pasternak foundation (bending + Winkler + shear).

    Parameters
    ----------
    nodal_loads : dict[int, float] | None
        Nodal transverse forces (downward positive).
    uniform_load : float
        Transverse load per unit width ``q``, applied as consistent Hermite
        nodal loads.
    fixed_dofs : sequence[int] | None
        P2 dof indices (``2*node`` deflection, ``2*node+1`` rotation) held at
        zero displacement.
    supports : dict[str, sequence[int]] | None
        Convenience form, e.g. ``{"clamped": [0, -1]}`` or ``{"pinned": [0]}``;
        negative indices count from the far end.
    diagnostics : bool
        Also evaluate bending moments, foundation reaction split and the
        strain-energy split (skip for very large meshes).
    dense : bool
        Assemble the full matrix as well; only for small systems (unit tests,
        matrix cross-checks).

    Returns
    -------
    dict
        ``u`` (dof vector), ``w``/``theta`` (nodal fields), ``M`` (sagging
        moment), ``reaction`` (Winkler/shear split), ``applied``,
        ``support_reaction``, ``energy``, ``residual``.
    """
    asm = assemble_pasternak_beam(model, nodal_loads, uniform_load, dense=dense)
    n_nodes = model.n_elements + 1
    fixed = []
    if fixed_dofs is not None:
        fixed.extend(int(d) for d in fixed_dofs)
    if supports:
        for kind, nodes in supports.items():
            if kind not in ("clamped", "pinned", "sliding"):
                raise ValueError("supports kind must be clamped, pinned or sliding")
            for node in nodes:
                node = int(node) % n_nodes
                fixed.append(2 * node)
                if kind == "clamped":
                    fixed.append(2 * node + 1)
    fixed_t = torch.as_tensor(sorted({d % model.ndof for d in fixed}),
                             dtype=torch.long)
    u = _block_thomas(asm["D"], asm["U"], asm["f"], fixed_t).reshape(-1)
    K = asm.get("K") if dense else _expand_blocks(asm, model.ndof)
    residual = K @ u - asm["f"].reshape(-1)
    out = {
        "u": u,
        "w": u[0::2].clone(),
        "theta": u[1::2].clone(),
        "nodes": model.nodes,
        "applied": float(asm["f"][:, 0].sum()),
        "support_reaction": float(residual[fixed_t].sum()) if fixed_t.numel() else 0.0,
        "residual": float(residual.abs().max()),
        "fixed_dofs": fixed_t,
        "mesh": model.h,
    }
    if dense:
        out["K"] = asm["K"]
    if diagnostics:
        out["M"] = bending_moments_pasternak(model, u)
        out["reaction"] = foundation_reaction(model, u)
        out["energy"] = beam_energy(model, u)
    return out


# ---------------------------------------------------------------------------
# mode extraction and scaling-law helpers
# ---------------------------------------------------------------------------
def decay_modes_from_nodes(values, spacing: float,
                           start: int = 2) -> tuple[complex, complex]:
    """Extract the two decay rates of uniformly spaced two-exponential data.

    A function ``w = A exp(-l1 x) + B exp(-l2 x)`` sampled at ``x_n = n h``
    satisfies the three-term linear recurrence ``w_n = p w_{n-1} - q w_{n-2}``
    with ``p = e^{-l1 h} + e^{-l2 h}`` and ``q = e^{-(l1+l2) h}`` (Prony /
    linear-prediction form).  ``(p, q)`` are recovered by least squares over all
    samples from ``start`` onwards (so a kink at the load point is excluded) and
    the rates follow from the roots of ``z^2 - p z + q = 0``.  Returns the rates
    ordered by decreasing real part, ``(lambda_fast, lambda_slow)``.
    """
    w = torch.as_tensor(values, dtype=FLOAT).flatten()
    h = float(spacing)
    if h <= 0.0:
        raise ValueError("spacing must be positive")
    if w.numel() < start + 3:
        raise ValueError("need at least start+3 samples")
    rows = []
    rhs = []
    for n in range(max(start, 2), w.numel()):
        rows.append([float(w[n - 1]), -float(w[n - 2])])
        rhs.append(float(w[n]))
    A = torch.tensor(rows, dtype=FLOAT)
    b = torch.tensor(rhs, dtype=FLOAT)
    sol = torch.linalg.lstsq(A, b).solution
    p, q = float(sol[0]), float(sol[1])
    # z^2 - p z + q = 0
    disc = complex(p * p - 4.0 * q)
    root = cmath.sqrt(disc)
    z1 = (p + root) / 2.0
    z2 = (p - root) / 2.0
    rates = []
    for z in (z1, z2):
        if abs(z) < 1e-300:
            rates.append(complex(float("inf"), 0.0))
        else:
            rates.append(-cmath.log(z) / h)
    rates.sort(key=lambda lam: -lam.real)
    return rates[0], rates[1]


def power_law_exponent(x_values, y_values) -> dict:
    """Fit ``y = c x^m`` by least squares on ``log`` and report the exponent."""
    xs = torch.as_tensor(list(x_values), dtype=FLOAT)
    ys = torch.as_tensor(list(y_values), dtype=FLOAT)
    if xs.numel() != ys.numel() or xs.numel() < 2:
        raise ValueError("need at least two paired samples")
    if bool((xs <= 0).any()) or bool((ys <= 0).any()):
        raise ValueError("power-law fit needs strictly positive data")
    lx, ly = torch.log(xs), torch.log(ys)
    mx, my = float(lx.mean()), float(ly.mean())
    num = float(((lx - mx) * (ly - my)).sum())
    den = float(((lx - mx) ** 2).sum())
    slope = num / den
    intercept = my - slope * mx
    predicted = slope * lx + intercept
    ss_res = float(((ly - predicted) ** 2).sum())
    ss_tot = float(((ly - my) ** 2).sum())
    return {
        "exponent": slope,
        "coefficient": math.exp(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
        "n": int(xs.numel()),
    }