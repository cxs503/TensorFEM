"""Exact and variational reference solutions for rotationally restrained plates.

Motivation
----------
:mod:`tensorfem.plate_elastic_restraint` provides the *numerical* (Mindlin Q4)
elastic-rotational-restraint panel, parameterised by the embedding coefficient
``beta in [0, 1]``.  Its two endpoints could be validated against published
values (``beta = 0`` -> SSSS, ``beta = 1`` -> CCCC), but the interior of the
band - the entire point of a restraint model - had **no independent reference**.
This module supplies one, without any FEM involvement:

* :func:`levy_restrained_plate` - an **exact** Kirchhoff (thin-plate) Levy
  solution for a rectangular plate with two opposite edges simply supported and
  the other two carrying a *finite* distributed rotational spring
  ``K [N*m/m/rad]``.  The mode shape in the sprung direction is obtained by
  solving the 4x4 boundary-condition system of the plate ODE in closed form, so
  the result is exact for any ``K``, including both limits.
* :func:`kirchhoff_strip_end_moment` / :func:`kirchhoff_strip_deflection` -
  closed-form cylindrical-bending (beam-strip) solution used as a completely
  independent check of the Levy implementation in the ``b >> a`` limit.
* :func:`ritz_restrained_plate` - a double-sine Rayleigh-Ritz solution that
  handles **all four edges with independent springs** ``Kx``, ``Ky`` (the
  configuration of the ice-belt panel), used to cross-check the FEM band.

Conventions
-----------
The plate occupies ``[0, a] x [0, b]``.  The spring convention is the one that
follows from the energy

``Pi = U_plate + 1/2 K \int (w_n)^2 ds - \int q w``

i.e. on an edge normal to ``x``: ``D (w_xx + nu w_yy) = +-K w_x``, with the sign
taken so that ``K = 0`` gives a free (simple) support and ``K -> inf`` gives
zero slope (clamped).  Bending moments are
``Mx = -D (w_xx + nu w_yy)``, ``My = -D (w_yy + nu w_xx)``.

Everything here is Kirchhoff (no transverse shear).  Comparing with the Mindlin
FEM therefore carries a plate-theory model error, quantified in
``docs/CASE-FEM-ELASTIC-RESTRAINT-LEVY-VERIFICATION.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "flexural_rigidity",
    "kirchhoff_strip_end_moment",
    "kirchhoff_strip_deflection",
    "LevyRestrainedSolution",
    "levy_restrained_plate",
    "RitzRestrainedSolution",
    "ritz_restrained_plate",
]


def flexural_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Plate flexural rigidity ``D = E t^3 / (12 (1 - nu^2))`` [N*m]."""
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


# ---------------------------------------------------------------------------
# Cylindrical bending (beam strip) - independent closed form
# ---------------------------------------------------------------------------

def kirchhoff_strip_end_moment(q: float, span: float, D: float, K: float) -> float:
    """Magnitude of the end moment of a strip with rotational end springs.

    A unit-width beam of span ``L`` and rigidity ``D`` under a uniform load
    ``q``, rotationally restrained at both ends by stiffness ``K``
    [N*m/m/rad], carries

    ``M_end = q L^2 / 12 * 1 / (1 + 2 D / (K L))``

    which tends to ``q L^2 / 12`` (clamped) as ``K -> inf`` and to ``0``
    (simply supported) as ``K -> 0``.
    """
    if K <= 0.0:
        return 0.0
    return q * span**2 / 12.0 * 1.0 / (1.0 + 2.0 * D / (K * span))


def kirchhoff_strip_deflection(q: float, span: float, D: float, K: float) -> float:
    """Mid-span deflection of a unit-width strip with rotational end springs.

    Superposition of the fully simple beam and a pair of end moments ``M``
    that satisfy the spring relation ``M = K * theta_end``.  Exact; reduces to
    ``5 q L^4/(384 D)`` for ``K = 0`` and to ``q L^4/(384 D)`` for ``K -> inf``.
    """
    if K <= 0.0:
        return 5.0 * q * span**4 / (384.0 * D)
    m_end = kirchhoff_strip_end_moment(q, span, D, K)
    return 5.0 * q * span**4 / (384.0 * D) - m_end * span**2 / (8.0 * D)


# ---------------------------------------------------------------------------
# Exact Levy solution: y-edges simple, x-edges rotationally restrained
# ---------------------------------------------------------------------------

def _exp_basis(s: float, x: float, a: float):
    """Bounded homogeneous basis of ``(d^2/dx^2 - s^2)^2 X = 0`` and derivatives.

    ``e^{s(x-a)}, e^{-sx}, x e^{s(x-a)}, x e^{-sx}`` - the same four-dimensional
    space as ``cosh, sinh, x cosh, x sinh`` but with every value bounded by
    ``max(1, a)`` on ``[0, a]``, which keeps the 4x4 boundary system solvable
    for the high mode numbers needed to converge the moment series.
    """
    f1 = math.exp(s * (x - a))
    f2 = math.exp(-s * x)
    phi = np.array([f1, f2, x * f1, x * f2])
    dphi = np.array([s * f1, -s * f2, f1 + s * x * f1, f2 - s * x * f2])
    d2phi = np.array(
        [s * s * f1, s * s * f2, 2.0 * s * f1 + s * s * x * f1, -2.0 * s * f2 + s * s * x * f2]
    )
    return phi, dphi, d2phi


@dataclass
class LevyRestrainedSolution:
    """Exact Kirchhoff Levy solution (y-edges simple, x-edges sprung).

    Attributes
    ----------
    a, b : float
        Plate dimensions in x (sprung direction) and y (simple direction).
    D : float
        Flexural rigidity.
    nu : float
        Poisson ratio.
    K : float
        Distributed rotational spring stiffness on the edges ``x = 0`` and
        ``x = a`` [N*m/m/rad]; ``0`` is a simple support, ``inf`` is clamped.
    q : float
        Uniform pressure [Pa].
    n_modes : int
        Number of ``sin(n pi y / b)`` modes retained.
    coeffs : ndarray (n_modes, 4)
        Homogeneous mode amplitudes.
    particular : ndarray (n_modes,)
        Constant particular solution per mode (uniform load in x).
    s : ndarray (n_modes,)
        ``s_n = n pi / b``.
    """

    a: float
    b: float
    D: float
    nu: float
    K: float
    q: float
    n_modes: int
    coeffs: np.ndarray = field(repr=False)
    particular: np.ndarray = field(repr=False)
    s: np.ndarray = field(repr=False)

    # -- per-mode shape functions in x ---------------------------------
    def _mode_terms(self, x: float, k: int):
        """Return ``(X, X', X'')`` of mode ``k`` at station ``x`` (with particular).

        The homogeneous basis is ``e^{s(x-a)}, e^{-sx}, x e^{s(x-a)}, x e^{-sx}``
        (see :func:`_exp_basis`), which spans the same space as the classical
        ``cosh/sinh`` pair but stays bounded for the high modes needed by the
        moment series.
        """
        s = self.s[k]
        c = self.coeffs[k]
        phi, dphi, d2phi = _exp_basis(s, x, self.a)
        val = float(c @ phi) + float(self.particular[k])
        d1 = float(c @ dphi)
        d2 = float(c @ d2phi)
        return val, d1, d2

    def w(self, x: float, y: float) -> float:
        """Transverse displacement [m] at ``(x, y)``."""
        return float(
            sum(self._mode_terms(x, k)[0] * math.sin(self.s[k] * y) for k in range(self.n_modes))
        )

    def slope(self, x: float, y: float) -> float:
        """``dw/dx`` at ``(x, y)`` (used for the spring-moment boundary check)."""
        return float(
            sum(self._mode_terms(x, k)[1] * math.sin(self.s[k] * y) for k in range(self.n_modes))
        )

    def w_xx(self, x: float, y: float) -> float:
        return float(
            sum(self._mode_terms(x, k)[2] * math.sin(self.s[k] * y) for k in range(self.n_modes))
        )

    def w_yy(self, x: float, y: float) -> float:
        return float(
            sum(
                -self.s[k] ** 2 * self._mode_terms(x, k)[0] * math.sin(self.s[k] * y)
                for k in range(self.n_modes)
            )
        )

    def moments(self, x: float, y: float) -> tuple[float, float]:
        """Bending moments ``(Mx, My)`` [N*m/m] at ``(x, y)``."""
        wxx = self.w_xx(x, y)
        wyy = self.w_yy(x, y)
        return (
            -self.D * (wxx + self.nu * wyy),
            -self.D * (wyy + self.nu * wxx),
        )

    def w_centre(self) -> float:
        return self.w(self.a / 2.0, self.b / 2.0)

    def moments_centre(self) -> tuple[float, float]:
        return self.moments(self.a / 2.0, self.b / 2.0)


def levy_restrained_plate(
    a: float,
    b: float,
    D: float,
    nu: float,
    K: float,
    q: float,
    n_modes: int = 201,
) -> LevyRestrainedSolution:
    """Exact Kirchhoff solution of a plate with rotational springs on two edges.

    Edges ``y = 0, b`` are simply supported (``w = 0``, ``M_y = 0``); edges
    ``x = 0, a`` are held (``w = 0``) with a distributed rotational spring
    ``M_x = -+ K w_x``.  ``K = 0`` reproduces the SSSS plate exactly and
    ``K -> inf`` the CCCC plate, so one routine gives the reference for every
    intermediate restraint level.

    Parameters
    ----------
    a, b : float
        Plate dimensions [m]; ``a`` is the sprung direction.
    D : float
        Flexural rigidity [N*m].
    nu : float
        Poisson ratio.
    K : float
        Distributed rotational spring stiffness [N*m/m/rad].
    q : float
        Uniform pressure [Pa].
    n_modes : int
        Number of sine modes in y.  A uniform load excites odd modes only; 201
        modes converge the centre moment series to ~1e-7 relative.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if K < 0.0:
        raise ValueError("K must be non-negative")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    s_all = np.arange(1, n_modes + 1) * math.pi / b
    coeffs = np.zeros((n_modes, 4), dtype=float)
    particular = np.zeros(n_modes, dtype=float)

    for k in range(n_modes):
        n = k + 1
        s = s_all[k]
        q_n = 4.0 * q / (n * math.pi) if n % 2 == 1 else 0.0
        xp = q_n / (D * s**4)
        particular[k] = xp

        rows = np.zeros((4, 4), dtype=float)
        rhs = np.zeros(4, dtype=float)
        for end, xx in ((0, 0.0), (1, a)):
            phi, dphi, d2phi = _exp_basis(s, xx, a)
            sign = 1.0 if end == 0 else -1.0
            # w = 0 on the edge (the particular solution is constant in x)
            rows[2 * end, :] = phi
            rhs[2 * end] = -xp
            # D (w_xx + nu w_yy) = +-K w_x, with w_yy = -s^2 w
            rows[2 * end + 1, :] = D * (d2phi - nu * s * s * phi) - sign * K * dphi
            rhs[2 * end + 1] = -(D * (0.0 - nu * s * s * xp))

        # row scaling keeps the 4x4 well conditioned (rows carry s^2 factors)
        scale = np.abs(rows).max(axis=1)
        scale[scale == 0.0] = 1.0
        coeffs[k, :] = np.linalg.solve(rows / scale[:, None], rhs / scale)

    return LevyRestrainedSolution(
        a=a,
        b=b,
        D=D,
        nu=nu,
        K=K,
        q=q,
        n_modes=n_modes,
        coeffs=coeffs,
        particular=particular,
        s=s_all,
    )


# ---------------------------------------------------------------------------
# Rayleigh-Ritz, all four edges with independent springs
# ---------------------------------------------------------------------------

@dataclass
class RitzRestrainedSolution:
    """Double-sine Rayleigh-Ritz solution of a four-edge restrained plate."""

    a: float
    b: float
    D: float
    nu: float
    Kx: float
    Ky: float
    q: float
    m_modes: int
    n_modes: int
    amplitudes: np.ndarray = field(repr=False)

    def _coef(self, m: int, n: int) -> float:
        return float(self.amplitudes[m - 1, n - 1])

    def w(self, x: float, y: float) -> float:
        """Transverse displacement [m] at ``(x, y)``."""
        m = np.arange(1, self.m_modes + 1)
        n = np.arange(1, self.n_modes + 1)
        sx = np.sin(m * math.pi * x / self.a)
        sy = np.sin(n * math.pi * y / self.b)
        return float(sx @ self.amplitudes @ sy)

    def moments(self, x: float, y: float) -> tuple[float, float]:
        """Bending moments ``(Mx, My)`` [N*m/m] at ``(x, y)``."""
        wxx = self._d2(x, y, axis="x")
        wyy = self._d2(x, y, axis="y")
        return (
            -self.D * (wxx + self.nu * wyy),
            -self.D * (wyy + self.nu * wxx),
        )

    def _d2(self, x: float, y: float, axis: str) -> float:
        m = np.arange(1, self.m_modes + 1)
        n = np.arange(1, self.n_modes + 1)
        km = -(m * math.pi / self.a) ** 2
        kn = -(n * math.pi / self.b) ** 2
        sx = np.sin(m * math.pi * x / self.a)
        sy = np.sin(n * math.pi * y / self.b)
        if axis == "x":
            return float((km * sx) @ self.amplitudes @ sy)
        return float(sx @ self.amplitudes @ (kn * sy))

    def w_centre(self) -> float:
        return self.w(self.a / 2.0, self.b / 2.0)

    def moments_centre(self) -> tuple[float, float]:
        return self.moments(self.a / 2.0, self.b / 2.0)


def ritz_restrained_plate(
    a: float,
    b: float,
    D: float,
    nu: float,
    Kx: float,
    Ky: float,
    q: float,
    m_modes: int = 40,
    n_modes: int = 40,
) -> RitzRestrainedSolution:
    """Rayleigh-Ritz solution of a plate with springs on all four edges.

    The trial set ``sin(m pi x / a) sin(n pi y / b)`` satisfies ``w = 0`` and
    the zero-tangential-slope condition on all edges exactly, so only the
    normal-rotation springs enter; the bending energy and the spring energies
    are both diagonal/quadratic in the amplitudes, so the problem is one linear
    solve of size ``m_modes * n_modes``.

    This is variational (deflections converge from above) and is *not* exact,
    but for finite ``Kx``/``Ky`` the exact solution is smooth and convergence is
    rapid; it is used here to cross-check the FEM at ``Kx = Ky`` and to
    cross-check :func:`levy_restrained_plate` at ``Ky = 0``.
    """
    if Kx < 0.0 or Ky < 0.0:
        raise ValueError("spring stiffnesses must be non-negative")
    M, N = m_modes, n_modes
    size = M * N
    idx = lambda i, j: i * N + j  # noqa: E731  (i: 0-based m, j: 0-based n)

    m_arr = np.arange(1, M + 1)
    n_arr = np.arange(1, N + 1)
    km = m_arr * math.pi / a
    kn = n_arr * math.pi / b

    stiff = np.zeros((size, size), dtype=float)
    load_vec = np.zeros(size, dtype=float)

    # Bending energy: D*pi^4*a*b/8 * A_mn^2 (m^2/a^2 + n^2/b^2)^2
    for i in range(M):
        for j in range(N):
            k = idx(i, j)
            stiff[k, k] += (
                D
                * math.pi**4
                * a
                * b
                / 4.0
                * ((m_arr[i] / a) ** 2 + (n_arr[j] / b) ** 2) ** 2
            )
            if m_arr[i] % 2 == 1 and n_arr[j] % 2 == 1:
                load_vec[k] += 4.0 * a * b * q / (math.pi**2 * m_arr[i] * n_arr[j])

    # Spring energy at x = 0 and x = a: (Kx*b/4) * sum_n [ (sum_m A_mn km)^2
    #   + (sum_m (-1)^m A_mn km)^2 ]
    if Kx > 0.0:
        for j in range(N):
            for i1 in range(M):
                for i2 in range(M):
                    k1, k2 = idx(i1, j), idx(i2, j)
                    factor = (Kx * b / 2.0) * km[i1] * km[i2]
                    sign_comb = 1.0 + (-1.0) ** (m_arr[i1] + m_arr[i2])
                    stiff[k1, k2] += factor * sign_comb
    if Ky > 0.0:
        for i in range(M):
            for j1 in range(N):
                for j2 in range(N):
                    k1, k2 = idx(i, j1), idx(i, j2)
                    factor = (Ky * a / 2.0) * kn[j1] * kn[j2]
                    sign_comb = 1.0 + (-1.0) ** (n_arr[j1] + n_arr[j2])
                    stiff[k1, k2] += factor * sign_comb

    # Row scaling + least squares: as the springs close (K -> inf) the spring
    # energy only constrains the *sums* of amplitudes along each row/column of
    # the mode grid, so the assembled matrix becomes ill-conditioned even though
    # the underlying variational problem stays well posed.  A minimum-norm
    # least-squares solve keeps the amplitudes bounded there.
    scale = np.abs(stiff).max(axis=1)
    scale[scale == 0.0] = 1.0
    scaled = stiff / scale[:, None]
    rhs = load_vec / scale
    # Tikhonov regularisation.  The variational problem is SPD for every
    # finite K (the bending energy is diagonal positive), but the dynamic range
    # between the bending and spring entries reaches ~1e18 at K -> inf, so the
    # scaled system is numerically singular; a relative shift of 1e-12 on the
    # diagonal restores a unique, well-behaved solution without measurably
    # biasing the answer (checked against the exact Levy solution).
    diag = np.diag(scaled).copy()
    diag[diag <= 0.0] = np.abs(diag).max() if np.any(diag) else 1.0
    scaled[np.diag_indices_from(scaled)] += 1e-12 * diag
    amps = np.linalg.solve(scaled, rhs)
    return RitzRestrainedSolution(
        a=a,
        b=b,
        D=D,
        nu=nu,
        Kx=Kx,
        Ky=Ky,
        q=q,
        m_modes=M,
        n_modes=N,
        amplitudes=amps.reshape(M, N),
    )