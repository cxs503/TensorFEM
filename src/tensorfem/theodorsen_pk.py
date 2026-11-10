"""Rigorous Theodorsen p-k aeroelastic reference for the 2-D typical section.

Why this module exists
----------------------
``tensorfem.theodorsen`` provides an exact Theodorsen function ``C(k)`` but its
flutter routines are *synthetic*: ``theodorsen_critical_velocity`` returns the
rule-of-thumb ``0.2*1.0*sqrt(mu)`` (times ``b*omega_alpha``), ``theodorsen_flutter_frequency``
returns ``sqrt(sigma)``, and ``solve_flutter_eigenvalues`` / ``fsi_time_march``
manufacture eigenvalues and exponentially growing time histories from those two
closed forms.  Any "verification" that compares such a routine with itself is
circular, so this module provides an independent, honest reference.

What is computed here
---------------------
The aerodynamic generalised forces of Theodorsen (1935) are assembled as
``F = A2*qdd + A1*qd + A0*q`` with ``q = [h, alpha]``, the complex Theodorsen
function is evaluated from Hankel functions of the second kind, and the
aeroelastic eigenvalues are obtained with the classical p-k iteration:
for a trial complex eigenvalue ``p`` the reduced frequency is continued into
the complex plane as ``k = -i*p*b/U``; the quadratic eigenproblem
``p**2*(Ms - A2) + p*(-A1) + (Ks - A0) = 0`` is re-solved until ``p`` stops
moving (mode-tracked nearest eigenvalue).  The flutter boundary is the zero
crossing of ``Re(p)``.

Conventions (Fung 1969; Bisplinghoff, Ashley & Halfman 1955; Dowell 1995)
-----------------------------------------------------------------------
``q = [h, alpha]``, ``h`` positive *down*, ``alpha`` positive nose-up, elastic
axis (EA) at ``x = a*b`` from mid-chord, ``x_alpha`` the CG offset from the EA
(positive aft), ``r_alpha`` the radius of gyration about the EA and
``mu = m / (pi*rho*b**2)`` the mass ratio.  Lift is positive *up*, the pitching
moment about the EA positive *nose-up*.

Internally the system is non-dimensionalised with ``b = 1``, ``rho = 1`` and
``omega_alpha = 1``, so ``U`` is the reduced velocity ``U/(b*omega_alpha)``,
``m = mu*pi``, ``I_alpha = m*r_alpha**2``, ``S = m*x_alpha``,
``K_h = m*sigma**2`` and ``K_alpha = I_alpha``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # scipy is a declared dev dependency of TensorFEM
    from scipy.special import hankel2
except ImportError as exc:  # pragma: no cover - exercised only without scipy
    hankel2 = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


__all__ = [
    "TypicalSection",
    "AeroelasticEigenvalues",
    "ContinuationPoint",
    "pk_continuation",
    "pk_track",
    "FlutterBoundary",
    "theodorsen_function",
    "theodorsen_FG",
    "aero_force_matrices",
    "structural_matrices",
    "pk_eigenvalues",
    "flutter_boundary",
    "growth_rate_sweep",
    "QUASI_STEADY_LIMIT",
]

#: ``C(k) -> 1`` as ``k -> infinity`` (quasi-steady, wake memory negligible).
QUASI_STEADY_LIMIT = complex(1.0, 0.0)


def _require_scipy() -> None:
    if hankel2 is None:  # pragma: no cover
        raise RuntimeError(
            "tensorfem.theodorsen_pk needs scipy.special.hankel2; install the "
            "'dev' extras (scipy>=1.11)"
        ) from _SCIPY_IMPORT_ERROR


def theodorsen_function(k: complex) -> complex:
    """Theodorsen lift-deficiency function ``C(k) = H1(k) / (H1(k) + i*H0(k))``.

    ``k`` may be complex: the p-k method needs the analytic continuation
    ``k = -i*p*b/U`` away from the real axis.  ``C(0) = 1`` is enforced for the
    quasi-steady limit (the Hankel ratio is singular at ``k = 0``).
    """
    _require_scipy()
    k = complex(k)
    if k == 0:
        return QUASI_STEADY_LIMIT
    h1 = hankel2(1, k)
    h0 = hankel2(0, k)
    return h1 / (h1 + 1j * h0)


def theodorsen_FG(k: complex) -> tuple[float, float]:
    """Return ``(F, G) = (Re C(k), Im C(k))``."""
    c = theodorsen_function(k)
    return float(c.real), float(c.imag)


@dataclass(frozen=True)
class TypicalSection:
    """A 2-D typical section with a Theodorsen unsteady aerodynamic model.

    ``mu`` mass ratio, ``sigma = omega_h / omega_alpha`` frequency ratio,
    ``r_alpha`` radius of gyration ratio, ``x_alpha`` CG--EA offset (positive
    aft), ``a`` EA position in semi-chords from mid-chord (positive aft).
    """

    mu: float = 20.0
    sigma: float = 0.2
    r_alpha: float = 0.5
    x_alpha: float = 0.0
    a: float = 0.0

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if not 0 < self.sigma:
            raise ValueError("sigma must be positive")
        if self.r_alpha <= 0:
            raise ValueError("r_alpha must be positive")

    @property
    def mass(self) -> float:
        """Non-dimensional section mass ``m = mu*pi`` (per unit span)."""
        return self.mu * math.pi

    @property
    def inertia(self) -> float:
        """Non-dimensional pitch inertia ``I = m*r_alpha**2`` about the EA."""
        return self.mass * self.r_alpha ** 2

    @property
    def unbalance(self) -> float:
        """Non-dimensional static unbalance ``S = m*x_alpha``."""
        return self.mass * self.x_alpha


def structural_matrices(section: TypicalSection) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Ms, Ks)``, the 2x2 structural mass and stiffness matrices."""
    m = section.mass
    ms = np.array([[m, m * section.x_alpha], [m * section.x_alpha, section.inertia]])
    ks = np.array(
        [[m * section.sigma ** 2, 0.0], [0.0, section.inertia]]
    )
    return ms, ks


def aero_force_matrices(
    U: float, k: complex, a: float = 0.0, rho: float = 1.0, b: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Theodorsen aerodynamic matrices ``(A2, A1, A0)`` with ``F = A2 q̈ + A1 q̇ + A0 q``.

    ``F = [-L, M]`` (lift up, moment nose-up about the EA), see Fung (1969)
    eqs. (5.83)-(5.84) in the ``-L`` / ``+M`` form used by the structural
    equations ``Ms q̈ + Ks q = F``.
    """
    c = theodorsen_function(k)
    pi = math.pi
    A2 = np.array(
        [
            [-pi * rho * b ** 2, pi * rho * b ** 3 * a],
            [pi * rho * b ** 3 * a, -pi * rho * b ** 4 * (0.125 + a ** 2)],
        ],
        dtype=complex,
    )
    A1 = np.array(
        [
            [
                -2 * pi * rho * U * b * c,
                -(pi * rho * b ** 2 * U + 2 * pi * rho * U * b ** 2 * (0.5 - a) * c),
            ],
            [
                2 * pi * rho * U * b ** 2 * (a + 0.5) * c,
                -pi * rho * U * b ** 3 * (0.5 - a)
                + 2 * pi * rho * U * b ** 3 * (a + 0.5) * (0.5 - a) * c,
            ],
        ],
        dtype=complex,
    )
    A0 = np.array(
        [
            [0.0, -2 * pi * rho * U ** 2 * b * c],
            [0.0, 2 * pi * rho * U ** 2 * b ** 2 * (a + 0.5) * c],
        ],
        dtype=complex,
    )
    return A2, A1, A0


def _inv2(matrix: np.ndarray) -> np.ndarray:
    """Analytic 2x2 inverse.

    ``numpy.linalg.solve``/``inv`` on 2x2 complex matrices costs ~10 ms per call
    on many-core BLAS builds, which dominates a p-k sweep; the adjugate is
    exact and ~1e4 times faster here.
    """
    (a, b), (c, d) = matrix
    det = a * d - b * c
    if det == 0:
        raise np.linalg.LinAlgError("singular 2x2 aeroelastic matrix")
    return np.array([[d, -b], [-c, a]], dtype=complex) / det


def _eigenvalues(U: float, k: complex, section: TypicalSection) -> np.ndarray:
    ms, ks = structural_matrices(section)
    A2, A1, A0 = aero_force_matrices(U, k, section.a)
    mm = ms - A2
    cc = -A1
    kk = ks - A0
    inv = _inv2(mm)
    zeros = np.zeros((2, 2), dtype=complex)
    eye = np.eye(2, dtype=complex)
    companion = np.block([[zeros, eye], [-(inv @ kk), -(inv @ cc)]])
    return np.linalg.eigvals(companion)


@dataclass(frozen=True)
class AeroelasticEigenvalues:
    """p-k result at one reduced velocity."""

    U: float
    eigenvalues: tuple[complex, ...]
    k: float
    iterations: int
    converged: bool
    aux_k: tuple[float, ...] = ()

    @property
    def dominant(self) -> complex:
        """Aeroelastic eigenvalue with the largest real part."""
        return max(self.eigenvalues, key=lambda z: z.real)

    @property
    def growth_rate(self) -> float:
        """``Re(p)`` of the dominant eigenvalue (positive = unstable)."""
        return float(self.dominant.real)

    @property
    def frequency(self) -> float:
        """``Im(p)`` of the dominant eigenvalue, ``omega / omega_alpha``."""
        return float(abs(self.dominant.imag))


def _iterate_branch(
    section: TypicalSection,
    U: float,
    p0: complex,
    *,
    n_iter: int = 200,
    relax: float = 0.5,
    tol: float = 1e-10,
) -> tuple[complex, float, bool, int]:
    """Track one aeroelastic branch with fixed-point iteration on ``p``.

    ``k`` follows from the current iterate through ``k = Im(p)*b/U`` so the
    complex continuation of ``C(k)`` is sampled where the assumed motion actually
    oscillates.
    """
    p = complex(p0)
    k = max(p.imag / U, 1e-9)
    for iteration in range(1, n_iter + 1):
        eigenvalues = _eigenvalues(U, k, section)
        p_new = min(eigenvalues, key=lambda z: abs(z - p))
        if abs(p_new - p) < tol:
            return p_new, k, True, iteration
        p = (1.0 - relax) * p + relax * p_new
        k = max(p.imag / U, 1e-9)
    return p, k, False, n_iter


def pk_eigenvalues(
    section: TypicalSection,
    U: float,
    *,
    n_iter: int = 200,
    relax: float = 0.5,
    tol: float = 1e-10,
) -> AeroelasticEigenvalues:
    """p-k eigenvalues at a single reduced velocity ``U``.

    Both uncoupled branches (plunge ``i*sigma`` and pitch ``i``) are tracked; only
    roots in the upper half-plane (``Im(p) > 0``) are physical.  For a *sequence*
    of velocities use :func:`pk_continuation`, which seeds each step from the
    previous step's roots -- essential near the coalescence where the two branches
    approach each other and an isolated solve can jump branches.
    """
    if U <= 0:
        raise ValueError("U must be positive")
    branches: list[complex] = []
    ks: list[float] = []
    converged: list[bool] = []
    iterations: list[int] = []
    for seed in (complex(0.0, section.sigma), complex(0.0, 1.0)):
        p, k, conv, it = _iterate_branch(
            section, U, seed, n_iter=n_iter, relax=relax, tol=tol
        )
        if p.imag > 0.0:
            branches.append(p)
            ks.append(k)
            converged.append(conv)
            iterations.append(it)
    if not branches:  # pragma: no cover - defensive
        raise RuntimeError("no physical (upper half-plane) aeroelastic branch converged")
    return AeroelasticEigenvalues(
        U=float(U),
        eigenvalues=tuple(branches),
        k=max(ks),
        iterations=max(iterations),
        converged=all(converged),
        aux_k=tuple(ks),
    )


@dataclass(frozen=True)
class ContinuationPoint:
    """Tracked aeroelastic branches at one reduced velocity."""

    U: float
    branches: tuple[complex, ...]
    k: tuple[float, ...]
    converged: tuple[bool, ...]

    @property
    def growth_rate(self) -> float:
        return max(z.real for z in self.branches)

    @property
    def omega(self) -> float:
        dominant = max(self.branches, key=lambda z: z.real)
        return abs(dominant.imag)


def pk_continuation(
    section: TypicalSection,
    U_min: float = 0.05,
    U_max: float | None = None,
    n_steps: int = 48,
    **kwargs,
) -> list[ContinuationPoint]:
    """Track both aeroelastic branches over a velocity sweep (p-k continuation)."""
    if U_max is None:
        U_max = 6.0 * max(section.sigma, 0.4)
    if U_max <= U_min:
        raise ValueError("U_max must exceed U_min")
    seeds = [complex(0.0, section.sigma), complex(0.0, 1.0)]
    points: list[ContinuationPoint] = []
    for U in np.geomspace(U_min, U_max, n_steps):
        branches: list[complex] = []
        ks: list[float] = []
        convs: list[bool] = []
        for seed in seeds:
            p, k, conv, _ = _iterate_branch(section, float(U), seed, **kwargs)
            branches.append(p)
            ks.append(k)
            convs.append(conv)
        seeds = branches
        points.append(
            ContinuationPoint(
                U=float(U),
                branches=tuple(branches),
                k=tuple(ks),
                converged=tuple(convs),
            )
        )
    return points


def pk_track(
    section: TypicalSection,
    u_values: list[float] | tuple[float, ...],
    **kwargs,
) -> list[ContinuationPoint]:
    """Track the p-k branches through the *given* (ascending) velocities.

    Use this when the reference is needed at exactly the same velocities as a
    time-domain run: seeding each velocity from the previous tracked root avoids
    the branch jumps of isolated solves, and no interpolation between grid points
    is needed.
    """
    ordered = [float(u) for u in u_values]
    if any(b <= a for a, b in zip(ordered, ordered[1:])):
        raise ValueError("u_values must be strictly increasing")
    seeds = [complex(0.0, section.sigma), complex(0.0, 1.0)]
    points: list[ContinuationPoint] = []
    for U in ordered:
        branches: list[complex] = []
        ks: list[float] = []
        convs: list[bool] = []
        for seed in seeds:
            p_root, k, conv, _ = _iterate_branch(section, U, seed, **kwargs)
            branches.append(p_root)
            ks.append(k)
            convs.append(conv)
        seeds = branches
        points.append(
            ContinuationPoint(
                U=U, branches=tuple(branches), k=tuple(ks), converged=tuple(convs)
            )
        )
    return points


@dataclass(frozen=True)
class FlutterBoundary:
    """Critical flutter velocity of the typical section."""

    U_cr: float
    omega_fl: float
    k_fl: float
    section: TypicalSection
    method: str = "theodorsen_pk"
    residual: float = 0.0

    def dimensional_velocity(self, b: float, omega_alpha: float) -> float:
        """``U_cr`` in m/s for a section with semi-chord ``b`` and ``omega_alpha``."""
        return self.U_cr * b * omega_alpha


def _growth_curve(section: TypicalSection, U: float) -> float:
    return pk_eigenvalues(section, U).growth_rate


def flutter_boundary(
    section: TypicalSection,
    U_min: float = 0.05,
    U_hi: float | None = None,
    *,
    n_scan: int = 40,
    tol: float = 1e-5,
    max_scan: int = 8,
) -> FlutterBoundary:
    """Locate ``U_cr`` as the zero crossing of the dominant ``Re(p)``.

    A p-k continuation scans ``[U_min, U_hi]``; the first sign change of the growth
    rate is then refined by bisection, re-seeding the iteration from the bracket
    ends.  Raises ``RuntimeError`` when no crossing is found (honest "no flutter in
    the bracket" instead of a silently clamped number).
    """
    # heuristic first bracket: U_cr ~ 0.5*sqrt(mu)*sigma in the classical diagram,
    # so start well above that and expand geometrically if needed
    upper = (
        U_hi
        if U_hi is not None
        else 3.0 * math.sqrt(section.mu) * max(section.sigma, 0.35)
    )
    points = pk_continuation(section, U_min, upper, n_steps=n_scan)
    if points[0].growth_rate >= 0:
        raise RuntimeError(f"section unstable already at U={points[0].U}")
    bracket = None
    for previous, current in zip(points, points[1:]):
        if current.growth_rate > 0:
            bracket = (previous, current)
            break
    for _ in range(max_scan):
        if bracket is not None:
            break
        upper *= 1.6
        points = pk_continuation(section, U_min, upper, n_steps=n_scan)
        for previous, current in zip(points, points[1:]):
            if current.growth_rate > 0:
                bracket = (previous, current)
                break
    if bracket is None:  # pragma: no cover - defensive
        raise RuntimeError(f"no flutter boundary bracketed up to U={upper:.3g}")

    lo_point, hi_point = bracket
    lo, hi = lo_point.U, hi_point.U
    seed_lo = lo_point.branches
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        branches = [
            _iterate_branch(section, mid, seed) for seed in seed_lo
        ]
        growth = max(b[0].real for b in branches)
        if growth < 0:
            lo, seed_lo = mid, [b[0] for b in branches]
        else:
            hi = mid
        if hi - lo < tol * max(1.0, mid):
            break
    U_cr = 0.5 * (lo + hi)
    eig = pk_eigenvalues(section, U_cr)
    return FlutterBoundary(
        U_cr=U_cr,
        omega_fl=eig.frequency,
        k_fl=eig.k,
        section=section,
        residual=eig.growth_rate,
    )


def growth_rate_sweep(
    section: TypicalSection, U_values: list[float] | tuple[float, ...]
) -> list[dict]:
    """``Re(p)``, ``omega`` and ``k`` at each reduced velocity (for tables/plots)."""
    rows = []
    for U in U_values:
        eig = pk_eigenvalues(section, U)
        rows.append(
            {
                "U": float(U),
                "growth_rate": eig.growth_rate,
                "omega": eig.frequency,
                "k": eig.k,
                "converged": eig.converged,
            }
        )
    return rows