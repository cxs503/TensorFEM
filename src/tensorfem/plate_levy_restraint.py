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
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "flexural_rigidity",
    "kirchhoff_strip_end_moment",
    "kirchhoff_strip_deflection",
    "LevyRestrainedSolution",
    "levy_restrained_plate",
    "LevyStepLoadSolution",
    "levy_restrained_plate_step_load",
    "levy_restrained_plate_patch",
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


def _exp_basis3(s: float, x: float, a: float):
    """:func:`_exp_basis` plus the third derivative.

    The homogeneous space of ``(d^2/dx^2 - s^2)^2 X = 0`` is fixed by four
    conditions; joining two segments of a piecewise-constant load requires
    ``X, X', X'', X'''`` to be continuous, hence the third derivative here.
    """
    f1 = math.exp(s * (x - a))
    f2 = math.exp(-s * x)
    phi = np.array([f1, f2, x * f1, x * f2])
    dphi = np.array([s * f1, -s * f2, f1 + s * x * f1, f2 - s * x * f2])
    d2phi = np.array(
        [s * s * f1, s * s * f2, 2.0 * s * f1 + s * s * x * f1, -2.0 * s * f2 + s * s * x * f2]
    )
    d3phi = np.array(
        [
            s**3 * f1,
            -(s**3) * f2,
            3.0 * s * s * f1 + s**3 * x * f1,
            3.0 * s * s * f2 - s**3 * x * f2,
        ]
    )
    return phi, dphi, d2phi, d3phi


#: Condition-number ceiling for one mode of the step-load system.  The
#: boundary-condition matrix of a mode with wavenumber ``s`` has dynamic range
#: ``exp(2 s a)``, so beyond this ceiling the coefficients are dominated by
#: round-off.  Such modes are *dropped*: their particular amplitude is
#: ``q_n / (D s^4)`` and their bending-moment scale ``q_n / s^2``, both far below
#: the series value, and the convergence plateau is asserted in
#: ``tests/test_plate_levy_patch.py``.
_STEP_LOAD_COND_LIMIT = 1e14


def _normalise_pieces(segments, a: float):
    """Clip, sort and gap-fill a piecewise-constant pressure description.

    ``segments`` is a sequence of ``(x0, x1, q)``; gaps inside ``[0, a]`` are
    filled with ``q = 0`` so the returned list always covers ``[0, a]``
    contiguously (required by the continuity join).
    """
    pieces = []
    for seg in segments:
        if len(seg) != 3:
            raise ValueError("each segment must be (x0, x1, q)")
        x0, x1, q = (float(v) for v in seg)
        if x1 < x0:
            raise ValueError("segment x1 must be >= x0")
        if x1 == x0:
            continue
        if x1 <= 0.0 or x0 >= a:
            continue
        pieces.append([max(0.0, x0), min(a, x1), q])
    pieces.sort(key=lambda p: (p[0], p[1]))
    out: list[list[float]] = []
    for x0, x1, q in pieces:
        if out and x0 < out[-1][1] - 1e-12 * a:
            raise ValueError("overlapping load segments")
        if out and x0 > out[-1][1] + 1e-12 * a:
            out.append([out[-1][1], x0, 0.0])
        out.append([x0, x1, q])
    if not out:
        raise ValueError("empty load description")
    if out[0][0] > 0.0:
        out.insert(0, [0.0, out[0][0], 0.0])
    if out[-1][1] < a:
        out.append([out[-1][1], a, 0.0])
    return tuple((float(x0), float(x1), float(q)) for x0, x1, q in out)


@dataclass
class LevyStepLoadSolution:
    """Exact Kirchhoff Levy solution for a *piecewise-constant* patch load.

    Same boundary configuration as :class:`LevyRestrainedSolution` (``y = 0, b``
    simply supported, ``x = 0, a`` on rotational springs), but the pressure is
    ``q(x, y) = q_j`` on the strip ``x in [x0_j, x1_j]`` and independent of ``y``
    - the JOUR-ICE-STRUCT patch idealisation (full breadth x partial length).

    Exactness: with ``w = X(x) sin(s_n y)``, ``s_n = n pi / b``, the plate PDE
    ``D lap^2 w = q`` becomes one ODE per mode,
    ``D (X'''' - 2 s^2 X'' + s^4 X) = q_n(x)``, with
    ``q_n = 4 q(x) / (n pi)`` for odd ``n`` (zero otherwise).  On a segment of
    constant ``q_n`` a particular integral is the constant ``q_n / (D s^4)``, and
    the general solution is that constant plus the four-dimensional homogeneous
    basis of :func:`_exp_basis3`.  The ``4 x n_seg`` amplitudes are obtained from
    the two edge conditions (``w = 0`` and ``D(w_xx + nu w_yy) = +-K w_x``) plus
    ``X, X', X'', X'''`` continuity at the internal segment breaks, i.e. one
    closed-form linear solve per mode and **no discretisation in x**.

    Accuracy domain
    ---------------
    Only *interior* stations are certified.  The series for an interior load
    discontinuity carries a dynamic range ``exp(2 s d)`` (``d`` = distance from
    the station to the load), so at stations far from the patch - in particular
    on the **unloaded edges** - the high-mode sum is round-off dominated and is
    *not* usable, even though the centre values are converged.  Edge moments /
    spring rotations are therefore deliberately not exposed here; take them from
    the FEM spring-consistent recovery (``edge_spring_stress``) or from
    :func:`ritz_restrained_plate`.  See
    ``docs/CASE-FEM-ELASTIC-RESTRAINT-PATCH-LEVY.md`` Sec. 5.
    """

    a: float
    b: float
    D: float
    nu: float
    K: float
    segments: tuple
    n_modes: int
    s: np.ndarray = field(repr=False)
    coeffs: np.ndarray = field(repr=False)
    particular: np.ndarray = field(repr=False)
    dropped_modes: tuple = ()

    # -- geometry helpers ------------------------------------------------
    def segment_index(self, x: float) -> int:
        """Index of the constant-load segment containing ``x``."""
        eps = 1e-12 * max(self.a, 1.0)
        for j, (x0, x1, _q) in enumerate(self.segments):
            if x0 - eps <= x <= x1 + eps:
                return j
        raise ValueError(f"x={x} outside [0, {self.a}]")

    def load(self, x: float) -> float:
        """Pressure [Pa] at station ``x`` (independent of ``y``)."""
        return self.segments[self.segment_index(x)][2]

    def _mode_terms(self, x: float, k: int):
        """``(X, X', X'')`` of mode ``k`` at ``x`` including its particular part."""
        j = self.segment_index(x)
        s = self.s[k]
        c = self.coeffs[k, j]
        phi, dphi, d2phi, _d3 = _exp_basis3(s, x, self.a)
        return (
            float(c @ phi) + float(self.particular[k, j]),
            float(c @ dphi),
            float(c @ d2phi),
        )

    # -- fields ----------------------------------------------------------
    def w(self, x: float, y: float) -> float:
        """Transverse displacement [m]."""
        return float(
            sum(self._mode_terms(x, k)[0] * math.sin(self.s[k] * y) for k in range(self.n_modes))
        )

    def slope(self, x: float, y: float) -> float:
        """``dw/dx`` [rad]."""
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
        """Bending moments ``(Mx, My)`` [N*m/m]."""
        wxx = self.w_xx(x, y)
        wyy = self.w_yy(x, y)
        return (-self.D * (wxx + self.nu * wyy), -self.D * (wyy + self.nu * wxx))

    def w_centre(self) -> float:
        return self.w(0.5 * self.a, 0.5 * self.b)

    def moments_centre(self) -> tuple[float, float]:
        return self.moments(0.5 * self.a, 0.5 * self.b)


def levy_restrained_plate_step_load(
    a: float,
    b: float,
    D: float,
    nu: float,
    K: float,
    segments,
    n_modes: int = 201,
) -> LevyStepLoadSolution:
    """Exact Kirchhoff Levy solution for a patch load on rotationally sprung edges.

    Parameters
    ----------
    a, b : float
        Plate dimensions [m]; ``a`` is the sprung direction, and the load is a
        function of ``x`` only (full breadth in ``y``).
    D, nu : float
        Flexural rigidity [N*m] and Poisson ratio.
    K : float
        Distributed rotational spring stiffness [N*m/m/rad] on ``x = 0, a``.
    segments : sequence of (x0, x1, q)
        Piecewise-constant pressure [Pa] on ``x in [x0, x1]``.  Gaps are treated
        as zero pressure; overlap is an error.
    n_modes : int
        Number of ``sin(n pi y / b)`` modes.  For a load independent of ``y``
        only odd modes are loaded, and 201 modes converge the centre moment to
        ~1e-7 relative.

    Notes
    -----
    ``K = 0`` is the simply supported plate and ``K -> inf`` the clamped one, so
    one routine supplies an exact reference for the whole restraint band - the
    thing the uniform-load routine already did and this one adds for the
    discontinuous panel patch the manuscript actually applies.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if K < 0.0:
        raise ValueError("K must be non-negative")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    segs = _normalise_pieces(segments, a)
    nseg = len(segs)
    s_all = np.arange(1, n_modes + 1) * math.pi / b
    coeffs = np.zeros((n_modes, nseg, 4), dtype=float)
    particular = np.zeros((n_modes, nseg), dtype=float)
    dropped: list[int] = []
    size = 4 * nseg

    for k in range(n_modes):
        n = k + 1
        s = s_all[k]
        # y-expansion of a load that is constant across the full breadth
        factor = 4.0 / (n * math.pi) if n % 2 == 1 else 0.0
        q_n = np.array([factor * seg[2] for seg in segs], dtype=float)
        xp = q_n / (D * s**4)
        particular[k, :] = xp

        rows = np.zeros((size, size), dtype=float)
        rhs = np.zeros(size, dtype=float)

        # --- edge conditions at x = 0 (segment 0) and x = a (last segment) ---
        phi, dphi, d2phi, _d3 = _exp_basis3(s, 0.0, a)
        rows[0, 0:4] = phi
        rhs[0] = -xp[0]
        # D(w_xx + nu w_yy) = +K w_x  with w_yy = -s^2 w   (sign = +1 at x = 0)
        rows[1, 0:4] = D * (d2phi - nu * s * s * phi) - K * dphi
        rhs[1] = -(D * (-nu * s * s * xp[0]))

        sl = slice(4 * (nseg - 1), 4 * nseg)
        phi, dphi, d2phi, _d3 = _exp_basis3(s, a, a)
        rows[2, sl] = phi
        rhs[2] = -xp[-1]
        # sign = -1 at x = a
        rows[3, sl] = D * (d2phi - nu * s * s * phi) + K * dphi
        rhs[3] = -(D * (-nu * s * s * xp[-1]))

        # --- C0..C2 continuity at the internal load breaks -------------------
        row = 4
        for j in range(nseg - 1):
            xk = segs[j][1]
            phi, dphi, d2phi, d3phi = _exp_basis3(s, xk, a)
            sa = slice(4 * j, 4 * j + 4)
            sb = slice(4 * (j + 1), 4 * (j + 1) + 4)
            rows[row, sa] = phi
            rows[row, sb] = -phi
            rhs[row] = xp[j + 1] - xp[j]
            row += 1
            rows[row, sa] = dphi
            rows[row, sb] = -dphi
            row += 1
            rows[row, sa] = d2phi
            rows[row, sb] = -d2phi
            row += 1
            rows[row, sa] = d3phi
            rows[row, sb] = -d3phi
            row += 1

        scale = np.abs(rows).max(axis=1)
        scale[scale == 0.0] = 1.0
        scaled = rows / scale[:, None]
        try:
            if np.linalg.cond(scaled) > _STEP_LOAD_COND_LIMIT:
                raise np.linalg.LinAlgError("ill-conditioned mode")
            sol = np.linalg.solve(scaled, rhs / scale)
            if not np.all(np.isfinite(sol)):  # overflow, not just round-off
                raise np.linalg.LinAlgError("non-finite mode amplitudes")
        except np.linalg.LinAlgError:
            # Very high modes: the boundary-condition matrix has dynamic range
            # ``exp(2 s a)`` and underflows to a singular matrix once
            # ``s a >~ 700``.  Such a mode's field is already below the double
            # precision floor wherever it can be evaluated (its particular
            # amplitude is ``q_n / (D s^4)`` and its moment scale
            # ``q_n / s^2``), so dropping it is not an approximation at the
            # accuracy of the series - see the convergence check against the
            # 401-term Navier baseline in
            # ``docs/CASE-FEM-ELASTIC-RESTRAINT-PATCH-LEVY.md``.
            dropped.append(n)
            continue
        coeffs[k, :, :] = sol.reshape(nseg, 4)

    return LevyStepLoadSolution(
        a=a,
        b=b,
        D=D,
        nu=nu,
        K=K,
        n_modes=n_modes,
        segments=segs,
        s=s_all,
        coeffs=coeffs,
        particular=particular,
        dropped_modes=tuple(dropped),
    )


def levy_restrained_plate_patch(
    a: float,
    b: float,
    D: float,
    nu: float,
    K: float,
    q: float,
    patch: tuple[float, float],
    n_modes: int = 201,
) -> LevyStepLoadSolution:
    """Exact reference for a full-breadth pressure patch ``x in patch``.

    Convenience wrapper around :func:`levy_restrained_plate_step_load` for the
    JOUR-ICE-STRUCT load case: ``q`` [Pa] on ``x in [patch[0], patch[1]]`` and
    the full breadth ``y in [0, b]``, zero elsewhere.
    """
    x0, x1 = (float(v) for v in patch)
    return levy_restrained_plate_step_load(
        a, b, D, nu, K, ((x0, x1, q),), n_modes=n_modes
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


def _modal_integral(n: int, lo: float, hi: float, span: float) -> float:
    """``integral_lo^hi sin(n pi s / span) ds`` for a positive integer ``n``."""
    k = n * math.pi / span
    return (math.cos(k * lo) - math.cos(k * hi)) / k


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
    patch: Optional[tuple[float, float, float, float]] = None,
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

    ``patch = (x1, x2, y1, y2)`` applies ``q`` only on that rectangle instead of
    the whole plate (the JOUR-ICE-STRUCT panel load case).  The modal load then
    is the exact double integral of ``sin(m pi x / a) sin(n pi y / b)`` over the
    patch, so a load discontinuity inside the domain only slows the series; it
    does not introduce an approximation of the load.
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
            if patch is None:
                if m_arr[i] % 2 == 1 and n_arr[j] % 2 == 1:
                    load_vec[k] += (
                        4.0 * a * b * q / (math.pi**2 * m_arr[i] * n_arr[j])
                    )
            else:
                x1, x2, y1, y2 = patch
                load_vec[k] += q * _modal_integral(
                    m_arr[i], x1, x2, a) * _modal_integral(n_arr[j], y1, y2, b)

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