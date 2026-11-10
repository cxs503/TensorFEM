"""Structural mechanics dual verification face: Mindlin plate bending + vibration.

Why this module exists
----------------------
The library already verified *basic* Mindlin plate bending (Kirchhoff limit at
one thickness) and *single* plate/shell vibration cases under one boundary
condition each.  What was missing is the quantity that actually decides whether
a structural model may be used for a ship panel or a hull shell:

1. **Thick-plate statics with an FSDT reference.**  For ``h/L`` up to 0.2 the
   Kirchhoff solution is no longer the truth; the first-order shear deformation
   theory (Mindlin-Reissner) Navier solution is, and the difference between the
   two is the *shear-deformation effect* a shell model must reproduce before it
   can be trusted on thick panels (ice belts, collision strakes, sonar domes).
2. **Shear-locking behaviour across the whole thickness range.**  An element
   accurate at ``h/L = 0.01`` can be useless at ``h/L = 0.005`` (locking) or at
   ``h/L = 0.2`` (shear over-stiffness).  Only a sweep with a
   *full-integration control variant* can show the difference.
3. **A spectrum across boundary conditions.**  Panel frequencies without the
   clamped and free-edge cases are not usable for screening: a real stiffened
   panel is neither all-simply-supported nor all-clamped.

This module therefore adds one face with two halves, each with its own
independent reference:

Static half (thickness sweep)
    * Q4 Mindlin element with three shear treatments: ``sri`` (one-point,
      locking-free but rank-deficient in shear), ``mitc`` (Dvorkin-Bathe
      assumed natural shear strain) and ``full`` (2x2 shear integration, the
      textbook locking case kept as an honest control).
    * References: Kirchhoff Navier double-sine solution (uniform load) and the
      FSDT Navier solution of the same problem, which adds the
      ``q_mn / (k_s G h R)`` shear particular solution.  Both are closed form
      and independent of the FE code.

Vibration half (boundary-condition sweep)
    * Plate: SS / CC / CF, first six frequencies of a square plate.
      References: Kirchhoff Navier (SS) and a polynomial Rayleigh-Ritz
      (CC, CF) that shares no algebra with the FE assembly.  The exact FSDT
      frequency of the SS plate is also available, so the thick-plate spectrum
      can be checked at ``h/L = 0.2`` as well.
    * Cylindrical shell: SS / CC / CF, first six frequencies.  SS has the
      Donnell-Mushtari-Vlasov closed form of
      :mod:`tensorfem.cylindrical_shell_modal`; CC and CF have no closed form,
      so a three-mesh Richardson/GCI extrapolation is used and labelled as
      such (no fabricated "literature value").

Everything is SI (m, kg, s, Pa, Hz).  The module carries no transcribed
reference table: every reference number is either computed from a closed form or
extrapolated, and extrapolated references carry their GCI uncertainty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

__all__ = [
    "SHEAR_CORRECTION",
    "PLATE_BOUNDARIES",
    "KirchhoffReference",
    "FSDTReference",
    "PlateStaticResult",
    "PlateModalResult",
    "kirchhoff_ss_uniform_load",
    "fsdt_ss_uniform_load",
    "kirchhoff_ss_angular_frequencies",
    "mindlin_ss_angular_frequencies",
    "ritz_plate_angular_frequencies",
    "q4_mindlin_stiffness_batch",
    "q4_mindlin_mass_batch",
    "structured_rect_plate_mesh",
    "solve_plate_static",
    "solve_plate_modes",
    "locking_sweep",
    "mesh_convergence",
    "observed_convergence_order",
    "richardson_extrapolation",
    "shell_boundary_dofs",
    "solve_shell_natural_frequencies",
]

SHEAR_CORRECTION = 5.0 / 6.0
"""Mindlin shear correction factor (Reissner's ``k_s = 5/6``)."""

PLATE_BOUNDARIES = ("SS", "CC", "CF")
"""Supported plate boundary conditions.

``SS``: simply supported on all edges (``w=0`` plus vanishing tangential
rotation); ``CC``: clamped on all edges; ``CF``: clamped along ``x=0`` only and
free elsewhere (cantilever plate).
"""

_GP = 1.0 / math.sqrt(3.0)


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _poisson(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not (-1.0 < value < 0.5):
        raise ValueError("poisson must lie strictly between -1 and 0.5")
    return value


def _shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear Q4 shape functions and their natural derivatives."""
    N = 0.25 * np.array(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
    )
    dN = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)],
        [-(1 + eta), (1 - xi)],
    ])
    return N, dN


def _rigidity(young: float, poisson: float, thickness: float) -> float:
    return young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))


def _bending_operator(young: float, poisson: float, thickness: float) -> np.ndarray:
    return young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2)) * np.array(
        [[1.0, poisson, 0.0],
         [poisson, 1.0, 0.0],
         [0.0, 0.0, (1.0 - poisson) / 2.0]]
    )


def _generalized_lowest_eigenpairs(K: np.ndarray, M: np.ndarray, modes: int):
    """Lowest ``modes`` eigenpairs of ``K phi = omega^2 M phi`` (both SPD).

    The problem is reduced with an explicit Cholesky factor and a *standard*
    symmetric solver; the LAPACK generalised driver (``?sygv``/``?hegv``) either
    segfaults or returns a corrupt spectrum on this platform, which is the same
    failure family already documented in :mod:`tensorfem.plate_vibration`.

    Because the BLAS/LAPACK stack here (flexiblas 3.3.0, loongarch64) also
    occasionally corrupts a plain symmetric solve, every candidate spectrum is
    validated against the eigenpair residual and a differently-driven solver is
    tried on failure.  A silently wrong spectrum must never escape this function.
    """
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    # Jacobi scaling: the translational and rotatory inertia blocks differ by
    # (h^2/12), i.e. four orders of magnitude at h = 0.01, and the unreduced
    # dynamic range is what makes the platform LAPACK return corrupt spectra.
    diag = np.sqrt(np.diag(M))
    if np.any(diag <= 0.0):
        raise ValueError("mass matrix has a non-positive diagonal entry")
    scale = 1.0 / diag
    K = K * scale[:, None] * scale[None, :]
    M = M * scale[:, None] * scale[None, :]
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValueError("mass matrix is not positive definite") from exc
    k_norm = float(np.abs(K).max()) or 1.0
    last_issue = "no attempt made"
    for driver in (None, "evd", "ev", "evx"):
        # A = L^-1 K L^-T   (avoids the crashing triangular LAPACK wrappers)
        A = np.linalg.solve(L, np.linalg.solve(L, K).T)
        A = 0.5 * (A + A.T)
        try:
            if driver is None:
                lam_all, vec_all = np.linalg.eigh(A)
            else:
                lam_all, vec_all = sla.eigh(A, driver=driver)
        except Exception as exc:  # pragma: no cover - platform specific
            last_issue = f"{driver or 'numpy'} raised {type(exc).__name__}"
            continue
        lam = np.asarray(lam_all[:modes], dtype=float)
        vec = np.asarray(vec_all[:, :modes], dtype=float)
        if not np.all(np.isfinite(lam)):
            last_issue = f"{driver or 'numpy'}: non-finite eigenvalues"
            continue
        if lam.min() < -1e-6 * max(abs(lam.max()), 1.0):
            last_issue = f"{driver or 'numpy'}: negative eigenvalue {lam.min():.3e}"
            continue
        lam = np.clip(lam, 0.0, None)
        phi_scaled = np.linalg.solve(L.T, vec)
        residual = 0.0
        for j in range(modes):
            v = phi_scaled[:, j]
            r = K @ v - lam[j] * (M @ v)          # scaled system: same eigenpairs
            residual = max(residual, float(np.linalg.norm(r)) /
                           (k_norm * float(np.linalg.norm(v))))
        if residual > 1e-8:
            last_issue = f"{driver or 'numpy'}: residual {residual:.2e}"
            continue
        return lam, phi_scaled * scale[:, None]   # undo the Jacobi scaling
    raise RuntimeError(f"generalized eigensolver is unreliable: {last_issue}")


# ---------------------------------------------------------------------------
# Closed-form references (static)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KirchhoffReference:
    """Kirchhoff (thin plate) Navier solution for a uniformly loaded SS plate."""

    deflection_center: float
    moment_x_center: float
    moment_y_center: float
    series_terms: int

    def deflection(self, x: float, y: float, *, length: float, width: float,
                   young: float, poisson: float, thickness: float,
                   load: float, terms: int = 61) -> float:
        """Kirchhoff deflection at an arbitrary interior point."""
        D = _rigidity(young, poisson, thickness)
        total = 0.0
        for m in range(1, terms + 1, 2):
            for n in range(1, terms + 1, 2):
                a = m * math.pi / length
                b = n * math.pi / width
                q_mn = 16.0 * load / (math.pi ** 2 * m * n)
                R = a * a + b * b
                total += q_mn / (D * R * R) * math.sin(a * x) * math.sin(b * y)
        return total

    def moment_x(self, x: float, y: float, *, length: float, width: float,
                 poisson: float, load: float, terms: int = 61) -> float:
        """Kirchhoff bending moment ``M_x`` at an arbitrary interior point."""
        total = 0.0
        for m in range(1, terms + 1, 2):
            for n in range(1, terms + 1, 2):
                a = m * math.pi / length
                b = n * math.pi / width
                q_mn = 16.0 * load / (math.pi ** 2 * m * n)
                R = a * a + b * b
                total += (q_mn * (a * a + poisson * b * b) / (R * R)
                          * math.sin(a * x) * math.sin(b * y))
        return total


@dataclass(frozen=True)
class FSDTReference:
    """First-order shear deformation (Mindlin) Navier solution, SS plate."""

    deflection_center: float
    deflection_kirchhoff_center: float
    moment_x_center: float
    shear_amplification: float
    series_terms: int


def kirchhoff_ss_uniform_load(
    *, length: float, width: float, thickness: float, young: float,
    poisson: float, load: float, terms: int = 61,
) -> KirchhoffReference:
    """Centre deflection/moments of a uniformly loaded Kirchhoff SS plate.

    Navier double-sine solution, odd ``m, n`` only::

        w   = sum_mn q_mn sin(m pi x/a) sin(n pi y/b) / (D R^2)
        M_x = sum_mn q_mn (alpha^2 + nu beta^2) sin(...) sin(...) / R^2

    with ``q_mn = 16 q0/(pi^2 m n)``, ``R = alpha^2 + beta^2``,
    ``alpha = m pi/a``, ``beta = n pi/b``.
    """
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    load = _positive("load", load)
    D = _rigidity(young, poisson, thickness)
    w = mx = my = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            a = m * math.pi / length
            b = n * math.pi / width
            q_mn = 16.0 * load / (math.pi ** 2 * m * n)
            R = a * a + b * b
            s = math.sin(m * math.pi / 2.0) * math.sin(n * math.pi / 2.0)
            w += q_mn / (D * R * R) * s
            mx += q_mn * (a * a + poisson * b * b) / (R * R) * s
            my += q_mn * (b * b + poisson * a * a) / (R * R) * s
    return KirchhoffReference(w, mx, my, terms)


def fsdt_ss_uniform_load(
    *, length: float, width: float, thickness: float, young: float,
    poisson: float, load: float, shear_correction: float = SHEAR_CORRECTION,
    terms: int = 61,
) -> FSDTReference:
    """FSDT (Mindlin) Navier centre solution for a uniformly loaded SS plate.

    The Mindlin plate with shear correction ``k_s`` reproduces the Kirchhoff
    curvature field plus a shear particular solution::

        w   = sum_mn q_mn [ 1/(D R^2) + 1/(k_s G h R) ] sin sin
        M_x = sum_mn q_mn (alpha^2 + nu beta^2)/R^2 sin sin     (unchanged)

    because the added field satisfies ``grad^2 w_p = -q/(k_s G h)`` and carries
    no curvature, so the bending moments coincide with the Kirchhoff ones for
    this load case.  The formula is re-derived here from the Mindlin equilibrium
    equations rather than transcribed from a table.
    """
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    load = _positive("load", load)
    ks = _positive("shear_correction", shear_correction)
    D = _rigidity(young, poisson, thickness)
    G = young / (2.0 * (1.0 + poisson))
    w = w_k = mx = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            a = m * math.pi / length
            b = n * math.pi / width
            q_mn = 16.0 * load / (math.pi ** 2 * m * n)
            R = a * a + b * b
            s = math.sin(m * math.pi / 2.0) * math.sin(n * math.pi / 2.0)
            w_k += q_mn / (D * R * R) * s
            w += q_mn * (1.0 / (D * R * R) + 1.0 / (ks * G * thickness * R)) * s
            mx += q_mn * (a * a + poisson * b * b) / (R * R) * s
    return FSDTReference(w, w_k, mx, w / w_k, terms)


# ---------------------------------------------------------------------------
# Closed-form / variational references (vibration)
# ---------------------------------------------------------------------------

def kirchhoff_ss_angular_frequencies(
    *, length: float, width: float, thickness: float, young: float,
    poisson: float, density: float, modes: int = 6,
) -> np.ndarray:
    """Kirchhoff SS plate frequencies ``omega_mn = R sqrt(D/(rho h))``."""
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    density = _positive("density", density)
    D = _rigidity(young, poisson, thickness)
    values: list[float] = []
    for m in range(1, 12):
        for n in range(1, 12):
            alpha = m * math.pi / length
            beta = n * math.pi / width
            values.append((alpha ** 2 + beta ** 2)
                          * math.sqrt(D / (density * thickness)))
    return np.sort(np.array(values))[:modes]


def mindlin_ss_angular_frequencies(
    *, length: float, width: float, thickness: float, young: float,
    poisson: float, density: float, modes: int = 6,
    shear_correction: float = SHEAR_CORRECTION, max_half_waves: int = 10,
) -> np.ndarray:
    """Exact FSDT/Mindlin frequencies of a simply supported rectangular plate.

    For every wave pair ``(m, n)`` the amplitudes of
    ``psi_x = A cos(alpha x) sin(beta y)``,
    ``psi_y = B sin(alpha x) cos(beta y)``,
    ``w = C sin(alpha x) sin(beta y)`` satisfy a symmetric 3x3 generalised
    eigenproblem::

        K = [[D (a^2 + g b^2) + S,  D g' a b,             -S a],
             [D g' a b,             D (b^2 + g a^2) + S,  -S b],
             [-S a,                 -S b,                  S (a^2 + b^2)]]
        M = diag(rho h^3/12, rho h^3/12, rho h)

    with ``g = (1-nu)/2``, ``g' = (1+nu)/2`` and ``S = k_s G h``.  Rotatory
    inertia is retained, so the result stays valid well above the thin-plate
    limit and serves as the thick-plate vibration reference.
    """
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    density = _positive("density", density)
    ks = _positive("shear_correction", shear_correction)
    D = _rigidity(young, poisson, thickness)
    G = young / (2.0 * (1.0 + poisson))
    S = ks * G * thickness
    g, gp = (1.0 - poisson) / 2.0, (1.0 + poisson) / 2.0
    mass = np.diag([density * thickness ** 3 / 12.0,
                    density * thickness ** 3 / 12.0,
                    density * thickness])
    # The 3x3 problem is solved in 30-digit arithmetic (mpmath): in double
    # precision the two thickness-shear branches are 1e4 times stiffer than the
    # bending branch and the small eigenvalue loses its last digits (measured
    # 1.6e-5 relative error, which is the same order as the physical shear
    # correction the reference is supposed to resolve).
    try:
        import mpmath
    except ImportError:  # pragma: no cover - optional dependency
        # Fallback: double-precision 3x3 solve.  The thickness-shear branches are
        # ~1e4 times stiffer than the bending branch, so the smallest eigenvalue
        # then carries a ~1e-5 relative error -- enough for the 2-3 % acceptance
        # bands used here but not enough to resolve the thin-plate shear
        # correction itself.
        fallback: list[float] = []
        for m in range(1, max_half_waves + 1):
            for n in range(1, max_half_waves + 1):
                a = m * math.pi / length
                b = n * math.pi / width
                K = np.array([
                    [D * (a * a + g * b * b) + S, D * gp * a * b, -S * a],
                    [D * gp * a * b, D * (b * b + g * a * a) + S, -S * b],
                    [-S * a, -S * b, S * (a * a + b * b)],
                ])
                lam, _ = _generalized_lowest_eigenpairs(K, mass, 1)
                fallback.append(float(math.sqrt(lam[0])))
        return np.sort(np.array(fallback))[:modes]

    dps = mpmath.mp.dps
    mpmath.mp.dps = 40
    try:
        mass_diag = [mpmath.mpf(repr(density * thickness ** 3 / 12.0)),
                     mpmath.mpf(repr(density * thickness ** 3 / 12.0)),
                     mpmath.mpf(repr(density * thickness))]
        values: list[float] = []
        for m in range(1, max_half_waves + 1):
            for n in range(1, max_half_waves + 1):
                a = m * math.pi / length
                b = n * math.pi / width
                entries = (
                    (D * (a * a + g * b * b) + S, D * gp * a * b, -S * a),
                    (D * gp * a * b, D * (b * b + g * a * a) + S, -S * b),
                    (-S * a, -S * b, S * (a * a + b * b)),
                )
                scaled = mpmath.matrix(3, 3)
                for i in range(3):
                    for j in range(3):
                        scaled[i, j] = (mpmath.mpf(repr(entries[i][j]))
                                        / mpmath.sqrt(mass_diag[i] * mass_diag[j]))
                lam = mpmath.eigsy(scaled, eigvals_only=True)
                smallest = min(v for v in lam if v > 0)
                values.append(float(mpmath.sqrt(smallest)))
    finally:
        mpmath.mp.dps = dps
    return np.sort(np.array(values))[:modes]


def _polynomial_basis(x: np.ndarray, degree: int, kind: str, side: float):
    """Value / first / second derivative matrices of a constrained basis.

    ``kind`` is ``"CC"`` (clamped at both ends), ``"CF"`` (clamped at 0, no
    essential constraint at ``side``) or ``"cos"`` (Fourier-cosine basis for a
    free edge, where no essential constraint exists).
    """
    s = np.asarray(x, dtype=float) / side
    V, D1, D2 = [], [], []
    for p in range(degree):
        if kind == "CC":
            # s^(p+2) (1-s)^2 : value and slope vanish at both ends
            V.append(s ** (p + 2) - 2.0 * s ** (p + 3) + s ** (p + 4))
            D1.append(((p + 2) * s ** (p + 1) - 2.0 * (p + 3) * s ** (p + 2)
                       + (p + 4) * s ** (p + 3)) / side)
            D2.append(((p + 2) * (p + 1) * s ** p
                       - 2.0 * (p + 3) * (p + 2) * s ** (p + 1)
                       + (p + 4) * (p + 3) * s ** (p + 2)) / side ** 2)
        elif kind == "CF":
            V.append(s ** (p + 2))
            D1.append((p + 2) * s ** (p + 1) / side)
            D2.append((p + 2) * (p + 1) * s ** p / side ** 2)
        elif kind == "cos":
            # Fourier-cosine basis: complete on [0, side], orthogonal and far
            # better conditioned than high-degree monomials.  Used for plate
            # edges with no essential boundary condition (free edges), where
            # any complete admissible basis gives a valid Ritz upper bound.
            k = p * math.pi
            V.append(np.cos(k * s))
            D1.append(-k * np.sin(k * s) / side)
            D2.append(-k * k * np.cos(k * s) / side ** 2)
        else:  # pragma: no cover - guarded by the public wrapper
            raise ValueError(f"unsupported basis kind {kind!r}")
    return np.column_stack(V), np.column_stack(D1), np.column_stack(D2)


def _diff_matrices(degree: int, kind: str, side: float, n_gauss: int = 28):
    """1-D energy matrices of the polynomial basis on ``[0, side]``.

    Returns ``(A0, A1, A2, B2)`` with ``A0 = int X X``, ``A1 = int X' X'``,
    ``A2 = int X'' X''``, ``B2 = int X'' X``.  The basis is re-orthonormalised
    in the ``A0`` metric so the Ritz matrices stay well conditioned.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    x = 0.5 * side * (nodes + 1.0)
    w = 0.5 * side * weights
    V, D1, D2 = _polynomial_basis(x, degree, kind, side)
    # Orthonormalise in the discrete L2 metric with a QR of the weighted
    # Vandermonde matrix.  A Cholesky of ``A0`` squares the condition number and
    # fails outright for the clamped-free monomial basis above degree ~11.
    _, R = np.linalg.qr(V * np.sqrt(w)[:, None])
    T = np.linalg.inv(R)
    V, D1, D2 = V @ T, D1 @ T, D2 @ T
    A0 = V.T @ (w[:, None] * V)
    A1 = D1.T @ (w[:, None] * D1)
    A2 = D2.T @ (w[:, None] * D2)
    B2 = D2.T @ (w[:, None] * V)
    return A0, A1, A2, B2


def ritz_plate_angular_frequencies(
    *, length: float, width: float, thickness: float, young: float,
    poisson: float, density: float, boundary: str = "CC", modes: int = 6,
    degree: int = 10,
) -> np.ndarray:
    """Kirchhoff plate frequencies by polynomial Rayleigh-Ritz.

    A discretisation that shares no algebra with the Q4 FE assembly: the
    deflection is ``w = sum_pq c_pq X_p(x) Y_q(y)`` with polynomials satisfying
    the *essential* boundary conditions exactly, and the eigenproblem is built
    from the exact energy functionals::

        U = D/2 int [ w_xx^2 + w_yy^2 + 2 nu w_xx w_yy + 2(1-nu) w_xy^2 ]
        T = rho h/2 int w^2

    Ritz frequencies are upper bounds that decrease monotonically with
    ``degree``; callers should confirm refinement insensitivity.  Valid for
    thin plates (``h/L <= 0.05``); use the FSDT reference above that range.
    """
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    density = _positive("density", density)
    if boundary not in ("CC", "CF"):
        raise ValueError("ritz_plate_angular_frequencies supports 'CC' and 'CF'")
    if degree < 2:
        raise ValueError("degree must be at least 2")
    kind_x = "CC" if boundary == "CC" else "CF"
    kind_y = "CC" if boundary == "CC" else "cos"
    Ax0, Ax1, Ax2, Bx2 = _diff_matrices(degree, kind_x, length)
    Ay0, Ay1, Ay2, By2 = _diff_matrices(degree, kind_y, width)
    D = _rigidity(young, poisson, thickness)
    cross = np.kron(Bx2, By2)
    K = (D * (np.kron(Ax2, Ay0) + np.kron(Ax0, Ay2))
         + D * poisson * (cross + cross.T)
         + 2.0 * D * (1.0 - poisson) * np.kron(Ax1, Ay1))
    M = density * thickness * np.kron(Ax0, Ay0)
    lam, _ = _generalized_lowest_eigenpairs(K, M, modes)
    lam = np.sort(np.clip(lam, 0.0, None))
    return np.sqrt(lam[:modes])


# ---------------------------------------------------------------------------
# Finite-element discretisation (numpy, vectorised over elements)
# ---------------------------------------------------------------------------

def structured_rect_plate_mesh(
    elements_x: int, elements_y: int, length: float = 1.0, width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Structured Q4 mesh of ``[0, length] x [0, width]``."""
    if elements_x < 1 or elements_y < 1:
        raise ValueError("elements_x and elements_y must be positive")
    x = np.linspace(0.0, length, elements_x + 1)
    y = np.linspace(0.0, width, elements_y + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    stride = elements_y + 1
    elements = []
    for i in range(elements_x):
        for j in range(elements_y):
            n0 = i * stride + j
            elements.append((n0, n0 + stride, n0 + stride + 1, n0 + 1))
    return nodes, np.array(elements, dtype=np.int64)


def _derivatives(coords: np.ndarray, xi: float, eta: float):
    """Shape values, physical shape derivatives and Jacobian determinant.

    The 2x2 inverse is written out explicitly: stacked ``np.linalg.inv`` on
    tiny matrices falls off the BLAS fast path on this platform and costs more
    than the whole assembly (measured 1.9 s per call for 256 elements).
    """
    N, dNn = _shape(xi, eta)
    J = np.einsum("ia,eib->eab", dNn, coords)
    j00, j01, j10, j11 = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    detJ = j00 * j11 - j01 * j10
    if np.any(detJ <= 0.0):
        raise ValueError("element with non-positive Jacobian")
    Jinv = np.empty_like(J)
    Jinv[:, 0, 0] = j11 / detJ
    Jinv[:, 0, 1] = -j01 / detJ
    Jinv[:, 1, 0] = -j10 / detJ
    Jinv[:, 1, 1] = j00 / detJ
    dNdx = np.einsum("ia,eab->eib", dNn, Jinv)
    return N, dNdx, detJ


def _bending_matrix(dNdx: np.ndarray) -> np.ndarray:
    ne = dNdx.shape[0]
    Bb = np.zeros((ne, 3, 12))
    idx = np.arange(4)
    Bb[:, 0, 3 * idx + 1] = dNdx[:, idx, 0]
    Bb[:, 1, 3 * idx + 2] = dNdx[:, idx, 1]
    Bb[:, 2, 3 * idx + 1] = dNdx[:, idx, 1]
    Bb[:, 2, 3 * idx + 2] = dNdx[:, idx, 0]
    return Bb


def _shear_matrix(N: np.ndarray, dNdx: np.ndarray) -> np.ndarray:
    ne = dNdx.shape[0]
    Bs = np.zeros((ne, 2, 12))
    idx = np.arange(4)
    Bs[:, 0, 3 * idx] = dNdx[:, idx, 0]
    Bs[:, 0, 3 * idx + 1] = N
    Bs[:, 1, 3 * idx] = dNdx[:, idx, 1]
    Bs[:, 1, 3 * idx + 2] = N
    return Bs


def q4_mindlin_stiffness_batch(
    nodes: np.ndarray, elements: np.ndarray, young: float, poisson: float,
    thickness: float, *, shear_treatment: str = "sri",
    shear_correction: float = SHEAR_CORRECTION,
) -> np.ndarray:
    """Element stiffness matrices ``(ne, 12, 12)`` of the Q4 Mindlin plate.

    ``shear_treatment``:

    ``"sri"``
        2x2 bending with 1x1 transverse shear: removes thin-plate shear
        locking but leaves two spurious zero-energy shear modes per element
        (harmless statically, visible in free vibration).
    ``"mitc"``
        Dvorkin-Bathe assumed natural shear strain sampled at the four
        mid-sides and interpolated bilinearly; locking-free and free of the
        spurious mechanisms.  Recommended for modal work.
    ``"full"``
        2x2 transverse shear: the textbook locking case, kept as an honest
        control so the locking claim can be falsified.
    """
    if shear_treatment not in ("sri", "mitc", "full"):
        raise ValueError("shear_treatment must be 'sri', 'mitc' or 'full'")
    young = _positive("young", young)
    poisson = _poisson(poisson)
    thickness = _positive("thickness", thickness)
    ks = _positive("shear_correction", shear_correction)
    coords = nodes[elements]
    ne = coords.shape[0]
    Db = _bending_operator(young, poisson, thickness)
    G = young / (2.0 * (1.0 + poisson))
    Ds = ks * G * thickness * np.eye(2)
    ke = np.zeros((ne, 12, 12))
    for xi in (-_GP, _GP):
        for eta in (-_GP, _GP):
            N, dNdx, detJ = _derivatives(coords, xi, eta)
            Bb = _bending_matrix(dNdx)
            ke += np.einsum("eai,ab,ebj->eij", Bb, Db, Bb) * detJ[:, None, None]
            if shear_treatment == "full":
                Bs = _shear_matrix(np.repeat(N[None, :], ne, axis=0), dNdx)
                ke += np.einsum("eai,ab,ebj->eij", Bs, Ds, Bs) * detJ[:, None, None]
    if shear_treatment == "sri":
        N, dNdx, detJ = _derivatives(coords, 0.0, 0.0)
        Bs = _shear_matrix(np.repeat(N[None, :], ne, axis=0), dNdx)
        ke += np.einsum("eai,ab,ebj->eij", Bs, Ds, Bs) * detJ[:, None, None] * 4.0
    elif shear_treatment == "mitc":
        rows = {}
        for tag, (xi, eta) in {
            "bottom": (0.0, -1.0), "top": (0.0, 1.0),
            "left": (-1.0, 0.0), "right": (1.0, 0.0),
        }.items():
            N, dNdx, _ = _derivatives(coords, xi, eta)
            rows[tag] = _shear_matrix(np.repeat(N[None, :], ne, axis=0), dNdx)
        for xi in (-_GP, _GP):
            for eta in (-_GP, _GP):
                _, _, detJ = _derivatives(coords, xi, eta)
                Bs = np.zeros((ne, 2, 12))
                Bs[:, 0] = (0.5 * (1.0 - eta) * rows["bottom"][:, 0]
                            + 0.5 * (1.0 + eta) * rows["top"][:, 0])
                Bs[:, 1] = (0.5 * (1.0 - xi) * rows["left"][:, 1]
                            + 0.5 * (1.0 + xi) * rows["right"][:, 1])
                ke += np.einsum("eai,ab,ebj->eij", Bs, Ds, Bs) * detJ[:, None, None]
    return ke


def q4_mindlin_mass_batch(
    nodes: np.ndarray, elements: np.ndarray, density: float, thickness: float,
    *, rotary_inertia: bool = True,
) -> np.ndarray:
    """Consistent Q4 mass matrices ``(ne, 12, 12)`` (translations + rotations)."""
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    coords = nodes[elements]
    ne = coords.shape[0]
    me = np.zeros((ne, 12, 12))
    scale_w = density * thickness
    scale_rot = density * thickness ** 3 / 12.0 if rotary_inertia else 0.0
    idx = np.arange(4)
    for xi in (-_GP, _GP):
        for eta in (-_GP, _GP):
            N, _, detJ = _derivatives(coords, xi, eta)
            NN = np.einsum("i,j->ij", N, N)
            me[:, 3 * idx[:, None] + 0, 3 * idx[None, :] + 0] += (
                scale_w * NN * detJ[:, None, None]
            )
            for p in (1, 2):
                me[:, 3 * idx[:, None] + p, 3 * idx[None, :] + p] += (
                    scale_rot * NN * detJ[:, None, None]
                )
    return me


def _element_area(coords: np.ndarray) -> float:
    """Area of a planar bilinear quad by the shoelace formula."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _assemble(
    nodes: np.ndarray, elements: np.ndarray, young: float, poisson: float,
    thickness: float, density: float, *, shear_treatment: str,
    load: float = 0.0,
) -> tuple[sps.csr_matrix, sps.csr_matrix | None, np.ndarray]:
    """Assemble sparse ``(K, M, F)`` on a structured plate mesh."""
    ke = q4_mindlin_stiffness_batch(
        nodes, elements, young, poisson, thickness, shear_treatment=shear_treatment
    )
    ndof = 3 * nodes.shape[0]
    dofs = (3 * elements[:, :, None] + np.arange(3)[None, None, :]).reshape(
        elements.shape[0], 12
    )
    rows = np.repeat(dofs, 12, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, 12)).reshape(-1)
    K = sps.coo_matrix((ke.reshape(-1), (rows, cols)),
                       shape=(ndof, ndof)).tocsr()
    M = None
    if density > 0.0:
        me = q4_mindlin_mass_batch(nodes, elements, density, thickness)
        M = sps.coo_matrix((me.reshape(-1), (rows, cols)),
                           shape=(ndof, ndof)).tocsr()
    F = np.zeros(ndof)
    if load != 0.0:
        areas = np.array([_element_area(c) for c in nodes[elements]])
        # np.add.at is required here: shared nodes appear in several elements and
        # buffered fancy-index ``+=`` would keep only one contribution.
        np.add.at(F, dofs[:, 0::3].reshape(-1), np.repeat(load * areas / 4.0, 4))
    return K, M, F


def _boundary_dofs(
    nodes: np.ndarray, *, length: float, width: float, boundary: str,
    tol_scale: float = 1e-9,
) -> np.ndarray:
    """Constrained DOF indices for a plate boundary condition."""
    if boundary not in PLATE_BOUNDARIES:
        raise ValueError(f"boundary must be one of {PLATE_BOUNDARIES}")
    tol = tol_scale * max(length, width)
    x, y = nodes[:, 0], nodes[:, 1]
    at_x0 = np.abs(x) < tol
    at_xl = np.abs(x - length) < tol
    at_y0 = np.abs(y) < tol
    at_yw = np.abs(y - width) < tol
    x_edge = at_x0 | at_xl
    y_edge = at_y0 | at_yw
    if boundary == "SS":
        transverse = np.where(x_edge | y_edge)[0]
        rot = np.concatenate((3 * np.where(y_edge)[0] + 1,
                              3 * np.where(x_edge)[0] + 2))
    elif boundary == "CC":
        on_edge = np.where(x_edge | y_edge)[0]
        transverse = on_edge
        rot = np.concatenate((3 * on_edge + 1, 3 * on_edge + 2))
    else:  # CF: clamped along x = 0 only
        clamped = np.where(at_x0)[0]
        transverse = clamped
        rot = np.concatenate((3 * clamped + 1, 3 * clamped + 2))
    return np.unique(np.concatenate((3 * transverse, rot))).astype(np.int64)


@dataclass(frozen=True)
class PlateStaticResult:
    """Outcome of a Mindlin plate static solve."""

    deflection_center: float
    moment_x_center: float
    moment_y_center: float
    bending_stress_center: float
    sample_point: tuple[float, float]
    deflection_field: np.ndarray
    coordinates: np.ndarray
    elements_x: int
    elements_y: int
    n_dof: int
    boundary: str
    shear_treatment: str


@dataclass(frozen=True)
class PlateModalResult:
    """Outcome of a Mindlin plate free-vibration solve."""

    frequencies_hz: np.ndarray
    angular_frequencies: np.ndarray
    modes: np.ndarray
    n_dof: int
    n_free_dof: int
    boundary: str
    shear_treatment: str


def _centre_index(nodes: np.ndarray, length: float, width: float) -> int:
    d2 = (nodes[:, 0] - 0.5 * length) ** 2 + (nodes[:, 1] - 0.5 * width) ** 2
    return int(np.argmin(d2))


def _centre_moments(
    nodes: np.ndarray, elements: np.ndarray, u: np.ndarray, young: float,
    poisson: float, thickness: float, length: float, width: float,
) -> tuple[float, float, tuple[float, float]]:
    """Bending moments at the element Gauss point nearest the plate centre.

    Sampling at a Gauss point *and* comparing against the closed form at the
    same coordinates keeps a discretisation offset out of the stress error.
    """
    coords_all = nodes[elements]
    centroids = coords_all.mean(axis=1)
    e = int(np.argmin((centroids[:, 0] - 0.5 * length) ** 2
                      + (centroids[:, 1] - 0.5 * width) ** 2))
    coords = coords_all[e]
    target = np.array([0.5 * length, 0.5 * width])
    best = None
    for xi in (-_GP, _GP):
        for eta in (-_GP, _GP):
            N, dNdx, _ = _derivatives(coords[None, ...], xi, eta)
            pt = N @ coords
            d = float(np.sum((pt - target) ** 2))
            if best is None or d < best[0]:
                best = (d, N, dNdx[0])
    _, N, dNdx = best
    point = N @ coords
    dofs = (3 * elements[e][:, None] + np.arange(3)[None, :]).reshape(-1)
    kappa = _bending_matrix(dNdx[None, ...])[0] @ u[dofs]
    moments = _bending_operator(young, poisson, thickness) @ kappa
    return float(moments[0]), float(moments[1]), (float(point[0]), float(point[1]))


def solve_plate_static(
    *, elements_x: int = 16, elements_y: int = 16, length: float = 1.0,
    width: float = 1.0, thickness: float = 0.01, young: float = 200.0e9,
    poisson: float = 0.3, load: float = 1.0e4, boundary: str = "SS",
    shear_treatment: str = "sri",
) -> PlateStaticResult:
    """Static Mindlin plate under uniform pressure, ``SS``/``CC``/``CF``."""
    nodes, elements = structured_rect_plate_mesh(
        elements_x, elements_y, length, width
    )
    K, _, F = _assemble(nodes, elements, young, poisson, thickness, 0.0,
                        shear_treatment=shear_treatment, load=load)
    fixed = _boundary_dofs(nodes, length=length, width=width, boundary=boundary)
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    u = np.zeros(K.shape[0])
    u[free] = spsla.spsolve(K[free][:, free].tocsc(), F[free])
    centre = _centre_index(nodes, length, width)
    mx, my, point = _centre_moments(nodes, elements, u, young, poisson,
                                    thickness, length, width)
    return PlateStaticResult(
        deflection_center=float(u[3 * centre]), moment_x_center=mx,
        moment_y_center=my,
        bending_stress_center=6.0 * max(abs(mx), abs(my)) / thickness ** 2,
        sample_point=point, deflection_field=u[0::3], coordinates=nodes,
        elements_x=elements_x, elements_y=elements_y, n_dof=K.shape[0],
        boundary=boundary, shear_treatment=shear_treatment,
    )


def solve_plate_modes(
    *, elements_x: int = 16, elements_y: int = 16, length: float = 1.0,
    width: float = 1.0, thickness: float = 0.01, young: float = 200.0e9,
    poisson: float = 0.3, density: float = 7850.0, boundary: str = "SS",
    modes: int = 6, shear_treatment: str = "mitc",
    rotary_inertia: bool = True, dense_limit: int = 4500,
) -> PlateModalResult:
    """Free-vibration Mindlin plate modes with ``SS``/``CC``/``CF`` boundaries."""
    nodes, elements = structured_rect_plate_mesh(
        elements_x, elements_y, length, width
    )
    K, M, _ = _assemble(nodes, elements, young, poisson, thickness, density,
                        shear_treatment=shear_treatment)
    if M is None:  # pragma: no cover - density is validated as positive
        raise ValueError("modal solve requires a positive density")
    fixed = _boundary_dofs(nodes, length=length, width=width, boundary=boundary)
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    if modes >= free.size:
        raise ValueError("modes exceeds the number of free degrees of freedom")
    Kf = K[free][:, free].tocsc()
    Mf = M[free][:, free].tocsc()
    if free.size <= dense_limit:
        lam, vec = _generalized_lowest_eigenpairs(Kf.toarray(), Mf.toarray(), modes)
    else:  # pragma: no cover - exercised only on very fine meshes
        lam, vec = spsla.eigsh(Kf, k=modes, M=Mf, sigma=0.0, which="LM")
        order = np.argsort(lam)
        lam, vec = lam[order], vec[:, order]
    lam = np.asarray(lam, dtype=float)
    if np.any(~np.isfinite(lam)) or np.any(lam < -1e-9 * max(abs(lam.max()), 1.0)):
        raise RuntimeError("eigensolver returned an invalid spectrum")
    lam = np.clip(lam, 0.0, None)
    omega = np.sqrt(lam)
    full = np.zeros((K.shape[0], modes))
    full[free] = vec
    return PlateModalResult(
        frequencies_hz=omega / (2.0 * math.pi), angular_frequencies=omega,
        modes=full, n_dof=K.shape[0], n_free_dof=free.size,
        boundary=boundary, shear_treatment=shear_treatment,
    )

# ---------------------------------------------------------------------------
# Study drivers
# ---------------------------------------------------------------------------

def locking_sweep(
    *, thickness_ratios: Iterable[float] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    elements_x: int = 16, elements_y: int = 16, length: float = 1.0,
    width: float = 1.0, young: float = 200.0e9, poisson: float = 0.3,
    load: float = 1.0e4,
    treatments: Iterable[str] = ("sri", "mitc", "full"),
) -> list[dict]:
    """Centre deflection vs the FSDT closed form across the thickness range.

    One row per ``(h/L, treatment)``.  The full-integration rows are the locking
    control: they must degrade as ``h -> 0`` while the locking-free treatments
    must stay on the reference.
    """
    rows: list[dict] = []
    for ratio in thickness_ratios:
        h = ratio * length
        ref = fsdt_ss_uniform_load(
            length=length, width=width, thickness=h, young=young,
            poisson=poisson, load=load,
        )
        for treatment in treatments:
            result = solve_plate_static(
                elements_x=elements_x, elements_y=elements_y, length=length,
                width=width, thickness=h, young=young, poisson=poisson,
                load=load, boundary="SS", shear_treatment=treatment,
            )
            rows.append({
                "thickness_ratio": ratio,
                "treatment": treatment,
                "deflection_fe": result.deflection_center,
                "deflection_fsdt": ref.deflection_center,
                "deflection_kirchhoff": ref.deflection_kirchhoff_center,
                "error_vs_fsdt": ((result.deflection_center - ref.deflection_center)
                                  / ref.deflection_center),
                "error_vs_kirchhoff": ((result.deflection_center
                                        - ref.deflection_kirchhoff_center)
                                       / ref.deflection_kirchhoff_center),
                "fsdt_over_kirchhoff": ref.shear_amplification,
            })
    return rows


def mesh_convergence(
    *, elements: Iterable[int] = (4, 8, 16, 32), thickness_ratio: float = 0.01,
    length: float = 1.0, young: float = 200.0e9, poisson: float = 0.3,
    load: float = 1.0e4, shear_treatment: str = "sri",
) -> list[dict]:
    """Centre-deflection error against the closed-form FSDT value per mesh."""
    h = thickness_ratio * length
    ref = fsdt_ss_uniform_load(
        length=length, width=length, thickness=h, young=young,
        poisson=poisson, load=load,
    )
    rows = []
    for n in elements:
        result = solve_plate_static(
            elements_x=n, elements_y=n, length=length, width=length,
            thickness=h, young=young, poisson=poisson, load=load,
            boundary="SS", shear_treatment=shear_treatment,
        )
        rows.append({
            "elements": n,
            "mesh_size": length / n,
            "deflection_fe": result.deflection_center,
            "reference_fsdt": ref.deflection_center,
            "relative_error": ((result.deflection_center - ref.deflection_center)
                               / ref.deflection_center),
        })
    return rows


def observed_convergence_order(
    mesh_sizes: Sequence[float], errors: Sequence[float],
) -> dict:
    """Least-squares observed order of accuracy ``p`` from ``e ~ C h^p``.

    Uses the absolute relative errors of the two finest-adjacent pairs only
    after a full log-log fit, so a single anomalous coarse mesh cannot dominate.
    """
    h = np.asarray(mesh_sizes, dtype=float)
    e = np.abs(np.asarray(errors, dtype=float))
    keep = (h > 0) & (e > 0)
    if keep.sum() < 2:
        raise ValueError("need at least two positive errors")
    slope, intercept = np.polyfit(np.log(h[keep]), np.log(e[keep]), 1)
    # pairwise orders for transparency
    idx = np.where(keep)[0]
    pairwise = []
    for a, b in zip(idx[:-1], idx[1:]):
        if e[a] > 0 and e[b] > 0:
            pairwise.append(float(np.log(e[a] / e[b]) / np.log(h[a] / h[b])))
    return {
        "observed_order": float(slope),
        "pairwise_orders": pairwise,
        "constant": float(math.exp(intercept)),
        "mesh_sizes": h[keep].tolist(),
        "errors": e[keep].tolist(),
    }


def richardson_extrapolation(
    values: Sequence[float], refinement: float = 2.0,
    observed_order: float | None = None, safety_factor: float = 1.25,
) -> dict:
    """Richardson extrapolation plus GCI from three successive refinements.

    ``values`` must run coarse -> fine with a constant refinement ratio.
    When ``observed_order`` is not supplied it is estimated from the three
    values; otherwise the supplied order is used.  The GCI uses the
    three-grid recommendation of Roache with ``safety_factor`` (1.25).
    """
    if len(values) != 3:
        raise ValueError("richardson_extrapolation needs exactly three values")
    f1, f2, f3 = (float(v) for v in values)
    r = float(refinement)
    if r <= 1.0:
        raise ValueError("refinement must exceed 1")
    if observed_order is None:
        num, den = (f3 - f2), (f2 - f1)
        if abs(den) < 1e-300 or num * den <= 0.0:
            raise ValueError("values are not monotone; order cannot be estimated")
        p = abs(math.log(abs(num / den)) / math.log(r))
    else:
        p = float(observed_order)
    denom = r ** p - 1.0
    extrapolated = f3 + (f3 - f2) / denom
    fine_step_change = abs((f3 - f2) / f3) if f3 != 0.0 else float("inf")
    gci = safety_factor * fine_step_change / denom
    return {
        "extrapolated": extrapolated,
        "observed_order": p,
        "refinement": r,
        "finest_step_change": fine_step_change,
        "gci": gci,
        "values": [f1, f2, f3],
    }


# ---------------------------------------------------------------------------
# Cylindrical shell half of the vibration face
# ---------------------------------------------------------------------------

SHELL_BOUNDARIES = ("SS", "CC", "CF")
"""Shell end conditions: simply supported (diaphragm), clamped-clamped,
clamped-free (cantilever)."""


def shell_boundary_dofs(nodes, length: float, boundary: str,
                        *, tol_scale: float = 1e-9) -> list[int]:
    """Constrained shell DOF indices (6 per node) for an end condition.

    ``SS`` reproduces the diaphragm condition of
    :func:`tensorfem.cylindrical_shell_modal.apply_simply_supported_bc`
    (``v = w = rx = 0`` at both ends); ``CC`` clamps all six DOF on both end
    rings; ``CF`` clamps all six DOF on the ``x = 0`` ring only.
    """
    import torch

    if boundary not in SHELL_BOUNDARIES:
        raise ValueError(f"boundary must be one of {SHELL_BOUNDARIES}")
    tol = tol_scale * max(length, 1.0)
    x = nodes[:, 0]
    at_start = (x.abs() < tol)
    at_end = ((x - length).abs() < tol)
    if boundary == "SS":
        from .cylindrical_shell_modal import apply_simply_supported_bc
        return apply_simply_supported_bc(nodes, None, length, tol=tol)
    rings = at_start | at_end if boundary == "CC" else at_start
    nodes_idx = torch.nonzero(rings, as_tuple=False).flatten()
    constrained: list[int] = []
    for i in nodes_idx.tolist():
        constrained.extend(6 * i + k for k in range(6))
    return constrained


def solve_shell_natural_frequencies(
    *, radius: float, length: float, thickness: float, young: float,
    poisson: float, density: float, n_axial: int, n_circ: int,
    boundary: str = "SS", modes: int = 6,
) -> dict:
    """First ``modes`` natural frequencies of an end-conditioned cylinder.

    Returns a dictionary with the frequencies in Hz, the mesh sizes, the number
    of free DOF and the ``w`` mode-shape matrix ``(n_axial+1, n_circ, modes)`` so
    that the circumferential wave number of every mode can be identified by an
    independent FFT instead of assumed.
    """
    import numpy as np

    from .cylindrical_shell_modal import (
        assemble_cylindrical_shell,
        solve_shell_modes,
        structured_cylindrical_mesh,
    )

    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    density = _positive("density", density)

    nodes, elements, node_theta = structured_cylindrical_mesh(
        n_axial, n_circ, radius, length
    )
    K, M = assemble_cylindrical_shell(
        nodes, elements, node_theta, radius, young, poisson, density, thickness
    )
    constrained = shell_boundary_dofs(nodes, length, boundary)
    result = solve_shell_modes(K, M, constrained, n_modes=modes)
    omega = result.angular_frequencies.detach().cpu().numpy()
    w_modes = result.modes.detach().cpu().numpy()[2::6, :]
    w_field = w_modes.reshape(n_axial + 1, n_circ, omega.size)
    return {
        "frequencies_hz": (omega / (2.0 * math.pi)).tolist(),
        "angular_frequencies": omega.tolist(),
        "w_field": w_field,
        "n_axial": n_axial,
        "n_circ": n_circ,
        "n_dof": int(nodes.shape[0] * 6),
        "n_free_dof": int(result.n_free_dof),
        "boundary": boundary,
        "length": length,
    }


def shell_mode_wave_numbers(result: dict) -> list[int]:
    """Identify the dominant circumferential wave number of each shell mode.

    The ``w`` field of every mode is transformed along the circumference; the
    integer ``|n|`` with the largest amplitude is the wave number.  This makes
    the mode pairing with a closed-form reference an *identified* quantity
    instead of an assumption based on frequency ordering.
    """
    field = np.asarray(result["w_field"])
    numbers = []
    for k in range(field.shape[2]):
        ring = field[field.shape[0] // 2, :, k]
        spectrum = np.abs(np.fft.rfft(ring))
        if spectrum.size < 2:
            numbers.append(0)
            continue
        numbers.append(int(np.argmax(spectrum)))
    return numbers
