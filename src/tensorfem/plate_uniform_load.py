"""Uniform lateral pressure on a square plate: exact and independent references.

Why this module exists
----------------------
The merged plate path (``plate.q4_mindlin_stiffness`` driving
``plate_mindlin.solve_plate_on_winkler_foundation``) was only ever checked
against a *single-mode* sinusoidal pressure, ``q0 sin(pi x/a) sin(pi y/a)``,
whose exact Mindlin answer is one algebraic line
(:func:`tensorfem.plate.mindlin_sine_center_deflection`).  That load is
eigen-aligned: it excites exactly one Navier mode, so it cannot detect a wrong
load-vector, a wrong higher-mode bending stiffness, or an error in the
transverse-shear split -- all it measures is mode ``(1,1)``.

Uniform pressure is the ship-structure load case (hydrostatic head on shell
plating) and it excites *every* odd Navier mode, with alternating signs at the
centre.  This module supplies the two references needed to use it as a
verification case:

``ss_uniform_kirchhoff`` / ``ss_uniform_mindlin``
    The exact Navier double series for a simply supported square under uniform
    pressure (Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*,
    Sec. 30 and 39), summed to convergence: deflection, bending moments, and
    the Mindlin transverse-shear correction to the deflection.  With 401 odd
    terms the centre values are ``w = 0.00406235 q a^4 / D`` and
    ``M = 0.0478864 q a^2`` (``nu = 0.3``), i.e. Timoshenko's tabulated
    ``0.00406`` and ``0.0479``.

``clamped_uniform_ritz``
    A Rayleigh-Ritz solution for the clamped square using products of
    clamped-clamped Euler-Bernoulli beam modes.  The stiffness is assembled
    from the exact one-dimensional integrals of the separable basis
    (``G2 (x) A0 + A0 (x) G2 + 2 nu H (x) H + 2(1-nu) A1 (x) A1``), so no
    two-dimensional quadrature is required.  The same machinery run with
    simply supported (sine) modes reproduces the exact Navier value
    ``0.00406235`` (function :func:`ss_uniform_ritz`), which is the machinery
    anchor: the Ritz code is validated against the closed form before it is
    used for the case that has no closed form.

Finite-element side
-------------------
:func:`solve_uniform_square_plate` calls the merged solver with
``winkler_k = 0`` and a uniform ``pressure``, and :func:`element_center_moments`
recovers ``(Mx, My, Mxy)`` at element centres from the Mindlin curvatures.  All
sampling is location matched: a finite-element value is only ever compared with
a reference evaluated at the same physical point.  Peak values that live on a
clamped edge (where pointwise moment series converge slowly in *both* methods)
are outside the certified scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .plate import _shape
from .plate_mindlin import MindlinPlateResult, solve_plate_on_winkler_foundation

SHEAR_CORRECTION = 5.0 / 6.0

#: roots of ``cosh(lam) cos(lam) = 1``: clamped-clamped beam eigenvalues.
CLAMPED_BEAM_EIGENVALUES = (
    4.7300407448627040, 7.8532046240958376, 10.9956078380016706,
    14.1371654912574642, 17.2787596573994810, 20.4203522456260610,
    23.5619449019234483, 26.7035375555132425, 29.8451302091030356,
    32.9867228626928287, 36.1283155162826221, 39.2699081698724155,
)


# --------------------------------------------------------------------------
# exact Navier series, simply supported square under uniform pressure
# --------------------------------------------------------------------------
def ss_uniform_kirchhoff(
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    x: float = 0.5,
    y: float = 0.5,
    terms: int = 401,
) -> dict[str, float]:
    """Exact thin-plate (Kirchhoff) Navier series for uniform pressure.

    ``w(x,y) = 16 q a^4/(pi^6 D) sum_{m,n odd} sin sin / (m n (m^2+n^2)^2)``
    ``Mx     = 16 q a^2/pi^4     sum_{m,n odd} (m^2+nu n^2) sin sin / (m n (m^2+n^2)^2)``

    The series is alternating at the centre and converges to machine-relevant
    accuracy for ``terms >= 401`` (centre deflection changes by less than
    1e-9 relative between 401 and 801 terms).
    """
    if terms < 1 or terms % 2 == 0:
        raise ValueError("terms must be a positive odd number")
    if side <= 0 or thickness <= 0 or pressure < 0:
        raise ValueError("side/thickness must be positive and pressure non-negative")
    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    s_w = s_mx = s_my = 0.0
    for m in range(1, terms + 1, 2):
        sm = math.sin(m * math.pi * x / side)
        for n in range(1, terms + 1, 2):
            s = sm * math.sin(n * math.pi * y / side)
            den = m * n * (m * m + n * n) ** 2
            s_w += s / den
            s_mx += (m * m + poisson * n * n) * s / den
            s_my += (n * n + poisson * m * m) * s / den
    return {
        "bending_stiffness": D,
        "deflection": 16.0 * pressure * side**4 / (math.pi**6 * D) * s_w,
        "moment_x": 16.0 * pressure * side**2 / math.pi**4 * s_mx,
        "moment_y": 16.0 * pressure * side**2 / math.pi**4 * s_my,
    }


def ss_uniform_mindlin(
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    x: float = 0.5,
    y: float = 0.5,
    terms: int = 401,
    shear_correction: float = SHEAR_CORRECTION,
) -> dict[str, float]:
    """Exact Mindlin deflection for the same case: bending plus shear.

    For a Navier load ``q_mn`` the Mindlin deflection is
    ``w_mn = q_mn (1 + lam^2 D/(kappa G h)) / (D lam^4)``, so the shear part of
    the uniform-pressure series is
    ``16 q a^2/(pi^4 kappa G h) sum sin sin / (m n (m^2+n^2))``.  Bending
    moments are unchanged by the shear correction for sinusoidal data, so the
    Kirchhoff moment series remains the reference.
    """
    kirch = ss_uniform_kirchhoff(side, young, poisson, thickness, pressure, x, y, terms)
    G = young / (2.0 * (1.0 + poisson))
    s_sh = 0.0
    for m in range(1, terms + 1, 2):
        sm = math.sin(m * math.pi * x / side)
        for n in range(1, terms + 1, 2):
            s_sh += sm * math.sin(n * math.pi * y / side) / (m * n * (m * m + n * n))
    shear = 16.0 * pressure * side**2 / (math.pi**4 * shear_correction * G * thickness) * s_sh
    out = dict(kirch)
    out["deflection_kirchhoff"] = kirch["deflection"]
    out["deflection_shear"] = shear
    out["deflection"] = kirch["deflection"] + shear
    return out


# --------------------------------------------------------------------------
# Gauss-Legendre and separable Rayleigh-Ritz references
# --------------------------------------------------------------------------
def _gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """Gauss-Legendre nodes/weights on [-1, 1] (Newton on the Legendre ODE)."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, 0.0
            for j in range(1, n + 1):
                p2 = p1
                p1 = p0
                p0 = ((2 * j - 1) * z * p1 - (j - 1) * p2) / j
            dp = n * (z * p0 - p1) / (z * z - 1.0)
            dz = -p0 / dp
            z += dz
            if abs(dz) < 1e-16:
                break
        nodes.append(z)
        weights.append(2.0 / ((1.0 - z * z) * dp * dp))
    return nodes, weights


def _clamped_mode(m: int, x: float) -> tuple[float, float, float]:
    """Clamped-clamped beam mode ``m`` on the unit interval: (psi, psi', psi'')."""
    lam = CLAMPED_BEAM_EIGENVALUES[m]
    ch, cl = math.cosh(lam), math.cos(lam)
    sh, sl = math.sinh(lam), math.sin(lam)
    sigma = (ch - cl) / (sh - sl)
    C, S = math.cosh(lam * x), math.sinh(lam * x)
    c, s = math.cos(lam * x), math.sin(lam * x)
    psi = (C - c) - sigma * (S - s)
    d1 = lam * (S + s) - sigma * lam * (C - c)
    d2 = lam**2 * (C + c) - sigma * lam**2 * (S + s)
    return psi, d1, d2


def _sine_mode(m: int, x: float) -> tuple[float, float, float]:
    """Simply supported beam mode ``m+1``: sin((m+1) pi x) and derivatives."""
    k = (m + 1) * math.pi
    return math.sin(k * x), k * math.cos(k * x), -(k * k) * math.sin(k * x)


@dataclass(frozen=True)
class RitzPlateSolution:
    """Separable Rayleigh-Ritz solution of the Kirchhoff square plate."""

    side: float
    bending_stiffness: float
    pressure: float
    poisson: float
    n_modes: int
    coefficients: tuple[tuple[float, ...], ...]
    mode: str  # "clamped" or "sine"

    def _mode(self, m: int, x: float) -> tuple[float, float, float]:
        if self.mode == "clamped":
            return _clamped_mode(m, x * self.side)
        return _sine_mode(m, x * self.side)

    def deflection(self, x: float, y: float) -> float:
        """Transverse deflection at physical point ``(x, y)``."""
        xr, yr = x / self.side, y / self.side
        total = 0.0
        for i, row in enumerate(self.coefficients):
            fi = self._mode(i, xr)[0]
            for j, cij in enumerate(row):
                if cij:
                    total += cij * fi * self._mode(j, yr)[0]
        return total

    def moments(self, x: float, y: float) -> tuple[float, float]:
        """``(Mx, My)`` at ``(x, y)``; derivatives are w.r.t. physical length."""
        xr, yr = x / self.side, y / self.side
        wxx = wyy = 0.0
        for i, row in enumerate(self.coefficients):
            d2i = self._mode(i, xr)[2]
            f2i = self._mode(i, xr)[0]
            for j, cij in enumerate(row):
                if cij:
                    wxx += cij * d2i * self._mode(j, yr)[0]
                    wyy += cij * f2i * self._mode(j, yr)[2]
        D, nu = self.bending_stiffness, self.poisson
        return -D * (wxx + nu * wyy), -D * (wyy + nu * wxx)


def _ritz_solve(
    mode: str,
    n_modes: int,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    pressure: float,
    n_quad: int = 60,
) -> RitzPlateSolution:
    """Assemble the separable Ritz system and solve for the modal coefficients."""
    if mode not in ("clamped", "sine"):
        raise ValueError("mode must be 'clamped' or 'sine'")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    nodes, weights = _gauss_legendre(n_quad)
    # Map [-1, 1] -> [0, 1] and fold the side length into the physical length.
    xs = [0.5 * (z + 1.0) for z in nodes]
    ws = [0.5 * w for w in weights]
    func = _clamped_mode if mode == "clamped" else _sine_mode
    values = [[func(m, x)[0] for x in xs] for m in range(n_modes)]
    d1 = [[func(m, x)[1] for x in xs] for m in range(n_modes)]
    d2 = [[func(m, x)[2] for x in xs] for m in range(n_modes)]
    A0 = [[sum(w * values[p][k] * values[q][k] for k, w in enumerate(ws))
           for q in range(n_modes)] for p in range(n_modes)]
    A1 = [[sum(w * d1[p][k] * d1[q][k] for k, w in enumerate(ws))
           for q in range(n_modes)] for p in range(n_modes)]
    G2 = [[sum(w * d2[p][k] * d2[q][k] for k, w in enumerate(ws))
           for q in range(n_modes)] for p in range(n_modes)]
    H = [[sum(w * d2[p][k] * values[q][k] for k, w in enumerate(ws))
          for q in range(n_modes)] for p in range(n_modes)]
    load_1d = [sum(w * values[p][k] for k, w in enumerate(ws)) for p in range(n_modes)]
    # Physical-length derivatives are (1/a) d/dx_normalised, so every second
    # derivative integral carries a 1/a^2 and dA carries a factor a^2.
    scale = D / side**2
    ndof = n_modes * n_modes
    K = [[0.0] * ndof for _ in range(ndof)]
    f = [0.0] * ndof
    for a in range(n_modes):
        for b in range(n_modes):
            gi = a * n_modes + b
            f[gi] = pressure * side**2 * load_1d[a] * load_1d[b]
            for c in range(n_modes):
                for d in range(n_modes):
                    gj = c * n_modes + d
                    K[gi][gj] = scale * (
                        G2[a][c] * A0[b][d] + A0[a][c] * G2[b][d]
                        + 2.0 * poisson * H[a][c] * H[b][d]
                        + 2.0 * (1.0 - poisson) * A1[a][c] * A1[b][d]
                    )
    # Symmetric positive definite 100x100 system: Gaussian elimination with
    # partial pivoting in pure Python keeps torch out of the reference path.
    coeff = _solve_dense(K, f)
    rows = tuple(tuple(coeff[a * n_modes + b] for b in range(n_modes))
                 for a in range(n_modes))
    return RitzPlateSolution(side, D, pressure, poisson, n_modes, rows, mode)


def _solve_dense(K: list[list[float]], f: list[float]) -> list[float]:
    n = len(f)
    A = [row[:] + [f[i]] for i, row in enumerate(K)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-300:
            raise ValueError("singular Ritz system")
        A[col], A[piv] = A[piv], A[col]
        piv_val = A[col][col]
        for r in range(col + 1, n):
            factor = A[r][col] / piv_val
            if factor:
                for c in range(col, n + 1):
                    A[r][c] -= factor * A[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = A[r][n] - sum(A[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / A[r][r]
    return x


def clamped_uniform_ritz(
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    n_modes: int = 10,
) -> RitzPlateSolution:
    """Rayleigh-Ritz reference for the clamped square under uniform pressure.

    ``n_modes = 10`` reproduces Timoshenko's tabulated
    ``w_max = 0.00126 q a^4/D`` and centre moment ``0.0231 q a^2`` (nu = 0.3);
    the N=8/10/12 sequence agrees within 1e-3 relative for both.
    """
    return _ritz_solve("clamped", n_modes, side, young, poisson, thickness, pressure)


def ss_uniform_ritz(
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    n_modes: int = 8,
) -> RitzPlateSolution:
    """Machinery anchor: the Ritz code run with sine modes must hit the exact series.

    This is not a separate physical reference -- it exists so that the clamped
    Ritz result (which has no closed form) is produced by a code path that has
    first reproduced ``0.00406235 q a^4/D`` against the Navier series.
    """
    return _ritz_solve("sine", n_modes, side, young, poisson, thickness, pressure)


# --------------------------------------------------------------------------
# finite-element side
# --------------------------------------------------------------------------
def solve_uniform_square_plate(
    nx: int,
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    boundary: str = "simply_supported",
) -> MindlinPlateResult:
    """Uniform pressure on a square plate through the merged Q4 Mindlin path.

    ``winkler_k = 0`` reduces the merged Winkler solver to the plain plate, so
    this function adds no new element or assembly code: it is the verification
    entry point for the already merged feature.
    """
    if boundary not in ("simply_supported", "clamped"):
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    return solve_plate_on_winkler_foundation(
        nx=nx,
        side=side,
        young=young,
        poisson=poisson,
        thickness=thickness,
        winkler_k=0.0,
        pressure=pressure,
        boundary=boundary,
    )


def centre_node_deflection(result: MindlinPlateResult) -> float:
    """Deflection at the plate centre node (the mesh must have an odd node count)."""
    nodes = result.nodes
    side = float(nodes[:, 0].max())
    tol = side * 1e-9
    mask = (nodes[:, 0] - 0.5 * side).abs() < tol
    idx = torch.nonzero(mask)[:, 0]
    idx = idx[(nodes[idx, 1] - 0.5 * side).abs() < tol]
    if idx.numel() != 1:
        raise ValueError("mesh must contain exactly one node at the plate centre")
    return float(result.displacement[3 * int(idx)].item())


def node_values_along_x(result: MindlinPlateResult, y: float) -> tuple[list[float], list[float]]:
    """Nodes and deflection along the line ``y = const`` (sorted by ``x``)."""
    nodes = result.nodes
    side = float(nodes[:, 0].max())
    tol = side * 1e-9
    idx = torch.nonzero((nodes[:, 1] - y).abs() < tol)[:, 0]
    order = torch.argsort(nodes[idx, 0])
    idx = idx[order]
    xs = [float(v) for v in nodes[idx, 0]]
    ws = [float(result.displacement[3 * int(i)].item()) for i in idx]
    return xs, ws


def element_center_moments(result: MindlinPlateResult) -> dict[str, list[float]]:
    """``(Mx, My, Mxy)`` at element centres from the Mindlin curvatures.

    Same curvature-to-moment relation as ``plate_mindlin.compute_face_stresses``
    (``Mx = D(kappa_x + nu kappa_y)``), evaluated at the element centre only, so
    every value can be compared with a reference at the identical point.
    """
    D = result.young * result.thickness**3 / (12.0 * (1.0 - result.poisson**2))
    nu = result.poisson
    cx, cy, mx, my, mxy = [], [], [], [], []
    for conn in result.elements:
        xy = result.nodes[conn]
        _, dN_nat = _shape(xy.new_tensor(0.0), xy.new_tensor(0.0))
        J = dN_nat.T @ xy
        dN = dN_nat @ torch.linalg.inv(J).T
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        u = result.displacement[dofs]
        tx, ty = u[1::3], u[2::3]
        kx = float(dN[:, 0] @ tx)
        ky = float(dN[:, 1] @ ty)
        kxy = float(dN[:, 0] @ ty + dN[:, 1] @ tx)
        cent = xy.mean(dim=0)
        cx.append(float(cent[0]))
        cy.append(float(cent[1]))
        mx.append(D * (kx + nu * ky))
        my.append(D * (ky + nu * kx))
        mxy.append(D * (1.0 - nu) / 2.0 * kxy)
    return {"x": cx, "y": cy, "moment_x": mx, "moment_y": my, "moment_xy": mxy}


def nearest_element_center(moments: dict[str, list[float]], x: float, y: float) -> int:
    """Index of the element centre closest to ``(x, y)``."""
    best, best_d = 0, None
    for i, (cx, cy) in enumerate(zip(moments["x"], moments["y"])):
        d = (cx - x) ** 2 + (cy - y) ** 2
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best