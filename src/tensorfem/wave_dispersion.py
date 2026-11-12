"""Rayleigh--Lamb wave dispersion: analytic oracle + Bloch--Floquet FEM cell.

Verification surface for the *dynamics* of the production plane-strain Q4
continuum element (`tensorfem.continuum.q4_stiffness`) plus a consistent mass
matrix.  The oracle is the exact Rayleigh--Lamb frequency equation for a
traction-free elastic strip (plane strain, thickness ``2h``), independent of any
finite element discretisation:

* symmetric (extensional) family  S_n:
  ``(q^2-k^2)^2 sin(qh) cos(ph) + 4 k^2 p q cos(qh) sin(ph) = 0``
* antisymmetric (flexural) family A_n:
  ``4 k^2 p q sin(qh) cos(ph) + (q^2-k^2)^2 cos(qh) sin(ph) = 0``

with ``p^2 = w^2/cL^2 - k^2``, ``q^2 = w^2/cT^2 - k^2``.  ``p`` and ``q`` become
imaginary below the corresponding cut-offs, so both are evaluated through
complex square roots; the residual is real for real ``k``.

The FEM side builds one rectangular Q4 cell of length ``L`` and thickness
``2h``, assembles the real stiffness/mass matrices of the production element and
imposes the Floquet condition ``u(x+L) = exp(i k L) u(x)`` by a complex
reduction ``T``: ``K_red = T^H K T``, ``M_red = T^H M T``.  The generalised
Hermitian eigenproblem then gives the discrete branches ``w_n(k)`` whose
long-wavelength limit must reproduce the analytic branches.

Neither path consumes the other, so agreement is a verification statement rather
than a self-consistency check.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
import torch

from .continuum import elasticity_matrix, q4_stiffness, rectangular_q4_mesh

__all__ = [
    "shear_speed",
    "longitudinal_speed",
    "plate_speed",
    "rayleigh_speed",
    "flexural_phase_velocity",
    "lamb_residual",
    "rayleigh_lamb_frequencies",
    "q4_consistent_mass",
    "bloch_cell_matrices",
    "bloch_cell_frequencies",
]

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# material wave speeds (exact)
# ---------------------------------------------------------------------------
def shear_speed(young: float, poisson: float, density: float) -> float:
    """``cT = sqrt(mu/rho)``."""
    mu = young / (2.0 * (1.0 + poisson))
    return math.sqrt(mu / density)


def longitudinal_speed(young: float, poisson: float, density: float) -> float:
    """``cL = sqrt((lambda+2mu)/rho)``."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return math.sqrt((lam + 2.0 * mu) / density)


def plate_speed(young: float, poisson: float, density: float) -> float:
    """Plane-strain extensional plate velocity ``sqrt(E/(rho (1-nu^2)))``.

    Derived ``k -> 0`` limit of the S0 branch, not a fitted constant.
    """
    return math.sqrt(young / ((1.0 - poisson ** 2) * density))


def rayleigh_speed(young: float, poisson: float, density: float) -> float:
    """Rayleigh surface-wave speed from the classical cubic in ``z = c^2/cT^2``.

    ``z^3 - 8 z^2 + 8 z (3 - 2 eta) - 16 (1 - eta) = 0`` with
    ``eta = cT^2/cL^2 = (1-2 nu)/(2 (1-nu))``; the physical root is the real one
    in ``(0, 1)``.
    """
    cT = shear_speed(young, poisson, density)
    eta = (1.0 - 2.0 * poisson) / (2.0 * (1.0 - poisson))
    roots = np.roots([1.0, -8.0, 8.0 * (3.0 - 2.0 * eta), -16.0 * (1.0 - eta)])
    for root in roots:
        if abs(root.imag) < 1e-10 and 1e-8 < root.real < 1.0:
            return cT * math.sqrt(float(root.real))
    raise RuntimeError("no physical Rayleigh root found")


def flexural_phase_velocity(wave_number: float, young: float, poisson: float,
                            density: float, half_thickness: float) -> float:
    """Kirchhoff thin-plate flexural phase velocity (A0 long-wavelength limit).

    ``c_ph = k h sqrt(E' / (3 rho))`` with ``E' = E/(1-nu^2)``, from
    ``D = E'(2h)^3/12`` and ``c_ph^2 = D k^2/(rho 2h)``.  Independent of the
    Rayleigh--Lamb equation and of the finite element formulation.
    """
    e_plate = young / (1.0 - poisson ** 2)
    return wave_number * half_thickness * math.sqrt(e_plate / (3.0 * density))


# ---------------------------------------------------------------------------
# exact Rayleigh--Lamb equation
# ---------------------------------------------------------------------------
def lamb_residual(omega: float, wave_number: float, c_long: float, c_shear: float,
                  half_thickness: float, branch: str) -> float:
    """Normalised real residual of the Rayleigh--Lamb equation.

    ``branch`` is ``"symmetric"`` (extensional S family) or ``"antisymmetric"``
    (flexural A family).  The three frequency regimes are written with real
    arithmetic only, because ``tan(i x) = i tanh(x)``: below the shear and
    longitudinal cut-offs the trigonometric tangents become hyperbolic and the
    sign of the residual would otherwise be lost in a complex phase.  The
    residual is divided by the amplitude of its own terms so that the bisection
    tolerance is dimensionless.
    """
    if branch not in ("symmetric", "antisymmetric"):
        raise ValueError("branch must be 'symmetric' or 'antisymmetric'")
    k = float(wave_number)
    w = float(omega)
    h = float(half_thickness)
    symmetric = branch == "symmetric"
    p2 = (w / c_long) ** 2 - k * k
    q2 = (w / c_shear) ** 2 - k * k
    if p2 > 0.0 and q2 > 0.0:
        p, q = math.sqrt(p2), math.sqrt(q2)
        tp, tq = math.tan(p * h), math.tan(q * h)
        cross = 4.0 * k * k * p * q
        qk = q2 - k * k
        num, den = (qk * qk, cross) if symmetric else (cross, qk * qk)
        # tan(qh)/tan(ph) = -num/den, multiplied through by cos(ph) to stay finite.
        value = tq * math.cos(p * h) + (num / den) * math.sin(p * h)
        scale = abs(tq) + abs(num / den) + 1.0
        return value / scale
    if q2 > 0.0:  # k*cT < w < k*cL: p evanescent, q propagating
        p, q = math.sqrt(-p2), math.sqrt(q2)
        ratio = math.tan(q * h) / math.tanh(p * h)
        qk = q2 - k * k
        cross = 4.0 * k * k * p * q
        if cross == 0.0 or qk == 0.0:
            return 1.0
        rhs = cross / (qk * qk) if symmetric else -(qk * qk) / cross
        return (ratio - rhs) / (abs(ratio) + abs(rhs) + 1.0)
    # w < k*cT: both evanescent (the thin-plate flexural regime)
    p, q = math.sqrt(-p2), math.sqrt(-q2)
    ratio = math.tanh(q * h) / math.tanh(p * h)
    qk = k * k - q2  # (k^2 + q_evanescent^2) because q2 < 0 here
    cross = 4.0 * k * k * p * q
    if cross == 0.0 or qk == 0.0:
        return 1.0
    rhs = cross / (qk * qk) if symmetric else (qk * qk) / cross
    return (ratio - rhs) / (abs(ratio) + abs(rhs) + 1.0)


def _bisect(func, lo: float, hi: float, f_lo: float, f_hi: float,
            tol: float = 1e-15, max_iter: int = 200) -> float:
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise ValueError("bracket does not contain a sign change")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0 or (hi - lo) < tol * max(1.0, abs(mid)):
            return mid
        if f_lo * f_mid < 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def rayleigh_lamb_frequencies(wave_number: float, half_thickness: float, young: float,
                              poisson: float, density: float, *, n_roots: int = 4,
                              scan_per_branch: int = 1000) -> dict[str, list[float]]:
    """Lowest Rayleigh--Lamb frequencies at a given real wave number.

    Returns ``{"symmetric": [...], "antisymmetric": [...]}`` sorted ascending and
    truncated to ``n_roots``.  The scan spans
    ``0 < w <= (n_roots+1) pi cT/(2h)``, which contains at least ``n_roots``
    roots of each family because the cut-off spacing is ``pi cT/(2h)``.
    """
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if half_thickness <= 0.0:
        raise ValueError("half_thickness must be positive")
    c_long = longitudinal_speed(young, poisson, density)
    c_shear = shear_speed(young, poisson, density)
    # Window must cover both the cut-off ladder and the propagating range at
    # large k (where w ~ cL k can exceed the cut-off scale by orders of magnitude).
    omega_max = max((n_roots + 1) * math.pi * c_shear / (2.0 * half_thickness),
                    3.0 * longitudinal_speed(young, poisson, density) * wave_number)
    # Log-spaced scan: the flexural A0 root can sit orders of magnitude below the
    # cut-off scale pi*cT/(2h), while the trigonometric residual still oscillates
    # on that cut-off scale.  A geometric grid resolves both ends.
    steps = scan_per_branch * (n_roots + 1)
    omega_floor = 1e-6 * wave_number * min(c_long, c_shear)
    log_lo, log_hi = math.log(omega_floor), math.log(omega_max)
    grid = [math.exp(log_lo + (log_hi - log_lo) * (i + 1) / steps) for i in range(steps)]
    out: dict[str, list[float]] = {}
    for branch in ("symmetric", "antisymmetric"):
        func = lambda w, b=branch: lamb_residual(w, wave_number, c_long, c_shear,
                                                 half_thickness, b)
        roots: list[float] = []
        f_prev, w_prev = func(grid[0]), grid[0]
        for omega in grid[1:]:
            f_cur = func(omega)
            if f_prev * f_cur < 0.0:
                root = _bisect(func, w_prev, omega, f_prev, f_cur)
                # A sign change across a tan/ tanh pole is not a root: the
                # normalised residual jumps between +-1 there, so a genuine root
                # is the only case with |residual| below this tolerance.
                if abs(func(root)) < 1e-6:
                    roots.append(root)
            f_prev, w_prev = f_cur, omega
        out[branch] = roots[:n_roots]
    return out


# ---------------------------------------------------------------------------
# production element + consistent mass
# ---------------------------------------------------------------------------
def q4_consistent_mass(coordinates: torch.Tensor, density: float,
                       thickness: float = 1.0) -> torch.Tensor:
    """Consistent mass matrix of the bilinear quad, ``rho t int N_i N_j detJ``.

    Exact for the bilinear interpolation (2x2 Gauss integrates the quartic
    integrand of a bilinear map exactly).  ``coordinates`` has shape ``(e,4,2)``
    and the result has shape ``(e,8,8)``.
    """
    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 2):
        raise ValueError("coordinates must have shape (elements, 4, 2)")
    g = 1.0 / 3.0 ** 0.5
    points = ((-g, -g), (g, -g), (g, g), (-g, g))
    ke = torch.zeros((len(coordinates), 8, 8), dtype=coordinates.dtype,
                     device=coordinates.device)
    for xi, eta in points:
        shape = coordinates.new_tensor([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                        (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
        natural = coordinates.new_tensor([
            [-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]
        ]) / 4
        det = torch.linalg.det(torch.einsum("eia,ib->eab", coordinates, natural))
        if torch.any(det <= 0):
            raise ValueError("Q4 element has non-positive Jacobian")
        nmat = torch.outer(shape, shape)
        pattern = torch.zeros((8, 8), dtype=coordinates.dtype)
        pattern[0::2, 0::2] = nmat
        pattern[1::2, 1::2] = nmat
        ke = ke + pattern * (det * density * thickness)[:, None, None]
    return ke


@dataclass(frozen=True)
class BlochCell:
    """Assembled reduced (Floquet) cell matrices and the mesh it used."""

    wave_number: float
    cell_length: float
    half_thickness: float
    nodes: torch.Tensor
    elements: torch.Tensor
    stiffness: torch.Tensor
    mass: torch.Tensor
    total_mass: float


def bloch_cell_matrices(wave_number: float, *, half_thickness: float, young: float,
                        poisson: float, density: float, nx: int = 16, ny: int = 8,
                        cells_per_wavelength: float | None = None,
                        cell_length_in_half_thickness: float = 2.0) -> BlochCell:
    """Assemble the Floquet-reduced stiffness and mass matrices of one cell.

    The cell length is a *fixed physical length*,
    ``cell_length_in_half_thickness * half_thickness``, independent of ``k``:
    every wave number then samples a different phase advance ``k L`` over one and
    the same cell, which is what a dispersion relation is.  (An earlier version
    set ``L = 2 pi / (k * cells_per_wavelength)``, i.e. a constant ``k L``; the
    discrete problem is then scale invariant and the "dispersion curve" collapses
    to a single phase velocity -- a defect recorded in section 8 of the anchor
    document.)  ``cells_per_wavelength`` is kept only as a compatibility shim for
    callers that explicitly want the old geometry; pass ``None`` (default) for the
    physical cell.

    The reduced unknown set keeps *both* displacement components per retained
    node: collapsing ``u_x`` and ``u_y`` onto one column would add their stiffness
    and mass entries together (the defect fixed here).
    """
    if wave_number < 0.0:
        raise ValueError("wave_number must be non-negative")
    if nx < 2 or ny < 1:
        raise ValueError("nx must be >= 2 and ny >= 1")
    # k = 0 is the periodic (Gamma point) cell: used by the rigid-mode count check.
    if cells_per_wavelength is not None:            # legacy 1/k-scaled geometry
        cell_length = (2.0 * math.pi / (wave_number * cells_per_wavelength)
                       if wave_number > 0.0 else 10.0 * half_thickness)
    else:                                           # fixed physical cell
        cell_length = float(cell_length_in_half_thickness) * half_thickness
    nodes, elements = rectangular_q4_mesh(cell_length, 2.0 * half_thickness, nx, ny,
                                          dtype=_DTYPE)
    dmat = elasticity_matrix(torch.tensor([young], dtype=_DTYPE),
                             torch.tensor([poisson], dtype=_DTYPE), "strain")[0]
    coords = nodes[elements]
    ke, _ = q4_stiffness(coords, dmat, torch.tensor(1.0, dtype=_DTYPE))
    me = q4_consistent_mass(coords, density, 1.0)
    ndof = 2 * len(nodes)
    stiff = torch.zeros((ndof, ndof), dtype=_DTYPE)
    mass = torch.zeros((ndof, ndof), dtype=_DTYPE)
    edofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), 2).reshape(-1, 8).long()
    rows = edofs[:, :, None].expand(-1, -1, 8).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 8, -1).reshape(-1)
    stiff = stiff.index_put((rows, cols), ke.reshape(-1), accumulate=True)
    mass = mass.index_put((rows, cols), me.reshape(-1), accumulate=True)

    # Floquet reduction: right-hand column (x = L) maps onto the left column.
    xs = nodes[:, 0]
    left = torch.nonzero(torch.isclose(xs, torch.tensor(0.0, dtype=_DTYPE))).flatten()
    right = torch.nonzero(torch.isclose(xs, torch.tensor(cell_length, dtype=_DTYPE))).flatten()
    partner = {int(r): int(l) for r, l in zip(right.tolist(), left.tolist())}
    phase = cmath.exp(1j * wave_number * cell_length)
    # Retained node -> compact index; the right column is eliminated by
    # u_right = exp(i k L) u_left, so both columns share one compact index.
    reduced: dict[int, int] = {}
    next_index = 0
    for node in range(len(nodes)):
        if node in partner:
            reduced[node] = reduced[partner[node]]
        else:
            reduced[node] = next_index
            next_index += 1
    nred = 2 * next_index           # one unknown per component per retained node
    trans = torch.zeros((ndof, nred), dtype=torch.complex128)
    for node in range(len(nodes)):
        coeff = complex(phase if node in partner else 1.0)
        for d in (0, 1):
            trans[2 * node + d, 2 * reduced[node] + d] = coeff
    stiff_red = trans.conj().T @ stiff.to(torch.complex128) @ trans
    mass_red = trans.conj().T @ mass.to(torch.complex128) @ trans
    return BlochCell(wave_number, cell_length, half_thickness, nodes, elements,
                     stiff_red, mass_red, float(mass.sum()))


def bloch_cell_frequencies(wave_number: float, *, half_thickness: float, young: float,
                           poisson: float, density: float, nx: int = 16, ny: int = 8,
                           n_modes: int = 4, cells_per_wavelength: float | None = None,
                           cell_length_in_half_thickness: float = 2.0) -> np.ndarray:
    """Lowest ``n_modes`` angular frequencies of the Floquet cell (ascending)."""
    cell = bloch_cell_matrices(wave_number, half_thickness=half_thickness, young=young,
                               poisson=poisson, density=density, nx=nx, ny=ny,
                               cells_per_wavelength=cells_per_wavelength,
                               cell_length_in_half_thickness=cell_length_in_half_thickness)
    k_red = cell.stiffness.numpy()
    m_red = cell.mass.numpy()
    # Symmetrise away the round-off of the Hermitian reduction.
    k_red = 0.5 * (k_red + k_red.conj().T)
    m_red = 0.5 * (m_red + m_red.conj().T)
    from scipy.linalg import eigh

    values = eigh(k_red, m_red, eigvals_only=True)
    values = np.clip(np.asarray(values).real, 0.0, None)
    omega = np.sqrt(values)
    return np.sort(omega)[:n_modes]