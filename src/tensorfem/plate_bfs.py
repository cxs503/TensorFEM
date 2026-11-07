"""Bogner--Fox--Schmit (BFS) C1 bicubic plate element and Kirchhoff oracles.

The BFS rectangle carries four degrees of freedom per node,
``(w, w_,x, w_,y, w_,xy)``, so the transverse displacement field is a
bicubic Hermite polynomial that is ``C1`` across element edges.  That is the
conforming element for the *thin* (Kirchhoff) plate, and it converges in the
energy norm for both displacement and slope boundary conditions.

What this module provides
-------------------------
* :func:`bfs_element_matrices` -- bending stiffness, consistent mass and
  consistent transverse load of one element, built from the tensor product of
  cubic Hermite functions with 3x3 (stiffness/load) and 4x4 (mass) Gauss rules.
* :func:`assemble_bfs_plate` -- sparse assembly on a structured
  ``n_x`` by ``n_y`` mesh of a rectangular plate.
* :func:`bfs_boundary_dofs` -- the two Kirchhoff boundary conditions used by
  the verification ladder: hard simple support (``w = 0`` and the *tangential*
  slope zero, which is the exact Kirchhoff condition on a simply supported
  edge) and full clamping (``w = w_,x = w_,y = 0``).
* :func:`bfs_evaluate` -- recover ``w`` and its first/second derivatives at
  arbitrary points (needed for deflections, slopes and bending moments).
* Analytic oracles used by ``benchmarks/verified/fem_*``: Euler--Bernoulli
  beam solutions, the Navier double-sine solution of the simply supported
  plate (deflection *and* bending moments), the Navier eigenfrequencies of a
  rectangular simply supported plate, and the Timoshenko table coefficients.

Conventions
-----------
Positive ``w`` is downward (the load ``q`` is positive downward) and the
bending moments follow ``M_x = -D (w_,xx + nu w_,yy)``, so a sagging plate has
``M_x > 0``.  The dof ordering per element is node-major over
``(i, j) = (0,0), (1,0), (0,1), (1,1)`` with ``k = 0..3`` equal to
``(w, w_,x, w_,y, w_,xy)``.  Slope dofs carry physical derivatives (1/m), i.e.
the shape functions are the *length-scaled* Hermite cubics; this keeps the
element matrices in physical units.

The bending matrix is also available from
:func:`tensorfem.plate_buckling_hermite.bfs_plate_element_matrices`; the test
suite asserts the two independently written implementations agree to round-off
(``tests/test_plate_bfs.py``), which is why the duplicate is kept rather than
imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "TIMOSHENKO_SSSS_ALPHA",
    "TIMOSHENKO_CCCC_ALPHA",
    "TIMOSHENKO_SSSS_MOMENT",
    "TIMOSHENKO_CCCC_MOMENT",
    "BFSAssembledPlate",
    "bfs_element_matrices",
    "assemble_bfs_plate",
    "bfs_boundary_dofs",
    "bfs_evaluate",
    "bfs_moments",
    "solve_bfs_static",
    "solve_bfs_modes",
    "solve_bfs_mode_shapes",
    "plate_rigidity",
    "ssss_navier_centre",
    "ssss_navier_deflection",
    "ssss_navier_moments",
    "ssss_plate_angular_frequencies",
    "simply_supported_uniform_load_deflection",
    "cantilever_tip_load_deflection",
    "ss_beam_angular_frequencies",
]

# ---------------------------------------------------------------------------
# Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., 1959.
# Table 8  (SSSS square, nu = 0.3): w_max = 0.00406 q a^4 / D, M = 0.0479 q a^2
# Table 35 (CCCC square, nu = 0.3): w_max = 0.00126 q a^4 / D, M = 0.0231 q a^2
# The table values are quoted to three significant digits, so they are graded
# with the campaign's 2 % plate tolerance; the Navier series is the
# higher-precision oracle where it applies (simply supported edges only).
# ---------------------------------------------------------------------------
TIMOSHENKO_SSSS_ALPHA = 0.00406
TIMOSHENKO_CCCC_ALPHA = 0.00126
TIMOSHENKO_SSSS_MOMENT = 0.0479
TIMOSHENKO_CCCC_MOMENT = 0.0231

_INV_PHI = 1.0 / math.sqrt(3.0)


def _gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre rule on the unit interval ``[0, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _hermite(s: np.ndarray | float, length: float):
    """Value, first and second ``x``-derivatives of the four cubic Hermites.

    Ordering ``(w@0, w_,x@0, w@1, w_,x@1)``; the slope entries are scaled by
    ``length`` so that the returned first/second derivatives are derivatives
    with respect to the physical coordinate ``x = length * s``.
    """
    s = np.asarray(s, dtype=float)
    values = np.array(
        [
            1.0 - 3.0 * s**2 + 2.0 * s**3,
            length * (s - 2.0 * s**2 + s**3),
            3.0 * s**2 - 2.0 * s**3,
            length * (-(s**2) + s**3),
        ]
    )
    first = np.array(
        [
            (-6.0 * s + 6.0 * s**2) / length,
            1.0 - 4.0 * s + 3.0 * s**2,
            (6.0 * s - 6.0 * s**2) / length,
            -2.0 * s + 3.0 * s**2,
        ]
    )
    second = np.array(
        [
            (-6.0 + 12.0 * s) / length**2,
            (-4.0 + 6.0 * s) / length,
            (6.0 - 12.0 * s) / length**2,
            (-2.0 + 6.0 * s) / length,
        ]
    )
    return values, first, second


def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Kirchhoff bending stiffness ``D = E t^3 / (12 (1 - nu^2))``."""
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def bfs_element_matrices(
    length_x: float,
    length_y: float,
    young: float,
    poisson: float,
    thickness: float,
    density: float | None = None,
    surface_load: float = 0.0,
) -> dict[str, np.ndarray]:
    """Element matrices of one BFS rectangle.

    Returns a dict with ``bending`` (16x16), ``mass`` (16x16 or ``None`` when
    ``density`` is not given) and ``load`` (16) carrying
    ``surface_load`` (force per unit area, positive along ``+w``).
    """
    for name, value in (("length_x", length_x), ("length_y", length_y),
                        ("young", young), ("thickness", thickness)):
        if not float(value) > 0.0:
            raise ValueError(f"{name} must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie in (-1, 0.5)")

    stiffness_d = plate_rigidity(young, poisson, thickness)
    constitutive = stiffness_d * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]]
    )
    jacobian = length_x * length_y  # s, t in [0, 1] -> dx dy = lx ly ds dt
    ix = (0, 1, 0, 1)
    iy = (0, 0, 1, 1)

    bending = np.zeros((16, 16))
    mass = np.zeros((16, 16)) if density is not None else None
    load = np.zeros(16)

    rule_k = _gauss_rule(3)
    for gx, wx in zip(*rule_k):
        hx, dx_, ddx_ = _hermite(gx, length_x)
        for gy, wy in zip(*rule_k):
            hy, dy_, ddy_ = _hermite(gy, length_y)
            weight = wx * wy * jacobian
            operator = np.zeros((3, 16))
            shape = np.zeros(16)
            for node in range(4):
                di, dj = node % 2, node // 2
                for k in range(4):
                    dof = 4 * node + k
                    a = ix[k] + 2 * di
                    b = iy[k] + 2 * dj
                    operator[0, dof] = ddx_[a] * hy[b]
                    operator[1, dof] = hx[a] * ddy_[b]
                    operator[2, dof] = 2.0 * dx_[a] * dy_[b]
                    shape[dof] = hx[a] * hy[b]
            bending += weight * (operator.T @ constitutive @ operator)
            load += weight * surface_load * shape

    if mass is not None:
        rule_m = _gauss_rule(4)
        for gx, wx in zip(*rule_m):
            hx = _hermite(gx, length_x)[0]
            for gy, wy in zip(*rule_m):
                hy = _hermite(gy, length_y)[0]
                weight = wx * wy * jacobian * density * thickness
                shape = np.zeros(16)
                for node in range(4):
                    di, dj = node % 2, node // 2
                    for k in range(4):
                        shape[4 * node + k] = hx[ix[k] + 2 * di] * hy[iy[k] + 2 * dj]
                mass += weight * np.outer(shape, shape)

    return {"bending": bending, "mass": mass, "load": load}


def _dof_count(n_x: int, n_y: int) -> int:
    return 4 * (n_x + 1) * (n_y + 1)


def _node_dof(n_x: int, i: int, j: int, k: int) -> int:
    return 4 * (j * (n_x + 1) + i) + k


@dataclass(frozen=True)
class BFSAssembledPlate:
    """Sparse BFS mesh operators on a rectangular ``a x b`` plate."""

    n_x: int
    n_y: int
    length_x: float
    length_y: float
    stiffness: sp.csc_matrix
    mass: sp.csc_matrix | None
    load: np.ndarray

    @property
    def n_dof(self) -> int:
        return self.stiffness.shape[0]

    @property
    def h(self) -> float:
        """Largest element side (mesh size measure used for the ladders)."""
        return max(self.length_x / self.n_x, self.length_y / self.n_y)


def assemble_bfs_plate(
    n_x: int,
    n_y: int,
    length_x: float,
    length_y: float,
    young: float,
    poisson: float,
    thickness: float,
    *,
    density: float | None = None,
    surface_load: float = 0.0,
) -> BFSAssembledPlate:
    """Assemble stiffness (and optionally mass and load) for a BFS plate mesh."""
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if int(value) < 1:
            raise ValueError(f"{name} must be at least 1")
    element = bfs_element_matrices(
        length_x / n_x,
        length_y / n_y,
        young,
        poisson,
        thickness,
        density=density,
        surface_load=surface_load,
    )
    ndof = _dof_count(n_x, n_y)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    mass_rows: list[int] = []
    mass_cols: list[int] = []
    mass_vals: list[float] = []
    load = np.zeros(ndof)
    for j in range(n_y):
        for i in range(n_x):
            dofs = np.array(
                [
                    _node_dof(n_x, i + di, j + dj, k)
                    for dj in (0, 1)
                    for di in (0, 1)
                    for k in range(4)
                ]
            )
            rows.extend(np.repeat(dofs, 16))
            cols.extend(np.tile(dofs, 16))
            vals.extend(element["bending"].ravel())
            load[dofs] += element["load"]
            if element["mass"] is not None:
                mass_rows.extend(np.repeat(dofs, 16))
                mass_cols.extend(np.tile(dofs, 16))
                mass_vals.extend(element["mass"].ravel())
    stiffness = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    mass = (
        sp.coo_matrix((mass_vals, (mass_rows, mass_cols)), shape=(ndof, ndof)).tocsc()
        if element["mass"] is not None
        else None
    )
    return BFSAssembledPlate(
        n_x=n_x,
        n_y=n_y,
        length_x=length_x,
        length_y=length_y,
        stiffness=stiffness,
        mass=mass,
        load=load,
    )


def bfs_boundary_dofs(
    n_x: int, n_y: int, boundary: str
) -> np.ndarray:
    """Constrained dofs of a BFS mesh for ``"ssss"`` or ``"cccc"`` edges.

    ``"ssss"`` (hard simple support): ``w = 0`` and the *tangential* slope zero
    on every edge -- the exact Kirchhoff condition, which also removes the
    corner twist dof and is the convention validated in
    ``benchmarks/verified/fem_plate``.  ``"cccc"``: ``w = 0`` and both slopes
    zero on every edge (the corner twist dof is left free, the classical BFS
    clamping used by Zienkiewicz; the convergence ladder confirms it).
    """
    if boundary not in ("ssss", "cccc"):
        raise ValueError("boundary must be 'ssss' or 'cccc'")
    nodes_x, nodes_y = n_x + 1, n_y + 1
    fixed: set[int] = set()
    for i in range(nodes_x):
        for j in (0, n_y):
            fixed.add(_node_dof(n_x, i, j, 0))
            fixed.add(_node_dof(n_x, i, j, 1))  # tangential slope on a y edge
            if boundary == "cccc":
                fixed.add(_node_dof(n_x, i, j, 2))
    for j in range(nodes_y):
        for i in (0, n_x):
            fixed.add(_node_dof(n_x, i, j, 0))
            fixed.add(_node_dof(n_x, i, j, 2))  # tangential slope on an x edge
            if boundary == "cccc":
                fixed.add(_node_dof(n_x, i, j, 1))
    return np.array(sorted(fixed), dtype=int)


def _locate(n_x: int, n_y: int, lx: float, ly: float, x: float, y: float):
    i = min(int(math.floor(x / lx)), n_x - 1)
    j = min(int(math.floor(y / ly)), n_y - 1)
    i = max(i, 0)
    j = max(j, 0)
    s = (x - i * lx) / lx
    t = (y - j * ly) / ly
    return i, j, s, t


def bfs_evaluate(
    plate: BFSAssembledPlate, displacement: np.ndarray, points: np.ndarray
) -> dict[str, np.ndarray]:
    """Recover ``w`` and its derivatives at ``points`` (shape ``(n, 2)``)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    lx = plate.length_x / plate.n_x
    ly = plate.length_y / plate.n_y
    ix = (0, 1, 0, 1)
    iy = (0, 0, 1, 1)
    out = {key: np.zeros(len(points)) for key in ("w", "w_x", "w_y", "w_xx", "w_yy", "w_xy")}
    for p, (x, y) in enumerate(points):
        i, j, s, t = _locate(plate.n_x, plate.n_y, lx, ly, x, y)
        hx, dx_, ddx_ = _hermite(s, lx)
        hy, dy_, ddy_ = _hermite(t, ly)
        local = np.zeros(16)
        for dj in (0, 1):
            for di in (0, 1):
                node = dj * 2 + di
                for k in range(4):
                    local[4 * node + k] = displacement[_node_dof(plate.n_x, i + di, j + dj, k)]
        for node in range(4):
            di, dj = node % 2, node // 2
            for k in range(4):
                a = ix[k] + 2 * di
                b = iy[k] + 2 * dj
                u = local[4 * node + k]
                out["w"][p] += u * hx[a] * hy[b]
                out["w_x"][p] += u * dx_[a] * hy[b]
                out["w_y"][p] += u * hx[a] * dy_[b]
                out["w_xx"][p] += u * ddx_[a] * hy[b]
                out["w_yy"][p] += u * hx[a] * ddy_[b]
                out["w_xy"][p] += u * dx_[a] * dy_[b]
    return out


def bfs_moments(
    plate: BFSAssembledPlate,
    displacement: np.ndarray,
    points: np.ndarray,
    young: float,
    poisson: float,
    thickness: float,
) -> dict[str, np.ndarray]:
    """Bending moments ``M_x = -D (w,xx + nu w,yy)`` (and ``M_y``/``M_xy``)."""
    stiffness_d = plate_rigidity(young, poisson, thickness)
    fields = bfs_evaluate(plate, displacement, points)
    return {
        "M_x": -stiffness_d * (fields["w_xx"] + poisson * fields["w_yy"]),
        "M_y": -stiffness_d * (fields["w_yy"] + poisson * fields["w_xx"]),
        "M_xy": -stiffness_d * (1.0 - poisson) * fields["w_xy"],
    }


# ---------------------------------------------------------------------------
# Analytic oracles
# ---------------------------------------------------------------------------

def simply_supported_uniform_load_deflection(
    x: float, length: float, load: float, EI: float
) -> float:
    """Euler--Bernoulli SS beam, uniform load ``q`` (positive down).

    ``w(x) = q x (L^3 - 2 L x^2 + x^3) / (24 EI)``, ``w(0) = w(L) = 0``.
    """
    return load * x * (length**3 - 2.0 * length * x**2 + x**3) / (24.0 * EI)


def cantilever_tip_load_deflection(length: float, load: float, EI: float) -> float:
    """Cantilever tip deflection ``P L^3 / (3 EI)`` and tip rotation ``P L^2 / (2 EI)``."""
    return load * length**3 / (3.0 * EI)


def ss_beam_angular_frequencies(
    length: float, EI: float, rho_a: float, modes: int = 5
) -> np.ndarray:
    """``omega_n = (n pi / L)^2 sqrt(EI / (rho A))`` for a simply supported beam."""
    n = np.arange(1, modes + 1, dtype=float)
    return (n * math.pi / length) ** 2 * math.sqrt(EI / rho_a)


def ssss_navier_deflection(
    x: float,
    y: float,
    length_x: float,
    length_y: float,
    load: float,
    rigidity: float,
    terms: int = 11,
) -> float:
    """Navier double-sine solution of the SSSS plate, truncated at ``terms``.

    ``w = 16 q / (pi^6 D) sum_{m,n odd} sin(m pi x/a) sin(n pi y/b) /
    [m n ((m/a)^2 + (n/b)^2)^2]``.  ``terms`` counts the odd harmonics kept in
    *each* direction (11 -> m, n <= 11), matching the campaign mandate.
    """
    total = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            total += (
                math.sin(m * math.pi * x / length_x)
                * math.sin(n * math.pi * y / length_y)
                / (m * n * ((m / length_x) ** 2 + (n / length_y) ** 2) ** 2)
            )
    return 16.0 * load / (math.pi**6 * rigidity) * total


def ssss_navier_centre(
    length_x: float, length_y: float, load: float, rigidity: float, terms: int = 11
) -> float:
    """Centre deflection of the SSSS plate from the truncated Navier series."""
    return ssss_navier_deflection(
        0.5 * length_x, 0.5 * length_y, length_x, length_y, load, rigidity, terms
    )


def ssss_navier_moments(
    x: float,
    y: float,
    length_x: float,
    length_y: float,
    load: float,
    poisson: float,
    terms: int = 11,
) -> tuple[float, float]:
    """Centre bending moments ``(M_x, M_y)`` of the SSSS plate (Navier series).

    Differentiating the sine series term by term:
    ``w,xx = -16 q / (pi^4 D) sum m/n sin sin / ((m/a)^2+(n/b)^2)^2`` and the
    analogous ``w,yy`` expression, so moments stay finite term by term.
    """
    sxx = 0.0
    syy = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            denom = (m * n * ((m / length_x) ** 2 + (n / length_y) ** 2) ** 2)
            trig = math.sin(m * math.pi * x / length_x) * math.sin(n * math.pi * y / length_y)
            sxx += (m / length_x) ** 2 / denom * trig
            syy += (n / length_y) ** 2 / denom * trig
    factor = 16.0 * load / math.pi**4
    w_xx = -factor * sxx
    w_yy = -factor * syy
    return (-(w_xx + poisson * w_yy), -(w_yy + poisson * w_xx))


def ssss_plate_angular_frequencies(
    length_x: float, length_y: float, young: float, poisson: float,
    thickness: float, density: float, m_max: int = 6, n_max: int = 6,
) -> list[tuple[int, int, float]]:
    """Navier eigenpairs of the SSSS rectangular plate, sorted ascending.

    ``omega_mn = pi^2 sqrt(D / (rho t)) [(m/a)^2 + (n/b)^2]``; returns
    ``(m, n, omega)`` triples sorted by ``omega``.
    """
    rigidity = plate_rigidity(young, poisson, thickness)
    scale = math.pi**2 * math.sqrt(rigidity / (density * thickness))
    pairs = [
        (m, n, scale * ((m / length_x) ** 2 + (n / length_y) ** 2))
        for m in range(1, m_max + 1)
        for n in range(1, n_max + 1)
    ]
    return sorted(pairs, key=lambda item: item[2])

def _free_dofs(n_dof: int, fixed: np.ndarray) -> np.ndarray:
    mask = np.ones(n_dof, dtype=bool)
    mask[np.asarray(fixed, dtype=int)] = False
    return np.flatnonzero(mask)


def solve_bfs_static(
    plate: BFSAssembledPlate, fixed_dofs: np.ndarray
) -> np.ndarray:
    """Solve ``K u = f`` with the given dofs constrained to zero (sparse LU)."""
    free = _free_dofs(plate.n_dof, fixed_dofs)
    u = np.zeros(plate.n_dof)
    if free.size:
        u[free] = spla.spsolve(
            plate.stiffness[free][:, free].tocsc(), plate.load[free]
        )
    return u


def solve_bfs_modes(
    plate: BFSAssembledPlate, fixed_dofs: np.ndarray, modes: int = 6
) -> np.ndarray:
    """Lowest ``modes`` angular frequencies of ``K phi = omega^2 M phi``.

    The generalised problem is solved by shift-invert Lanczos with ``sigma = 0``
    (``scipy.sparse.linalg.eigsh``), which is exact to round-off for the
    discrete problem; the returned values are sorted ascending.
    """
    if plate.mass is None:
        raise ValueError("assemble the plate with a density to get a mass matrix")
    free = _free_dofs(plate.n_dof, fixed_dofs)
    stiffness = plate.stiffness[free][:, free]
    mass = plate.mass[free][:, free]
    values = spla.eigsh(
        stiffness, k=modes, M=mass, sigma=0.0, which="LM", return_eigenvectors=False
    )
    return np.sort(np.sqrt(np.abs(values)))


def solve_bfs_mode_shapes(
    plate: BFSAssembledPlate, fixed_dofs: np.ndarray, modes: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``modes`` angular frequencies *and* full-length eigenvectors.

    Same shift-invert Lanczos solve as :func:`solve_bfs_modes`; the vectors are
    expanded back onto all dofs (constrained dofs zero) and normalised to unit
    maximum transverse displacement, which is what the mode-shape figures use.
    """
    if plate.mass is None:
        raise ValueError("assemble the plate with a density to get a mass matrix")
    free = _free_dofs(plate.n_dof, fixed_dofs)
    values, vectors = spla.eigsh(
        plate.stiffness[free][:, free],
        k=modes,
        M=plate.mass[free][:, free],
        sigma=0.0,
        which="LM",
    )
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    full = np.zeros((plate.n_dof, modes))
    full[free] = vectors
    for column in range(modes):
        scale = np.abs(full[0::4, column]).max()
        if scale > 0.0:
            full[:, column] /= scale
    return np.sqrt(np.abs(values)), full
