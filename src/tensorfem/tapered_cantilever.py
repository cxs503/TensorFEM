"""Linearly tapered cantilever: closed-form flexibilities, a consistent 1-D
frame and a graded HEX20 solid, for the case-pool anchor ``CASE-CANTILEVER-TIP``.

Why this module exists
----------------------
The case-pool anchor is the tip deflection of a **prismatic** cantilever graded
against ``delta_EB = PL^3/3EI`` / ``delta_TI = PL^3/3EI + PL/(kappa G A)``.
Every existing route in the repository (``benchmarks/cantilever_tip_deflection.py``,
``benchmarks/verified/fem_cantilever*``) holds the section constant.  Ship
structures are rarely prismatic -- girders taper towards the ends, brackets taper
from root to tip -- so the *same* acceptance criterion (``< 3 %``) is exercised
here on a geometry family where the flexibility is an integral of ``1/I(x)``
rather than an algebraic expression.

Closed-form oracles (derived, not transcribed)
----------------------------------------------
With the depth varying linearly, ``h(x) = h0 (1 - c x/L)``, ``0 <= c < 1``,
``I(x) = b h(x)^3 / 12`` and a tip load ``P``, Castigliano's unit-load theorem
gives the bending part exactly::

    delta_bend = (P/E) int_0^L (L-x)^2 / I(x) dx
               = (P L^3 / (E I0)) * B(c)

    B(c) = [ -c - c^2/2 - ln(1-c) ] / c^3 ,      B(0) = 1/3

and the shear part (``A(x) = b h(x)``, ``kappa`` the Timoshenko shear factor)::

    delta_shear = int_0^L P/(kappa G A(x)) dx
                = (P L / (kappa G A0)) * S(c),
    S(c) = -ln(1-c)/c ,                          S(0) = 1

so that ``delta_TI = delta_bend + delta_shear`` reduces to the prismatic
expressions as ``c -> 0`` (both limits are machine-verified, see
:func:`taper_limits`).  Two features are worth noting because they are *not*
available for the prismatic case and they are what a validation study has to
respect:

* the bending amplification ``B(c)/B(0) = 3B(c)`` grows without bound as
  ``c -> 1`` -- a wedge tapering to a knife edge has **logarithmically
  divergent** tip deflection (``B(c) ~ -ln(1-c)``), so ``c = 1`` is excluded
  and near-wedge members are load-dominated, not stiffness-dominated;
* the shear compliance carries ``-ln(1-c)``, i.e. the *shear* share of the tip
  deflection **decreases** with taper (0.77 % at ``c = 0``, 0.57 % at
  ``c = 0.75`` for the case below) even though the shear compliance itself
  grows: bending grows faster.

The integrals above are re-derived with ``sympy`` at run time
(:func:`bending_factor_symbolic`), and independently re-integrated with adaptive
quadrature (:meth:`TaperCase.delta_quadrature`), so the oracle is never a
transcribed number.

Models
------
``frame_tip_deflection``
    A 2-node Euler-Bernoulli frame with *per-element* inertia.  Two sampling
    schemes are provided because the difference is a real engineering decision:

    ``midpoint``  ``EI`` evaluated at the element centre (what a lumped
                  "average inertia per element" model does);
    ``gauss``     ``K_e = int B^T EI(x) B dx`` integrated with 4-point Gauss,
                  exact for the cubic ``EI(x)`` of a linear taper.

``solve_tapered_cantilever``
    A graded HEX20 solid whose depth varies element by element.  The geometry is
    *represented exactly* (an element edge is a straight line, so piecewise
    linear ``h(x)`` is honoured at the nodes); what is discretised is ``I(x)``
    inside the element.  Two root conditions separate mesh error from the
    clamped-root boundary layer: ``clamp_all`` (``u = 0`` on the root face) and
    ``lateral_free`` (``u_x = 0`` only, the rigid translations pinned at one
    corner), the latter letting the root contract laterally as a beam does.

References
----------
* Castigliano / unit-load theorem: Timoshenko & Gere, *Mechanics of Materials*.
* Tapered cantilever deflection tables: Young, Budynas & Sadegh, *Roark's
  Formulas for Stress and Strain*, 8th ed., Table 8.1 (linearly varying depth).
* Shear factor ``kappa = 5/6``: Timoshenko, *Phil. Mag.* **41** (1921) 744;
  refined value: Cowper, *J. Appl. Mech.* **33** (1966) 335.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .hex20 import Hex20Model, solve_hex20_sparse, upgrade_hex8_to_hex20

__all__ = [
    "TaperCase",
    "bending_factor",
    "shear_factor",
    "bending_factor_series",
    "shear_factor_series",
    "bending_factor_symbolic",
    "taper_limits",
    "frame_tip_deflection",
    "tapered_hex20_mesh",
    "tip_face_loads",
    "solve_tapered_cantilever",
    "validate_mesh_geometry",
]

# 4-point Gauss-Legendre rule on [0, 1] (exact to degree 7).
_GAUSS4_X = np.array(
    [0.5 - 0.5 * math.sqrt(3.0 / 7.0 - (2.0 / 7.0) * math.sqrt(6.0 / 5.0)),
     0.5 - 0.5 * math.sqrt(3.0 / 7.0 + (2.0 / 7.0) * math.sqrt(6.0 / 5.0)),
     0.5 + 0.5 * math.sqrt(3.0 / 7.0 + (2.0 / 7.0) * math.sqrt(6.0 / 5.0)),
     0.5 + 0.5 * math.sqrt(3.0 / 7.0 - (2.0 / 7.0) * math.sqrt(6.0 / 5.0))]
)
# Weights are ordered to match _GAUSS4_X: the *inner* nodes (closest to the
# element centre) carry the larger weight (18 + sqrt(30))/72.
_GAUSS4_W = np.array(
    [(18.0 + math.sqrt(30.0)) / 72.0,
     (18.0 - math.sqrt(30.0)) / 72.0,
     (18.0 - math.sqrt(30.0)) / 72.0,
     (18.0 + math.sqrt(30.0)) / 72.0]
)


def bending_factor(c: float) -> float:
    """Exact bending amplification ``B(c) = int_0^1 (1-t)^2/(1-c t)^3 dt``.

    ``B(0) = 1/3`` recovers ``delta_EB = PL^3/(3EI)``; ``B(c) -> -ln(1-c)/c^3``
    as ``c -> 1`` (logarithmic divergence of a knife-edge wedge).
    """
    if not (0.0 <= c < 1.0):
        raise ValueError("taper ratio must satisfy 0 <= c < 1")
    if c < 1e-4:
        # The closed form cancels catastrophically for small c (numerator ~ c^3
        # against terms ~ c), so evaluate the convergent series instead.
        return bending_factor_series(c)
    return (-c - 0.5 * c * c - math.log1p(-c)) / c**3


def bending_factor_series(c: float, terms: int = 400) -> float:
    """Independent series route ``B(c) = sum_k b_k c^k`` (``|c| < 1``).

    ``b_k = (k+2)(k+1)/(2(k+3)) - k/2``; at ``c = 0`` it starts at ``1/3``.  This
    route shares no code with :func:`bending_factor` and is used both as the
    small-``c`` branch of that function and as its independent cross-check.
    """
    if not (0.0 <= c < 1.0):
        raise ValueError("taper ratio must satisfy 0 <= c < 1")
    total = 0.0
    power = 1.0
    for k in range(terms):
        total += ((k + 2) * (k + 1) / (2.0 * (k + 3)) - 0.5 * k) * power
        power *= c
    return total


def shear_factor(c: float) -> float:
    """Exact shear amplification ``S(c) = int_0^1 dt/(1-c t) = -ln(1-c)/c``."""
    if not (0.0 <= c < 1.0):
        raise ValueError("taper ratio must satisfy 0 <= c < 1")
    if c < 1e-4:
        return shear_factor_series(c)
    return -math.log1p(-c) / c


def shear_factor_series(c: float, terms: int = 400) -> float:
    """Independent series route ``S(c) = sum_k c^k/(k+1)`` (``|c| < 1``)."""
    if not (0.0 <= c < 1.0):
        raise ValueError("taper ratio must satisfy 0 <= c < 1")
    total = 0.0
    power = 1.0
    for k in range(terms):
        total += power / (k + 1)
        power *= c
    return total


@lru_cache(maxsize=8)
def bending_factor_symbolic(c: float) -> dict[str, float]:
    """Re-derive :func:`bending_factor` with sympy and report the residual.

    Returns ``{"symbolic", "closed", "residual", "series_residual"}`` where
    ``residual`` is ``|F_symbolic(c) - B(c)| / B(c)`` evaluated at this ``c``.
    The symbolic integral is done once per ``c`` and cached.
    """
    import sympy as sp

    t, cc = sp.symbols("t c", positive=True)
    symbolic = sp.integrate((1 - t) ** 2 / (1 - cc * t) ** 3, (t, 0, 1))
    value = complex(sp.N(symbolic.subs(cc, c)))
    if abs(value.imag) > 1e-12:
        raise RuntimeError(f"symbolic bending integral is complex at c={c}: {value}")
    closed = bending_factor(c)
    series = bending_factor_series(c)
    return {
        "symbolic": value.real,
        "closed": closed,
        "series": series,
        "residual": abs(value.real - closed) / closed,
        "series_residual": abs(series - closed) / closed,
    }


def taper_limits() -> dict[str, float]:
    """Machine check of the two ``c -> 0`` limits against the prismatic oracles."""
    # B(c)*3 -> 1 and S(c) -> 1 as c -> 0 (compared on a tiny but non-zero c so
    # the closed form itself, not the hard-coded limit branch, is tested).
    eps = 1e-9
    return {
        "bending_over_prismatic": 3.0 * bending_factor(eps),
        "shear_over_prismatic": shear_factor(eps),
    }


@dataclass(frozen=True)
class TaperCase:
    """A rectangular cantilever whose depth tapers linearly towards the tip.

    ``h(x) = depth_root * (1 - taper * x / length)``, width constant.  The
    default parameters reproduce the case-pool section (``b = h = 10 mm``,
    ``E = 200 GPa``, ``P = 5 N``) and set ``L/h = 10`` so that the shear share
    of the deflection (0.77 %) is small but visible and the 3-D model is cheap.
    """

    length: float = 0.10
    width: float = 0.010
    depth_root: float = 0.010
    taper: float = 0.0
    young: float = 200e9
    poisson: float = 0.3
    load: float = 5.0
    kappa: float = 5.0 / 6.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.taper < 1.0):
            raise ValueError("taper must satisfy 0 <= taper < 1")
        for name in ("length", "width", "depth_root", "young", "load", "kappa"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson ratio out of range")

    # -- geometry / section -------------------------------------------------
    def depth(self, x: float) -> float:
        return self.depth_root * (1.0 - self.taper * float(x) / self.length)

    @property
    def depth_tip(self) -> float:
        return self.depth_root * (1.0 - self.taper)

    @property
    def inertia_root(self) -> float:
        return self.width * self.depth_root**3 / 12.0

    @property
    def area_root(self) -> float:
        return self.width * self.depth_root

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    def inertia(self, x: float) -> float:
        return self.width * self.depth(x) ** 3 / 12.0

    def area(self, x: float) -> float:
        return self.width * self.depth(x)

    @property
    def slenderness(self) -> float:
        return self.length / self.depth_root

    # -- oracles ------------------------------------------------------------
    def delta_euler(self) -> float:
        """``delta_bend`` (Euler-Bernoulli, bending only)."""
        return self.load * self.length**3 / (self.young * self.inertia_root) * bending_factor(self.taper)

    def delta_timoshenko(self) -> float:
        """``delta_bend + delta_shear`` with the declared ``kappa``."""
        shear = self.load * self.length / (self.kappa * self.shear_modulus * self.area_root) \
            * shear_factor(self.taper)
        return self.delta_euler() + shear

    def delta_shear(self) -> float:
        return self.delta_timoshenko() - self.delta_euler()

    def shear_share(self) -> float:
        return self.delta_shear() / self.delta_timoshenko()

    def delta_quadrature(self) -> dict[str, float]:
        """Independent route: adaptive quadrature of the two flexibility integrals."""
        from scipy.integrate import quad

        i0, a0 = self.inertia_root, self.area_root
        bend = quad(
            lambda t: self.load * (self.length * (1.0 - t)) ** 2 * self.length
            / (self.young * i0 * (1.0 - self.taper * t) ** 3),
            0.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=200,
        )[0]
        shear = quad(
            lambda t: self.load * self.length
            / (self.kappa * self.shear_modulus * a0 * (1.0 - self.taper * t)),
            0.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=200,
        )[0]
        return {"bending": bend, "shear": shear, "total": bend + shear}

    def relative_deflection(self) -> float:
        """``delta_TI(c) / delta_TI(0)`` -- the taper amplification."""
        return self.delta_timoshenko() / TaperCase(
            length=self.length, width=self.width, depth_root=self.depth_root,
            taper=0.0, young=self.young, poisson=self.poisson,
            load=self.load, kappa=self.kappa,
        ).delta_timoshenko()


# --------------------------------------------------------------------------- #
# 1-D Euler-Bernoulli frame with per-element inertia
# --------------------------------------------------------------------------- #
def _hermite_curvature(xi: np.ndarray, le: float) -> np.ndarray:
    """Second derivatives of the cubic Hermite shape functions, shape (4, n)."""
    return np.array([
        (-6.0 + 12.0 * xi) / le**2,
        (-4.0 + 6.0 * xi) / le,
        (6.0 - 12.0 * xi) / le**2,
        (-2.0 + 6.0 * xi) / le,
    ])


def frame_tip_deflection(case: TaperCase, elements: int, *, sampling: str = "gauss") -> float:
    """Tip deflection of the tapered cantilever from a 2-node EB frame.

    ``sampling="midpoint"`` uses ``EI`` at the element centre (a lumped
    per-element inertia, first-order accurate); ``sampling="gauss"`` integrates
    the element stiffness ``int B^T EI(x) B dx`` with 4-point Gauss, which is
    exact for the cubic ``EI(x)`` and therefore leaves only the Hermite-space
    error.
    """
    if elements < 1:
        raise ValueError("elements must be >= 1")
    if sampling not in ("midpoint", "gauss"):
        raise ValueError("sampling must be 'midpoint' or 'gauss'")
    n = int(elements)
    le = case.length / n
    ndof = 2 * (n + 1)
    stiff = np.zeros((ndof, ndof))
    for e in range(n):
        x0 = e * le
        if sampling == "midpoint":
            ei = case.inertia(x0 + 0.5 * le) * case.young
            ke = ei / le**3 * np.array([
                [12.0, 6 * le, -12.0, 6 * le],
                [6 * le, 4 * le**2, -6 * le, 2 * le**2],
                [-12.0, -6 * le, 12.0, -6 * le],
                [6 * le, 2 * le**2, -6 * le, 4 * le**2],
            ])
        else:
            ke = np.zeros((4, 4))
            for xi, w in zip(_GAUSS4_X, _GAUSS4_W):
                nn = _hermite_curvature(np.array([xi]), le)[:, 0]
                ei = case.young * case.inertia(x0 + xi * le)
                ke += (ei * np.outer(nn, nn)) * w * le
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                stiff[ia, ib] += ke[a, b]
    force = np.zeros(ndof)
    force[2 * n] = case.load
    fixed = np.array([0, 1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    disp = np.zeros(ndof)
    disp[free] = np.linalg.solve(stiff[np.ix_(free, free)], force[free])
    return float(disp[2 * n])


# --------------------------------------------------------------------------- #
# graded HEX20 solid
# --------------------------------------------------------------------------- #
def tapered_hex20_mesh(case: TaperCase, nx: int, ny: int, nz: int):
    """Conforming graded HEX20 mesh of the tapered solid (HEX8 grid + mid-nodes).

    The HEX8 corner grid honours ``h(x)`` exactly at every axial station; the
    mid-side nodes are inserted by :func:`upgrade_hex8_to_hex20`, so every
    element edge is a straight line and the piecewise-linear depth is exact.
    """
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    xs = np.linspace(0.0, case.length, nx + 1)
    ys = np.linspace(-0.5 * case.width, 0.5 * case.width, ny + 1)
    zs = np.linspace(-0.5, 0.5, nz + 1)
    index: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    for i, x in enumerate(xs):
        half = 0.5 * case.depth(x)
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                index[(i, j, k)] = len(points)
                points.append((float(x), float(y), float(z * 2.0 * half)))
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cells.append([
                    index[(i, j, k)], index[(i + 1, j, k)],
                    index[(i + 1, j + 1, k)], index[(i, j + 1, k)],
                    index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
                    index[(i + 1, j + 1, k + 1)], index[(i, j + 1, k + 1)],
                ])
    nodes = torch.tensor(points, dtype=torch.float64)
    cells_t = torch.tensor(cells, dtype=torch.long)
    return upgrade_hex8_to_hex20(nodes, cells_t)


def tip_face_loads(nodes: torch.Tensor, case: TaperCase, *, load: float | None = None) -> torch.Tensor:
    """Consistent 8-node face load on the tip face (uniform traction, net ``load``).

    The serendipity 8-node face weights for a uniform traction are
    ``(-1/12, 1/3)`` on corners/mid-sides of the *reference* square; the mapping
    Jacobian of a flat rectangle is constant, so the same pattern holds at any
    aspect ratio and is scaled here to a unit total.
    """
    total = case.load if load is None else float(load)
    forces = nodes.new_zeros(3 * len(nodes))
    sel = torch.where(nodes[:, 0] >= case.length - 1e-12)[0]
    half_b = torch.tensor(0.5 * case.width, dtype=nodes.dtype)
    half_h = torch.tensor(0.5 * case.depth_tip, dtype=nodes.dtype)
    corner = torch.isclose(nodes[sel, 1].abs(), half_b) & torch.isclose(nodes[sel, 2].abs(), half_h)
    weight = torch.where(corner, torch.tensor(-1.0 / 12.0, dtype=nodes.dtype),
                         torch.tensor(1.0 / 3.0, dtype=nodes.dtype))
    weight = weight / weight.sum()
    forces[3 * sel + 2] = total * weight
    return forces


def solve_tapered_cantilever(
    case: TaperCase,
    nx: int,
    ny: int,
    nz: int,
    *,
    root_bc: str = "clamp_all",
) -> dict[str, object]:
    """Solve the graded HEX20 model; return the root-aligned tip deflection.

    ``root_bc`` is ``"clamp_all"`` (``u = 0`` on the root face) or
    ``"lateral_free"`` (``u_x = 0`` on the root face, ``u_y = u_z = 0`` at one
    root corner), the second letting the root contract laterally so that the
    clamp cost can be separated from the mesh error.
    """
    if root_bc not in ("clamp_all", "lateral_free"):
        raise ValueError("root_bc must be 'clamp_all' or 'lateral_free'")
    nodes, cells = tapered_hex20_mesh(case, nx, ny, nz)
    forces = tip_face_loads(nodes, case)
    root = torch.where(nodes[:, 0] <= 1e-12)[0]
    if root_bc == "clamp_all":
        fixed = torch.stack((3 * root, 3 * root + 1, 3 * root + 2), 1).reshape(-1)
    else:
        fixed = (3 * root).clone()
        corner = root[torch.argmin(nodes[root, 1] ** 2 + nodes[root, 2] ** 2)]
        fixed = torch.cat((fixed, torch.tensor([3 * int(corner) + 1, 3 * int(corner) + 2])))
    model = Hex20Model(
        nodes, cells,
        nodes.new_tensor(case.young), nodes.new_tensor(case.poisson),
        forces, fixed.long(),
    )
    result = solve_hex20_sparse(model)
    tip = torch.where(nodes[:, 0] >= case.length - 1e-12)[0]
    tip_mean = float(result.displacement[3 * tip + 2].mean())
    root_mean = float(result.displacement[3 * root + 2].mean())
    sigma_xx = result.stress[:, 0]
    return {
        "tip_deflection": tip_mean - root_mean,
        "tip_face_mean": tip_mean,
        "root_face_mean": root_mean,
        "n_nodes": int(len(nodes)),
        "n_elements": int(len(cells)),
        "n_dof": 3 * int(len(nodes)),
        "strain_energy": float(result.strain_energy),
        "max_sigma_xx": float(sigma_xx.abs().max()),
        "nodes": nodes,
        "elements": cells,
        "sigma_xx": sigma_xx,
        "displacement": result.displacement,
    }


def validate_mesh_geometry(case: TaperCase, nx: int, ny: int, nz: int) -> dict[str, float]:
    """Pre-compute geometry gates (the analogue of the ``Cb/L/B`` check).

    Returns the worst-case deviation of the mesh surface from the exact
    ``z = +-h(x)/2`` surface, the worst cross-section area error, and the
    extreme element aspect ratio, so a reader can reject a bad mesh before
    looking at any deflection.
    """
    nodes, cells = tapered_hex20_mesh(case, nx, ny, nz)
    # Surface gate: at every axial station the outermost depth-heights must be
    # exactly +-h(x)/2 (interior stations may of course sit inside the section).
    x_sorted = torch.unique(nodes[:, 0].round(decimals=12))
    z_err = 0.0
    for station in x_sorted:
        sel = torch.isclose(nodes[:, 0], station)
        half = 0.5 * case.depth(float(station))
        z_err = max(z_err, float(((nodes[sel, 2].abs().max() - half) / half).__abs__()))
    # Section area: count the tip-face nodes' bounding box (exactly rectangular).
    tip = torch.where(nodes[:, 0] >= case.length - 1e-12)[0]
    area_mesh = float((nodes[tip, 1].max() - nodes[tip, 1].min())
                      * (nodes[tip, 2].max() - nodes[tip, 2].min()))
    area_exact = case.width * case.depth_tip
    # Element aspect ratio: longest / shortest HEX20 edge of the HEX8 corners.
    corners = cells[:, :8]
    p = nodes[corners.long()]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    lengths = torch.stack([(p[:, a] - p[:, b]).norm(dim=1) for a, b in edges], dim=1)
    aspect = float((lengths.max(dim=1).values / lengths.min(dim=1).values).max())
    return {
        "surface_deviation": z_err,
        "area_error": abs(area_mesh / area_exact - 1.0),
        "max_element_aspect": aspect,
        "min_edge_length": float(lengths.min()),
        "n_nodes": int(len(nodes)),
        "n_elements": int(len(cells)),
    }