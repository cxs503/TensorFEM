"""Boussinesq/Love elastic half-space oracle and a truncated HEX8 quarter model.

Why this module exists
----------------------
Every 3-D solid check in this repository loads a *closed* body: torsion bars,
cylinders, plates, spheres, Hertzian pairs.  A half-space is a different kind of
problem, because the reference solution lives on an unbounded domain while a
displacement-based FE model must truncate it, and the engineering quantity of
interest -- the settlement of a loaded patch, i.e. the compliance of the ground a
structure stands on -- is carried by the *far field*.  The Boussinesq surface
kernel decays like ``1/r``, so a large share of the deflection of a loaded patch
comes from material many patch widths away.  Truncating the mesh too close
therefore moves the settlement by ten percent or more -- in *either* direction,
because confinement raises the stress under the load while suppressing the far
field -- while *leaving no trace* in the residual, the energy balance or any
reaction check: the measured 4-patch-wide box is 16% high and the 20-patch-wide
box 5% low, both with a reaction balance of 1e-11.  A closed-body benchmark
cannot expose that class of error at all.  This module is the discriminating
case, plus the closed forms needed to grade it.

Exact solution used as the reference (Boussinesq 1885 / Love 1929)
------------------------------------------------------------------
A vertical point load ``P`` on the surface of an isotropic half-space
(``E``, ``nu``) produces the surface settlement

    w(r) = P (1 - nu^2) / (pi E r),

so a uniform pressure ``p`` over an area ``A`` gives
``w(x) = (p (1 - nu^2) / (pi E)) * integral_A dA / |x - xi|``.  For the rectangle
``[0, lx] x [0, ly]`` the corner value of that integral is closed::

    I(u, v) = u asinh(v/u) + v asinh(u/v) = int_0^u int_0^v dr ds / sqrt(r^2+s^2)

(the antiderivative identity ``d^2 I / du dv = 1 / sqrt(u^2 + v^2)`` is verified
numerically in the test suite).  ``I`` is homogeneous of degree one and even in
each argument up to a quadrant sign, so any field point is reached by the usual
2x2 corner inclusion-exclusion of the Newmark influence chart::

    w(x, y) = C [ I~(x,y) - I~(x-lx,y) - I~(x,y-ly) + I~(x-lx,y-ly) ],  C = p(1-nu^2)/(pi E)

with ``I~(u,v) = sign(u) sign(v) I(|u|,|v|)``.

The vertical stress uses the same inclusion-exclusion on the Newmark corner
factor (Newmark 1935; Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed.,
art. 138; Poulos & Davis 1974)::

    sigma_z/p = (1/4pi) [ 2mn sqrt(m^2+n^2+1) (m^2+n^2+2) / ((m^2+n^2+1+m^2n^2)(m^2+n^2+1))
                         + atan2(2mn sqrt(m^2+n^2+1), m^2+n^2+1-m^2n^2) ],  m = lx/z, n = ly/z

written with ``atan2`` so the branch stays correct for ``m^2+n^2+1 < m^2n^2``
(deep and wide patch -- the ``arctan`` form of the textbook silently loses ``pi``
there).  Neither closed form is trusted on its own: the test suite also carries a
composite Gauss quadrature of the Boussinesq *point-load* kernels.

Finite-element side
-------------------
Quarter model ``[0, L] x [0, L] x [0, D]`` of the half-space, loaded by a uniform
pressure on ``[0, a] x [0, a]`` -- the quarter of a ``2a x 2a`` patch centred on
the origin.  Symmetry gives ``u_x = 0`` on ``x = 0`` and ``u_y = 0`` on ``y = 0``;
the truncation is closed with rollers on the outer vertical faces (``u_x = 0`` at
``x = L``, ``u_y = 0`` at ``y = L``) and ``u_z = 0`` on the bottom.  Those are the
weakest admissible truncation constraints -- clamping the bottom would stiffen the
model without modelling anything -- and they are what makes the domain-size study
meaningful.  The mesh is graded geometrically away from the patch so the far
field can be pushed to ``20a`` without paying for it in DOF.

Reported checks: the two patch settlements (centre and far corner), the vertical
stress along the load axis at element centroids against the exact value *at the
same point*, and the reaction sum against the applied load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .solid3d import elasticity_matrix_3d, hex8_stiffness
from .sparse_core import assemble_coo, solve_sparse_static

__all__ = [
    "influence",
    "settlement",
    "corner_vertical_stress",
    "vertical_stress",
    "point_load_settlement",
    "auto_growth",
    "graded_axis",
    "graded_box_mesh",
    "HalfSpaceResult",
    "solve_halfspace_patch",
    "settlement_study",
]


# --------------------------------------------------------------------------- #
# exact closed forms
# --------------------------------------------------------------------------- #
def influence(u: float, v: float) -> float:
    """Quadrant Boussinesq integral ``I(u,v) = int_0^u int_0^v dr ds / sqrt(r^2+s^2)``."""
    u, v = abs(float(u)), abs(float(v))
    if u == 0.0 or v == 0.0:
        return 0.0
    return u * math.asinh(v / u) + v * math.asinh(u / v)


def _signed_influence(u: float, v: float) -> float:
    """``I`` continued to signed offsets (even in each argument, up to the quadrant sign)."""
    return math.copysign(1.0, u) * math.copysign(1.0, v) * influence(u, v)


def point_load_settlement(load: float, young: float, poisson: float, radius: float) -> float:
    """Boussinesq surface settlement ``P (1 - nu^2) / (pi E r)`` at distance ``r``."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if young <= 0.0:
        raise ValueError("young must be positive")
    return load * (1.0 - poisson ** 2) / (math.pi * young * radius)


def settlement(pressure: float, young: float, poisson: float,
               lx: float, ly: float, x: float, y: float) -> float:
    """Downward surface settlement at ``(x,y)`` under a uniform pressure on ``[0,lx]x[0,ly]``."""
    if young <= 0.0:
        raise ValueError("young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson out of range")
    if pressure < 0.0 or lx <= 0.0 or ly <= 0.0:
        raise ValueError("pressure must be non-negative and the patch non-degenerate")
    scale = pressure * (1.0 - poisson ** 2) / (math.pi * young)
    g = _signed_influence
    return scale * (g(x, y) - g(x - lx, y) - g(x, y - ly) + g(x - lx, y - ly))


def corner_vertical_stress(pressure: float, lx: float, ly: float, z: float) -> float:
    """``sigma_z`` at depth ``z`` directly below a corner of the loaded rectangle."""
    if z <= 0.0:
        raise ValueError("depth must be positive")
    m, n = lx / z, ly / z
    root = math.sqrt(m * m + n * n + 1.0)
    quad = m * m + n * n + 1.0
    t1 = 2.0 * m * n * root / (quad + m * m * n * n) * (quad + 1.0) / quad
    t2 = math.atan2(2.0 * m * n * root, quad - m * m * n * n)
    return pressure * (t1 + t2) / (4.0 * math.pi)


def vertical_stress(pressure: float, lx: float, ly: float,
                    x: float, y: float, z: float) -> float:
    """``sigma_z(x,y,z)`` under a uniform pressure on ``[0,lx]x[0,ly]`` (Newmark chart)."""
    if pressure < 0.0 or lx <= 0.0 or ly <= 0.0:
        raise ValueError("pressure must be non-negative and the patch non-degenerate")
    if z <= 0.0:
        raise ValueError("depth must be positive")
    def term(dx: float, dy: float) -> float:
        """Antiderivative of the corner factor, continued to signed offsets."""
        return (math.copysign(1.0, dx) * math.copysign(1.0, dy)
                * corner_vertical_stress(pressure, abs(dx), abs(dy), z))

    # F(lx-x, ly-y) - F(-x, ly-y) - F(lx-x, -y) + F(-x, -y): the four corners of the
    # loaded rectangle seen from the field point.  Note the offsets run corner-minus-
    # field-point; using field-minus-corner instead flips two of the four signs and
    # collapses the value to ~0 for field points *inside* the patch.
    return term(lx - x, ly - y) - term(-x, ly - y) - term(lx - x, -y) + term(-x, -y)


# --------------------------------------------------------------------------- #
# graded structured hexahedral mesh
# --------------------------------------------------------------------------- #
def auto_growth(h0: float, n_grow: int, span: float) -> float:
    """Growth ratio whose ``n_grow`` geometric cells run from ``h0`` to cover ``span``.

    Keeping the *first* cell at ``h0`` (instead of rescaling a fixed ratio) is what
    stops the near-surface layer from degenerating into a flat sliver: a fixed ratio
    tuned for a twenty-patch-wide domain shrinks the loaded layer about sevenfold and
    the surface stresses inherit that aspect ratio.
    """
    if h0 <= 0.0 or n_grow < 1 or span <= 0.0:
        raise ValueError("h0 and span must be positive and n_grow >= 1")
    target = span / h0
    if target <= n_grow:
        raise ValueError("too few cells to reach the requested extent at this spacing")

    def total(r):
        return float(n_grow) if abs(r - 1.0) < 1e-12 else r * (r ** n_grow - 1.0) / (r - 1.0)

    lo, hi = 1.0, 2.0
    while total(hi) < target:
        hi *= 1.5
        if hi > 1e6:
            raise ValueError("growth-ratio search diverged")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def graded_axis(h0: float, n_fine: int, growth, n_grow: int, extent: float) -> torch.Tensor:
    """1-D nodes: ``n_fine`` uniform cells of size ``h0``, then ``n_grow`` growing ones.

    ``growth=None`` picks the ratio that lands exactly on ``extent`` with the first
    growing cell still ``h0``; a numeric ``growth`` is honoured and the tail rescaled
    so the last node is exactly ``extent``.  Either way the axis is strictly
    increasing, i.e. every HEX8 Jacobian stays positive.
    """
    if h0 <= 0.0 or extent <= 0.0:
        raise ValueError("h0 and extent must be positive")
    if n_fine < 0 or n_grow < 0 or (n_fine == 0 and n_grow == 0):
        raise ValueError("need at least one cell")
    fine = h0 * n_fine
    if n_grow and extent - fine <= 0.0:
        raise ValueError("extent is smaller than the uniform part of the axis")
    if growth is None:
        if n_grow == 0:
            raise ValueError("a purely uniform axis must reach the extent exactly")
        growth = auto_growth(h0, n_grow, extent - fine)
    if growth <= 1.0:
        raise ValueError("growth must exceed one for a graded mesh")
    sizes = [h0] * n_fine + [h0 * growth ** k for k in range(1, n_grow + 1)]
    tail = math.fsum(sizes[n_fine:])
    scale = (extent - fine) / tail if tail > 0.0 else 1.0
    coords = [0.0]
    for index, size in enumerate(sizes):
        coords.append(coords[-1] + (size if index < n_fine else size * scale))
    coords[-1] = extent
    axis = torch.tensor(coords, dtype=torch.float64)
    if torch.any(axis[1:] <= axis[:-1]):
        raise ValueError("graded axis is not strictly increasing")
    return axis


def graded_box_mesh(xs: torch.Tensor, ys: torch.Tensor, zs: torch.Tensor):
    """Structured HEX8 mesh of a graded box; node ``(i,j,k)`` sits at ``(xs[i],ys[j],zs[k])``."""
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    nodes = torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), 1)

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cells.append((node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k),
                              node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                              node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)))
    return nodes, torch.tensor(cells, dtype=torch.long), (nx, ny, nz)


# --------------------------------------------------------------------------- #
# FE driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HalfSpaceResult:
    """Outcome of one quarter-model run against the closed-form half-space."""

    young: float
    poisson: float
    pressure: float
    half_width: float
    lateral: float
    depth: float
    n_nodes: int
    n_elements: int
    n_dofs: int
    w_center: float
    w_center_exact: float
    w_corner: float
    w_corner_exact: float
    depths: tuple
    stress_fe: tuple
    stress_exact: tuple
    reaction_balance: float
    finest_spacing: float
    profile_x: tuple = ()
    profile_w: tuple = ()
    profile_exact: tuple = ()
    slice_xz: tuple = ()

    @property
    def center_error(self) -> float:
        return abs(self.w_center - self.w_center_exact) / self.w_center_exact

    @property
    def corner_error(self) -> float:
        return abs(self.w_corner - self.w_corner_exact) / self.w_corner_exact

    @property
    def stress_error(self) -> float:
        """Worst relative error of the axis stresses where the reference exceeds 5% of p."""
        floor = 0.05 * self.pressure
        errors = [abs(fe - ex) / abs(ex) for fe, ex in zip(self.stress_fe, self.stress_exact)
                  if abs(ex) >= floor]
        return max(errors) if errors else math.nan

    def summary(self) -> dict:
        return {
            "young": self.young,
            "poisson": self.poisson,
            "pressure": self.pressure,
            "half_width": self.half_width,
            "lateral_over_patch": self.lateral,
            "depth_over_patch": self.depth,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "n_dofs": self.n_dofs,
            "finest_spacing_over_patch": self.finest_spacing,
            "w_center_fe": self.w_center,
            "w_center_exact": self.w_center_exact,
            "w_center_rel_error": self.center_error,
            "w_corner_fe": self.w_corner,
            "w_corner_exact": self.w_corner_exact,
            "w_corner_rel_error": self.corner_error,
            "depth_over_a": list(self.depths),
            "sigma_z_fe": list(self.stress_fe),
            "sigma_z_exact": list(self.stress_exact),
            "sigma_z_max_rel_error": self.stress_error,
            "reaction_balance_rel": self.reaction_balance,
        }


def solve_halfspace_patch(*, young: float = 1.0, poisson: float = 0.3,
                          pressure: float = 1.0, half_width: float = 1.0,
                          lateral: float = 20.0, depth: float = 20.0,
                          n_fine: int = 4, n_grow: int = 14, growth=None,
                          nz_layers: int = 14, rtol: float = 1e-10) -> HalfSpaceResult:
    """Load a ``2a x 2a`` patch and grade the truncated half-space against Boussinesq.

    ``lateral``/``depth`` are multiples of ``half_width`` ``a``; ``n_fine`` cells of
    size ``a/n_fine`` cover the loaded quarter patch, then ``n_grow`` geometrically
    growing cells reach ``lateral*a``.
    """
    if half_width <= 0.0 or pressure < 0.0:
        raise ValueError("half_width must be positive and pressure non-negative")
    if lateral <= 1.0 or depth <= 0.0:
        raise ValueError("the truncated domain must extend beyond the patch")
    if n_fine < 1 or n_grow < 0 or nz_layers < 1:
        raise ValueError("n_fine and nz_layers must be >= 1, n_grow >= 0")
    a = float(half_width)
    h0 = a / n_fine
    xs = graded_axis(h0, n_fine, growth, n_grow, lateral * a)
    ys = xs.clone()
    zs = graded_axis(h0, 0, growth, nz_layers, depth * a)
    nodes, cells, (nx, ny, nz) = graded_box_mesh(xs, ys, zs)
    n_elements = len(cells)

    modulus = elasticity_matrix_3d(nodes.new_tensor(young).expand(n_elements),
                                   nodes.new_tensor(poisson).expand(n_elements))
    ke, b_matrix = hex8_stiffness(nodes[cells], modulus)
    element_dofs = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(n_elements, 24).long()
    n_dofs = 3 * len(nodes)
    K = assemble_coo(ke, element_dofs, n_dofs, csr=True)

    # consistent nodal loads of the uniform pressure on the loaded quarter patch
    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    forces = nodes.new_zeros(n_dofs)
    for i in range(n_fine):
        for j in range(n_fine):
            share = pressure * (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]) / 4.0
            for corner in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                forces[3 * node(corner[0], corner[1], 0) + 2] += share
    applied = float(pressure * a * a)

    # symmetry on x=0 / y=0, rollers on the outer faces, bottom fixed in z
    fixed = {}
    for j in range(ny + 1):
        for k in range(nz + 1):
            fixed[3 * node(0, j, k)] = 0.0                 # u_x = 0 on x = 0
            fixed[3 * node(nx, j, k)] = 0.0                # u_x = 0 on x = L
    for i in range(nx + 1):
        for k in range(nz + 1):
            fixed[3 * node(i, 0, k) + 1] = 0.0             # u_y = 0 on y = 0
            fixed[3 * node(i, ny, k) + 1] = 0.0            # u_y = 0 on y = L
    for i in range(nx + 1):
        for j in range(ny + 1):
            fixed[3 * node(i, j, nz) + 2] = 0.0            # u_z = 0 on z = D
    solution = solve_sparse_static(K, forces, dirichlet=fixed, rtol=rtol)
    displacement = solution.displacement

    centre = node(0, 0, 0)
    corner = node(n_fine, n_fine, 0)
    w_center = float(displacement[3 * centre + 2])
    w_corner = float(displacement[3 * corner + 2])
    w_center_exact = settlement(pressure, young, poisson, 2 * a, 2 * a, a, a)
    w_corner_exact = settlement(pressure, young, poisson, 2 * a, 2 * a, 2 * a, 2 * a)

    reaction = float(-solution.reaction[2::3][[node(i, j, nz) for i in range(nx + 1)
                                               for j in range(ny + 1)]].sum())

    balance = abs(reaction - applied) / applied if applied > 0.0 else 0.0

    strain = torch.einsum("eij,ej->ei", b_matrix, displacement[element_dofs])
    stress = torch.einsum("eij,ej->ei", modulus, strain)
    centroids = nodes[cells].mean(1)

    depths, fe_values, exact_values = [], [], []
    for k in range(nz):
        column = k                       # element (i=0, j=0, k) straddles the load axis
        centroid = centroids[column]
        depths.append(float(centroid[2]) / a)
        fe_values.append(float(stress[column, 2]))
        # the loaded quarter patch is [0,a]^2, so the full patch is [-a,a]^2 and the
        # closed form -- written for a patch anchored at the origin -- must be asked
        # at (x + a, y + a).  Omitting the shift compares the FE centre with the
        # closed-form *corner* and reports a fictitious 30-60% stress error.
        exact_values.append(-vertical_stress(pressure, 2 * a, 2 * a,
                                             float(centroid[0]) + a, float(centroid[1]) + a,
                                             float(centroid[2])))
    # surface settlement profile along y = 0 (the symmetry plane that carries the patch edge)
    profile_x, profile_w, profile_exact = [], [], []
    for i in range(nx + 1):
        index = node(i, 0, 0)
        x = float(nodes[index, 0])
        profile_x.append(x / a)
        profile_w.append(float(displacement[3 * index + 2]))
        profile_exact.append(settlement(pressure, young, poisson, 2 * a, 2 * a, x + a, a))
    # sigma_zz over the x-z slice at y = first cell centre, for contour plotting
    slice_xz = tuple((float(centroids[(i * ny) * nz + k][0]) / a,
                      float(centroids[(i * ny) * nz + k][2]) / a,
                      float(stress[(i * ny) * nz + k, 2]) / pressure)
                     for i in range(nx) for k in range(nz))
    return HalfSpaceResult(
        profile_x=tuple(profile_x), profile_w=tuple(profile_w),
        profile_exact=tuple(profile_exact), slice_xz=slice_xz,
        young=young, poisson=poisson, pressure=pressure, half_width=a,
        lateral=lateral, depth=depth, n_nodes=len(nodes), n_elements=n_elements,
        n_dofs=n_dofs, w_center=w_center, w_center_exact=w_center_exact,
        w_corner=w_corner, w_corner_exact=w_corner_exact, depths=tuple(depths),
        stress_fe=tuple(fe_values), stress_exact=tuple(exact_values),
        reaction_balance=balance, finest_spacing=h0 / a,
    )


def settlement_study(levels=None, **kwargs):
    """Run a ladder of quarter models (mesh and/or domain refinement) for grading."""
    if levels is None:
        levels = (
            dict(n_fine=2, n_grow=12, nz_layers=12, lateral=20.0),
            dict(n_fine=4, n_grow=14, nz_layers=14, lateral=20.0),
            dict(n_fine=8, n_grow=16, nz_layers=16, lateral=20.0),
        )
    return tuple(solve_halfspace_patch(**{**kwargs, **level}) for level in levels)