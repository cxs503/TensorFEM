"""Cracked plate bending on a Winkler foundation: FE model and energy release rate.

Model
-----
A Mindlin-Reissner plate (the 4-node ``q4_mindlin_stiffness`` element of
:mod:`tensorfem.plate`, MITC transverse shear by default) rests on a Winkler
foundation of modulus ``k`` (``p = k w``; for a floating ice sheet
``k = rho_w g``).  A straight, through-thickness crack of length ``2a`` lies on
the grid line ``y = y_c``.  The crack is *explicit*: the nodes on the crack line
inside the crack span are duplicated into a lower and an upper copy, so the two
banks are kinematically independent (a free surface) while the plate outside the
crack stays conforming.  This is the natural crack representation for a
level-set front tracker (the level-set zero contour *is* the crack), and it
gives a sharp, mesh-resolved crack instead of a smeared stiffness reduction.

Energy release rate
-------------------
:func:`energy_release_rate_vce` advances the tip by a whole number of element
columns and evaluates ``G = -(dPi/da)/(2 t)`` from the strain + foundation
energy at fixed load (``Pi = -u^T f / 2``); i.e. the compliance (virtual crack
extension) method.  It is first-order accurate in the advance size ``da``, and
:func:`energy_release_rate_vce` returns both the raw value at ``da`` and the
Richardson-extrapolated value from ``da``/``2da``.

Analytic anchors
----------------
* **Uncracked plate on Winkler foundation, circular patch load** ``p`` of radius
  ``a_l``.  The exact Kelvin-function solution of the point-load problem gives
  ``M_r + M_theta = (P/2 pi) ker(r/l)`` and, using ``d/ds (s kei'(s)) = s ker(s)``,
  the centre moment of the patch load collapses to a single quadrature-free
  expression

      M_max = p l^2 X kei'(X),            X = a_l/l,   l = (D/k)^(1/4).

  (:func:`patch_max_moment`).  The folklore Westergaard/Nevel form
  ``M_max = (P/4 pi)(ln(l/a) + 0.6159)`` is available as
  :func:`patch_max_moment_nevel` for comparison; the two are compared in
  ``docs/levelset-ice-plate-fracture.md``.
* **Griffith crack in a plate under remote bending moment** ``M``:
  ``G = pi M^2 a/(D h)`` (see
  :func:`tensorfem.levelset.crack_criteria.bending_energy_release_rate`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from ..hetenyi_reference import kelvin_kei

__all__ = [
    "PlateCrackResult",
    "solve_cracked_plate",
    "energy_release_rate_vce",
    "patch_max_moment",
    "patch_moment_sum",
    "patch_max_moment_nevel",
    "kelvin_kei_derivative",
    "plate_rigidity",
    "winkler_length",
    "graded_axis",
    "rect_mindlin_stiffness",
]

_GP = 1.0 / math.sqrt(3.0)


def graded_axis(half_length: float, cells: int, h_min: float) -> np.ndarray:
    """Symmetric tensor-grid axis clustered towards the centre.

    ``x(t) = half*((1-g) t + g t^3)`` with ``g`` chosen so the centre spacing is
    ``h_min``; the spacing grows smoothly to the domain edge, which lets a small
    crack (``a ~ l/6``) be resolved without paying for a uniformly fine mesh.
    """
    if cells <= 0 or cells % 2:
        raise ValueError("cells must be even and positive")
    if h_min <= 0.0 or h_min * cells >= 2.0 * half_length:
        raise ValueError("h_min incompatible with the domain/cell count")
    g = 1.0 - h_min * cells / (2.0 * half_length)
    t = np.linspace(-1.0, 1.0, cells + 1)
    return half_length * ((1.0 - g) * t + g * t ** 3)


def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Flexural rigidity ``D = E h^3 / (12 (1 - nu^2))``."""
    return young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))


def winkler_length(rigidity: float, foundation: float) -> float:
    """Characteristic length ``l = (D/k)^(1/4)``."""
    if foundation <= 0.0:
        raise ValueError("foundation modulus must be positive")
    return (rigidity / foundation) ** 0.25


def kelvin_kei_derivative(x: float) -> float:
    """``kei'(x)`` by a 4th-order central difference of the exact ``kei`` series."""
    xs = torch.tensor([x - 2e-3, x - 1e-3, x + 1e-3, x + 2e-3], dtype=torch.float64)
    f = kelvin_kei(xs).numpy()
    step = 1e-3
    return float((f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * step))


def patch_moment_sum(pressure: float, radius: float, rigidity: float,
                     foundation: float) -> float:
    """Exact centre value of ``M_r + M_theta`` for a circular patch load.

    ``(M_r + M_theta)(0) = (P/2 pi)(ln(l/a) + 0.6159) = p l^2 X kei'(X)`` exactly,
    with ``X = a/l`` and ``l = (D/k)^(1/4)``; the logarithmic form is the
    small-``X`` limit, where ``ln 2 - gamma + 1/2 = 0.6159``.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    l = winkler_length(rigidity, foundation)
    x = radius / l
    return pressure * l ** 2 * x * kelvin_kei_derivative(x)


def patch_max_moment(pressure: float, radius: float, rigidity: float,
                     foundation: float, poisson: float = 0.3) -> float:
    """Exact centre bending moment ``M_r = M_theta`` of a circular patch load.

    ``M_c = (1+nu)/2 (M_r + M_theta)(0)`` because ``M_r + M_theta = -(1+nu) D lap w``
    for an axisymmetric Kirchhoff plate; the factor ``(1+nu)/2`` is confirmed by
    the finite-element moment field to about 1 % in
    ``docs/levelset-ice-plate-fracture.md``.
    """
    return 0.5 * (1.0 + poisson) * patch_moment_sum(
        pressure, radius, rigidity, foundation
    )


def patch_max_moment_nevel(total_load: float, radius: float, rigidity: float,
                           foundation: float, poisson: float = 0.3) -> float:
    """Westergaard/Nevel closed form ``(P/2 pi)(ln(l/a) + 0.6159)`` (for comparison).

    The constant is ``ln 2 - gamma + 1/2 = 0.6159``; it follows from the small
    argument expansion of the Kelvin solution, which also fixes the prefactor at
    ``P/(2 pi)`` (a ``4 pi`` prefactor appears in part of the literature and is
    off by a factor two -- see the numerical check in
    ``docs/levelset-ice-plate-fracture.md``).
    """
    l = winkler_length(rigidity, foundation)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return (1.0 + poisson) * total_load / (4.0 * math.pi) * (
        math.log(l / radius) + 0.6159
    )


@dataclass
class PlateCrackResult:
    """Solution fields of one cracked-plate solve."""

    nodes: np.ndarray            # (N, 2)
    elements: np.ndarray         # (M, 4) counter-clockwise
    u: np.ndarray                # (3N,)
    energy: float                # total potential energy Pi = -u.f/2
    total_load: float            # applied vertical load P
    stiffness: object            # csr matrix (for diagnostics)
    force: np.ndarray            # (3N,)
    tip_right: np.ndarray        # right tip coordinates
    tip_left: np.ndarray         # left tip coordinates
    crack_half_length: float
    gdof: np.ndarray             # (nnode, 3) global dof table
    grid: np.ndarray             # (ny+1, nx+1) node ids of the grid points

    def moment_field(self, *, young: float, poisson: float, thickness: float
                     ) -> dict[str, np.ndarray]:
        """Bending/twisting moments at the element centres."""
        db = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
        m = np.zeros((self.elements.shape[0], 3))
        for e, conn in enumerate(self.elements):
            xy = self.nodes[conn]
            ue = np.stack([self.u[self.gdof[conn, k]] for k in range(3)], axis=1)
            width = float(np.ptp(xy[:, 0]))
            height = float(np.ptp(xy[:, 1]))
            kappa = np.zeros(3)
            for xi, eta in ((-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)):
                dN = _dshape(xi, eta, width, height)
                kx = dN[:, 0] @ ue[:, 1]
                ky = dN[:, 1] @ ue[:, 2]
                kxy = dN[:, 1] @ ue[:, 1] + dN[:, 0] @ ue[:, 2]
                kappa += np.array([kx, ky, kxy]) / 4.0
            m[e] = db * np.array([
                kappa[0] + poisson * kappa[1],
                kappa[1] + poisson * kappa[0],
                (1.0 - poisson) * 0.5 * kappa[2],
            ])
        return {"Mx": m[:, 0], "My": m[:, 1], "Mxy": m[:, 2],
                "centres": np.array([self.nodes[c].mean(axis=0) for c in self.elements])}

    def surface_stress(self, **kw) -> np.ndarray:
        """Maximum bending surface stress ``6 M/h^2`` magnitude per element."""
        mom = self.moment_field(**kw)
        thickness = kw["thickness"]
        return 6.0 * np.maximum(np.abs(mom["Mx"]), np.abs(mom["My"])) / thickness ** 2


def rect_mindlin_stiffness(width: float, height: float, young: float, poisson: float,
                           thickness: float, *, shear_correction: float = 5.0 / 6.0,
                           shear_treatment: str = "mitc") -> np.ndarray:
    """12x12 Mindlin Q4 stiffness of an axis-aligned rectangle (numpy).

    Identical formulation to :func:`tensorfem.plate.q4_mindlin_stiffness` (2x2
    bending rule; MITC assumed natural shear field, or the one-point SRI shear
    rule), re-implemented in numpy because ``torch.linalg.inv`` costs ~17 ms per
    call on this host, which dominates the assembly of a cracked plate model that
    has one distinct element size per grid row.  Equality with the torch element
    to machine precision is asserted in ``tests/test_levelset_fracture.py``.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("element sides must be positive")
    e, nu, t = float(young), float(poisson), float(thickness)
    db = e * t ** 3 / (12.0 * (1.0 - nu ** 2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
    )
    ds = shear_correction * e / (2.0 * (1.0 + nu)) * t * np.eye(2)

    # the Jacobian of an axis-aligned rectangle is diagonal, so its inverse and
    # determinant are analytic: no LAPACK call (BLAS/LAPACK calls cost ~50 ms
    # each on this host, which would make the assembly unusable).
    inv_j_t_diag = np.array([2.0 / width, 2.0 / height])
    det_j = 0.25 * width * height

    def grad(xi: float, eta: float):
        n, dn = _shape(xi, eta)
        return n, dn * inv_j_t_diag, det_j

    def shear_rows(xi: float, eta: float) -> np.ndarray:
        n, dn, _ = grad(xi, eta)
        rows = np.zeros((2, 12))
        for i in range(4):
            rows[0, 3 * i] = dn[i, 0]
            rows[0, 3 * i + 1] = n[i]
            rows[1, 3 * i] = dn[i, 1]
            rows[1, 3 * i + 2] = n[i]
        return rows

    gp = 1.0 / math.sqrt(3.0)
    ke = np.zeros((12, 12))
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            _, dn, detj = grad(xif, etaf)
            bb = np.zeros((3, 12))
            for i in range(4):
                bb[0, 3 * i + 1] = dn[i, 0]
                bb[1, 3 * i + 2] = dn[i, 1]
                bb[2, 3 * i + 1] = dn[i, 1]
                bb[2, 3 * i + 2] = dn[i, 0]
            ke += bb.T @ db @ bb * detj
    if shear_treatment == "sri":
        n0, dn, detj = grad(0.0, 0.0)
        bs = np.zeros((2, 12))
        for i in range(4):
            bs[0, 3 * i] = dn[i, 0]
            bs[0, 3 * i + 1] = n0[i]
            bs[1, 3 * i] = dn[i, 1]
            bs[1, 3 * i + 2] = n0[i]
        return ke + bs.T @ ds @ bs * detj * 4.0
    if shear_treatment != "mitc":
        raise ValueError("shear_treatment must be 'sri' or 'mitc'")
    bottom = shear_rows(0.0, -1.0)
    top = shear_rows(0.0, 1.0)
    left = shear_rows(-1.0, 0.0)
    right = shear_rows(1.0, 0.0)
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            _, _, detj = grad(xif, etaf)
            bs = np.zeros((2, 12))
            bs[0] = 0.5 * (1.0 - etaf) * bottom[0] + 0.5 * (1.0 + etaf) * top[0]
            bs[1] = 0.5 * (1.0 - xif) * left[1] + 0.5 * (1.0 + xif) * right[1]
            ke += bs.T @ ds @ bs * detj
    return ke


def _shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    n = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dn = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]])
    return n, dn


def _dshape(xi: float, eta: float, width: float, height: float) -> np.ndarray:
    """Shape-function derivatives w.r.t. the physical coordinates (rectangular element)."""
    _, dn = _shape(xi, eta)
    return np.stack((dn[:, 0] * 2.0 / width, dn[:, 1] * 2.0 / height), axis=1)


def _build_mesh(xs: np.ndarray, ys: np.ndarray, jc: int, i_left: int, i_right: int):
    """Node/element arrays with the crack-line nodes duplicated over the span.

    Returns ``(nodes, elements, grid)`` where ``grid[j, i]`` is the node id of
    the grid point ``(xs[i], ys[j])`` on the *lower* side of the crack line (so
    boundary loads and supports can address the grid directly).
    """
    nx, ny = xs.size - 1, ys.size - 1
    base = lambda i, j: j * (nx + 1) + i  # noqa: E731
    extra: dict[tuple[int, int], int] = {}
    next_id = (nx + 1) * (ny + 1)

    def crack_node(i: int, side: int) -> int:
        nonlocal next_id
        if i < i_left or i > i_right:
            return base(i, jc)
        key = (i, side)
        if key not in extra:
            extra[key] = next_id
            next_id += 1
        return extra[key]

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = base(i, j)
            n1 = base(i + 1, j)
            n2 = base(i + 1, j + 1)
            n3 = base(i, j + 1)
            if j == jc - 1:  # element below the crack line: top nodes are the "low" copy
                n2 = crack_node(i + 1, 0)
                n3 = crack_node(i, 0)
            elif j == jc:    # element above the crack line: bottom nodes are the "up" copy
                n0 = crack_node(i, 1)
                n1 = crack_node(i + 1, 1)
            elements.append((n0, n1, n2, n3))
    nodes = np.zeros((next_id, 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[base(i, j)] = (xs[i], ys[j])
    for (i, _side), nid in extra.items():
        nodes[nid] = (xs[i], ys[jc])
    elements = np.asarray(elements, dtype=np.int64)
    # drop the base copies of the crack-line nodes that no element refers to;
    # leaving them in the system would add unconnected (zero-stiffness) dofs
    used = np.unique(elements)
    remap = -np.ones(nodes.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    grid_raw = np.array([[base(i, j) for i in range(nx + 1)] for j in range(ny + 1)],
                        dtype=np.int64)
    grid = remap[grid_raw]
    if i_right >= i_left:
        for i in range(int(i_left), int(i_right) + 1):
            grid[jc, i] = remap[extra[(i, 0)]]  # lower bank for boundary addressing
    return nodes[used], remap[elements], grid


def solve_cracked_plate(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    jc: int,
    i_left: int | None,
    i_right: int | None,
    thickness: float,
    young: float,
    poisson: float,
    foundation: float = 0.0,
    pressure: float = 0.0,
    patch_centre: tuple[float, float] = (0.0, 0.0),
    patch_radius: float = 0.0,
    edge_moment: float = 0.0,
    shear_treatment: str = "mitc",
    load_samples: int = 6,
    hinge_penalty: float = 0.0,
    stiffness_cache: dict | None = None,
) -> PlateCrackResult:
    """Solve the cracked plate problem (loads: pressure patch and/or edge moments).

    Rectangular elements are assumed (tensor-product grid), so the element
    matrices depend only on the element size and are cached.  ``edge_moment``
    applies biaxial edge moments ``M_xx = M_yy = +-M`` (equibiaxial bending);
    the patch pressure is integrated with ``load_samples^2`` sub-samples per
    element and the total then rescaled to ``pressure*pi*radius^2`` exactly, so
    the discrete load matches the analytic one to round-off.

    Both crack banks keep all three generalised displacements free, i.e. the
    free-surface condition of Reissner-Mindlin plate theory (the transverse
    shear force, the bending moment and the twisting moment all vanish on the
    crack faces).  Tying the transverse displacement across the crack (a pure
    "hinge") produces a singular system for this element, so that variant is not
    offered; the choice is documented in
    ``docs/levelset-ice-plate-fracture.md``.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    nx, ny = xs.size - 1, ys.size - 1
    if i_left is None or i_right is None or i_right < i_left:
        i_left = 1
        i_right = 0
    nodes, elements, grid = _build_mesh(xs, ys, jc, int(i_left), int(i_right))
    nnode = nodes.shape[0]

    # dof table: [w, theta_x, theta_y] per node; the "up" copy of the crack line
    # shares its w dof with the "low" copy when tie_transverse is on
    gdof = np.empty((nnode, 3), dtype=np.int64)
    gdof[:, 0] = np.arange(nnode)
    rot = nnode + 2 * np.arange(nnode)
    gdof[:, 1] = rot
    gdof[:, 2] = rot + 1
    ndof = 3 * nnode

    cache = {} if stiffness_cache is None else stiffness_cache
    widths = np.ptp(nodes[elements][:, :, 0], axis=1)
    heights = np.ptp(nodes[elements][:, :, 1], axis=1)
    areas = widths * heights
    keys = [(round(float(w), 12), round(float(hh), 12)) for w, hh in zip(widths, heights)]
    local = {}
    for (w, hh), key in zip(zip(widths, heights), keys):
        if key in local:
            continue
        ck = (key, foundation, thickness, young, poisson, shear_treatment)
        if ck in cache:
            local[key] = cache[ck]
            continue
        ke = rect_mindlin_stiffness(w, hh, young, poisson, thickness,
                                    shear_treatment=shear_treatment)
        if foundation > 0.0:
            kf = np.zeros((12, 12))
            for xi, eta in ((-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)):
                n, _ = _shape(xi, eta)
                kf[np.ix_([0, 3, 6, 9], [0, 3, 6, 9])] += (
                    foundation * np.outer(n, n) * (w * hh) / 4.0
                )
            ke = ke + kf
        local[key] = ke
        cache[ck] = ke

    rows, cols, vals = [], [], []
    for conn, key in zip(elements, keys):
        ke = local[key]
        dofs = gdof[conn].reshape(-1)
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(ke.reshape(-1))

    f = np.zeros(ndof)
    total_patch = 0.0
    if pressure != 0.0 and patch_radius > 0.0:
        s2 = load_samples ** 2
        gpts = -1.0 + 2.0 * (np.arange(load_samples) + 0.5) / load_samples
        xi, eta = np.meshgrid(gpts, gpts, indexing="ij")
        nmat = np.stack(
            [_shape(float(a), float(b))[0] for a, b in zip(xi.ravel(), eta.ravel())]
        )  # (s2, 4)
        xe = nodes[elements]  # (M, 4, 2)
        pos = np.einsum("sk,mkj->msj", nmat, xe)  # (M, s2, 2)
        inside = np.hypot(pos[:, :, 0] - patch_centre[0],
                          pos[:, :, 1] - patch_centre[1]) <= patch_radius
        wgt = areas[:, None] * pressure / s2 * inside  # (M, s2)
        total_patch = float(wgt.sum())
        for k in range(4):
            contrib = np.einsum("ms,s->m", wgt, nmat[:, k])
            np.add.at(f, gdof[elements[:, k], 0], contrib)
        if total_patch > 0.0:
            target = pressure * math.pi * patch_radius ** 2
            f *= target / total_patch
            total_patch = target
    if edge_moment != 0.0:
        # biaxial edge moments M_xx = M_yy = +-M; on all four edges the exact
        # solution is a uniform equibiaxial moment field with no lateral
        # free-edge layer, so this is a clean verification loading.
        trib_y = np.array([_tributary(ys, j) for j in range(ny + 1)])
        trib_x = np.array([_tributary(xs, i) for i in range(nx + 1)])
        left = grid[:, 0]
        right = grid[:, nx]
        bottom = grid[0, :]
        top = grid[ny, :]
        np.add.at(f, gdof[left, 1], -edge_moment * trib_y)
        np.add.at(f, gdof[right, 1], edge_moment * trib_y)
        np.add.at(f, gdof[bottom, 2], -edge_moment * trib_x)
        np.add.at(f, gdof[top, 2], edge_moment * trib_x)

    K = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(ndof, ndof)).tocsr()
    if hinge_penalty > 0.0 and i_right >= i_left:
        # optional "hinge" variant: a stiff spring ties the transverse
        # displacement of the two crack banks (w continuous, rotations free).
        scale = float(abs(K.diagonal()).mean()) * hinge_penalty
        pairs = _crack_bank_pairs(nodes, i_left, i_right, ys[jc])
        prow, pcol, pval = [], [], []
        for lo, up in pairs:
            a, b = int(gdof[lo, 0]), int(gdof[up, 0])
            prow += [a, b, a, b]
            pcol += [a, b, b, a]
            pval += [scale, scale, -scale, -scale]
        K = K + coo_matrix((pval, (prow, pcol)), shape=(ndof, ndof)).tocsr()
    # remove the rigid-body modes: the two constant-rotation modes are null
    # modes even with a Winkler foundation (which only penalises w); the
    # constant-w mode is null only for k = 0.  One node is enough for each.
    fixed = [int(gdof[0, 0])] if foundation <= 0.0 else []
    fixed += [int(gdof[0, 1]), int(gdof[0, 2])]
    free = np.setdiff1d(np.arange(ndof), np.asarray(fixed, dtype=np.int64))
    u = np.zeros(ndof)
    lu = splu(K.tocsr()[free, :].tocsc()[:, free])
    u[free] = lu.solve(f[free])
    energy = -0.5 * float(u @ f)
    if i_right >= i_left:
        tip_left = np.array([xs[int(i_left)], ys[jc]])
        tip_right = np.array([xs[int(i_right)], ys[jc]])
        half = 0.5 * (xs[int(i_right)] - xs[int(i_left)])
    else:
        tip_left = tip_right = np.array([np.nan, np.nan])
        half = 0.0
    gdof_out = np.stack((gdof[:, 0], gdof[:, 1], gdof[:, 2]), axis=1)
    return PlateCrackResult(
        nodes=nodes, elements=elements, u=u, energy=energy, total_load=total_patch,
        stiffness=K, force=f, tip_right=tip_right, tip_left=tip_left,
        crack_half_length=half, gdof=gdof_out, grid=grid,
    )


def _crack_bank_pairs(nodes: np.ndarray, i_left: int, i_right: int, yc: float):
    """Pairs ``(lower, upper)`` of the duplicated nodes on the crack line."""
    by_coord: dict[tuple[float, float], list[int]] = {}
    for n, (x, y) in enumerate(nodes):
        if abs(y - yc) < 1e-12:
            by_coord.setdefault((round(float(x), 12), 0.0), []).append(n)
    return [tuple(sorted(ids)) for ids in by_coord.values() if len(ids) == 2]


def _add_stiffness(rows, cols, vals, a: int, b: int, k: float) -> None:
    rows.extend([np.array([a, b, a, b])])
    cols.extend([np.array([a, b, b, a])])
    vals.extend([np.array([k, k, -k, -k])])


def _elem_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _tributary(coords: np.ndarray, j: int) -> float:
    n = coords.size
    if j == 0:
        return 0.5 * (coords[1] - coords[0])
    if j == n - 1:
        return 0.5 * (coords[-1] - coords[-2])
    return 0.5 * (coords[j + 1] - coords[j - 1])


def energy_release_rate_vce(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    jc: int,
    i_left: int,
    i_right: int,
    advances: tuple[int, ...] = (1, 2),
    **solve_kwargs,
) -> dict[str, float]:
    """Compliance-method (virtual crack extension) energy release rate.

    The crack is advanced symmetrically (both tips) by ``advances`` element
    columns; ``G(da) = (Pi(0) - Pi(da)) / (2 t da)``.  A first-order
    Richardson extrapolation ``2 G(da) - G(2 da)`` is returned as
    ``G_richardson``.
    """
    thickness = solve_kwargs["thickness"]
    base = solve_cracked_plate(xs, ys, jc=jc, i_left=i_left, i_right=i_right,
                               **solve_kwargs)
    out: dict[str, float] = {"energy_base": base.energy}
    g = {}
    for n in advances:
        adv = solve_cracked_plate(xs, ys, jc=jc, i_left=i_left - n, i_right=i_right + n,
                                  **solve_kwargs)
        da = 0.5 * ((xs[i_right + n] - xs[i_right]) + (xs[i_left] - xs[i_left - n]))
        g[n] = (base.energy - adv.energy) / (2.0 * thickness * da)
        out[f"G_da{n}"] = float(g[n])
        out[f"energy_da{n}"] = float(adv.energy)
    if 1 in g and 2 in g:
        out["G_richardson"] = float(2.0 * g[1] - g[2])
    out["G"] = float(g[min(g)])
    return out
