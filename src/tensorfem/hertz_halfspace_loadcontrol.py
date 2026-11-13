"""Load-controlled Hertz contact of a rigid sphere on an elastic half-space.

Why this module exists
----------------------
The library already ships ``hertz_axisymmetric`` (prescribed-indentation
*penalty* contact, dense ``torch.linalg.solve``).  Two properties of that model
prevent it from serving as a *load-controlled* benchmark face:

* the indentation is an input, whereas case FEM-HERTZ-CONTACT prescribes the
  normal load ``P`` and asks for the contact radius ``a`` and the indentation
  ``delta`` as outputs;
* the penalty regularisation lets the surfaces overlap by ``p / epsilon``, which
  is a *systematic bias on delta* (measured on the shipped model: 2.9% of delta
  at nr=nz=24 with its default penalty factor) that mesh refinement cannot
  remove -- refining only makes the penalty error relatively larger.

This module therefore solves the discrete Signorini (unilateral) problem by an
**augmented Lagrangian** and closes an outer **load-control loop** on the
indentation so that the total normal reaction equals the prescribed load.

Method lineage (see ``docs/hertz_contact_loadcontrol_report.md`` §2)
--------------------------------------------------------------------
* Hertz (1882) closed form -- the independent oracle.
* Signorini problem / variational inequality: Kikuchi & Oden (1988) ch. 3-4.
* Augmented Lagrangian for unilateral contact: Alart & Curnier (1991)
  *Comput. Mech.* 7:283; Simo & Laursen (1992) *Comput. Struct.* 42:97;
  Wriggers, *Computational Contact Mechanics* (2006) §6.2 (multiplier update
  ``lambda <- <lambda - rho g>_+``).  The node-wise KKT projection replaces the
  naive "toggle on penetration / off on tension" active-set loop, which
  chatters at the contact edge on coarse meshes (observed; see the failure-mode
  section of the report).
* Axisymmetric Q4 kinematics with the ``1/r`` hoop term: Zienkiewicz & Taylor.
* Load control by regula falsi on the monotone response ``P_fe(delta)``; the
  Hertz slope ``dP/ddelta = 2 E* sqrt(R* delta) > 0`` guarantees a unique root.

Conventions
-----------
The body occupies ``r in [0, W]``, ``z in [-H, 0]``; the axis ``r = 0`` is
symmetry, the top surface ``z = 0`` is traction free outside the contact zone,
the bottom ``z = -H`` and the outer radius ``r = W`` are truncation boundaries
(kinematically constrained; the truncation study quantifies the error).  The
rigid indenter is the *paraboloid* ``z_ind(r) = r^2/(2R) - delta``, i.e. exactly
the profile for which the Hertz closed form holds (Hertz 1882).  The nodal gap
is ``g_i = u_z(r_i) - z_ind(r_i)`` and the contact multiplier ``lambda_i >= 0``
is the *downward* nodal force applied by the indenter, so ``lambda_i / A_i`` is
the nodal contact pressure with ``A_i`` the consistent ring area.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "HertzLoadReference",
    "hertz_load_reference",
    "axisymmetric_mesh",
    "mesh_from_axes",
    "two_zone_axis",
    "graded_depth",
    "surface_ring_areas",
    "assemble_axisymmetric_stiffness",
    "solve_axisymmetric_contact_load_control",
    "ContactState",
]


# ---------------------------------------------------------------------------
# Analytical oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HertzLoadReference:
    """Hertz solution driven by the normal load (Hertz 1882; Johnson 1985 §4.2)."""

    load: float
    radius: float
    young: float
    poisson: float
    effective_radius: float
    effective_modulus: float
    contact_radius: float
    indentation: float
    peak_pressure: float
    mean_pressure: float

    def pressure(self, r):
        r = np.asarray(r, dtype=float)
        p = np.zeros_like(r)
        inside = r <= self.contact_radius
        p[inside] = self.peak_pressure * np.sqrt(
            np.clip(1.0 - (r[inside] / self.contact_radius) ** 2, 0.0, None)
        )
        return p


def hertz_load_reference(load, radius, young, poisson) -> HertzLoadReference:
    """Closed-form Hertz contact: rigid sphere on an elastic half-space.

    ``a = (3 P R/(4 E*))^(1/3)``, ``delta = a^2/R``, ``p0 = 3P/(2 pi a^2)`` with
    ``E* = E/(1-nu^2)`` (Johnson 1985 §4.2, rigid-indenter limit).
    """
    if load <= 0 or radius <= 0 or young <= 0:
        raise ValueError("load, radius and Young's modulus must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("Poisson ratio must lie in (-1, 0.5)")
    e_star = young / (1.0 - poisson**2)
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    delta = a * a / radius
    p0 = 3.0 * load / (2.0 * math.pi * a * a)
    return HertzLoadReference(
        load=float(load),
        radius=float(radius),
        young=float(young),
        poisson=float(poisson),
        effective_radius=float(radius),
        effective_modulus=float(e_star),
        contact_radius=float(a),
        indentation=float(delta),
        peak_pressure=float(p0),
        mean_pressure=float(load / (math.pi * a * a)),
    )


# ---------------------------------------------------------------------------
# Mesh and geometry helpers
# ---------------------------------------------------------------------------


def axisymmetric_mesh(nr, nz, width, depth, *, r_grading=1.0, z_grading=1.0):
    """Structured rectangular ``(r, z)`` mesh with the top surface at ``z = 0``.

    ``r = width * u^r_grading``, ``z = -depth * v^z_grading`` for uniform
    ``u, v``, so ``grading > 1`` clusters nodes on the axis and on the loaded
    surface.  Node ``(i, j)`` has global index ``i*(nz+1) + j``, so the surface
    nodes are ``j = 0`` (ordered by increasing radius).
    """
    if nr < 2 or nz < 2:
        raise ValueError("need at least 2 elements per direction")
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    if r_grading < 1.0 or z_grading < 1.0:
        raise ValueError("grading must be at least one")

    u = np.linspace(0.0, 1.0, nr + 1)
    v = np.linspace(0.0, 1.0, nz + 1)
    r = width * u**r_grading
    z = -depth * v**z_grading
    return mesh_from_axes(r, z)


def mesh_from_axes(r, z):
    """Tensor-product ``(r, z)`` mesh from explicit 1-D node arrays.

    Used when the mesh must be refined in a prescribed band (e.g. uniform in
    the contact zone, geometric outside) instead of a global power grading.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if r.ndim != 1 or z.ndim != 1 or r.size < 2 or z.size < 2:
        raise ValueError("need at least two nodes per direction")
    if not np.all(np.diff(r) > 0) or not np.all(np.diff(z) < 0):
        raise ValueError("r must increase and z must decrease from the surface")
    nr, nz = r.size - 1, z.size - 1
    nodes = np.empty((nr + 1, nz + 1, 2))
    nodes[:, :, 0] = r[:, None]
    nodes[:, :, 1] = z[None, :]
    nodes = nodes.reshape(-1, 2)

    i, j = np.meshgrid(np.arange(nr), np.arange(nz), indexing="ij")
    i = i.ravel()
    j = j.ravel()

    def nid(ii, jj):
        return ii * (nz + 1) + jj

    # Vertex order (i,j+1), (i+1,j+1), (i+1,j), (i,j): the local eta axis then
    # runs towards +z, so the isoparametric Jacobian is positive.
    quads = np.stack(
        [nid(i, j + 1), nid(i + 1, j + 1), nid(i + 1, j), nid(i, j)], axis=1
    )
    return nodes, quads


def two_zone_axis(r_inner, r_outer, n_inner, n_outer, *, growth=1.35):
    """Radial nodes: uniform spacing over the contact band, geometric outside.

    ``growth`` is the element-size ratio of the geometric outer zone, which
    lets a 40 a outer boundary be reached with a few dozen elements while the
    contact band keeps a prescribed ``h/a``.
    """
    if not (0 < r_inner < r_outer) or n_inner < 2 or n_outer < 2:
        raise ValueError("invalid two-zone axis parameters")
    inner = np.linspace(0.0, r_inner, n_inner + 1)
    ratio = growth ** (1.0 / (n_outer - 1)) if n_outer > 1 else 1.0
    # geometric growth of the element size, normalised to span r_inner..r_outer
    sizes = growth ** np.arange(n_outer)
    sizes /= sizes.sum()
    outer = r_inner + (r_outer - r_inner) * np.concatenate(([0.0], np.cumsum(sizes)))
    return np.concatenate([inner[:-1], outer])


def graded_depth(depth, band, n_fine, n_grow, *, growth=1.6):
    """Vertical nodes: fine band of thickness ``band`` under the surface, then
    geometric growth down to ``depth``."""
    if depth <= 0 or n_fine < 2 or n_grow < 2 or not (0 < band < depth):
        raise ValueError("invalid depth grading parameters")
    fine = np.linspace(0.0, band, n_fine + 1)
    sizes = growth ** np.arange(n_grow)
    sizes /= sizes.sum()
    rest = band + (depth - band) * np.concatenate(([0.0], np.cumsum(sizes)))
    return -np.concatenate([fine[:-1], rest])


def surface_ring_areas(radii):
    """Consistent lumped ring areas of the surface nodes of a radial mesh.

    ``A_i = pi (r_{i+1/2}^2 - r_{i-1/2}^2)`` with half-cell interfaces and
    ``r_{-1/2} = 0`` on the axis, so ``sum_i p_i A_i`` is exactly the resultant
    of the piecewise-constant traction field ``p_i``.
    """
    radii = np.asarray(radii, dtype=float)
    edges = np.empty(len(radii) + 1)
    edges[0] = 0.0
    edges[-1] = radii[-1]
    edges[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_GP = np.array([-1.0, 1.0]) / math.sqrt(3.0)


def _shape_functions(xi, eta):
    n = 0.25 * np.array(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ]
    )
    dn = 0.25 * np.array(
        [
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)],
        ]
    )
    return n, dn


def _isotropic_D(young, poisson):
    c = young / ((1 + poisson) * (1 - 2 * poisson))
    return c * np.array(
        [
            [1 - poisson, poisson, poisson, 0.0],
            [poisson, 1 - poisson, poisson, 0.0],
            [poisson, poisson, 1 - poisson, 0.0],
            [0.0, 0.0, 0.0, 0.5 * (1 - 2 * poisson)],
        ]
    )


def assemble_axisymmetric_stiffness(nodes, quads, young, poisson):
    """Vectorised sparse assembly of the axisymmetric Q4 stiffness matrix.

    Checked element-by-element against the dense reference implementation
    ``tensorfem.hertz_axisymmetric.assemble_axisymmetric_stiffness``:
    max |difference| = 1.9e-9 against a peak magnitude of 1.46e7.
    """
    coords = nodes[quads]  # (Ne, 4, 2)
    d = _isotropic_D(young, poisson)
    rows, cols, vals = [], [], []
    dofs = np.stack((2 * quads, 2 * quads + 1), axis=2).reshape(len(quads), 8)

    for a in _GP:
        for b in _GP:
            n, dn = _shape_functions(a, b)
            jac = np.einsum("ib,eia->eab", dn, coords)
            # Analytic 2x2 inverse: np.linalg.inv on a stacked (Ne,2,2) array
            # costs ~9 s for 3k elements (measured), the closed form ~0 ms.
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            if np.any(det <= 0):
                raise ValueError("non-positive element Jacobian")
            jinv = np.empty_like(jac)
            jinv[:, 0, 0] = jac[:, 1, 1] / det
            jinv[:, 1, 1] = jac[:, 0, 0] / det
            jinv[:, 0, 1] = -jac[:, 0, 1] / det
            jinv[:, 1, 0] = -jac[:, 1, 0] / det
            dndx = np.einsum("ib,eba->eia", dn, jinv)
            radius = np.einsum("i,ei->e", n, coords[:, :, 0])
            ne = len(quads)
            bmat = np.zeros((ne, 4, 8))
            bmat[:, 0, 0::2] = dndx[:, :, 0]
            bmat[:, 1, 1::2] = dndx[:, :, 1]
            bmat[:, 2, 0::2] = n[None, :] / radius[:, None]
            bmat[:, 3, 0::2] = dndx[:, :, 1]
            bmat[:, 3, 1::2] = dndx[:, :, 0]
            weight = 2.0 * math.pi * radius * det
            ke = np.einsum("eai,ab,ebj->eij", bmat, d, bmat) * weight[:, None, None]
            rows.append(np.repeat(dofs, 8, axis=1).ravel())
            cols.append(np.tile(dofs, (1, 8)).ravel())
            vals.append(ke.ravel())

    ndof = 2 * len(nodes)
    k = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsr()
    k.sum_duplicates()
    return k


def consistent_surface_pressure(radii, nodal_forces):
    """Recover a piecewise-linear surface traction from consistent nodal forces.

    Solves the 1-D surface mass problem ``M p = f`` with
    ``M_ab = int N_a N_b 2 pi r dr`` on the surface segments, which is the
    variationally consistent inverse of the nodal-force integration.  The
    lumped estimate ``p_i = f_i / A_i`` is exact only for a piecewise-constant
    traction; it overestimates the pressure at the axis, where the ring area
    collapses (the source of the +31% lumped peak-pressure error measured on
    the Hertz case), and is noisier at the contact edge.
    """
    radii = np.asarray(radii, dtype=float)
    f = np.asarray(nodal_forces, dtype=float)
    n = radii.size
    m = sp.lil_matrix((n, n))
    # 2-point Gauss on each surface segment (exact for the quadratic integrand).
    g = 1.0 / math.sqrt(3.0)
    for i in range(n - 1):
        r1, r2 = radii[i], radii[i + 1]
        ke = np.zeros((2, 2))
        for xi in (-g, g):
            n1, n2 = 0.5 * (1 - xi), 0.5 * (1 + xi)
            r = 0.5 * (1 - xi) * r1 + 0.5 * (1 + xi) * r2
            jac = 0.5 * (r2 - r1)
            ke += np.outer([n1, n2], [n1, n2]) * 2.0 * math.pi * r * jac
        m[i, i] += ke[0, 0]
        m[i, i + 1] += ke[0, 1]
        m[i + 1, i] += ke[1, 0]
        m[i + 1, i + 1] += ke[1, 1]
    return spla.spsolve(m.tocsc(), f)


# ---------------------------------------------------------------------------
# Contact state container
# ---------------------------------------------------------------------------


@dataclass
class ContactState:
    """Converged state of one load-control sub-problem (fixed ``delta``)."""

    prescribed_indentation: float
    reaction: float
    contact_radius_outer: float
    contact_radius_linear: float
    contact_radius_fit: float
    peak_pressure: float
    peak_pressure_fit: float
    surface_radii: np.ndarray
    surface_pressure: np.ndarray
    surface_gap: np.ndarray
    multiplier: np.ndarray
    surface_pressure_consistent: np.ndarray
    active_nodes: np.ndarray
    alm_iterations: int
    maximum_penetration: float
    displacement: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Augmented-Lagrangian contact solve at fixed indentation
# ---------------------------------------------------------------------------


def _contact_state_fixed_indentation(
    kmat,
    *,
    radii,
    areas,
    obstacle,
    free_dofs,
    cz_all,
    rho,
    force_scale,
    alm_tol=1e-10,
    relaxation=0.5,
    max_iterations=80,
):
    """Augmented-Lagrangian solve of the nodal KKT system (fixed ``delta``).

    Unknowns are the nodal gap ``g_i = u_z(r_i) - obstacle_i`` and the
    multiplier ``lambda_i >= 0`` (downward nodal contact force) with
    ``g >= 0``, ``lambda >= 0``, ``g*lambda = 0`` and equilibrium
    ``K u + C^T lambda = 0`` on the free dofs.  The iteration is

        A = { i : lambda_i - rho * g_i > 0 }
        [K + rho C_A^T C_A] u = C_A^T (rho * obstacle_A - lambda_A)
        lambda <- <lambda - rho g>_+

    i.e. the Alart-Curnier / Simo-Laursen augmented Lagrangian.  ``rho`` scales
    the iteration only: at the fixed point the second line reduces to the exact
    complementarity conditions, so the converged state is rho-independent (the
    report carries a rho-sensitivity row that demonstrates this numerically).
    """
    ndof = kmat.shape[0]
    lam = np.zeros(len(radii))
    u = np.zeros(ndof)
    factor = None
    key = None
    # Separation is positive: the body surface must stay below the indenter,
    # gap = z_ind(r) - u_z >= 0 (contact when the gap closes).
    gap = obstacle - u[cz_all]
    iterations = 0
    max_penetration = 0.0
    best = None
    # The nodal multiplier update is under-relaxed (relaxation = 0.5): the edge
    # node of a node-to-node discretisation otherwise chatters between two
    # active sets (observed: k = 33 <-> 34 with a 0.14% delta penetration swing),
    # which costs a factorisation per iteration and never satisfies both KKT
    # tests at once.
    for iterations in range(1, max_iterations + 1):
        active = (lam - rho * gap) > 0.0
        if not np.any(active):
            raise RuntimeError("no contact detected for this indentation")
        key_new = active.tobytes()
        if key_new != key:
            idx = cz_all[active]
            aug = kmat + sp.coo_matrix(
                (np.full(idx.size, rho), (idx, idx)), shape=kmat.shape
            ).tocsr()
            factor = spla.splu(aug[free_dofs][:, free_dofs].tocsc())
            key = key_new
        rhs = np.zeros(ndof)
        # Stationarity of the Moreau-Yosida-regularised Lagrangian:
        #   (K u)_i + lambda_i - rho*g_i = 0  on the augmented set,
        # i.e. the augmented operator with rhs = rho*obstacle - lambda.  At the
        # fixed point (g = 0) this reduces to the exact equilibrium (K u)_i = -lambda_i.
        rhs[cz_all[active]] = rho * obstacle[active] - lam[active]
        u = np.zeros(ndof)
        u[free_dofs] = factor.solve(rhs[free_dofs])
        gap = obstacle - u[cz_all]
        lam_new = np.maximum(0.0, lam - rho * gap)
        lam_new = lam + relaxation * (lam_new - lam)
        # Both tests are relative (force change vs prescribed load, penetration
        # vs prescribed indentation) so the stopping rule is scale invariant --
        # an absolute tolerance makes the iteration stall on a dimensionless
        # problem whose loads are O(1e-4) and whose displacements are O(1e-3).
        delta_lam = np.max(np.abs(lam_new - lam)) / max(force_scale, 1e-300)
        lam = lam_new
        max_penetration = float(np.max(np.maximum(-gap, 0.0)))
        if best is None or max_penetration < best[0]:
            best = (max_penetration, lam.copy(), u.copy(), gap.copy())
        if delta_lam < alm_tol and max_penetration < alm_tol * abs(float(obstacle[0])):
            break

    if best is not None and best[0] < max_penetration:
        # accept the least-penetrating iterate (chatting edge node)
        max_penetration, lam, u, gap = best
    gap = obstacle - u[cz_all]
    # Contact membership uses a *relative* threshold: the under-relaxed
    # multiplier of a node that has left the contact decays geometrically and
    # stays positive at round-off level for many sweeps, and a single such node
    # outside the patch corrupts every edge-based radius estimator.
    # Contact membership uses a *physical* threshold on the contact pressure,
    # not on the multiplier: p/p0 = sqrt(1-(r/a)^2) for Hertz, so a node with
    # p/p0 = 1e-6 sits within 5e-13 a of the true edge and counting it changes
    # nothing.  The previous threshold (1e-10 * max(lambda)) was loose enough to
    # keep an edge node whose multiplier had decayed to ~2e-9 of the peak; that
    # spurious node extended the patch by one surface spacing and made the
    # primary radius estimator *mesh tied* (measured: a/R = 0.10, h/a = 0.0075,
    # a_linear = 0.9825 = the node radius exactly, error -1.75%, while the
    # coarser meshes gave -0.37% -> non-monotone "convergence").
    # See docs/AUDIT-m17149-hertz-anchor-evidence.md (finding F1).
    internal = kmat @ u
    _p_raw = lam / areas
    membership_ratio = 1e-6
    lam_threshold = membership_ratio * max(float(np.max(_p_raw)), 1e-300)
    pressure = np.where(_p_raw > lam_threshold, _p_raw, 0.0)
    pressure_consistent = consistent_surface_pressure(radii, lam)
    reaction = float(np.sum(lam))
    active = pressure > 0.0
    idx = np.nonzero(active)[0]
    rr = radii[idx]
    pp = pressure[idx]
    peak_index = int(np.argmax(pp))

    # --- three independent contact-radius estimators -----------------------
    # (1) outermost active node: hard upper bound of the discrete contact.
    a_outer = float(rr[-1])
    # (2) linear zero crossing of the computed pressure between the last two
    #     active nodes -- PRIMARY estimator, assumes no analytical shape.
    if idx.size >= 2 and pp[-1] < pp[-2]:
        a_linear = float(rr[-1] + (rr[-1] - rr[-2]) * pp[-1] / (pp[-2] - pp[-1]))
    else:
        a_linear = a_outer
    # (3) form-assisted: least squares p^2 = c0 + c1 r^2 over the computed
    #     pressure field (assumes the Hertz shape, hence "assisted"; reported
    #     separately and never used as the primary metric).
    a_fit, p_fit = float("nan"), float("nan")
    if idx.size >= 3:
        design = np.stack((np.ones_like(rr), rr * rr), axis=1)
        coeff, *_ = np.linalg.lstsq(design, pp * pp, rcond=None)
        if coeff[1] < 0 and coeff[0] > 0:
            a_fit = float(np.sqrt(-coeff[0] / coeff[1]))
            p_fit = float(np.sqrt(coeff[0]))
    # same fit on the consistent (mass-matrix) pressure field, whose edge and
    # axis behaviour is smooth
    a_fit_c = float("nan")
    idx_c = np.nonzero(pressure_consistent > 1e-9 * max(pressure_consistent.max(), 1e-30))[0]
    if idx_c.size >= 3:
        rr_c = radii[idx_c]
        pp_c = pressure_consistent[idx_c]
        design = np.stack((np.ones_like(rr_c), rr_c * rr_c), axis=1)
        coeff, *_ = np.linalg.lstsq(design, pp_c * pp_c, rcond=None)
        if coeff[1] < 0 and coeff[0] > 0:
            a_fit_c = float(np.sqrt(-coeff[0] / coeff[1]))

    # Equilibrium audit: the free-dof residual of (internal force + applied
    # contact force) must vanish.  ``lam`` is the nodal contact force the
    # indenter transmits to the body, assembled at the axial dofs with the same
    # sign as the internal force, so ``(K u)_free + lam == 0`` at the ALM fixed
    # point.  Using ``f_ext = -lam`` here (the pre-m166xx convention) reports a
    # residual of exactly ``2 |lam|`` -- i.e. the audit could never pass, and a
    # real equilibrium violation would have been invisible behind that offset.
    f_ext = np.zeros(ndof)
    f_ext[cz_all] = lam
    resid = (internal + f_ext)[free_dofs]
    return ContactState(
        prescribed_indentation=float(-obstacle[0]),
        reaction=reaction,
        contact_radius_outer=a_outer,
        contact_radius_linear=a_linear,
        contact_radius_fit=a_fit,
        peak_pressure=float(pp[peak_index]),
        peak_pressure_fit=p_fit,
        surface_radii=radii.copy(),
        surface_pressure=pressure,
        surface_gap=gap,
        multiplier=lam,
        surface_pressure_consistent=pressure_consistent,
        active_nodes=active,
        alm_iterations=iterations,
        maximum_penetration=max_penetration,
        displacement=u,
        metadata={
            "alm_converged": bool(iterations < max_iterations),
            "alm_relaxation": relaxation,
            "contact_radius_fit_consistent": a_fit_c,
            "peak_pressure_consistent": float(np.max(pressure_consistent)),
            "active_count": int(np.count_nonzero(active)),
            "equilibrium_residual_abs": float(np.linalg.norm(resid)),
            "equilibrium_residual_rel": float(
                np.linalg.norm(resid) / max(np.linalg.norm(f_ext), 1e-30)
            ),
            "last_active_node": int(idx[-1]),
            "membership_ratio": membership_ratio,
            "edge_pressure_ratio": float(pp[-1] / max(float(np.max(pp)), 1e-300)),
            "edge_spacing_over_a": float((radii[idx[-1] + 1] - radii[idx[-1]])
                                          / max(a_linear, 1e-300))
            if idx[-1] + 1 < radii.size else float("nan"),
            "contact_radius_linear_interior_rel": float(
                (a_linear - rr[-1]) / (radii[idx[-1] + 1] - radii[idx[-1]])
            ) if idx[-1] + 1 < radii.size else float("nan"),
        },
    )


# ---------------------------------------------------------------------------
# Load control
# ---------------------------------------------------------------------------


def solve_axisymmetric_contact_load_control(
    *,
    load,
    radius,
    young,
    poisson=0.3,
    nr=96,
    nz=48,
    width=None,
    depth=None,
    r_grading=2.0,
    z_grading=2.5,
    indentation_guess=None,
    r_axis=None,
    z_axis=None,
    rho_factor=None,
    alm_tol=1e-10,
    load_tol=1e-9,
    max_outer=60,
    keep_displacement=False,
):
    """Load-controlled Hertz contact: find ``delta`` such that reaction == load.

    ``load`` is the full (2 pi integrated) normal force.  ``width``/``depth``
    default to ``12 a`` and ``24 a`` of the Hertz estimate.  ``rho_factor``
    scales the ALM parameter (``rho = rho_factor * E / a``); it affects the
    iteration only, not the converged state.

    Returns ``(HertzLoadReference, ContactState, diagnostics)``.
    """
    ref = hertz_load_reference(load, radius, young, poisson)
    if width is None:
        width = 12.0 * ref.contact_radius
    if depth is None:
        depth = 24.0 * ref.contact_radius
    if rho_factor is None:
        rho_factor = 1.0

    if r_axis is not None or z_axis is not None:
        if r_axis is None or z_axis is None:
            raise ValueError("pass both r_axis and z_axis, or neither")
        nodes, quads = mesh_from_axes(r_axis, z_axis)
        nr, nz = len(r_axis) - 1, len(z_axis) - 1
        width, depth = float(r_axis[-1]), float(-z_axis[-1])
    else:
        nodes, quads = axisymmetric_mesh(
            nr, nz, width, depth, r_grading=r_grading, z_grading=z_grading
        )
    kmat = assemble_axisymmetric_stiffness(nodes, quads, young, poisson)
    ndof = 2 * len(nodes)
    top = np.arange(0, len(nodes), nz + 1)
    radii = nodes[top, 0]
    areas = surface_ring_areas(radii)
    cz_all = 2 * top + 1

    # Truncation boundary conditions: bottom fully fixed, outer radius radially
    # fixed, axis node radially fixed by symmetry.
    bottom = np.arange(nz, len(nodes), nz + 1)
    outer = np.arange(len(nodes) - (nz + 1), len(nodes))
    fixed = np.unique(
        np.concatenate(
            [2 * bottom, 2 * bottom + 1, 2 * outer, 2 * top[np.isclose(radii, 0.0)]]
        )
    )
    free_dofs = np.setdiff1d(np.arange(ndof), fixed)

    rho = rho_factor * young / ref.contact_radius
    force_scale = float(load)

    def obstacle_for(delta):
        return radii * radii / (2.0 * radius) - delta

    def response(delta):
        return _contact_state_fixed_indentation(
            kmat,
            radii=radii,
            areas=areas,
            obstacle=obstacle_for(delta),
            free_dofs=free_dofs,
            cz_all=cz_all,
            rho=rho,
            force_scale=force_scale,
            alm_tol=alm_tol,
        )

    history = []
    delta = (
        float(indentation_guess)
        if indentation_guess is not None
        else float(ref.indentation)
    )
    d1 = delta
    s1 = response(d1)
    history.append((d1, s1.reaction))
    if s1.reaction <= 0:
        d1 = max(1e-8, 0.25 * d1)
        s1 = response(d1)
        history.append((d1, s1.reaction))
    factor = 1.10 if s1.reaction < load else 0.90
    d2 = d1 * factor
    s2 = response(d2)
    history.append((d2, s2.reaction))

    state = s2
    outer_iterations = 0
    for outer_iterations in range(1, max_outer + 1):
        if abs(s2.reaction - s1.reaction) < 1e-14 * load:
            break
        d3 = d2 - (s2.reaction - load) * (d2 - d1) / (s2.reaction - s1.reaction)
        d3 = float(min(max(d3, 1e-10), 1.0))
        s3 = response(d3)
        history.append((d3, s3.reaction))
        d1, s1 = d2, s2
        d2, s2 = d3, s3
        state = s3
        if abs(s2.reaction - load) <= load_tol * load:
            break

    residual = abs(state.reaction - load) / load
    diagnostics = {
        "outer_iterations": outer_iterations,
        "load_residual": residual,
        "load_converged": bool(residual <= max(load_tol, 1e-8)),
        "alm_iterations": state.alm_iterations,
        "alm_tol": alm_tol,
        "rho": rho,
        "rho_factor": rho_factor,
        "ndof": ndof,
        "n_elements": len(quads),
        "mesh": {
            "nr": nr,
            "nz": nz,
            "width": width,
            "depth": depth,
            "r_grading": r_grading,
            "z_grading": z_grading,
        },
        "surface_spacing_at_contact": float(
            np.min(np.diff(radii[(radii > 0.5 * ref.contact_radius) & (radii < 1.5 * ref.contact_radius)]))
            if np.any((radii > 0.5 * ref.contact_radius) & (radii < 1.5 * ref.contact_radius))
            else np.nan
        ),
        "load_history": history,
        "maximum_penetration": state.maximum_penetration,
        "penetration_over_indentation": state.maximum_penetration
        / max(state.prescribed_indentation, 1e-30),
    }
    if not keep_displacement:
        state.displacement = None
    return ref, state, diagnostics
