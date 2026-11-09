"""Elastic-plastic indentation with unilateral contact: whole load cycle.

This module extends the elastic axisymmetric Hertz experiment of
:mod:`tensorfem.hertz_axisymmetric` (small strain, elasticity, monotonic load)
into the nonlinear regime that governs ice-structure interaction:

1. **Elastic contact** -- a rigid parabolic indenter pressed into a
   displacement-controlled axisymmetric half-space; the elastic backbone is
   verified against the closed-form Hertz solution.
2. **Elastic-plastic indentation** -- incremental ``J2`` flow theory with
   isotropic hardening at every Gauss point, radial return mapping and the
   consistent (algorithmic) tangent.  The mean contact pressure is compared
   with the Johnson (1985) cavity-model indentation law (see
   :func:`johnson_mean_pressure`).
3. **Unloading and residual deformation** -- the indenter is withdrawn and the
   contact is released entirely; the body relaxes elastically around the frozen
   plastic strain and the residual indentation depth is read off the centre
   node.
4. **Whole-cycle energy balance** -- the external work ``oint F dh`` is
   compared with the volume integrals of plastic dissipation and of the
   *stored* elastic energy of the residual stress field.  The two routes are
   independent (path integral of the contact force versus a Gauss-point volume
   sum), so the closure check is a genuine audit.
5. **Repeated cycles** -- repeated load/unload to the same peak indentation.
   For an isotropically hardening (or perfectly plastic) solid under
   displacement control the analytic prediction is *elastic shakedown*: the
   residual depth increment for cycles ``n >= 2`` vanishes.  That increment is
   measured from the FE state, not assumed.  For the kinematic-hardening case,
   where accumulation is genuinely non-zero,
   :func:`armstrong_frederick_ratchet_rate` supplies the closed-form ratcheting
   rate (verified against direct cycle-by-cycle integration in
   ``tests/test_indentation_plasticity.py``).

Conventions
-----------
* Strain/stress vectors are ``[rr, zz, tt, rz]`` with the **tensor** shear
  component (``e_rz``, not the engineering ``gamma_rz``); the ``2x`` weight of
  the double contraction is carried by :data:`_W`.
* ``z`` points down into the half-space from the free surface ``z = 0`` in the
  mesh of :func:`tensorfem.hertz_axisymmetric.axisymmetric_mesh`
  (``z in [-depth, 0]``), so a downward indenter displacement is negative
  ``u_z``.
* Units are whatever the caller uses; the study script works in dimensionless
  units (E, R, h all O(1)) and states the scaling.

References
----------
Johnson, K. L. (1985) *Contact Mechanics*, ch. 4 (Hertz) and 6 (cavity model,
elastic unloading of an indentation).
Hill, R. (1950) *The Mathematical Theory of Plasticity*.
Simo, J. C. & Hughes, T. J. R. (1998) *Computational Inelasticity* (radial
return mapping, algorithmic tangent).
Armstrong & Frederick (1966) CEGB RD/B/N731 (nonlinear kinematic hardening).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .hertz_axisymmetric import axisymmetric_mesh

# double-contraction weight for the [rr, zz, tt, rz] tensor-shear convention
_W = np.array([1.0, 1.0, 1.0, 2.0])
_ONE = np.array([1.0, 1.0, 1.0, 0.0])
# engineering-shear column scaling: d(gamma_rz)/d(e_rz) = 2
_HALF_GAMMA = np.array([1.0, 1.0, 1.0, 0.5])


# ---------------------------------------------------------------------------
# material and constitutive update
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class J2Material:
    """Isotropic linear-hardening von Mises material.

    ``hardening`` is the plastic slope ``H = d sigma_y / d eps_p`` (0 gives a
    perfectly plastic solid, the classical indentation idealisation).
    """

    young: float
    poisson: float
    yield_stress: float
    hardening: float = 0.0

    def __post_init__(self):
        if self.young <= 0.0 or not (-1.0 < self.poisson < 0.5):
            raise ValueError("invalid elastic constants")
        if self.yield_stress <= 0.0 or self.hardening < 0.0:
            raise ValueError("invalid plastic parameters")

    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def effective_modulus(self) -> float:
        """``E* = E / (1 - nu^2)`` for a rigid indenter on an elastic half-space."""
        return self.young / (1.0 - self.poisson ** 2)

    def elastic_tangent(self) -> np.ndarray:
        """4x4 elastic stiffness in the **tensor**-shear convention.

        The shear entry is ``C[3,3] = 2G = E/(1+nu)`` because the fourth strain
        component stored here is ``e_rz``, not the engineering ``gamma_rz``.
        (The engineering-convention ``D[3,3] = G`` of
        :func:`tensorfem.hertz_axisymmetric.assemble_axisymmetric_stiffness`
        would be wrong by a factor two with this storage; the elastic
        benchmark of the module checks the consequence.)
        """
        nu, e = self.poisson, self.young
        c = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array([
            [1.0 - nu, nu, nu, 0.0],
            [nu, 1.0 - nu, nu, 0.0],
            [nu, nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.0, (1.0 - 2.0 * nu)],
        ])

    def yield_at(self, eqp):
        return self.yield_stress + self.hardening * np.asarray(eqp)


def von_mises(stress: np.ndarray):
    """Return ``(q, s, |s|)`` for ``(..., 4)`` stress arrays."""
    stress = np.asarray(stress, dtype=float)
    p = stress[..., :3].mean(axis=-1)
    s = stress.copy()
    s[..., :3] -= p[..., None]
    rho = np.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2 + s[..., 2] ** 2
                  + 2.0 * s[..., 3] ** 2)
    return math.sqrt(1.5) * rho, s, rho


def j2_return(strain: np.ndarray, plastic_strain: np.ndarray,
              eqp: np.ndarray, material: J2Material) -> dict:
    """Vectorised radial return mapping with the algorithmic tangent.

    Parameters
    ----------
    strain, plastic_strain : (n, 4) arrays
        Total strain at the current iterate and the committed plastic strain.
    eqp : (n,) array
        Committed equivalent plastic strain.
    material : J2Material

    Returns
    -------
    dict
        ``stress`` (n, 4), ``plastic_strain`` (n, 4), ``eqp`` (n,),
        ``tangent`` (n, 4, 4) algorithmic tangent, ``plastic_work`` (n,)
        density increment (exact for linear hardening), ``dlam`` (n,) and
        ``equivalent_stress`` (n,).

    Notes
    -----
    Consistency for ``sigma_y = Y0 + H eps_p``::

        dlam   = (q_tr - Y0 - H eps_p) / (3 G + H)
        q_new  = q_tr - 3 G dlam,        beta = q_new / q_tr
        s_new  = beta s_tr
        C_ep   = K 1 x 1 + 2 G beta P_dev + 2 G (c - beta) n n
        c      = H / (3 G + H)

    with ``n = s_tr/|s_tr|`` the unit deviatoric direction and ``P_dev`` the
    deviatoric projector.  For pure proportional plastic flow
    (``d eps ~ n``) the ``n n`` term cancels the deviatoric elastic term
    exactly, which is the signature of a correct consistent tangent; the
    tangent is additionally audited against central differences in
    ``tests/test_indentation_plasticity.py``.
    """
    ce = material.elastic_tangent()
    strain = np.atleast_2d(np.asarray(strain, dtype=float))
    plastic_strain = np.atleast_2d(np.asarray(plastic_strain, dtype=float))
    eqp = np.atleast_1d(np.asarray(eqp, dtype=float))
    trial = (strain - plastic_strain) @ ce.T
    q_tr, s_tr, rho = von_mises(trial)
    f = q_tr - material.yield_at(eqp)
    plastic = f > 0.0
    G, H, K = material.shear, material.hardening, material.bulk

    dlam = np.zeros_like(q_tr)
    beta = np.ones_like(q_tr)
    if np.any(plastic):
        dlam[plastic] = f[plastic] / (3.0 * G + H)
        beta[plastic] = (q_tr[plastic] - 3.0 * G * dlam[plastic]) / q_tr[plastic]

    stress = trial.copy()
    plastic_strain_new = plastic_strain.copy()
    eqp_new = eqp.copy()
    plastic_work = np.zeros_like(q_tr)
    if np.any(plastic):
        p = trial[plastic, :3].mean(axis=-1)
        stress[plastic, :3] = p[:, None] + beta[plastic, None] * s_tr[plastic, :3]
        stress[plastic, 3] = beta[plastic] * s_tr[plastic, 3]
        flow = 1.5 * s_tr[plastic] / q_tr[plastic][:, None]
        plastic_strain_new[plastic] = plastic_strain[plastic] + dlam[plastic, None] * flow
        eqp_new[plastic] = eqp[plastic] + dlam[plastic]
        e0, e1 = eqp[plastic], eqp_new[plastic]
        plastic_work[plastic] = (material.yield_stress * dlam[plastic]
                                 + 0.5 * H * (e1 ** 2 - e0 ** 2))

    tangent = np.broadcast_to(ce, (len(strain), 4, 4)).copy()
    if np.any(plastic):
        n_dir = s_tr[plastic] / np.maximum(rho[plastic], 1e-300)[:, None]
        beta_p = beta[plastic][:, None, None]
        # d rho_new / d rho_tr with rho = |s| (q = sqrt(3/2) rho)
        c_hard = H / (3.0 * G + H)
        p_dev = np.eye(4) - np.outer(_ONE, _ONE) / 3.0
        # W-self-adjoint outer product: the double contraction weights the
        # shear component by two (tensor-shear convention)
        nn = n_dir[:, :, None] * (n_dir * _W)[:, None, :]
        tangent[plastic] = (K * np.outer(_ONE, _ONE)[None, :, :]
                           + 2.0 * G * beta_p * p_dev[None, :, :]
                           + 2.0 * G * (c_hard - beta_p) * nn)
    return {
        "stress": stress,
        "plastic_strain": plastic_strain_new,
        "eqp": eqp_new,
        "tangent": tangent,
        "plastic_work": plastic_work,
        "dlam": dlam,
        "equivalent_stress": von_mises(stress)[0],
    }


# ---------------------------------------------------------------------------
# axisymmetric Q4 discretisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisymmetricDiscretisation:
    nodes: np.ndarray
    elements: np.ndarray
    B: np.ndarray            # (ne, ngp, 4, 8)
    weight: np.ndarray       # (ne, ngp) with 2 pi r detJ
    dof_map: np.ndarray      # (ne, 8)
    top_nodes: np.ndarray
    ring_area: np.ndarray
    radii: np.ndarray        # radii of the contact (top) nodes
    width: float
    depth: float

    @property
    def n_dof(self) -> int:
        return 2 * len(self.nodes)


def axisymmetric_discretisation(nr: int = 24, nz: int = 24, width: float = 10.0,
                                depth: float = 10.0, grading: float = 2.0,
                                ) -> AxisymmetricDiscretisation:
    """Structured graded ``(r, z)`` mesh with Q4 kinematics precomputed.

    The mesh topology is exactly the one of
    :func:`tensorfem.hertz_axisymmetric.axisymmetric_mesh` (single source of
    truth for the discretisation); the Gauss-point kinematics, the axisymmetric
    volume weight ``2 pi r detJ`` and the lumped contact ring areas are added
    here.
    """
    nodes_t, elements_t = axisymmetric_mesh(nr, nz, width, depth, grading=grading)
    nodes = np.asarray(nodes_t.numpy(), dtype=float)
    elements = np.asarray(elements_t.numpy(), dtype=np.int64)
    ne = len(elements)
    g = 1.0 / math.sqrt(3.0)
    gauss = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    Nmat = np.zeros((4, 4))
    dnat = np.zeros((4, 4, 2))
    for gp, (a, b) in enumerate(gauss):
        Nmat[gp] = 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b),
                                    (1 + a) * (1 + b), (1 - a) * (1 + b)])
        dnat[gp] = 0.25 * np.array([
            [-(1 - b), -(1 - a)],
            [(1 - b), -(1 + a)],
            [(1 + b), (1 + a)],
            [-(1 + b), (1 - a)],
        ])
    coords = nodes[elements]                                  # (ne, 4, 2)
    # Explicit per-element loop (identical formulas to the dense reference
    # assembly of hertz_axisymmetric / the textbook axisymmetric Q4) -- a
    # vectorised einsum version was found to disagree with this on graded
    # meshes, so the transparent form is kept.
    B = np.zeros((ne, 4, 4, 8))
    weight = np.zeros((ne, 4))
    for e in range(ne):
        x = coords[e]
        for gp in range(4):
            N = Nmat[gp]
            dN = dnat[gp] @ np.linalg.inv(dnat[gp].T @ x)
            det = np.linalg.det(dnat[gp].T @ x)
            if det <= 0.0:
                raise ValueError("non-positive Jacobian in the axisymmetric mesh")
            radius = float(N @ x[:, 0])
            Be = np.zeros((4, 8))
            Be[0, 0::2] = dN[:, 0]
            Be[1, 1::2] = dN[:, 1]
            Be[2, 0::2] = N / radius
            # engineering shear row: gamma_rz = du_r/dz + du_z/dr
            Be[3, 0::2] = dN[:, 1]
            Be[3, 1::2] = dN[:, 0]
            B[e, gp] = Be
            weight[e, gp] = 2.0 * math.pi * radius * det
    dof_map = np.stack((2 * elements, 2 * elements + 1), axis=-1).reshape(ne, 8)
    top_nodes = np.arange(0, (nr + 1) * (nz + 1), nz + 1)
    radii = nodes[top_nodes, 0]
    edges = np.empty(len(radii) + 1)
    edges[0], edges[-1] = 0.0, radii[-1]
    edges[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    ring_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return AxisymmetricDiscretisation(nodes, elements, B, weight, dof_map,
                                      top_nodes, ring_area, radii, width, depth)


def boundary_dofs(discret: AxisymmetricDiscretisation) -> np.ndarray:
    """Symmetry at the axis, rigid base, and far-field radial truncation."""
    fixed = []
    for i, (r, z) in enumerate(discret.nodes):
        if abs(r) < 1e-12:
            fixed.append(2 * i)
        if abs(z + discret.depth) < 1e-12:
            fixed.extend((2 * i, 2 * i + 1))
        if abs(r - discret.width) < 1e-12:
            fixed.append(2 * i)
    return np.array(sorted(set(fixed)), dtype=np.int64)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
@dataclass
class IndentationStep:
    indentation: float
    load: float
    contact_radius: float
    peak_pressure: float
    mean_pressure: float
    max_penetration: float
    fitted_radius: float = 0.0


@dataclass
class IndentationCycle:
    steps: list
    residual_depth: float
    plastic_work: float
    stored_energy: float
    external_work: float
    load_work: float
    unload_work: float
    max_eqp: float
    release_plastic_flow: float
    penalty_energy: float
    energy_closure: float
    energy_closure_corrected: float
    iterations: int
    converged: bool

    def loading_curve(self):
        return (np.array([s.indentation for s in self.steps]),
                np.array([s.load for s in self.steps]))


@dataclass
class IndentationResult:
    cycles: list
    discret: AxisymmetricDiscretisation
    material: J2Material
    radius: float
    penalty: float
    surface_profile: dict = field(default_factory=dict)

    @property
    def residual_depths(self) -> np.ndarray:
        return np.array([c.residual_depth for c in self.cycles])


class IndentationSolver:
    """Displacement-controlled rigid-indenter solver on an axisymmetric mesh."""

    def __init__(self, discret: AxisymmetricDiscretisation, material: J2Material,
                 *, radius: float, penalty_factor: float = 300.0,
                 tolerance: float = 1e-9, max_iterations: int = 40):
        self.d = discret
        self.m = material
        self.radius = float(radius)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.free = np.setdiff1d(np.arange(discret.n_dof), boundary_dofs(discret))
        self.penalty = penalty_factor * material.effective_modulus / math.sqrt(radius)
        size = discret.B.shape[0] * discret.B.shape[1]
        self.plastic_strain = np.zeros((size, 4))
        self.eqp = np.zeros(size)
        self.u = np.zeros(discret.n_dof)
        self.iterations = 0
        # sparse assembly pattern
        # rows[e, a, b] = dof a, cols[e, a, b] = dof b (row/column scatter)
        self._rows = np.repeat(discret.dof_map[:, :, None], 8, axis=2)
        self._cols = np.repeat(discret.dof_map[:, None, :], 8, axis=1)
        self._shape = (discret.n_dof, discret.n_dof)

    # -- kinematics / assembly -------------------------------------------
    def strains(self, u: np.ndarray) -> np.ndarray:
        """Total strain at the Gauss points in the **tensor**-shear convention."""
        ue = u[self.d.dof_map]
        eps = np.einsum("egca,ea->egc", self.d.B, ue)
        eps[..., 3] *= 0.5            # gamma_rz -> e_rz
        return eps.reshape(-1, 4)

    def assemble(self, u: np.ndarray):
        out = j2_return(self.strains(u), self.plastic_strain, self.eqp, self.m)
        ne, ngp = self.d.B.shape[0], self.d.B.shape[1]
        stress = out["stress"].reshape(ne, ngp, 4)
        tangent = out["tangent"].reshape(ne, ngp, 4, 4)
        w = self.d.weight
        f_int = np.zeros(self.d.n_dof)
        blocks = np.zeros((self.d.B.shape[0], 8, 8))
        for gp in range(ngp):
            Bg = self.d.B[:, gp]                       # (ne, 4, 8), engineering
            w = self.d.weight[:, gp]
            np.add.at(f_int, self.d.dof_map.reshape(-1),
                      (np.einsum("eca,ec->ea", Bg, stress[:, gp])
                       * w[:, None]).reshape(-1))
            # tangent columns with a shear strain index map dgamma -> 2 de
            ceng = tangent[:, gp] * _HALF_GAMMA
            cb = np.einsum("ecd,ecb->edb", ceng, Bg)
            blocks += np.einsum("eca,ecb->eab", Bg, cb) * w[:, None, None]
        stiffness = sp.coo_matrix((blocks.reshape(-1),
                                   (self._rows.reshape(-1), self._cols.reshape(-1))),
                                  shape=self._shape).tocsr()
        return f_int, stiffness, out

    # -- contact ---------------------------------------------------------
    def obstacle(self, h: float) -> np.ndarray:
        return self.d.radii ** 2 / (2.0 * self.radius) - h

    def contact_pressure(self, u: np.ndarray, h: float):
        uz = u[2 * self.d.top_nodes + 1]
        pen = np.clip(uz - self.obstacle(h), 0.0, None)
        return pen, self.penalty * pen

    def contact_force(self, u: np.ndarray, h: float) -> np.ndarray:
        _, pressure = self.contact_pressure(u, h)
        return pressure * self.d.ring_area

    # -- Newton ----------------------------------------------------------
    def solve_step(self, h: float, *, contact: bool = True,
                   max_iterations: int = None) -> bool:
        limit = max_iterations or self.max_iterations
        u = self.u.copy()
        converged = False
        for _ in range(limit):
            self.iterations += 1
            f_int, stiffness, _ = self.assemble(u)
            residual = f_int.copy()
            tangent = stiffness
            if contact:
                force = self.contact_force(u, h)
                # external contact force acts along -z (the indenter pushes the
                # surface down); the out-of-balance residual is f_int - f_ext
                residual[2 * self.d.top_nodes + 1] += force
                active = force > 0.0
                if np.any(active):
                    idx = 2 * self.d.top_nodes[active] + 1
                    diagonal = tangent.diagonal().copy()
                    diagonal[idx] += self.penalty * self.d.ring_area[active]
                    tangent = tangent + sp.diags(
                        diagonal - tangent.diagonal())
            g = residual[self.free]
            scale = max(float(np.linalg.norm(residual)), 1.0)
            if np.linalg.norm(g) / scale <= self.tolerance:
                converged = True
                break
            du = spla.spsolve(tangent[self.free][:, self.free].tocsc(), -g)
            u[self.free] += du
        self.u = u
        return converged

    # -- derived metrics --------------------------------------------------
    def metrics(self, h: float) -> IndentationStep:
        pen, pressure = self.contact_pressure(self.u, h)
        load = float(np.sum(pressure * self.d.ring_area))
        active = np.nonzero(pen > 0.0)[0]
        radius_eff = 0.0
        peak = float(pressure.max()) if len(active) else 0.0
        if len(active) >= 3:
            # Hertz-consistent recovery p^2 = p0^2 (1 - r^2/a^2); used for the
            # elastic benchmark and reported as the "fitted" radius.
            r = self.d.radii[active]
            p = pressure[active]
            design = np.stack((np.ones_like(r), r * r), axis=1)
            coeff, *_ = np.linalg.lstsq(design, p * p, rcond=None)
            peak = float(np.sqrt(max(coeff[0], 0.0)))
            fitted_radius = float(np.sqrt(max(-coeff[0] / coeff[1], 0.0))) \
                if coeff[1] < 0.0 else 0.0
        else:
            fitted_radius = 0.0
        # contact edge from the pressure profile: linear extrapolation of the
        # last two active points to p = 0 (works for flat elastic-plastic
        # plateaus where the Hertzian quadratic fit is not applicable)
        if len(active):
            i = int(active[-1])
            r = self.d.radii
            if i > 0 and pressure[i - 1] > pressure[i]:
                radius_eff = float(r[i] + (r[i] - r[i - 1])
                                   * pressure[i] / (pressure[i - 1] - pressure[i]))
            else:
                radius_eff = float(r[i])
        mean = load / (math.pi * radius_eff ** 2) if radius_eff > 0 else 0.0
        return IndentationStep(float(h), load, radius_eff, peak, mean,
                               float(pen.max()), fitted_radius)

    def volume_energies(self) -> tuple:
        """Committed plastic work and stored elastic energy (volume sums)."""
        strains = self.strains(self.u)
        out = j2_return(strains, self.plastic_strain, self.eqp, self.m)
        elastic = strains - out["plastic_strain"]
        ce = self.m.elastic_tangent()
        density = 0.5 * np.einsum("ni,ij,nj->n", elastic * _W, ce, elastic)
        w = self.d.weight.reshape(-1)
        stored = float(np.sum(density * w))
        return stored, out

    def surface_profile(self) -> dict:
        return {"radii": self.d.radii.copy(),
                "uz": self.u[2 * self.d.top_nodes + 1].copy()}


def run_indentation_cycles(*, nr: int = 24, nz: int = 24, width: float = 10.0,
                           depth: float = 10.0, grading: float = 2.0,
                           radius: float = 1.0, peak_indentation: float = 0.05,
                           load_steps: int = 24, unload_steps: int = 8,
                           cycles: int = 1, material: J2Material | None = None,
                           penalty_factor: float = 300.0,
                           tolerance: float = 1e-9,
                           strict_unloading: bool = True) -> IndentationResult:
    """Run ``cycles`` complete load/unload cycles and return the audit data.

    Each cycle drives the indenter from ``h = 0`` to ``peak_indentation`` in
    ``load_steps`` equal increments, then withdraws it in ``unload_steps``
    increments until the contact force vanishes and finally releases the
    contact completely (one Newton solve with ``f_int = 0``) so that the
    residual state is the true equilibrium of the body with zero external
    load.
    """
    if material is None:
        material = J2Material(young=1000.0, poisson=0.3, yield_stress=18.2)
    discret = axisymmetric_discretisation(nr=nr, nz=nz, width=width,
                                          depth=depth, grading=grading)
    solver = IndentationSolver(discret, material, radius=radius,
                               penalty_factor=penalty_factor,
                               tolerance=tolerance)
    results = []
    for cycle in range(cycles):
        steps = []
        plastic_work = 0.0
        load_work = 0.0
        unload_work = 0.0
        converged = True

        h_prev, f_prev = 0.0, 0.0
        hs = np.linspace(0.0, peak_indentation, load_steps + 1)[1:]
        for h in hs:
            converged &= solver.solve_step(h)
            state = j2_return(solver.strains(solver.u), solver.plastic_strain,
                              solver.eqp, material)
            plastic_work += float(np.sum(state["plastic_work"]
                                         * discret.weight.reshape(-1)))
            solver.plastic_strain = state["plastic_strain"]
            solver.eqp = state["eqp"]
            step = solver.metrics(h)
            load_work += 0.5 * (f_prev + step.load) * (h - h_prev)
            h_prev, f_prev = h, step.load
            steps.append(step)
        peak_state = solver.metrics(peak_indentation)
        # Penalty-contact compliance energy stored at the peak load.  It is
        # returned by the *indenter* spring but is not part of the continuum
        # energy, so it disappears when the contact is released: without this
        # term the ledger under-counts the external work by exactly
        # sum(1/2 penalty u_pen^2 A) (a pure penalty-discretisation artifact,
        # O(1/penalty); it is measured here instead of being silently dropped).
        pen_peak, _ = solver.contact_pressure(solver.u, peak_indentation)
        penalty_energy = float(0.5 * solver.penalty
                               * np.sum(pen_peak ** 2 * discret.ring_area))
        for k in range(1, unload_steps + 1):
            h = peak_indentation * (1.0 - k / unload_steps)
            converged &= solver.solve_step(h, max_iterations=60)
            step = solver.metrics(h)
            unload_work += 0.5 * (f_prev + step.load) * (h - h_prev)
            h_prev, f_prev = h, step.load
            if step.load <= 1e-6 * max(peak_state.load, 1e-30):
                break
        # full release: no contact at all (elastic spring-back around the
        # frozen plastic strain; any further plastic flow here would be a bug)
        eqp_before_release = solver.eqp.copy()
        converged &= solver.solve_step(0.0, contact=False, max_iterations=60)
        residual_depth = -float(solver.u[2 * discret.top_nodes[0] + 1])
        stored, final_state = solver.volume_energies()
        release_flow = float(np.max(final_state["eqp"] - eqp_before_release))
        if strict_unloading and release_flow > 1e-6:
            # The unloading/release branch is treated as elastic (state frozen
            # at the end of loading).  Reverse plastic flow would invalidate
            # the ledger, so it is detected instead of being silently ignored.
            raise RuntimeError(
                f"reverse plastic flow during unloading ({release_flow:.3e}): "
                "refine the unloading path before trusting the energy ledger")
        external = load_work + unload_work
        denom = max(abs(external), 1e-30)
        closure = (external - (plastic_work + stored)) / max(denom,
                                                              abs(plastic_work) + abs(stored),
                                                              1e-12)
        results.append(IndentationCycle(
            steps=steps,
            residual_depth=residual_depth,
            plastic_work=plastic_work,
            stored_energy=stored,
            external_work=external,
            load_work=load_work,
            unload_work=unload_work,
            max_eqp=float(np.max(final_state["eqp"])),
            release_plastic_flow=release_flow,
            penalty_energy=penalty_energy,
            energy_closure=float(closure),
            energy_closure_corrected=float(
                (external - (plastic_work + stored + penalty_energy))
                / max(denom, abs(plastic_work) + abs(stored), 1e-12)),
            iterations=solver.iterations,
            converged=bool(converged),
        ))
    return IndentationResult(results, discret, material, radius,
                             solver.penalty, solver.surface_profile())

# ---------------------------------------------------------------------------
# Johnson (1985) elastic-plastic indentation law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JohnsonModel:
    """Cavity-model indentation law of Johnson (1985), §6.5.

    Parameters
    ----------
    effective_modulus : float
        ``E*`` of the indenter/half-space pair (rigid indenter: ``E/(1-nu^2)``).
    yield_stress : float
        Uniaxial yield (flow) stress ``Y``.
    radius : float
        Indenter radius ``R``.
    fully_plastic_factor : float
        Mean-pressure ceiling ``p_m = factor * Y`` once the plastic zone
        reaches the free surface (Johnson quotes ``2.85 ... 3.0`` for a sphere;
        Tabor's constraint factor is used here as ``2.85``).
    yield_pressure_factor : float
        Mean pressure at first yield.  The von Mises first-yield condition for
        a sphere is ``p_0 = 1.6 Y``, i.e. ``p_m = 2/3 p_0 = 1.0667 Y``
        (Johnson quotes the rounded ``1.07 Y``; the exact value is used so that
        the elastic and cavity branches are exactly continuous).
    calibrated : bool
        If ``True`` the logarithmic cavity law is shifted so that it is
        *continuous* with the elastic solution at first yield instead of
        jumping to the raw asymptotic value ``(2/3)Y[2 + ln(Lambda/3)]``.
        The raw Johnson law evaluates to ``1.21 Y`` at the elastic limit while
        the exact elastic result is ``1.07 Y``; the calibrated form removes
        that ~13 % discontinuity.  Both are reported by the study script so the
        consequence of the choice is visible.
    """

    effective_modulus: float
    yield_stress: float
    radius: float
    fully_plastic_factor: float = 2.85
    yield_pressure_factor: float = 1.6 * 2.0 / 3.0
    calibrated: bool = False

    @property
    def elastic_limit_radius(self) -> float:
        """Contact radius at first yield: ``p_0 = 2 E* a / (pi R) = 1.6 Y``."""
        return 1.6 * math.pi * self.yield_stress * self.radius / (2.0 * self.effective_modulus)

    @property
    def elastic_limit_pressure(self) -> float:
        return self.yield_pressure_factor * self.yield_stress

    def mean_pressure(self, contact_radius: float) -> float:
        """``p_m(a)``: elastic, elastic-plastic, then fully plastic."""
        a = float(contact_radius)
        y = self.yield_stress
        elastic = 4.0 * self.effective_modulus * a / (3.0 * math.pi * self.radius)
        if a <= self.elastic_limit_radius:
            return elastic
        lam = self.effective_modulus * a / (3.0 * y * self.radius)
        if self.calibrated:
            lam_y = (self.effective_modulus * self.elastic_limit_radius
                     / (3.0 * y * self.radius))
            plastic = self.elastic_limit_pressure + (2.0 / 3.0) * y * math.log(lam / lam_y)
        else:
            plastic = (2.0 / 3.0) * y * (2.0 + math.log(lam))
        return min(plastic, self.fully_plastic_factor * y)

    def load(self, indentation: float) -> float:
        """``F(h)`` with the kinematic assumption ``h = a^2 / R``."""
        a = math.sqrt(self.radius * float(indentation))
        return self.mean_pressure(a) * math.pi * a * a

    def load_curve(self, indentations) -> np.ndarray:
        return np.array([self.load(h) for h in np.atleast_1d(indentations)])

    def constraint_factor(self, indentation: float) -> float:
        """``Psi = F / F_hertz(h)``: elastic-plastic load enhancement."""
        h = float(indentation)
        hertz = (4.0 / 3.0) * self.effective_modulus * math.sqrt(self.radius) * h ** 1.5
        return self.load(h) / hertz if hertz > 0 else 1.0

    def residual_depth(self, peak_indentation: float) -> float:
        """Residual depth after elastic unloading (Johnson eq. 6.44 form).

        Unloading follows the Hertzian law with the *elastoplastic* peak load,
        ``F_max = (4/3) E* sqrt(R) (h_max - h_r)^{3/2}``, so
        ``h_r = h_max - (3 F_max / (4 E* sqrt(R)))^{2/3}``.
        """
        f_max = self.load(peak_indentation)
        return peak_indentation - (3.0 * f_max
                                   / (4.0 * self.effective_modulus
                                      * math.sqrt(self.radius))) ** (2.0 / 3.0)


def johnson_model(material: J2Material, radius: float, *,
                  calibrated: bool = False) -> JohnsonModel:
    """Convenience constructor for :class:`JohnsonModel`."""
    return JohnsonModel(material.effective_modulus, material.yield_stress,
                        float(radius), calibrated=calibrated)


# ---------------------------------------------------------------------------
# cyclic plasticity: Armstrong-Frederick ratcheting rate
# ---------------------------------------------------------------------------
def armstrong_frederick_ratchet_rate(*, yield_stress: float, hardening_modulus: float,
                                     gamma: float, stress_max: float,
                                     stress_min: float) -> float:
    """Steady-state plastic strain increment per cycle (uniaxial, AF hardening).

    Constitutive law (Armstrong & Frederick 1966), uniaxial::

        d alpha = C d eps_p - gamma alpha |d eps_p|,     |sigma - alpha| = Y

    Under *stress control* between ``stress_min`` and ``stress_max`` the
    hysteresis loop closes in back-stress but not in strain, and the
    per-cycle accumulation has the closed form::

        d eps_cycle = (1/gamma) ln[ (C^2 - gamma^2 s_lo^2)
                                  / (C^2 - gamma^2 s_hi^2) ]
        s_lo = stress_min + Y,   s_hi = stress_max - Y

    valid for ``C > gamma * s_hi`` (otherwise ratcheting is unbounded, the
    classic AF ratcheting failure).  Derivation: on the loading branch
    ``d sigma = d alpha = (C - gamma alpha) d eps_p`` integrates to
    ``d eps = -(1/gamma) ln((C - gamma alpha_1)/(C - gamma alpha_0))``;
    the unloading branch gives ``d eps = (1/gamma) ln((C + gamma alpha_2)/
    (C + gamma alpha_1))``; imposing a periodic back-stress
    (``alpha_0 = alpha_2``) and the yield conditions at both stress ends
    yields the expression above.  ``tests/test_indentation_plasticity.py``
    verifies it against direct cycle-by-cycle numerical integration.
    """
    c = float(hardening_modulus)
    g = float(gamma)
    y = float(yield_stress)
    if g <= 0.0 or c <= 0.0:
        raise ValueError("AF hardening needs C > 0 and gamma > 0")
    s_lo = stress_min + y
    s_hi = stress_max - y
    if s_hi <= s_lo:
        return 0.0
    numerator = c * c - g * g * s_lo * s_lo
    denominator = c * c - g * g * s_hi * s_hi
    if denominator <= 0.0 or numerator <= 0.0:
        return float("inf")
    return math.log(numerator / denominator) / g


def integrate_armstrong_frederick_cycles(*, yield_stress: float,
                                         hardening_modulus: float, gamma: float,
                                         stress_max: float, stress_min: float,
                                         cycles: int = 10, steps: int = 2000) -> np.ndarray:
    """Direct uniaxial integration of the AF law; returns strain per cycle.

    Used as the numerical oracle of :func:`armstrong_frederick_ratchet_rate`.
    Stress is driven in ``steps`` increments per half cycle with a fully
    implicit (backward Euler) update of the back-stress, followed by a return
    to the yield surface, so the result is independent of the step size to
    O(1/steps).
    """
    c, g, y = float(hardening_modulus), float(gamma), float(yield_stress)
    alpha = 0.0
    eps = 0.0
    per_cycle = []
    for _ in range(cycles):
        eps0 = eps
        for sign, (s_from, s_to) in ((1.0, (stress_min, stress_max)),
                                     (-1.0, (stress_max, stress_min))):
            ds = (s_to - s_from) / steps
            for k in range(steps):
                sigma = s_from + ds * (k + 1)
                trial = sigma - alpha
                if sign * trial <= y:      # still elastic in this direction
                    continue
                def residual(d):
                    a_new = (alpha + sign * c * d) / (1.0 + g * d)
                    return sigma - a_new - sign * y

                r0 = residual(0.0)
                lo, hi = 0.0, 1.0e-6
                while np.sign(residual(hi)) == np.sign(r0) and hi < 1.0:
                    hi *= 2.0
                for _ in range(90):
                    mid = 0.5 * (lo + hi)
                    if np.sign(residual(mid)) == np.sign(r0):
                        lo = mid
                    else:
                        hi = mid
                d = 0.5 * (lo + hi)
                alpha = (alpha + sign * c * d) / (1.0 + g * d)
                eps += sign * d
        per_cycle.append(eps - eps0)
    return np.array(per_cycle)
