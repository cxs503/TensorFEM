"""Elastoplastic thick-walled cylinder under internal pressure (Hill 1950).

Why this case: the repository verifies the *elastic* Lamé cylinder in plane
strain (``docs/lame-cylinder-plane-strain.md``) and in three dimensions
(``docs/thick-cylinder-3d-lame.md``), and it verifies J2 return mapping at the
material point (``docs/j2-plasticity-minimal-surface.md``).  What is missing is
the classical **elastic-plastic** pressure vessel problem, i.e. a case where the
number, location and stresses of the yielded integration points are the answer
that has to be predicted — not an input.

Closed-form oracle (Hill, *The Mathematical Theory of Plasticity*, 1950, §5.3;
also Lubliner §3.5): long tube ``a <= r <= b``, internal pressure ``p``, plane
strain (``eps_z = 0``), von Mises yield, **perfect** plasticity.  Plane strain
plus plastic incompressibility forces ``sigma_z = (sigma_r + sigma_theta)/2``, so
the von Mises equivalent stress reduces to ``q = (sqrt3/2)(sigma_theta - sigma_r)``
and yield reads ``sigma_theta - sigma_r = 2k`` with ``k = sigma_y/sqrt(3)``.
Equilibrium then integrates exactly::

    p(c) = 2k [ ln(c/a) + (b^2 - c^2) / (2 b^2) ]        (c = plastic radius)
    plastic zone  a <= r <= c:
        sigma_r = -2k [ ln(c/r) + (b^2-c^2)/(2b^2) ],  sigma_theta = sigma_r + 2k
    elastic zone  c <= r <= b  (Lamé with interface pressure p_c = 2k(b^2-c^2)/(2b^2)):
        sigma_r = A (1 - b^2/r^2),  sigma_theta = A (1 + b^2/r^2),
        A = p_c c^2/(b^2-c^2)
    first yield: p_y = k (1 - a^2/b^2);   limit pressure: p_p = 2k ln(b/a)

The finite-element model is a *plane-strain axisymmetric Q4* discretisation of
the annulus: axisymmetric kinematics ``(eps_r, eps_z, eps_theta, gamma_rz)`` with
``u_z = 0`` imposed everywhere, which realises ``eps_z = 0`` **exactly** and keeps
the cylindrical geometry exact (no faceting, unlike a Cartesian sector).  At
every Gauss point the full three-dimensional J2 return mapping of
:mod:`tensorfem.j2_saturation_hardening` is used, i.e. neither ``sigma_z`` nor the
yield condition is imposed by hand — both are *outputs* of the constitutive
update that the oracle then checks.

Scope/limitations: small strain, isotropic elasticity, axisymmetric loading,
perfect (or isotropic-hardening) plasticity; no axial force/closed-end pressure
coupling beyond the plane-strain constraint, no Tresca comparison, no fatigue or
ratcheting, no finite strain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .j2_saturation_hardening import (
    EPState,
    HardeningLaw,
    ReturnResult,
    consistent_tangent,
    return_map,
)

Tensor = torch.Tensor

__all__ = [
    "HillPlaneStrainTube",
    "AxisymmetricMesh",
    "annulus_mesh",
    "TubeSolution",
    "solve_plastic_tube",
    "elastic_plastic_radius",
    "field_errors",
    "qualify_plastic_tube",
]

# Voigt permutation between the 3-D engineering-shear order
# [xx, yy, zz, xy, yz, zx] and the axisymmetric 4-vector [rr, zz, theta-theta, rz].
# With tensor axes (x, y, z) = (r, theta, z): rr -> 0, theta-theta -> 1, zz -> 2, rz -> 5.
AXI_PERM = (0, 2, 1, 5)
SQRT3 = math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# closed-form reference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HillPlaneStrainTube:
    """Hill's plane-strain elastic-plastic tube (perfect plasticity, von Mises)."""

    inner_radius: float = 1.0
    outer_radius: float = 2.0
    yield_stress: float = 250.0e6
    young: float = 2.0e11
    poisson: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 < self.inner_radius < self.outer_radius:
            raise ValueError("require 0 < a < b")
        if not self.yield_stress > 0.0 or not self.young > 0.0:
            raise ValueError("yield_stress and young must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")

    @property
    def k(self) -> float:
        """Shear yield stress ``k = sigma_y / sqrt(3)``."""
        return self.yield_stress / SQRT3

    @property
    def pressure_first_yield(self) -> float:
        a, b = self.inner_radius, self.outer_radius
        return self.k * (1.0 - (a / b) ** 2)

    @property
    def pressure_limit(self) -> float:
        return 2.0 * self.k * math.log(self.outer_radius / self.inner_radius)

    # ---- pressure <-> plastic radius (exact inverse by safegouarded Newton) --
    def pressure_at_radius(self, c: float) -> float:
        a, b = self.inner_radius, self.outer_radius
        if not (a - 1e-15) <= c <= b:
            raise ValueError("plastic radius must lie inside the wall")
        return 2.0 * self.k * (math.log(c / a) + (b * b - c * c) / (2.0 * b * b))

    def plastic_radius(self, pressure: float, *, tolerance: float = 1e-14) -> float:
        """Radius of the elastic-plastic interface for an internal pressure."""
        a, b = self.inner_radius, self.outer_radius
        lo, hi = a, b
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.pressure_at_radius(mid) < pressure:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tolerance * (b - a):
                break
        return 0.5 * (lo + hi)

    def elastic_amplitude(self, pressure: float) -> float:
        """Lamé constant ``A`` of the *elastic* region, ``sig_r = A (1 - b^2/r^2)``.

        For ``p <= p_y`` the whole wall is elastic and ``A`` follows from the bore
        traction ``sig_r(a) = -p``; otherwise the elastic region starts at ``c``
        with the interface pressure ``p_c = k (1 - c^2/b^2)``.
        """
        a, b, k = self.inner_radius, self.outer_radius, self.k
        c = self.plastic_radius(pressure)
        if c <= a * (1.0 + 1e-12):
            return pressure * a * a / (b * b - a * a)
        interface = k * (1.0 - (c / b) ** 2)
        return interface * c * c / (b * b - c * c)

    # ---- stresses ---------------------------------------------------------
    def stresses(self, r, pressure: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(sigma_r, sigma_theta, sigma_z, q)`` at radius ``r``."""
        r = torch.as_tensor(r, dtype=torch.float64)
        b, k = self.outer_radius, self.k
        c = self.plastic_radius(pressure)
        elastic = r >= c * (1.0 - 1e-12)
        # plastic branch
        sig_r_p = -2.0 * k * (torch.log(torch.as_tensor(c) / r) + (b * b - c * c) / (2.0 * b * b))
        sig_t_p = sig_r_p + 2.0 * k
        # elastic branch
        amp = self.elastic_amplitude(pressure)
        sig_r_e = amp * (1.0 - b * b / r ** 2)
        sig_t_e = amp * (1.0 + b * b / r ** 2)
        sig_r = torch.where(elastic, sig_r_e, sig_r_p)
        sig_t = torch.where(elastic, sig_t_e, sig_t_p)
        # Plane strain: plastic zone has eps_z^p = 0 => sigma_z = (sig_r+sig_t)/2;
        # the *elastic* zone responds elastically under eps_z = 0 => sigma_z = nu(sig_r+sig_t).
        # (The classical formula assumes s_z = 0 in the plastic zone only; the
        # residual inconsistency for nu != 1/2 is quantified in the report.)
        sig_z = torch.where(elastic, self.poisson * (sig_r_e + sig_t_e), 0.5 * (sig_r + sig_t))
        q = SQRT3 / 2.0 * (sig_t - sig_r)
        return sig_r, sig_t, sig_z, q

    def equivalent_stress(self, r, pressure: float) -> Tensor:
        return self.stresses(r, pressure)[3]

    # ---- radial displacement (independent numerical integration) ----------
    def radial_displacement(self, r, pressure: float, *, samples: int = 4000) -> Tensor:
        """``u_r(r)`` from the strain-compatibility relation, by quadrature.

        With ``eps_z = 0`` and ``eps_theta = u/r`` the compatibility identity
        ``d(r eps_theta)/dr = eps_r = eps_vol - eps_theta`` becomes
        ``d(r^2 eps_theta)/dr = r eps_vol``, so

        ``u_r(r) = w(r)/r``,   ``w(r) = c^2 eps_theta(c) - int_r^c s eps_vol(s) ds``

        with ``eps_theta(c)`` taken from the elastic side (continuity of ``u_r``)
        and ``eps_vol = (1-2nu)(3/2)(sigma_r + sigma_theta)/E``.  The integral is
        evaluated on ``samples`` uniform sub-intervals, so the displacement oracle
        is independent of any closed form for ``u_r`` in the plastic zone.
        """
        a = self.inner_radius
        E, nu = self.young, self.poisson
        c = self.plastic_radius(pressure)
        target = torch.as_tensor(r, dtype=torch.float64)

        sig_r, sig_t, _, _ = self.stresses(torch.as_tensor(c), pressure)
        sig_z_elastic = nu * (sig_r + sig_t)   # continuity of u_r at r = c
        eps_theta_c = float((sig_t - nu * (sig_r + sig_z_elastic)) / E)

        grid = torch.linspace(a, c, samples + 1, dtype=torch.float64)
        sig_r_g, sig_t_g, _, _ = self.stresses(grid, pressure)
        integrand = grid * (1.0 - 2.0 * nu) * 1.5 * (sig_r_g + sig_t_g) / E
        cumulative = torch.cumulative_trapezoid(integrand, grid)
        partial = torch.cat((torch.zeros(1, dtype=torch.float64), cumulative))
        total = float(partial[-1])
        w = c * c * eps_theta_c - (total - partial)   # w = r^2 eps_theta
        u_plastic = w / grid                            # u_r = w / r
        radial = torch.where(target >= c * (1.0 - 1e-12),
                             self._elastic_theta(target, pressure),
                             _interp(grid, u_plastic, target))
        return radial

    def _theta_rate(self, r: float, eps_theta: float, pressure: float) -> float:
        sig_r, sig_t, _, _ = self.stresses(torch.as_tensor(r), pressure)
        eps_vol = (1.0 - 2.0 * self.poisson) * 1.5 * float(sig_r + sig_t) / self.young
        return (eps_vol - 2.0 * eps_theta) / r

    def _elastic_theta(self, r: Tensor, pressure: float) -> Tensor:
        b = self.outer_radius
        amp = self.elastic_amplitude(pressure)
        lame_b = amp * b * b                  # B in sig_r = A - B/r^2
        return (1.0 + self.poisson) / self.young * ((1.0 - 2.0 * self.poisson) * amp * r
                                                   + lame_b / r)


def _interp(x: Tensor, y: Tensor, xq: Tensor) -> Tensor:
    """Monotone linear interpolation (``x`` ascending) for the ODE table."""
    xq = torch.minimum(torch.maximum(xq, x[0]), x[-1])
    idx = torch.searchsorted(x, xq.contiguous(), right=True).clamp(1, x.numel() - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    weight = (xq - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def elastic_plastic_radius(tube: HillPlaneStrainTube, pressure: float) -> float:
    """Convenience wrapper mirroring the oracle used in the report."""
    return tube.plastic_radius(pressure)


# --------------------------------------------------------------------------- #
# finite-element model: plane-strain axisymmetric Q4
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxisymmetricMesh:
    """4-node axisymmetric annulus mesh in the ``(r, z)`` plane."""

    nodes: Tensor           # (n, 2)
    elements: Tensor        # (m, 4) counter-clockwise
    inner_radius: float
    outer_radius: float

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes

    def inner_faces(self) -> Tensor:
        """Node pairs on the pressurised cylindrical surface ``r = a``."""
        a = self.inner_radius
        on_surface = torch.nonzero(torch.isclose(self.nodes[:, 0], self.nodes.new_tensor(a)),
                                   as_tuple=False).flatten()
        faces = []
        for element in self.elements.tolist():
            local = [n for n in element if n in set(on_surface.tolist())]
            if len(local) == 2:
                pair = sorted(local, key=lambda n: float(self.nodes[n, 1]))
                faces.append(pair)
        return torch.tensor(sorted(faces), dtype=torch.long)


def annulus_mesh(n_r: int, inner_radius: float = 1.0, outer_radius: float = 2.0,
                 n_z: int = 1, height: float = 1.0, grading: float = 0.0,
                 dtype=torch.float64) -> AxisymmetricMesh:
    """Structured annulus mesh with optional radial grading towards ``r = a``.

    ``grading = 0`` is uniform; ``grading > 0`` clusters elements near the inner
    wall with a geometric progression of ratio ``1 + grading``.
    """
    if n_r < 1 or n_z < 1:
        raise ValueError("need at least one element in each direction")
    if grading > 0.0:
        ratio = 1.0 + grading
        lengths = torch.tensor([ratio ** i for i in range(n_r)], dtype=dtype)
        lengths = lengths / lengths.sum()
        radii = inner_radius + (outer_radius - inner_radius) * torch.cat(
            (torch.zeros(1, dtype=dtype), torch.cumsum(lengths, 0)))
    else:
        radii = torch.linspace(inner_radius, outer_radius, n_r + 1, dtype=dtype)
    z = torch.linspace(0.0, height, n_z + 1, dtype=dtype)
    nodes, index = [], {}
    for j in range(n_z + 1):
        for i in range(n_r + 1):
            index[(i, j)] = len(nodes)
            nodes.append([float(radii[i]), float(z[j])])
    elements = []
    for j in range(n_z):
        for i in range(n_r):
            elements.append([index[(i, j)], index[(i + 1, j)],
                             index[(i + 1, j + 1)], index[(i, j + 1)]])
    return AxisymmetricMesh(torch.tensor(nodes, dtype=dtype),
                            torch.tensor(elements, dtype=torch.long),
                            inner_radius, outer_radius)


@dataclass(frozen=True)
class TubeSolution:
    pressure: float
    displacement: Tensor          # (n, 2) u_r, u_z
    gauss_radius: Tensor
    gauss_equivalent_stress: Tensor
    gauss_alpha: Tensor
    gauss_stress: Tensor          # (m, 4) per Gauss point [sig_r, sig_z, sig_theta, tau_rz]
    gauss_strain: Tensor
    iterations: int
    residual: float
    states: tuple
    mesh: AxisymmetricMesh
    residual_history: tuple = ()

    # ---- derived quantities ----------------------------------------------
    def plastic_radius(self, yield_stress: float, *, window: int = 4) -> dict[str, float]:
        """Two estimators of the elastic-plastic interface radius.

        ``bracket``: midpoint between the outermost yielded Gauss point and the
        first elastic one (model-free, resolution ``h/2``).

        ``lame_fit``: in the elastic zone the Lamé solution gives exactly
        ``q(r) = C / r^2``, so ``C`` is fitted (least squares) over the ``window``
        elastic Gauss points nearest the interface and ``r_p = sqrt(C/sigma_y)``;
        this is second order in the mesh size.
        """
        order = torch.argsort(self.gauss_radius)
        r = self.gauss_radius[order]
        q = self.gauss_equivalent_stress[order]
        alpha = self.gauss_alpha[order]
        yielded = alpha > 0.0
        if not bool(yielded.any()):
            return {"bracket": float("nan"), "lame_fit": float("nan"),
                    "plastic_fraction": 0.0, "r_inner_yielded": float(r[0])}
        last = int(torch.nonzero(yielded, as_tuple=False).flatten().max())
        r_out = float(r[last])
        r_next = float(r[last + 1]) if last + 1 < r.numel() else float("nan")
        bracket = 0.5 * (r_out + r_next) if r_next == r_next else r_out
        elastic = ~yielded
        idx = torch.nonzero(elastic, as_tuple=False).flatten()[:window]
        if idx.numel() >= 2:
            c = q[idx] * r[idx] ** 2
            constant = float(c.mean())
            fitted = math.sqrt(constant / yield_stress)
        else:
            fitted = float("nan")
        return {"bracket": bracket, "lame_fit": fitted,
                "plastic_fraction": float(yielded.double().mean()),
                "r_inner_yielded": float(r[0])}


_SHAPE = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
_GAUSS = 1.0 / math.sqrt(3.0)


def _element_kinematics(radius: Tensor, z: Tensor):
    """Shape-function derivatives and axisymmetric ``B`` matrices at 2x2 Gauss points."""
    blocks, detj, gauss_r = [], [], []
    for eta in (-_GAUSS, _GAUSS):
        for xi in (-_GAUSS, _GAUSS):
            dN_dxi = torch.tensor(
                [[-0.25 * (1.0 - eta), 0.25 * (1.0 - eta), 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)],
                 [-0.25 * (1.0 - xi), -0.25 * (1.0 + xi), 0.25 * (1.0 + xi), 0.25 * (1.0 - xi)]],
                dtype=radius.dtype)
            N = torch.tensor([0.25 * (1.0 - xi) * (1.0 - eta), 0.25 * (1.0 + xi) * (1.0 - eta),
                              0.25 * (1.0 + xi) * (1.0 + eta), 0.25 * (1.0 - xi) * (1.0 + eta)],
                             dtype=radius.dtype)
            jacobian = dN_dxi @ torch.stack((radius, z), dim=1)
            det = float(torch.linalg.det(jacobian))
            if det <= 0.0:
                raise ValueError("element has non-positive Jacobian (check node ordering)")
            dN = torch.linalg.solve(jacobian, dN_dxi)
            dN_dr, dN_dz = dN[0], dN[1]
            r_gp = float(N @ radius)
            b = torch.zeros((4, 8), dtype=radius.dtype)
            b[0, 0::2] = dN_dr
            b[1, 1::2] = dN_dz
            b[2, 0::2] = N / r_gp
            b[3, 0::2] = dN_dz
            b[3, 1::2] = dN_dr
            blocks.append(b)
            detj.append(det)
            gauss_r.append(r_gp)
    return blocks, detj, gauss_r


def _strain_tensor(strain4: Tensor) -> Tensor:
    """Axisymmetric 4-vector ``[e_rr, e_zz, e_tt, gamma_rz]`` to a 3x3 tensor."""
    return torch.stack((
        torch.stack((strain4[0], torch.zeros((), dtype=strain4.dtype), 0.5 * strain4[3])),
        torch.stack((torch.zeros((), dtype=strain4.dtype), strain4[2], torch.zeros((), dtype=strain4.dtype))),
        torch.stack((0.5 * strain4[3], torch.zeros((), dtype=strain4.dtype), strain4[1])),
    ))


def _stress4(stress: Tensor) -> Tensor:
    """3x3 stress tensor to ``[sig_rr, sig_zz, sig_tt, tau_rz]``."""
    return torch.stack((stress[0, 0], stress[2, 2], stress[1, 1], stress[0, 2]))


def _pressure_load(mesh: AxisymmetricMesh, pressure: float) -> Tensor:
    """Consistent nodal loads of an internal pressure on the cylindrical surface."""
    force = torch.zeros(mesh.n_dofs, dtype=mesh.nodes.dtype)
    if pressure == 0.0:
        return force
    a = mesh.inner_radius
    for face in mesh.inner_faces().tolist():
        z0, z1 = mesh.nodes[face[0], 1], mesh.nodes[face[1], 1]
        length = float(z1 - z0)
        area = 2.0 * math.pi * a * length
        for local, node in enumerate(face):
            weight = 0.5 * area
            dof = 2 * node
            # traction t = -p n with n = -e_r the outward normal of the *material*
            # at the bore, i.e. t = +p e_r: the bore pushes the wall outwards.
            force[dof] += pressure * weight
    return force


def _material_tangent(state: EPState, law: HardeningLaw, young: float, poisson: float,
                      strain: Tensor) -> tuple[Tensor, ReturnResult]:
    result = return_map(strain, state, law, young, poisson)
    tangent6 = consistent_tangent(strain, state, law, young, poisson, result)
    perm = torch.tensor(AXI_PERM)
    return tangent6[perm][:, perm], result


def solve_plastic_tube(
    tube: HillPlaneStrainTube,
    mesh: AxisymmetricMesh,
    law: HardeningLaw,
    pressure: float,
    *,
    steps: int = 8,
    tolerance: float = 1e-11,
    max_iterations: int = 40,
) -> TubeSolution:
    """Incremental load-controlled Newton solution of the plane-strain tube.

    The committed state is only advanced after an increment converges; the inner
    Newton loop always re-integrates every Gauss point from that immutable state,
    which is what makes the algorithm tangent self-correcting.
    """
    if mesh.nodes.dtype != torch.float64:
        mesh = AxisymmetricMesh(mesh.nodes.double(), mesh.elements, mesh.inner_radius,
                                mesh.outer_radius)
    elements = mesh.elements
    kinematics = []
    for element in elements.tolist():
        radius = mesh.nodes[element, 0]
        z = mesh.nodes[element, 1]
        blocks, detj, gauss_r = _element_kinematics(radius, z)
        dofs = torch.tensor([[2 * n, 2 * n + 1] for n in element]).flatten()
        weights = [2.0 * math.pi * g * d for g, d in zip(gauss_r, detj)]
        kinematics.append((blocks, weights, dofs, gauss_r))

    n_gp = 4
    states = tuple(EPState.virgin() for _ in range(len(elements) * n_gp))
    u = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)

    # plane strain: eps_z = 0 everywhere -> u_z = 0 at every node
    fixed = torch.zeros(mesh.n_dofs, dtype=torch.bool)
    fixed[1::2] = True
    free = torch.nonzero(~fixed, as_tuple=False).flatten()

    final = None
    for step in range(1, steps + 1):
        target = pressure * step / steps
        external = _pressure_load(mesh, target)
        scale = max(float(torch.linalg.vector_norm(external[free])), 1.0)
        trial_states = states
        iterations = 0
        history = []
        for iterations in range(1, max_iterations + 1):
            internal = torch.zeros(mesh.n_dofs, dtype=torch.float64)
            tangent = torch.zeros((mesh.n_dofs, mesh.n_dofs), dtype=torch.float64)
            new_states = []
            for e, (blocks, weights, dofs, _) in enumerate(kinematics):
                ue = u[elements[e]].flatten()
                ke = torch.zeros((8, 8), dtype=torch.float64)
                fe = torch.zeros(8, dtype=torch.float64)
                for gp, (b, w) in enumerate(zip(blocks, weights)):
                    strain4 = b @ ue
                    strain = _strain_tensor(strain4)
                    c4, result = _material_tangent(states[e * n_gp + gp], law,
                                                   tube.young, tube.poisson, strain)
                    stress4 = _stress4(result.stress)
                    fe += (b.T @ stress4) * w
                    ke += (b.T @ c4 @ b) * w
                    new_states.append(result.state)
                internal.index_add_(0, dofs, fe)
                tangent.index_put_((dofs[:, None], dofs[None, :]), ke, accumulate=True)
            residual = external - internal
            norm = float(torch.linalg.vector_norm(residual[free]))
            history.append(norm / scale)
            if norm <= tolerance * scale:
                break
            du = torch.linalg.solve(tangent[free][:, free], residual[free])
            u = u.reshape(-1).index_add(0, free, du).reshape(-1, 2)
            trial_states = tuple(new_states)
        else:
            raise RuntimeError(f"tube Newton failed at p={target:.4e} after {max_iterations} iterations")
        states = trial_states
        final = (target, u.clone(), iterations, norm, tuple(history))

    target, u, iterations, norm, history = final
    # post-process Gauss-point fields from the converged state
    radii, qs, alphas, stresses, strains = [], [], [], [], []
    for e, (blocks, _, _, gauss_r) in enumerate(kinematics):
        ue = u[elements[e]].flatten()
        for gp, b in enumerate(blocks):
            strain4 = b @ ue
            strain = _strain_tensor(strain4)
            result = return_map(strain, states[e * n_gp + gp], law, tube.young, tube.poisson)
            dev = result.stress - torch.trace(result.stress) / 3.0 * torch.eye(
                3, dtype=torch.float64)
            radii.append(gauss_r[gp])
            qs.append(float(torch.sqrt(1.5 * torch.sum(dev * dev))))
            alphas.append(float(result.state.alpha))
            stresses.append(_stress4(result.stress))
            strains.append(strain4)
    return TubeSolution(target, u, torch.tensor(radii), torch.tensor(qs), torch.tensor(alphas),
                        torch.stack(stresses), torch.stack(strains), iterations, norm, states, mesh,
                        history)


# --------------------------------------------------------------------------- #
# verification driver
# --------------------------------------------------------------------------- #
def field_errors(solution: TubeSolution, tube: HillPlaneStrainTube,
                 pressure: float) -> dict[str, float]:
    """Relative errors of the computed Gauss-point fields against Hill's solution.

    Stresses are normalised by the applied pressure (``sigma_r`` vanishes at the
    outer surface, so a per-point relative error would be meaningless there).
    """
    radius = solution.gauss_radius
    sig_r, sig_t, sig_z, q = tube.stresses(radius, pressure)
    scale = abs(pressure)
    out = {}
    for name, computed, reference in (
        ("sigma_r", solution.gauss_stress[:, 0], sig_r),
        ("sigma_theta", solution.gauss_stress[:, 2], sig_t),
        ("sigma_z", solution.gauss_stress[:, 1], sig_z),
        ("equivalent_stress", solution.gauss_equivalent_stress, q),
    ):
        out[f"{name}_max_rel"] = float((computed - reference).abs().max()) / scale
    out["sigma_r_bore"] = float(solution.gauss_stress[:, 0].min()) / scale
    out["sigma_r_outer"] = float(solution.gauss_stress[:, 0].max()) / scale
    yielded = solution.gauss_alpha > 0.0
    if bool(yielded.any()):
        out["yield_residual"] = float((solution.gauss_equivalent_stress[yielded]
                                       - tube.yield_stress).abs().max()) / tube.yield_stress
    else:
        out["yield_residual"] = 0.0
    elastic = ~yielded
    if bool(elastic.any()):
        out["max_q_over_sigma_y_elastic"] = float(
            solution.gauss_equivalent_stress[elastic].max()) / tube.yield_stress
    else:
        out["max_q_over_sigma_y_elastic"] = float("nan")
    reference_u = float(tube.radial_displacement(
        torch.tensor(tube.inner_radius, dtype=torch.float64), pressure))
    out["u_bore"] = float(solution.displacement[0, 0])
    out["u_bore_reference"] = reference_u
    out["u_bore_rel"] = abs(out["u_bore"] - reference_u) / abs(reference_u)
    return out


def qualify_plastic_tube(
    tube: HillPlaneStrainTube | None = None,
    *,
    mesh_sizes: tuple[int, ...] = (8, 16, 24, 48),
    pressure_ratios: tuple[float, ...] = (1.0, 1.15),
    elastic_ratio: float = 0.5,
    steps: int = 6,
    grading: float = 0.0,
) -> dict[str, object]:
    """Run the mesh/pressure matrix behind ``docs/thick-cylinder-von-mises-hill.md``.

    Returns a JSON-serialisable dictionary: one row per ``(mesh, pressure)`` plus
    an elastic (pre-yield) element-check row.  ``pressure_ratios`` are multiples
    of the shear yield stress ``k``; ``p_y = 0.75 k`` and ``p_p = 1.386 k`` for the
    default geometry, so every ratio here lies strictly inside the contained
    plastic range.
    """
    tube = tube or HillPlaneStrainTube()
    law = HardeningLaw(tube.yield_stress)
    rows = []

    elastic_pressure = elastic_ratio * tube.pressure_first_yield
    for n_r in mesh_sizes:
        mesh = annulus_mesh(n_r, grading=grading)
        solution = solve_plastic_tube(tube, mesh, law, elastic_pressure, steps=4)
        errors = field_errors(solution, tube, elastic_pressure)
        rows.append({
            "case": "elastic_element_check",
            "n_r": n_r,
            "pressure": elastic_pressure,
            "pressure_over_k": elastic_pressure / tube.k,
            "yielded_fraction": 0.0,
            "errors": errors,
            "iterations": solution.iterations,
            "grading": grading,
        })

    for ratio in pressure_ratios:
        pressure = ratio * tube.k
        reference = tube.plastic_radius(pressure)
        for n_r in mesh_sizes:
            mesh = annulus_mesh(n_r, grading=grading)
            solution = solve_plastic_tube(tube, mesh, law, pressure, steps=steps)
            estimate = solution.plastic_radius(tube.yield_stress)
            errors = field_errors(solution, tube, pressure)
            rows.append({
                "case": "elastic_plastic",
                "n_r": n_r,
                "pressure": pressure,
                "pressure_over_k": ratio,
                "r_p_reference": reference,
                "r_p_bracket": estimate["bracket"],
                "r_p_lame_fit": estimate["lame_fit"],
                "r_p_bracket_error": estimate["bracket"] / reference - 1.0,
                "r_p_error": estimate["lame_fit"] / reference - 1.0,
                "yielded_fraction": estimate["plastic_fraction"],
                "errors": errors,
                "iterations": solution.iterations,
                "grading": grading,
                "residual_history": list(solution.residual_history),
                "final_residual": solution.residual,
            })
    return {
        "geometry": {"inner_radius": tube.inner_radius, "outer_radius": tube.outer_radius,
                     "young": tube.young, "poisson": tube.poisson,
                     "yield_stress": tube.yield_stress},
        "pressure_first_yield": tube.pressure_first_yield,
        "pressure_limit": tube.pressure_limit,
        "shear_yield_stress": tube.k,
        "rows": rows,
    }
