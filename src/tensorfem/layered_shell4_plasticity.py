"""Small-strain layered Shell4 with transactional plane-stress J2 points.

This module is deliberately limited to flat-facet, small-strain Mindlin shells.
Membrane strains and curvatures are evaluated at 2x2 in-plane Gauss points and
mapped to independent material points through the thickness.  Transverse shear
and drilling stabilization remain elastic.  It is not a finite-rotation or
post-buckling formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .plasticity import J2State, update_j2


def layered_shell4_local_frame(
    xyz: torch.Tensor, *, maximum_warp_ratio: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a shallow warped Q4 facet onto its best-fit reference plane.

    Initial plate imperfections generally make the four corners non-coplanar.
    A flat-facet constitutive law can still represent such a mesh when each
    facet is shallow: its integration geometry is the orthogonal projection
    onto the total-least-squares plane.  The normal sign follows the element
    node ordering and the first axis is the projected 0--1 edge, making the
    construction deterministic and objective under rigid transformations.

    This is deliberately *not* an arbitrary warped-shell formulation.  Deeply
    folded facets are rejected rather than silently projected.
    """
    if xyz.shape != (4, 3):
        raise ValueError("xyz must have shape (4, 3)")
    centred = xyz - xyz.mean(0)
    _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
    scale = singular[0]
    if bool(scale <= torch.finfo(xyz.dtype).eps):
        raise ValueError("degenerate layered Shell4 element")
    normal = vh[-1]
    ordered_normal = torch.linalg.cross(xyz[1]-xyz[0], xyz[3]-xyz[0], dim=0)
    if bool(torch.dot(normal, ordered_normal) < 0):
        normal = -normal
    edge = xyz[1] - xyz[0]
    e1 = edge - torch.dot(edge, normal) * normal
    e1_norm = torch.linalg.vector_norm(e1)
    if bool(e1_norm <= torch.finfo(xyz.dtype).eps * scale):
        raise ValueError("degenerate layered Shell4 0-1 edge")
    e1 = e1 / e1_norm
    e2 = torch.linalg.cross(normal, e1, dim=0)
    basis = torch.stack((e1, e2, normal))
    local = centred @ basis.T
    in_plane_scale = torch.linalg.vector_norm(local[:, :2], dim=1).max()
    if bool(in_plane_scale <= torch.finfo(xyz.dtype).eps):
        raise ValueError("degenerate layered Shell4 projected area")
    warp = local[:, 2].abs().max() / in_plane_scale
    if bool(warp > maximum_warp_ratio):
        raise ValueError(
            f"layered Shell4 warp ratio {float(warp):.3g} exceeds "
            f"limit {maximum_warp_ratio:.3g}"
        )
    return local[:, :2], basis


@dataclass(frozen=True)
class LayeredShell4Model:
    nodes: torch.Tensor
    elements: torch.Tensor
    young: float
    poisson: float
    thickness: float
    yield_stress: float
    hardening: float
    layers: int = 5
    shear_correction: float = 5.0 / 6.0
    drilling_factor: float = 1e-6
    residual_stress: torch.Tensor | None = None

    def validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape [n,3]")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("elements must have shape [m,4]")
        if self.elements.numel() == 0 or bool(torch.any(self.elements < 0)) or int(self.elements.max()) >= len(self.nodes):
            raise ValueError("invalid shell connectivity")
        if self.young <= 0 or self.thickness <= 0 or self.yield_stress <= 0 or self.hardening < 0:
            raise ValueError("invalid material constants")
        if not (-1.0 < self.poisson < .5) or self.layers < 1:
            raise ValueError("invalid poisson ratio or layer count")
        if self.residual_stress is not None:
            expected = (len(self.elements), 4, self.layers, 3)
            if self.residual_stress.shape != expected:
                raise ValueError(f"residual_stress must have shape {expected}")
            if self.residual_stress.dtype != self.nodes.dtype or self.residual_stress.device != self.nodes.device:
                raise ValueError("residual_stress must share node dtype and device")
            if not bool(torch.all(torch.isfinite(self.residual_stress))):
                raise ValueError("residual_stress must be finite")

    @property
    def n_dofs(self) -> int:
        return 6 * len(self.nodes)


@dataclass(frozen=True)
class LayeredShell4State:
    points: tuple[tuple[tuple[J2State, ...], ...], ...]

    @classmethod
    def virgin(cls, model: LayeredShell4Model) -> "LayeredShell4State":
        model.validate()
        zero = torch.zeros((3, 3), dtype=model.nodes.dtype, device=model.nodes.device)
        alpha = model.nodes.new_zeros(())
        return cls(tuple(tuple(tuple(J2State(zero.clone(), alpha.clone())
                                     for _ in range(model.layers)) for _ in range(4))
                         for _ in model.elements))


@dataclass(frozen=True)
class LayeredShell4Step:
    load_factor: float
    displacement: torch.Tensor
    reaction: torch.Tensor
    stress: torch.Tensor
    alpha: torch.Tensor
    state: LayeredShell4State
    iterations: int
    residual_norm: float


def _tensor_from_inplane(value: torch.Tensor, eps_zz: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        torch.stack((value[0], .5 * value[2], value.new_zeros(()))),
        torch.stack((.5 * value[2], value[1], value.new_zeros(()))),
        torch.stack((value.new_zeros(()), value.new_zeros(()), eps_zz)),
    ))


def _residual_strain(stress: torch.Tensor, young: float, poisson: float) -> torch.Tensor:
    """3-D elastic strain producing an admissible in-plane residual stress."""
    sx, sy, tau = stress
    return _tensor_from_inplane(torch.stack(((sx-poisson*sy)/young,
                                             (sy-poisson*sx)/young,
                                             tau/(young/(2*(1+poisson))))),
                                -poisson*(sx+sy)/young)


def plane_stress_j2_update(strain: torch.Tensor, model: LayeredShell4Model,
                           committed: J2State, residual_stress: torch.Tensor | None = None,
                           *, tangent: bool = True):
    """Return ``[sx,sy,txy]``, numerical consistent tangent and trial state.

    The out-of-plane strain is locally condensed by enforcing ``sigma_zz=0``.
    The tangent is the derivative of that same fully condensed update, rather
    than the uncondensed 3-D material matrix.
    """
    if strain.shape != (3,):
        raise ValueError("plane-stress strain must be [ex,ey,gxy]")
    initial = torch.zeros((3, 3), dtype=strain.dtype, device=strain.device)
    if residual_stress is not None:
        initial = _residual_strain(residual_stress, model.young, model.poisson)

    def response(value: torch.Tensor):
        # Elastic plane-stress value is a reliable initial guess, including
        # plastic loading because the scalar local solve is convex.
        zz = -model.poisson/(1-model.poisson) * (value[0]+value[1])
        for _ in range(30):
            full = _tensor_from_inplane(value, zz) + initial
            sigma, trial, _ = update_j2(full, model.young, model.poisson,
                                        model.yield_stress, model.hardening, committed)
            residual = sigma[2, 2]
            if abs(float(residual)) <= 2e-12 * max(model.yield_stress, 1.0):
                break
            h = math.sqrt(torch.finfo(value.dtype).eps) * max(1.0, abs(float(zz)))
            hp = value.new_tensor(h)
            sp = update_j2(_tensor_from_inplane(value, zz+hp)+initial,
                           model.young, model.poisson, model.yield_stress,
                           model.hardening, committed)[0][2, 2]
            sm = update_j2(_tensor_from_inplane(value, zz-hp)+initial,
                           model.young, model.poisson, model.yield_stress,
                           model.hardening, committed)[0][2, 2]
            derivative = (sp-sm)/(2*hp)
            if abs(float(derivative)) < torch.finfo(value.dtype).eps:
                raise RuntimeError("plane-stress local condensation has singular tangent")
            zz = zz-residual/derivative
        else:
            raise RuntimeError("plane-stress local condensation did not converge")
        return torch.stack((sigma[0, 0], sigma[1, 1], sigma[0, 1])), trial

    stress, trial = response(strain)
    if not tangent:
        return stress, None, trial
    # Symmetric difference differentiates the actual return map including the
    # plane-stress condensation and active plastic branch.
    columns = []
    # A strain-scale step avoids cancellation from the nested local Newton
    # solve (sqrt(eps) is too small for this condensed finite difference).
    base = torch.finfo(strain.dtype).eps ** (1.0/3.0)
    for j in range(3):
        h = base * max(1.0, abs(float(strain[j])))
        delta = torch.zeros_like(strain); delta[j] = h
        columns.append((response(strain+delta)[0]-response(strain-delta)[0])/(2*h))
    tangent = torch.stack(columns, 1)
    return stress, tangent, trial


def _shape_derivatives(xy: torch.Tensor, xi: float, eta: float):
    natural = xy.new_tensor([[-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
                             [(1+eta), (1+xi)], [-(1+eta), (1-xi)]]) / 4
    jacobian = natural.T @ xy
    determinant = torch.linalg.det(jacobian)
    if bool(determinant <= 0):
        raise ValueError("Shell4 element has non-positive Jacobian")
    return natural @ torch.linalg.inv(jacobian), determinant


def _generalized_b(xy: torch.Tensor, xi: float, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    derivative, determinant = _shape_derivatives(xy, xi, eta)
    b = torch.zeros((6, 24), dtype=xy.dtype, device=xy.device)
    for i in range(4):
        u, v, rx, ry = 6*i, 6*i+1, 6*i+3, 6*i+4
        dx, dy = derivative[i]
        b[0, u] = dx; b[1, v] = dy; b[2, u] = dy; b[2, v] = dx
        b[3, ry] = dx
        b[4, rx] = -dy
        b[5, ry] = dy; b[5, rx] = -dx
    return b, determinant


def _transform(basis: torch.Tensor, *, dtype, device) -> torch.Tensor:
    transform = torch.zeros((24, 24), dtype=dtype, device=device)
    for node in range(4):
        transform[6*node:6*node+3, 6*node:6*node+3] = basis
        transform[6*node+3:6*node+6, 6*node+3:6*node+6] = basis
    return transform


def _elastic_shear_drilling(xy: torch.Tensor, local: torch.Tensor,
                            model: LayeredShell4Model):
    derivative, determinant = _shape_derivatives(xy, 0., 0.)
    shape = xy.new_tensor([.25, .25, .25, .25])
    bs = torch.zeros((2, 24), dtype=xy.dtype, device=xy.device)
    for i in range(4):
        w, rx, ry = 6*i+2, 6*i+3, 6*i+4
        bs[0, w] = derivative[i, 0]; bs[0, ry] = shape[i]
        bs[1, w] = derivative[i, 1]; bs[1, rx] = -shape[i]
    shear = model.shear_correction*model.young/(2*(1+model.poisson))*model.thickness
    k = bs.T @ (shear*torch.eye(2, dtype=xy.dtype, device=xy.device)) @ bs * determinant*4
    area = determinant*4
    rz = 6*torch.arange(4, device=xy.device)+5
    lap = xy.new_tensor([[2.,-1.,0.,-1.],[-1.,2.,-1.,0.],
                         [0.,-1.,2.,-1.],[-1.,0.,-1.,2.]])
    k[rz[:, None], rz] += model.drilling_factor*model.young*model.thickness*area*lap
    return k @ local, k


def element_response(model: LayeredShell4Model, element: int, local_global: torch.Tensor,
                     committed: tuple[tuple[J2State, ...], ...], *, compute_tangent: bool = True):
    conn = model.elements[element].long(); xyz = model.nodes[conn]
    xy, basis = layered_shell4_local_frame(xyz); transform = _transform(basis, dtype=xyz.dtype, device=xyz.device)
    local = transform @ local_global
    force, tangent = _elastic_shear_drilling(xy, local, model)
    gp = 1/math.sqrt(3); points = ((-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp))
    dz = model.thickness/model.layers
    zvalues = [(-model.thickness/2+(layer+.5)*dz) for layer in range(model.layers)]
    stresses, trial_points = [], []
    residual = model.residual_stress
    for q, (xi, eta) in enumerate(points):
        b, det = _generalized_b(xy, xi, eta); generalized = b @ local
        qstresses, qtrials = [], []
        for layer, z in enumerate(zvalues):
            strain = generalized[:3] + z*generalized[3:]
            initial = None if residual is None else residual[element, q, layer]
            stress, material, trial = plane_stress_j2_update(
                strain, model, committed[q][layer], initial, tangent=compute_tangent
            )
            bz = b[:3] + z*b[3:]
            force += bz.T @ stress * det*dz
            if compute_tangent:
                tangent += bz.T @ material @ bz * det*dz
            qstresses.append(stress); qtrials.append(trial)
        stresses.append(torch.stack(qstresses)); trial_points.append(tuple(qtrials))
    return transform.T@force, transform.T@tangent@transform, torch.stack(stresses), tuple(trial_points)


def _element_dofs(conn: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(6*conn+k for k in range(6)), 1).reshape(-1).long()


def assemble_layered_shell4(model: LayeredShell4Model, displacement: torch.Tensor,
                            committed: LayeredShell4State):
    model.validate()
    if displacement.shape != (model.n_dofs,):
        raise ValueError("wrong displacement vector length")
    internal = torch.zeros_like(displacement)
    tangent = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    stresses, trial = [], []
    for e, conn in enumerate(model.elements):
        ids = _element_dofs(conn)
        force, stiffness, stress, points = element_response(model, e, displacement[ids], committed.points[e])
        internal.index_add_(0, ids, force)
        tangent.index_put_((ids[:, None], ids[None, :]), stiffness, accumulate=True)
        stresses.append(stress); trial.append(points)
    return internal, tangent, torch.stack(stresses), LayeredShell4State(tuple(trial))


def solve_layered_shell4_increment(model: LayeredShell4Model, external_force: torch.Tensor,
                                   load_factor: float, fixed_dofs: torch.Tensor,
                                   committed: LayeredShell4State, displacement: torch.Tensor,
                                   *, tolerance=1e-8, max_iterations=25):
    """Full Newton load increment; committed history is immutable on failure."""
    fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device); mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    load = external_force.to(model.nodes)*load_factor; u = displacement.clone()
    scale = max(float(torch.linalg.vector_norm(load[free])), 1.)
    for iteration in range(1, max_iterations+1):
        internal, tangent, stress, trial = assemble_layered_shell4(model, u, committed)
        residual = load-internal; norm = float(torch.linalg.vector_norm(residual[free]))
        if norm <= tolerance*scale:
            alpha = torch.stack([torch.stack([torch.stack([p.alpha for p in q]) for q in e]) for e in trial.points])
            return LayeredShell4Step(float(load_factor), u, internal-load, stress, alpha,
                                     trial, iteration, norm)
        try:
            increment = torch.linalg.solve(tangent[free][:, free], residual[free])
        except torch.linalg.LinAlgError as error:
            raise RuntimeError("layered Shell4 Newton tangent is singular") from error
        u = u.index_add(0, free, increment)
    raise RuntimeError(f"layered Shell4 Newton failed after {max_iterations} iterations")


def solve_layered_shell4_path(model: LayeredShell4Model, external_force: torch.Tensor,
                              fixed_dofs: torch.Tensor, load_factors, *,
                              initial_state: LayeredShell4State | None = None,
                              initial_displacement: torch.Tensor | None = None,
                              tolerance=1e-8, max_iterations=25):
    state = LayeredShell4State.virgin(model) if initial_state is None else initial_state
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    if initial_displacement is not None:
        u = initial_displacement.to(model.nodes).clone()
    results = []
    for factor in load_factors:
        result = solve_layered_shell4_increment(model, external_force, float(factor), fixed_dofs,
                                                state, u, tolerance=tolerance,
                                                max_iterations=max_iterations)
        results.append(result); state = result.state; u = result.displacement
    return tuple(results)
