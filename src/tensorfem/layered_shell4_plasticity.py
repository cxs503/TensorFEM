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

from .plasticity import (J2State, condense_plane_stress,
                         j2_consistent_tangent, update_j2)


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
    """Return ``[sx,sy,txy]``, exact condensed tangent and trial state.

    The out-of-plane strain is locally condensed by enforcing ``sigma_zz = 0``.
    Both the local condensation and the returned tangent use the analytical
    algorithmic tangent of the radial return (``plasticity.j2_consistent_tangent``
    plus a Schur condensation), never a finite difference of the return map: a
    centred difference straddles the yield surface, so just below yield it mixed
    the elastic and plastic branches and biased the tangent by tens of percent
    exactly where Newton needs it.
    """
    if strain.shape != (3,):
        raise ValueError("plane-stress strain must be [ex,ey,gxy]")
    initial = torch.zeros((3, 3), dtype=strain.dtype, device=strain.device)
    if residual_stress is not None:
        initial = _residual_strain(residual_stress, model.young, model.poisson)
    scale = max(model.young, model.yield_stress, 1.)
    # Both acceptance tests below are scaled by the arithmetic unit round-off and
    # by the size of the quantity actually being cancelled.  Fixed absolute
    # thresholds (formerly 1e-15*E on sigma_zz and 1e-17 on the Newton step) are
    # unreachable in double precision as soon as the trial strain is large: the
    # Newton iteration then stalls inside one unit in the last place of eps_zz
    # with a residual that is already pure round-off, and the element aborted with
    # "plane-stress local condensation did not converge" even though the
    # condensed state was exact to working precision (issue #8).
    unit_roundoff = torch.finfo(strain.dtype).eps
    accept = 256. * unit_roundoff

    def response(value: torch.Tensor):
        """Stress, exact condensed tangent and trial state at in-plane ``value``."""
        zz = -model.poisson/(1-model.poisson) * (value[0]+value[1])
        for _ in range(40):
            full = _tensor_from_inplane(value, zz) + initial
            sigma, trial, dgamma = update_j2(
                full, model.young, model.poisson, model.yield_stress,
                model.hardening, committed)
            material = j2_consistent_tangent(
                full, model.young, model.poisson, model.hardening, committed, dgamma)
            residual = float(sigma[2, 2])
            stress_scale = max(scale, float(sigma.abs().max()))
            if abs(residual) <= accept * stress_scale:
                break
            slope = float(material[2, 2])
            if not math.isfinite(slope) or slope <= 0.:
                raise RuntimeError(
                    "plane-stress local condensation has a non-positive "
                    f"sigma_zz tangent ({slope:g})")
            step = residual / slope
            zz -= step
            if abs(step) <= accept * max(1.0, abs(zz)):
                full = _tensor_from_inplane(value, zz) + initial
                sigma, trial, dgamma = update_j2(
                    full, model.young, model.poisson, model.yield_stress,
                    model.hardening, committed)
                material = j2_consistent_tangent(
                    full, model.young, model.poisson, model.hardening, committed, dgamma)
                break
        else:
            raise RuntimeError("plane-stress local condensation did not converge")
        stress = torch.stack((sigma[0, 0], sigma[1, 1], sigma[0, 1]))
        return stress, condense_plane_stress(material), trial

    stress, material, trial = response(strain)
    if not tangent:
        return stress, None, trial
    return stress, material, trial


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
                     committed: tuple[tuple[J2State, ...], ...], *, compute_tangent: bool = True,
                     geometric: bool = False):
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
    if geometric and compute_tangent:
        # Updated geometric stiffness: in-plane resultants from the
        # current trial stress, positive in compression, subtracted
        # from the tangent (K_E + K_M - K_G).
        nq = -torch.stack([s.sum(dim=0) for s in stresses]) * dz
        nodal = torch.stack([_q4_shape_values(k) for k in range(4)]).to(dtype=xyz.dtype) @ nq
        tangent -= layered_shell4_geometric_stiffness(model, element, nodal)
    return transform.T@force, transform.T@tangent@transform, torch.stack(stresses), tuple(trial_points)


def _element_dofs(conn: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(6*conn+k for k in range(6)), 1).reshape(-1).long()


def assemble_layered_shell4(model: LayeredShell4Model, displacement: torch.Tensor,
                            committed: LayeredShell4State, *, geometric: bool = False):
    model.validate()
    if displacement.shape != (model.n_dofs,):
        raise ValueError("wrong displacement vector length")
    internal = torch.zeros_like(displacement)
    tangent = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    stresses, trial = [], []
    for e, conn in enumerate(model.elements):
        ids = _element_dofs(conn)
        force, stiffness, stress, points = element_response(model, e, displacement[ids],
                committed.points[e], geometric=geometric)
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


# ---------------------------------------------------------------------------
# Geometric (initial-stress) stiffness
# ---------------------------------------------------------------------------

def _q4_shape_values(gi: int) -> torch.Tensor:
    """Bilinear Q4 shape values at the 2x2 Gauss point ``gi``."""
    gp = 1.0 / math.sqrt(3.0)
    xi, eta = ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp))[gi]
    return torch.tensor([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)],
                        dtype=torch.float64) / 4.0


def _q4_natural_gradients(xi: float, eta: float, dtype, device) -> torch.Tensor:
    """4x2 matrix of dN/d(xi), dN/d(eta) for the bilinear Q4."""
    return torch.tensor([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                         [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]],
                        dtype=dtype, device=device) / 4.0


def layered_shell4_geometric_stiffness(model: LayeredShell4Model, element: int,
                                       nodal_resultants: torch.Tensor) -> torch.Tensor:
    """24x24 initial-stress matrix for one element.

    ``nodal_resultants`` is a finite ``(4, 3)`` tensor of in-plane force
    resultants per unit length at the element nodes, ``[Nx, Ny, Nxy]``,
    **positive in compression**.  The returned matrix is positive
    semidefinite for compression; it is subtracted from the material tangent
    in the nonlinear path and used as ``K - lambda*Kg`` in the linearised
    eigenvalue problem (same construction as
    :func:`tensorfem.shell4_buckling.shell4_geometric_stiffness`, but in the
    layered element's local frame).
    """
    conn = model.elements[element].long()
    xyz = model.nodes[conn]
    xy, basis = layered_shell4_local_frame(xyz)
    n = torch.as_tensor(nodal_resultants, dtype=xyz.dtype, device=xyz.device)
    if n.shape != (4, 3) or not bool(torch.all(torch.isfinite(n))):
        raise ValueError("nodal_resultants must be a finite (4, 3) tensor")
    gp = 1.0 / math.sqrt(3.0)
    points = ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp))
    kg4 = xyz.new_zeros((4, 4))
    for k in range(4):
        dnat = _q4_natural_gradients(*points[k], xyz.dtype, xyz.device)
        j = xy.T @ dnat
        detj = torch.linalg.det(j)
        if bool(detj <= torch.finfo(xyz.dtype).eps):
            raise ValueError("inverted or degenerate shell element")
        grad = dnat @ torch.linalg.inv(j)
        nloc = _q4_shape_values(k).to(dtype=xyz.dtype, device=xyz.device) @ n
        smat = torch.stack((torch.stack((nloc[0], nloc[2])),
                           torch.stack((nloc[2], nloc[1]))))
        kg4 += grad @ smat @ grad.T * detj
    local = xyz.new_zeros((24, 24))
    w = 6 * torch.arange(4, device=xyz.device) + 2
    local[w[:, None], w] = kg4
    transform = _transform(basis, dtype=xyz.dtype, device=xyz.device)
    return transform.T @ local @ transform


def layered_shell4_buckling_eigenvalue(model: LayeredShell4Model, fixed_dofs: torch.Tensor,
                                       unit_resultant: torch.Tensor, modes: int = 3):
    """Lowest buckling loads of ``(K_E - lambda*K_G) v = 0``.

    ``unit_resultant`` is the in-plane resultant per unit length (3,) for a
    stress of one pascal, e.g. ``[t, 0, 0]`` for uniaxial compression; the
    eigenvalue ``lambda`` is therefore the critical stress in pascals.  The
    pencil is reduced to the bending space ``(w, rx, ry)`` and solved with the
    Cholesky mapping of :func:`tensorfem.stiffened_panel_buckling._solve_pencil`.
    Returns ``(loads, w_fields, residuals)`` where ``w_fields`` is
    ``(modes, n_nodes)`` of the transverse displacement mode shapes.
    """
    from .stiffened_panel_buckling import _solve_pencil

    model.validate()
    fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]

    internal, tangent, _stress, _trial = assemble_layered_shell4(
        model, torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device),
        LayeredShell4State.virgin(model))
    stiffness = 0.5 * (tangent + tangent.T)

    n_nodes = len(model.nodes)
    geometric = torch.zeros_like(stiffness)
    unit = torch.as_tensor(unit_resultant, dtype=model.nodes.dtype, device=model.nodes.device)
    if unit.shape != (3,):
        raise ValueError("unit_resultant must have shape (3,)")
    nodal = unit.reshape(1, 3).expand(4, 3).clone()
    for e in range(len(model.elements)):
        kg = layered_shell4_geometric_stiffness(model, e, nodal)
        ids = _element_dofs(model.elements[e].long())
        geometric.index_put_((ids[:, None], ids[None, :]), kg, accumulate=True)

    # restrict to the bending space (w, rx, ry) on free dofs
    index = torch.cat([6 * torch.arange(n_nodes) + k for k in (2, 3, 4)])
    is_bending = torch.zeros(stiffness.shape[0], dtype=torch.bool)
    is_bending[index] = True
    keep = free[is_bending[free]]
    k_bend = stiffness[keep[:, None], keep[None, :]].double()
    g_bend = torch.zeros_like(k_bend)
    # g_bend must be the w-w block of the full-space geometric matrix, mapped
    # into the keep space (same construction as
    # tensorfem.stiffened_panel_buckling._bending_pencil).
    w_full = 6 * torch.arange(n_nodes) + 2
    lookup = {int(dof): i for i, dof in enumerate(keep.tolist())}
    w_pairs = [(int(d), lookup[int(d)]) for d in w_full.tolist() if int(d) in lookup]
    rows = torch.tensor([p[1] for p in w_pairs], dtype=torch.long)
    cols = torch.tensor([p[0] for p in w_pairs], dtype=torch.long)
    g_bend[rows[:, None], rows[None, :]] = geometric[cols, :][:, cols].double()
    loads, vectors, residuals = _solve_pencil(k_bend, g_bend, modes)
    w_rows = [i for i, dof in enumerate(keep.tolist()) if int(dof) % 6 == 2]
    nodes = (keep[w_rows] // 6).numpy()
    fields = torch.zeros((len(loads), n_nodes), dtype=torch.float64)
    for column in range(vectors.shape[1]):
        amplitude = torch.as_tensor(vectors[w_rows, column], dtype=torch.float64)
        peak = float(amplitude.abs().max())
        if peak > 0:
            amplitude = amplitude / peak
        fields[column, nodes] = amplitude
    return torch.as_tensor(loads, dtype=torch.float64), fields, torch.as_tensor(residuals, dtype=torch.float64)
