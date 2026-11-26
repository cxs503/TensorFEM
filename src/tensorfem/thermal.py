"""Small, differentiable heat-conduction and sequential thermoelastic kernels.

The implementation intentionally mirrors the dense reference solvers used by
TensorFEM's verified element kernels.  It supports LINE2 and Q4 conduction,
consistent heat capacity, prescribed temperature, nodal heat input, boundary
convection and element volumetric heat generation.
"""
import math
from dataclasses import dataclass
from collections.abc import Callable

import torch


@dataclass(frozen=True)
class ThermalModel:
    """Q4 conductivity also accepts SPD tensors shaped (1, 2, 2) or (ne, 2, 2) in global coordinates."""

    nodes: torch.Tensor
    elements: torch.Tensor
    conductivity: torch.Tensor
    density: torch.Tensor
    specific_heat: torch.Tensor
    fixed_nodes: torch.Tensor
    fixed_temperatures: torch.Tensor
    nodal_heat: torch.Tensor | None = None
    # W/m³: uniform, per-element constant, or (ne, nen) interpolated within each element.
    heat_source: torch.Tensor | None = None
    thickness: torch.Tensor | None = None
    convection_edges: torch.Tensor | None = None
    convection_coefficient: torch.Tensor | None = None
    # Scalar, per-edge constant, or (n_edges, 2) endpoint values in edge order.
    ambient_temperature: torch.Tensor | None = None
    element_type: str = "line2"
    # Lumped two-node contacts: pair distinct temperature DOFs across an
    # interface. Each conductance is tributary area / areal resistance [W/K].
    contact_pairs: torch.Tensor | None = None
    contact_conductance: torch.Tensor | None = None
    # Additional concentrated storage in J/K, one entry per global node.
    nodal_heat_capacity: torch.Tensor | None = None
    # Consistent surface storage in J/(m² K), scalar or one value per convection edge.
    wall_areal_heat_capacity: torch.Tensor | None = None
    # Optional total exposed area in m², scalar or per edge (e.g. a stepped shoulder).
    convection_area: torch.Tensor | None = None
    # H in W/(m³ K), scalar, per-element or (ne, nen) local nodal values.
    # The bulk sink is H*T; add H*Ta
    # to heat_source to exchange with a prescribed ambient temperature Ta.
    heat_loss_coefficient: torch.Tensor | None = None

    # LINE2 additive A*x², scalar or per element in W/m⁵.
    # Q4 additive A*x²+B*x*y+C*y² volumetric heating in global coordinates.
    # Coefficients in W/m⁵, shared (3,) or element-specific (ne, 3).
    heat_source_quadratic: torch.Tensor | None = None

    # Conforming Q4 contact edges (n,2,2): two traces with endpoints in
    # matching physical order. Total edge conductance is area/R in W/K.
    contact_edge_pairs: torch.Tensor | None = None
    # Optional (n,2) profile: each endpoint coefficient is h(endpoint)*edge area
    # in W/K, linearly interpolated along the edge.
    contact_edge_conductance: torch.Tensor | None = None

    def __post_init__(self):
        nen = {"line2": 2, "q4": 4}.get(self.element_type.lower())
        if nen is None:
            raise ValueError("element_type must be 'line2' or 'q4'")
        dim = 1 if self.element_type.lower() == "line2" else 2
        if self.nodes.ndim != 2 or self.nodes.shape[1] != dim:
            raise ValueError(f"nodes must have shape [n, {dim}]")
        if self.elements.ndim != 2 or self.elements.shape[1] != nen:
            raise ValueError(f"elements must have shape [ne, {nen}]")
        if self.fixed_nodes.numel() != self.fixed_temperatures.numel():
            raise ValueError("fixed_nodes and fixed_temperatures must match")
        if self.elements.numel() and (self.elements.min() < 0 or self.elements.max() >= len(self.nodes)):
            raise ValueError("invalid element connectivity")
        if (self.contact_pairs is None) != (self.contact_conductance is None):
            raise ValueError("contact_pairs and contact_conductance must be supplied together")
        if self.contact_pairs is not None:
            pairs = self.contact_pairs
            if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype not in (torch.int32, torch.int64):
                raise ValueError("contact_pairs must be an integer tensor of shape [n_contacts, 2]")
            if torch.any(pairs < 0) or torch.any(pairs >= len(self.nodes)) or torch.any(pairs[:, 0] == pairs[:, 1]):
                raise ValueError("contacts must connect distinct, valid nodes")
        if (self.contact_edge_pairs is None) != (self.contact_edge_conductance is None):
            raise ValueError("supply both contact_edge_pairs and contact_edge_conductance")
        if self.contact_edge_pairs is not None:
            edges = self.contact_edge_pairs
            if (nen != 4 or edges.ndim != 3 or edges.shape[1:] != (2,2)
                    or edges.dtype not in (torch.int32,torch.int64)):
                raise ValueError("contact_edge_pairs requires Q4 endpoint indices shaped (n,2,2)")
            if (edges < 0).any() or (edges >= len(self.nodes)).any():
                raise ValueError("invalid contact edge indices")
            edges = edges.to(self.nodes.device).long()
            if (edges.flatten(1).sort(1).values.diff(dim=1) == 0).any():
                raise ValueError("each contact edge pair requires four distinct nodes")
            a,b = self.nodes[edges].unbind(1)
            lengths = (a[:,1]-a[:,0]).norm(dim=1)
            if (lengths <= 0).any() or ((a-b).norm(dim=2) >
                    128*torch.finfo(a.dtype).eps*lengths[:,None]).any():
                raise ValueError("contact edge endpoints must coincide in matching order")
            # A trace must be an edge of a Q4, not a diagonal or an interior line.
            def edge_keys(start,end):
                return torch.minimum(start,end)*len(self.nodes)+torch.maximum(start,end)
            conn = self.elements.to(edges.device).long()
            keys,counts = torch.unique(edge_keys(conn,conn.roll(-1,1)),return_counts=True)
            if not torch.isin(edge_keys(edges[:,:,0],edges[:,:,1]),keys[counts==1]).all():
                raise ValueError("contact traces must be exposed element edges")
        if self.convection_area is not None and self.convection_edges is None:
            raise ValueError("convection_area requires convection_edges")
        if self.wall_areal_heat_capacity is not None and self.convection_edges is None:
            raise ValueError("wall_areal_heat_capacity requires convection_edges")
        if self.convection_edges is not None:
            if self.convection_edges.ndim != 2 or self.convection_edges.shape[1] != 2:
                raise ValueError("convection_edges must have shape [n_edges, 2]")
            if self.convection_coefficient is None or self.ambient_temperature is None:
                raise ValueError("convection coefficient and ambient temperature are required")


@dataclass(frozen=True)
class ThermalMatrices:
    conductivity: torch.Tensor
    capacity: torch.Tensor
    heat: torch.Tensor


@dataclass(frozen=True)
class ThermalTransientResult:
    time: torch.Tensor
    temperature: torch.Tensor
    matrices: ThermalMatrices
    theta: float = 1.0

    # Extra nodal power (W) averaged by the theta scheme on each interval.
    additional_stage_power: torch.Tensor | None = None

    # Actual stage-weighted temperatures for multistage integration.
    stage_temperature: torch.Tensor | None = None

    @property
    def interval_temperature(self) -> torch.Tensor:
        if self.stage_temperature is not None:
            return self.stage_temperature
        return (1-self.theta)*self.temperature[:-1] + self.theta*self.temperature[1:]

    def nodal_reaction_heat_steps(self) -> torch.Tensor:
        """Return inward reaction heat per interval and node, shape (nt-1, nn)."""
        t = self.temperature
        dt = self.time.diff().to(t)[:, None]
        stages = self.interval_temperature
        m = self.matrices
        applied = m.heat if self.additional_stage_power is None else m.heat + self.additional_stage_power
        return ((m.capacity @ t.diff(dim=0).T + m.conductivity @ (dt*stages).T).T
                - dt*applied)

    def integrated_nodal_reactions(self) -> torch.Tensor:
        """Heat supplied by prescribed temperatures over the returned time interval.

        Positive heat enters the model. Free-node entries measure residuals
        of the time-discrete equations. Applied loads are already subtracted.
        """
        t = self.temperature
        dt = self.time.diff().to(t)
        stages = self.interval_temperature
        m = self.matrices
        applied_heat = dt.sum() * m.heat
        if self.additional_stage_power is not None:
            applied_heat = applied_heat + dt @ self.additional_stage_power
        return (m.capacity @ (t[-1] - t[0]) + m.conductivity @ (dt @ stages)
                - applied_heat)


def _expand(value, ne, like, default=1.0):
    if value is None:
        return like.new_full((ne,), default)
    return torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(-1).expand(ne)


def _element_conductivity(model):
    x, ne = model.nodes, len(model.elements)
    material = torch.as_tensor(1.0 if model.conductivity is None else model.conductivity,
                               dtype=x.dtype, device=x.device)
    anisotropic = model.element_type.lower() == "q4" and material.ndim == 3 and material.shape[-2:] == (2, 2)
    if anisotropic:
        if material.shape[0] not in (1, ne):
            raise ValueError("Q4 conductivity needs one tensor or one per element")
        if not torch.allclose(material, material.transpose(-1, -2)) or not torch.isfinite(material).all():
            raise ValueError("Q4 conductivity tensors must be finite and symmetric")
        if torch.any(torch.linalg.cholesky_ex(material).info != 0):
            raise ValueError("Q4 conductivity tensors must be positive definite")
        return material
    return _expand(material, ne, x)


def _q4_geometry(coords, xi, eta):
    n = coords.new_tensor([(1-xi)*(1-eta),(1+xi)*(1-eta),
                      (1+xi)*(1+eta),(1-xi)*(1+eta)]) / 4
    dn = coords.new_tensor([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],
                       [(1+eta),(1+xi)],[-(1+eta),(1-xi)]]) / 4
    jac = torch.einsum("eia,ib->eab", coords, dn)
    det = torch.linalg.det(jac)
    if torch.any(det <= 0):
        raise ValueError("Q4 element has non-positive Jacobian")
    grad = torch.einsum("ib,ebc->eic", dn, torch.linalg.inv(jac))
    return n, grad, det


def thermal_element_flux(model: ThermalModel, temperature: torch.Tensor, *, xi=0., eta=0.) -> torch.Tensor:
    """Return global Fourier heat flux (..., element, direction) in W/m².

    Temperature has shape (..., node), including transient (time, node) histories.

    For Q4, (xi, eta) selects one point of the parent square [-1,1]² in
    every element. LINE2 flux is constant. Complex input gives flux phasors.
    """
    if temperature.ndim < 1 or temperature.shape[-1] != len(model.nodes):
        raise ValueError("temperature must have one value per node")
    if not (-1 <= xi <= 1 and -1 <= eta <= 1):
        raise ValueError("parent coordinates must be in [-1,1]")
    conn = model.elements.long()
    coords = model.nodes[conn]
    values = temperature.to(device=coords.device)[..., conn]
    k = _element_conductivity(model)
    dtype = torch.promote_types(values.dtype, coords.dtype)
    if model.element_type.lower() == "line2":
        length = coords[:,1,0] - coords[:,0,0]
        if torch.any(length <= 0):
            raise ValueError("LINE2 element must have positive length")
        return (-k * (values[...,1].to(dtype)-values[...,0].to(dtype)) / length)[...,None]
    _, grad, _ = _q4_geometry(coords, xi, eta)
    gradient = torch.einsum("...ei,eia->...ea", values.to(dtype), grad.to(dtype))
    if k.ndim == 3:
        return -torch.einsum("eij,...ej->...ei", k.to(dtype), gradient)
    return -k[:,None]*gradient


def thermal_element_edge_power(model: ThermalModel, temperature: torch.Tensor) -> torch.Tensor:
    """Integrate outward Fourier power (W) on each Q4 edge using two Gauss points.

    Temperature has shape (..., node) and power has shape (..., element, 4).
    Columns follow local node pairs (0,1), (1,2), (2,3), (3,0). Thickness
    is per element. Adjacent elements use their own one-sided gradient, so
    their reported interior powers can differ in magnitude on a finite mesh.
    """
    if model.element_type.lower() != "q4":
        raise ValueError("edge power requires Q4 elements")
    coords = model.nodes[model.elements.long()]
    tangent = coords.roll(-1, dims=1)-coords
    normal_measure = torch.stack((tangent[:,:,1], -tangent[:,:,0]), dim=-1)
    g = 3.**-.5
    rules = (((-g,-1.), (g,-1.)), ((1.,-g), (1.,g)),
             ((g,1.), (-g,1.)), ((-1.,g), (-1.,-g)))
    powers = []
    for edge, points in enumerate(rules):
        flux = sum(thermal_element_flux(model, temperature, xi=xi, eta=eta)
                   for xi, eta in points)/2
        powers.append((flux*normal_measure[:,edge]).sum(dim=-1))
    thickness = _expand(model.thickness, len(coords), model.nodes)
    return torch.stack(powers, dim=-1)*thickness[:,None]


def recover_linear_edge_flux(nodal_power: torch.Tensor, edge_areas: torch.Tensor, *, closed: bool = False) -> torch.Tensor:
    """Recover continuous piecewise-linear normal flux (W/m²) on an ordered edge chain.

    Supply oriented powers (W) against each ordered node's linear edge shape
    function, shaped (..., n+1), and positive areas (length times thickness)
    for its n consecutive straight edges. For a loop set closed=True, supply n
    distinct nodes and n areas, and place the closing edge last. A single nodal
    value represents the normal flux on both incident edges at every vertex.
    To recover an interface trace from
    element powers, assemble one material side and subtract any other loads
    at its interface nodes, including exterior loads at the chain endpoints.
    Output has the input power shape and orientation. The product of each
    edge area and its two endpoint fluxes' average gives its integrated power.
    """
    power = torch.as_tensor(nodal_power)
    if power.ndim < 1 or power.shape[-1] < 2 or not (power.is_floating_point() or power.is_complex()):
        raise ValueError("supply real or complex nodal powers for at least two ordered nodes")
    area = torch.as_tensor(edge_areas, device=power.device)
    if area.is_complex() or area.shape != (power.shape[-1] if closed else power.shape[-1]-1,) or not torch.isfinite(area).all() or (area <= 0).any():
        raise ValueError("supply one positive finite real area per edge")
    if not torch.isfinite(power).all():
        raise ValueError("nodal powers must be finite")
    area = area.to(power.real.dtype)
    off = area/6
    if closed:
        if len(area) < 3:
            raise ValueError("a closed boundary requires at least three distinct nodes")
        diagonal = (area+area.roll(1))/3
        # Remove the rank-one closing term w*(e0+en)*(e0+en)^T, solve
        # the resulting tridiagonal system, then restore that term.
        ends = torch.zeros_like(area)
        ends[0] = ends[-1] = 1
        weight = off[-1]
        diagonal = diagonal-weight*ends
        field = _edge_mass_solve(diagonal, off[:-1], power)
        update = _edge_mass_solve(diagonal, off[:-1], ends)
        scale = weight*(field[...,0]+field[...,-1])/(1+weight*(update[0]+update[-1]))
        return field-scale[...,None]*update
    diagonal = torch.cat((area[:1], area[:-1]+area[1:], area[-1:]))/3
    return _edge_mass_solve(diagonal, off, power)


def _edge_mass_solve(diagonal, off, power):
    """Thomas sweep for the positive definite edge mass tridiagonal."""
    pivot = diagonal[0]
    rhs = [power[..., 0]/pivot]
    upper = []
    for i in range(len(off)):
        upper.append(off[i]/pivot)
        pivot = diagonal[i+1]-off[i]*upper[-1]
        rhs.append((power[..., i+1]-off[i]*rhs[-1])/pivot)
    flux = [rhs[-1]]
    for i in reversed(range(len(off))):
        flux.append(rhs[i]-upper[i]*flux[-1])
    return torch.stack(flux[::-1], dim=-1)


def recover_q4_boundary_power(model, temperature, boundary_elements, interior_elements, local_edge):
    """Estimate outward Fourier power (W) on a row of Q4 boundary edges.

    Pair each boundary element with the next inward element of the same
    material. Both must be parallelograms sharing the opposite edge, with
    centers collinear with the wall midpoint. Their widths may differ.
    Extrapolate center fluxes to the wall midpoint and integrate using the
    physical edge vector and boundary element thickness. This postprocessed
    estimate is independent of the assembled nodal reaction.
    """
    if model.element_type.lower() != "q4" or local_edge not in (0,1,2,3):
        raise ValueError("provide Q4 elements and a local edge index from 0 to 3")
    x = model.nodes
    ids = []
    for supplied in (boundary_elements,interior_elements):
        index = torch.as_tensor(supplied,device=x.device)
        if (index.ndim != 1 or index.dtype not in (torch.int32,torch.int64)
                or torch.any(index < 0) or torch.any(index >= len(model.elements))):
            raise ValueError("element indices must be valid integer vectors")
        ids.append(index.long())
    outer,inner = ids
    if outer.shape != inner.shape:
        raise ValueError("provide one interior element for each boundary element")
    conn = model.elements[outer].long()
    near,far = x[conn],x[model.elements[inner].long()]
    e = local_edge
    opposite = conn[:,[(e+2)%4,(e+3)%4]]
    inner_conn = model.elements[inner]
    # The neighbor must traverse the common edge in the opposite direction.
    shared = (inner_conn == opposite[:,1:]) & (inner_conn.roll(-1,1) == opposite[:,:1])
    if not shared.any(1).all():
        raise ValueError("the inward neighbor must share the opposite edge")
    tol = 128*torch.finfo(x.dtype).eps
    for coords in (near,far):
        sides = coords.roll(-1,1)-coords
        scale = sides.norm(dim=2).amax(1)
        if torch.any((sides[:,0]+sides[:,2]).norm(dim=1) > tol*scale):
            raise ValueError("boundary recovery requires parallelogram elements")
    tangent = near[:,(e+1)%4]-near[:,e]
    normal = torch.stack((tangent[:,1],-tangent[:,0]),1)
    center = near.mean(1)
    to_wall = (near[:,e]+near[:,(e+1)%4])/2-center
    outward = center-far.mean(1)
    ratio = (to_wall*outward).sum(1)/outward.square().sum(1)
    if (not torch.isfinite(ratio).all() or (ratio <= 0).any() or
            torch.any((to_wall-ratio[:,None]*outward).norm(dim=1) > tol*outward.norm(dim=1))):
        raise ValueError("element centers must align with the boundary midpoint")
    k = _element_conductivity(model)
    if k.shape[0] != 1 and not torch.equal(k[outer],k[inner]):
        raise ValueError("boundary recovery requires the same conductivity across each pair")
    flux = thermal_element_flux(model,temperature)
    estimate = flux[...,outer,:]+ratio[:,None]*(flux[...,outer,:]-flux[...,inner,:])
    thickness = _expand(model.thickness,len(model.elements),x)
    if not torch.equal(thickness[outer],thickness[inner]):
        raise ValueError("boundary recovery requires equal thickness across each pair")
    return (estimate*normal).sum(-1)*thickness[outer]


def recover_q4_patch_boundary_power(model, temperature, patches, local_edge, *, degree=2):
    """Fit a physical-coordinate polynomial to nodal temperatures on Q4 patches.

    Each row of patches lists distinct edge-connected elements, beginning
    with the element whose local edge is exposed. Use a single material and
    thickness per patch. At least six quadratically independent nodal samples
    are needed for degree=2. For degree=3 use ten cubically independent
    samples (a three-by-three patch usually suffices). The least-squares fit
    weights each element-node incidence equally. Integrating its flux
    on the straight edge gives outward power in W, including thickness.
    Temperature can contain leading time or load axes and complex phasors.
    """
    if model.element_type.lower() != "q4" or local_edge not in (0,1,2,3):
        raise ValueError("provide Q4 patches and a local edge index from 0 to 3")
    if degree not in (2,3):
        raise ValueError("boundary patch degree must be 2 or 3")
    x = model.nodes
    ids = torch.as_tensor(patches,device=x.device)
    if (ids.ndim != 2 or not ids.shape[1] or ids.dtype not in (torch.int32,torch.int64)
            or (ids < 0).any() or (ids >= len(model.elements)).any()):
        raise ValueError("provide a matrix of valid patch element indices")
    if temperature.ndim < 1 or temperature.shape[-1] != len(x):
        raise ValueError("temperature must have one value per node")
    ids = ids.long()
    if (ids.sort(dim=1).values.diff(dim=1) == 0).any():
        raise ValueError("each patch requires distinct elements")
    conn = model.elements[ids].long()
    # Traverse shared edges by node identity, preserving topological cracks.
    ends = conn.roll(-1,2)
    forward = conn*len(x)+ends
    reverse = ends*len(x)+conn
    matches = forward[:,:,None,:,None] == reverse[:,None,:,None,:]
    if matches[:,0,:,local_edge,:].any():
        raise ValueError("target edge must be exposed on the patch")
    adjacent = matches.any(-1).any(-1)
    reached = torch.zeros_like(ids,dtype=torch.bool)
    reached[:,0] = True
    for _ in range(ids.shape[1]-1):
        reached = reached | (adjacent & reached[:,None,:]).any(-1)
    if not reached.all():
        raise ValueError("each patch must be edge-connected")

    k = _element_conductivity(model)
    k = k.expand(len(model.elements),*k.shape[1:])[ids]
    thick = _expand(model.thickness,len(model.elements),x)[ids]
    if (k != k[:,:1]).any() or (thick != thick[:,:1]).any():
        raise ValueError("use one material and thickness per patch")
    coords = x[conn]
    sides = coords.roll(-1,2)-coords
    following = sides.roll(-1,2)
    if ((sides[...,0]*following[...,1]-sides[...,1]*following[...,0]) <= 0).any():
        raise ValueError("patches require counterclockwise convex Q4 elements")
    tangent = sides[:,0,local_edge]
    normal = torch.stack((tangent[:,1],-tangent[:,0]),1)
    midpoint = coords[:,0,local_edge]+tangent/2
    displacement = coords.flatten(1,2)-midpoint[:,None,:]
    scale = displacement.norm(dim=-1).amax(1)
    a,b = (displacement/scale[:,None,None]).unbind(-1)
    basis = torch.stack((torch.ones_like(a),a,b,a*a,a*b,b*b),dim=-1)
    if degree == 3:
        cubic = torch.stack((a**3,a*a*b,a*b*b,b**3),dim=-1)
        basis = torch.cat((basis,cubic),dim=-1)
    if (torch.linalg.matrix_rank(basis) < basis.shape[-1]).any():
        polynomial = "quadratic" if degree == 2 else "cubic"
        raise ValueError(f"patch nodes must span a complete {polynomial}")
    q,r = torch.linalg.qr(basis,mode="reduced")
    # Center the polynomial on the edge so odd derivative terms average to zero.
    direction = (k[:,0] @ normal.unsqueeze(-1)).squeeze(-1) if k.ndim == 4 else k[:,:1]*normal
    derivative = torch.zeros_like(r[:,:,0])
    linear = -direction*(thick[:,0]/scale)[:,None]
    derivative[:,1:3] = linear
    if degree == 3:
        # In normalized coordinates the edge is z*tangent/scale, for
        # -1/2 <= z <= 1/2. Its mean z² is 1/12.
        tx,ty = (tangent/scale[:,None]).unbind(1)
        gx,gy = linear.unbind(1)
        derivative[:,6:] = torch.stack((3*gx*tx**2,2*gx*tx*ty+gy*tx**2,
                                        gx*ty**2+2*gy*tx*ty,3*gy*ty**2),dim=1)/12
    weights = q @ torch.linalg.solve_triangular(r.transpose(1,2),derivative[:,:,None],upper=False)
    values = temperature.to(x.device)[...,conn.flatten(1,2)]
    return ((values-values[...,:1])*weights.squeeze(-1)).sum(-1)


def _thermal_element_matrices(model, load_like):
    x = model.nodes
    conn = model.elements.long()
    ne, nen = conn.shape
    k = _element_conductivity(model)
    rho = _expand(model.density, ne, x)
    cp = _expand(model.specific_heat, ne, x)
    thick = _expand(model.thickness, ne, x)
    source = torch.as_tensor(0. if model.heat_source is None else model.heat_source,
                             dtype=load_like.dtype, device=x.device)
    interpolated = source.shape == (ne, nen)
    if not interpolated:
        source = source.reshape(-1).expand(ne)
    ke = x.new_zeros((ne, nen, nen))
    ce = torch.zeros_like(ke)
    fe = load_like.new_zeros((ne, nen))
    coords = x[conn]
    loss = None
    variable_loss = False
    if model.heat_loss_coefficient is not None:
        raw = torch.as_tensor(model.heat_loss_coefficient, device=x.device)
        variable_loss = raw.shape==(ne,nen)
        if raw.is_complex() or not variable_loss and raw.shape not in ((),(1,),(ne,)):
            raise ValueError("heat_loss_coefficient must be real and scalar, per element or locally nodal")
        loss = torch.as_tensor(model.heat_loss_coefficient,dtype=x.dtype,device=x.device)
        if not variable_loss:
            loss = loss.expand(ne)
        if not torch.isfinite(loss).all() or (loss<0).any():
            raise ValueError("heat_loss_coefficient must be finite and nonnegative")
    sink = None
    quadratic = model.heat_source_quadratic
    if quadratic is not None:
        raw = torch.as_tensor(quadratic, device=x.device)
        if raw.is_complex() and not load_like.is_complex():
            raise ValueError("complex heat_source_quadratic requires complex loads")
        quadratic = torch.as_tensor(quadratic, device=x.device, dtype=load_like.dtype)
        if not torch.isfinite(quadratic).all():
            raise ValueError("heat_source_quadratic must be finite")
        if nen == 2:
            if quadratic.shape not in ((), (1,), (ne,)):
                raise ValueError("LINE2 heat_source_quadratic requires a scalar or one value per element")
            quadratic = quadratic.expand(ne)
        else:
            if quadratic.shape not in ((3,), (ne, 3)):
                raise ValueError("Q4 heat_source_quadratic requires one triple or one per element")
            quadratic = quadratic.expand(ne, 3)
    if model.element_type.lower() == "line2":
        length = coords[:, 1, 0] - coords[:, 0, 0]
        if torch.any(length <= 0):
            raise ValueError("LINE2 element must have positive length")
        base_k = x.new_tensor([[1., -1.], [-1., 1.]])
        base_c = x.new_tensor([[2., 1.], [1., 2.]])
        ke = k[:, None, None] * thick[:, None, None] / length[:, None, None] * base_k
        measure = (thick*length/6)[:,None,None]*base_c
        ce = (rho*cp)[:,None,None]*measure
        if variable_loss:
            # Integrals of H(x)*N_i*N_j for a linear coefficient on LINE2.
            left,right = loss.unbind(1)
            entries = torch.stack((3*left+right,left+right,left+right,left+3*right),1)
            sink = (thick*length/12)[:,None,None]*entries.reshape(ne,2,2)
        elif loss is not None:
            sink = loss[:,None,None]*measure
        if interpolated:
            fe = (thick*length/6)[:, None]*(source @ base_c.to(source.dtype))
        else:
            fe = (source*thick*length/2)[:, None] * x.new_ones((ne, 2))
        if quadratic is not None:
            # Two-point Gauss integrates N_i(x)*x² exactly on the affine line.
            for t in ((1-3**-.5)/2, (1+3**-.5)/2):
                n = x.new_tensor([1-t, t])
                position = coords[:, :, 0] @ n
                power = quadratic*position.square()*thick*length/2
                fe = fe + power[:, None]*n
    else:
        sink = torch.zeros_like(ke) if loss is not None else None
        g = 3.0**-0.5
        for xi, eta in ((-g,-g),(g,-g),(g,g),(-g,g)):
            n, grad, det = _q4_geometry(coords, xi, eta)
            weight = det*thick
            weighted_grad = grad @ k if k.ndim == 3 else grad * k[:, None, None]
            ke = ke + (weighted_grad @ grad.transpose(1, 2)) * weight[:, None, None]
            measure = weight[:,None,None]*(n[None,:,None]*n[None,None,:])
            ce = ce+(rho*cp)[:,None,None]*measure
            if loss is not None and not variable_loss:
                sink = sink+loss[:,None,None]*measure
            density = source @ n.to(source.dtype) if interpolated else source
            fe = fe + (density*weight)[:,None]*n[None,:]
        if variable_loss or quadratic is not None:
            # On a distorted bilinear map, H*N_i*N_j*det(J) can have
            # degree four in a reference coordinate. Three Gauss points suffice.
            g = (3/5)**.5
            rule = ((-g,5/9),(0.,8/9),(g,5/9))
            for xi,wx in rule:
                for eta,wy in rule:
                    n,_,det = _q4_geometry(coords,xi,eta)
                    weight = wx*wy*det*thick
                    if variable_loss:
                        sink = sink+(weight*(loss @ n))[:,None,None]*torch.outer(n,n)
                    if quadratic is not None:
                        # With a bilinear map, N_i*q(x,y)*det(J) has degree at
                        # most four in either parent coordinate.
                        px, py = (coords.transpose(1, 2) @ n).unbind(1)
                        a, b, c = quadratic.unbind(1)
                        power = weight*(a*px.square() + b*px*py + c*py.square())
                        fe = fe + power[:, None]*n
    return ke if sink is None else ke+sink, ce, fe, sink


def thermal_element_nodal_power(model: ThermalModel, temperature: torch.Tensor,
                                angular_frequency: float = 0., *, element_nodal_heat=None,
                                mean_temperature=None, mean_heating_rate=None, return_net_input: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Inward power phasors (W) at LINE2 or Q4 element nodes under exp(+iωt).

    Zero frequency recovers steady powers. The return dtype is complex.
    Includes the consistent storage term and subtracts the model's element
    volume loads. Supply any additional element-resolved nodal loads (ne, nen)
    in W to subtract them before assembly. Global point loads, films and
    contacts otherwise appear as exchanges. Q4 entries are weak-form nodal powers.
    When temperature is a relative field from separate_mean=True, pass its
    nodal mean_temperature or mean_heating_rate to restore uniform storage power.
    return_net_input also returns element-resolved distributed input minus bulk
    loss and storage, in W, for use as net_element_nodal_input in section recovery.
    """
    if not math.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValueError("angular_frequency must be finite and nonnegative")
    if temperature.shape != (len(model.nodes),):
        raise ValueError("temperature must have one value per node")
    dtype = torch.promote_types(model.nodes.dtype, torch.promote_types(temperature.dtype, torch.complex64))
    values = temperature.to(device=model.nodes.device, dtype=dtype)[model.elements.long()]
    ke, ce, fe, sink = _thermal_element_matrices(model, values)
    power = torch.einsum("eij,ej->ei", ke.to(dtype) + 1j*angular_frequency*ce.to(dtype), values) - fe
    uniform_storage = None
    if mean_temperature is not None and mean_heating_rate is not None:
        raise ValueError("supply only one of mean_temperature and mean_heating_rate")
    if mean_temperature is not None or mean_heating_rate is not None:
        offset = mean_temperature if mean_heating_rate is None else mean_heating_rate
        mean = torch.as_tensor(offset, dtype=dtype, device=values.device)
        if mean.shape != temperature.shape or not torch.isfinite(mean).all():
            raise ValueError("mean_temperature or mean_heating_rate requires a finite value at each node")
        element_mean = mean[model.elements.long()]
        if torch.any(element_mean != element_mean[:, :1]) or (sink is not None and torch.any(sink != 0)):
            raise ValueError("mean_temperature or mean_heating_rate must be uniform on each lossless element")
        rate = 1j*angular_frequency*element_mean if mean_heating_rate is None else element_mean
        uniform_storage = ce.sum(dim=2)*rate
        power = power+uniform_storage
    if element_nodal_heat is not None:
        load = torch.as_tensor(element_nodal_heat, dtype=dtype, device=values.device)
        if load.shape != values.shape or not torch.isfinite(load).all():
            raise ValueError("element_nodal_heat needs finite powers at every element node")
        power = power - load
    if return_net_input:
        net = fe-1j*angular_frequency*torch.einsum("eij,ej->ei",ce.to(dtype),values)
        if uniform_storage is not None:
            net = net-uniform_storage
        if element_nodal_heat is not None:
            net = net+load
        if sink is not None:
            net = net-torch.einsum("eij,ej->ei",sink.to(dtype),values)
        return power,net
    return power


def thermal_element_nodal_heat_steps(model: ThermalModel, result: ThermalTransientResult, *,
                                    element_nodal_heat_history: torch.Tensor | None = None,
                                    element_nodal_step_heat: torch.Tensor | None = None,
                                    return_net_input: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Inward energy at each element node, shaped (nt-1, ne, nen), in J.

    Uses local consistent capacity, conductivity and static volumetric sources.
    An optional (nt, ne, nen) array assigns additional applied powers (W) to
    individual element nodes, before global assembly. element_nodal_step_heat
    additionally assigns integrated joules of shape (nt-1, ne, nen). Supply each distributed
    load once. Global point loads, films and contacts appear as exchanges unless
    explicitly allocated to element nodes. For LINE2, negate the right entry
    for energy leaving the right end. Q4 entries are nodal weak-form quantities.
    return_net_input also returns the distributed net input in J per interval
    and element node. Divide both arrays by interval duration to recover power.
    """
    ke, ce, fe, sink = _thermal_element_matrices(model, model.nodes)
    t = result.temperature[:, model.elements.long()]
    stage = result.interval_temperature[:, model.elements.long()]
    stored = torch.einsum("eij,tej->tei", ce, t.diff(dim=0))
    dt = result.time.diff().to(t)[:,None,None]
    heat = stored
    net = dt*fe-stored if return_net_input else None
    power = torch.einsum("eij,tej->tei", ke, stage) - fe
    if element_nodal_heat_history is not None:
        load = torch.as_tensor(element_nodal_heat_history).to(t)
        if load.shape != t.shape or not torch.isfinite(load).all():
            raise ValueError("element_nodal_heat_history needs finite powers at every time and element node")
        average = (1-result.theta)*load[:-1] + result.theta*load[1:]
        power = power-average
        if return_net_input:
            net = net+dt*average
    if element_nodal_step_heat is not None:
        supplied = torch.as_tensor(element_nodal_step_heat).to(t)
        if supplied.shape != heat.shape or not torch.isfinite(supplied).all():
            raise ValueError("element_nodal_step_heat needs finite joules per interval and element node")
        heat = heat - supplied
        if return_net_input:
            net = net+supplied
    heat = heat+dt*power
    if return_net_input:
        if sink is not None:
            net = net-dt*torch.einsum("eij,tej->tei",sink,stage)
        return heat,net
    return heat


def thermal_element_section_power(model: ThermalModel, nodal_power: torch.Tensor, fractions, *,
                                  net_line_source_gradient=None, net_line_source_curvature=None,
                                  net_element_nodal_input=None) -> torch.Tensor:
    """Recover through-section power (W) from inward element nodal powers (W).

    For LINE2 the positive direction is node 0 to node 1. For a Q4 strip
    with one-dimensional heat transport and insulated lateral edges, it is
    from edge (0,3) toward edge (1,2). Nodal exchanges on the lateral edges
    must be absent or separately accounted for before using this reduction.

    Supply (..., element, local_node) phasors from thermal_element_nodal_power
    or interval-mean powers from thermal_element_nodal_heat_steps divided by
    the interval durations. fractions is a scalar or vector in [0,1].
    Returns (..., element, sample). Linear interpolation of the two section
    powers preserves each element's net heat balance, with constant power
    gradient in the local fractional coordinate. Area is already included.

    For a straight constant-section element, optional gradient (W/m²) and
    curvature (W/m³) describe the first and second axial derivatives of net
    heat input per length at its midpoint. Include section area and subtract
    storage and sinks when forming net heat input. For transient data use
    interval-mean derivatives. Scalars or arrays broadcastable to
    (..., element) are accepted. Integrating this quadratic source profile
    gives a cubic power profile with the same two endpoint powers.

    Alternatively, supply net_element_nodal_input in W with the same shape
    as nodal_power. Each entry is the integral of the net distributed input
    against its local shape function, after removing storage and sinks.
    The difference of the two axial nodal loads gives the linear source
    moment. This replaces the prescribed gradient and can be combined with
    a prescribed curvature; it is exact for quadratic net input per length.
    """
    if nodal_power.ndim<2 or nodal_power.shape[-2:]!=model.elements.shape:
        raise ValueError("nodal_power needs one inward power per element node")
    f = torch.as_tensor(fractions,device=nodal_power.device)
    if f.is_complex():
        raise ValueError("fractions must be real")
    # Convert the original input directly to preserve float64 Python samples.
    f = torch.atleast_1d(torch.as_tensor(fractions,dtype=model.nodes.dtype,device=nodal_power.device))
    if f.ndim!=1 or not torch.isfinite(f).all() or ((f<0)|(f>1)).any():
        raise ValueError("fractions must be finite local positions in [0,1]")
    entering,leaving = nodal_power[...,0],-nodal_power[...,1]
    if model.element_type.lower()=="q4":
        entering = entering+nodal_power[...,3]
        leaving = leaving-nodal_power[...,2]
    power = entering[...,None]*(1-f)+leaving[...,None]*f
    if net_element_nodal_input is not None:
        if net_line_source_gradient is not None:
            raise ValueError("supply either net nodal input or an axial source gradient")
        load = torch.as_tensor(net_element_nodal_input,device=power.device)
        if load.shape!=nodal_power.shape or not torch.isfinite(load).all():
            raise ValueError("net_element_nodal_input requires finite powers at every element node")
        moment = load[...,1]-load[...,0]
        if model.element_type.lower()=="q4":
            moment = moment+load[...,2]-load[...,3]
        # For g=g0+g1*x+g2*x²/2 about the midpoint, the right-minus-left
        # shape-function integral is g1*length²/6.
        power = power+3*moment[...,None]*(f*(f-1))
    if net_line_source_gradient is None and net_line_source_curvature is None:
        return power

    coords = model.nodes[model.elements.long()].to(device=power.device)
    span = coords[:,1]-coords[:,0]
    if model.element_type.lower()=="q4":
        span = (span+coords[:,2]-coords[:,3])/2
    length = torch.linalg.vector_norm(span,dim=-1)
    # Integrate about the midpoint and remove the endpoint interpolant.
    # Both basis functions vanish at f=0,1, preserving the nodal heat budget.
    bubble = f*(f-1)
    for value,basis in ((net_line_source_gradient,length[:,None]**2*bubble/2),
                        (net_line_source_curvature,length[:,None]**3*bubble*(f-.5)/6)):
        if value is None:
            continue
        raw = torch.as_tensor(value)
        dtype = torch.promote_types(power.dtype,raw.dtype)
        derivative = torch.as_tensor(value,dtype=dtype,device=power.device)
        if not torch.isfinite(derivative).all():
            raise ValueError("net line source derivatives must be finite")
        try:
            derivative = torch.broadcast_to(derivative,entering.shape)
        except RuntimeError as exc:
            raise ValueError("net line source derivatives must broadcast to the element power shape") from exc
        power = power+derivative[...,None]*basis
    return power


def _boundary_thickness(model, edges):
    """Gather the owning element's section area (LINE2) or thickness (Q4)."""
    x = model.nodes
    thickness = torch.as_tensor(1. if model.thickness is None else model.thickness,
                                dtype=x.dtype, device=x.device).reshape(-1)
    if thickness.numel() == 1 or edges.numel() == 0:
        return thickness.expand(len(edges)) if thickness.numel() == 1 else thickness[:0]
    conn = model.elements.to(device=x.device, dtype=torch.long)
    # Encode undirected faces so reversing either the element list or an edge
    # does not change the mapping. LINE2 boundary faces are repeated node IDs.
    other = conn if model.element_type.lower() == "line2" else conn.roll(-1,dims=1)
    keys = torch.minimum(conn,other)*len(x)+torch.maximum(conn,other)
    ordered, permutation = keys.flatten().sort()
    first, last = edges.unbind(dim=1)
    query = torch.minimum(first,last)*len(x)+torch.maximum(first,last)
    start = torch.searchsorted(ordered,query)
    stop = torch.searchsorted(ordered,query,right=True)
    if not torch.all(stop-start == 1):
        raise ValueError("per-element thickness requires convection edges with a unique owning element")
    owners = permutation[start] // conn.shape[1]
    return thickness[owners]


def _contact_edge_profile(model):
    """Return matched endpoints and area-scaled conductance at both ends."""
    x = model.nodes
    pairs = model.contact_edge_pairs.to(device=x.device,dtype=torch.long)
    g = torch.as_tensor(model.contact_edge_conductance,device=x.device)
    if g.is_complex():
        raise ValueError("contact edge conductance must be real")
    g = torch.as_tensor(model.contact_edge_conductance,dtype=x.dtype,device=x.device)
    if g.shape != (len(pairs),2):
        if g.numel() not in (1,len(pairs)):
            raise ValueError("provide scalar, per-edge or paired endpoint conductances")
        g = g.reshape(-1,1).expand(len(pairs),2)
    if not torch.isfinite(g).all() or (g < 0).any():
        raise ValueError("contact edge conductance must be finite and nonnegative")
    return pairs,g


def _contact_edge_matrix(model):
    """Integrate g(s) N_i(s) N_j(s) on the normalized edge."""
    pairs,g = _contact_edge_profile(model)
    left,right = g.unbind(1)
    mixed = left+right
    mass = torch.stack((3*left+right,mixed,mixed,left+3*right),dim=1).reshape(-1,2,2)/12
    return pairs,mass


def _contact_edge_traces(model, temperature):
    if model.contact_edge_pairs is None:
        raise ValueError("provide matching contact edge pairs on the model")
    if temperature.ndim < 1 or temperature.shape[-1] != len(model.nodes):
        raise ValueError("temperature must have one value per node")
    pairs,g = _contact_edge_profile(model)
    traces = temperature.to(model.nodes.device)[...,pairs]
    return pairs, g, traces


def _contact_edge_jump(model, temperature):
    pairs, g, traces = _contact_edge_traces(model, temperature)
    return pairs, g, traces[...,0,:]-traces[...,1,:]


def _contact_line_conductance(model, pairs, g, fraction):
    if not 0. <= fraction <= 1.:
        raise ValueError("contact edge fraction must lie in [0, 1]")
    first, last = model.nodes[pairs[:,0]].unbind(1)
    return ((1-fraction)*g[:,0] + fraction*g[:,1]) / (last-first).norm(dim=1)


def thermal_contact_edge_line_power_density(model: ThermalModel, temperature: torch.Tensor, *,
                                            fraction: float) -> torch.Tensor:
    """Local heat flow per physical edge length (W/m), positive out of the first trace.

    Evaluate all edges at the same fraction along their supplied endpoint order.
    Area-scaled conductance already includes thickness. Leading temperature axes
    and complex phasors are supported as in thermal_contact_edge_power.
    """
    pairs, g, jump = _contact_edge_jump(model, temperature)
    local_jump = (1-fraction)*jump[...,0] + fraction*jump[...,1]
    return _contact_line_conductance(model, pairs, g, fraction) * local_jump


def thermal_contact_edge_entropy_production_density(model: ThermalModel, temperature: torch.Tensor, *,
                                                   fraction: float) -> torch.Tensor:
    """Instantaneous contact entropy production per edge length in W/(m K).

    Supply absolute nodal temperatures in kelvins. Leading dimensions may
    index times or sampled phases. Each output is the local heat flow per
    length times the difference of reciprocal absolute trace temperatures.
    """
    pairs, g, traces = _positive_contact_traces(model, temperature)
    conductance = _contact_line_conductance(model, pairs, g, fraction)
    return conductance * _contact_entropy_factor(traces, fraction)


def _positive_contact_traces(model, temperature):
    if temperature.is_complex():
        raise ValueError("supply real instantaneous absolute temperatures")
    pairs, g, traces = _contact_edge_traces(model, temperature)
    if not torch.isfinite(traces).all() or (traces <= 0).any():
        raise ValueError("contact temperatures must be positive and finite in kelvins")
    return pairs, g, traces


def _contact_entropy_factor(traces, fraction):
    first, second = traces.unbind(-2)
    jump = first-second
    # Interpolate the nodal differences before dividing by absolute temperatures.
    def interpolate(values):
        return (1-fraction)*values[..., 0] + fraction*values[..., 1]
    local_jump = interpolate(jump)
    return (local_jump / interpolate(first)) * (local_jump / interpolate(second))



def thermal_contact_edge_entropy_production(model: ThermalModel, temperature: torch.Tensor, *,
                                            start: float = 0., end: float = 1.,
                                            quadrature_order: int = 8) -> torch.Tensor:
    """Integrate instantaneous entropy production over each Q4 contact edge (W/K).

    Supply absolute temperatures with nodes on the final axis. The normalized
    interval follows the paired edge orientation. Gauss-Legendre quadrature
    integrates the rational temperature dependence; increase its order when
    the relative variation of either temperature along the interval is large.
    """
    _, g, traces = _positive_contact_traces(model, temperature)
    return _integrate_contact_entropy(g, traces, start, end, quadrature_order)


def _integrate_contact_entropy(g, traces, start, end, quadrature_order):
    from numpy.polynomial.legendre import leggauss

    if not 0. <= start <= end <= 1.:
        raise ValueError("contact edge fractions must satisfy 0 <= start <= end <= 1")
    if type(quadrature_order) is not int or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    points, weights = leggauss(quadrature_order)
    half = (end-start)/2
    fractions = g.new_tensor(start + half*(1+points))
    conductance = g[:, :1]*(1-fractions) + g[:, 1:] * fractions
    # Insert the integration axis before the two trace and endpoint axes.
    factor = _contact_entropy_factor(traces.unsqueeze(-3), fractions)
    return half * (factor * conductance * g.new_tensor(weights)).sum(dim=-1)


def thermal_contact_edge_entropy_steps(model: ThermalModel, time: torch.Tensor, temperature: torch.Tensor, *,
                                       start: float = 0., end: float = 1., quadrature_order: int = 8,
                                       time_order: int = 4) -> torch.Tensor:
    """Entropy generated per sampled interval and Q4 contact edge (J/K).

    Interpolate instantaneous absolute nodal temperatures linearly between
    samples. Apply Gauss-Legendre quadrature in time and along the selected
    edge fraction. Returns shape (len(time)-1, n_edges).
    """
    from numpy.polynomial.legendre import leggauss

    if (time.ndim != 1 or time.numel() < 2 or time.is_complex() or not torch.isfinite(time).all()
            or not (time[1:] > time[:-1]).all()):
        raise ValueError("supply finite strictly increasing real sample times")
    if temperature.ndim != 2 or temperature.shape != (time.numel(), len(model.nodes)):
        raise ValueError("supply one nodal temperature field per sample time")
    if type(time_order) is not int or time_order < 1:
        raise ValueError("time_order must be a positive integer")
    # Validate the endpoints, including extrema that interior quadrature misses.
    _, g, traces = _positive_contact_traces(model, temperature)
    nodes, weights = leggauss(time_order)
    dtype = torch.promote_types(temperature.dtype, model.nodes.dtype)
    traces = traces.to(dtype)
    fraction = traces.new_tensor((nodes+1)/2).view(-1, 1, 1, 1)
    sampled = torch.lerp(traces[:-1, None], traces[1:, None], fraction)
    rate = _integrate_contact_entropy(g, sampled, start, end, quadrature_order)
    duration = time.to(device=rate.device).diff().to(rate.dtype)
    return duration[:, None] * (rate * rate.new_tensor(weights/2)[:, None]).sum(dim=1)


def _contact_pair_data(model):
    x = model.nodes
    pairs = model.contact_pairs.to(device=x.device, dtype=torch.long)
    g = torch.as_tensor(model.contact_conductance, dtype=x.dtype, device=x.device)
    if g.numel() not in (1, len(pairs)) or not torch.isfinite(g).all() or (g < 0).any():
        raise ValueError("contact_conductance must be finite, nonnegative, scalar or one per pair")
    g = g.reshape(-1).expand(len(pairs))
    return pairs,g


def thermal_contact_pair_power(model: ThermalModel, temperature: torch.Tensor) -> torch.Tensor:
    """Signed W through each lumped contact, positive from first node to second.

    Real fields and complex phasors may have arbitrary leading axes; the
    final temperature axis indexes global nodes, and the output indexes
    contacts in contact_pairs order. Conductance includes tributary area.
    """
    if model.contact_pairs is None:
        raise ValueError("provide contact_pairs on the model")
    if temperature.ndim < 1 or temperature.shape[-1] != len(model.nodes):
        raise ValueError("temperature must have one value per node")
    pairs,g = _contact_pair_data(model)
    left,right = pairs.unbind(dim=1)
    field = temperature.to(model.nodes.device)
    return g*(field[...,left]-field[...,right])


def _contact_pair_cycle_parameters(model, temperature, angular_frequency, steady_temperature):
    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    phasor = thermal_contact_pair_power(model,temperature)
    amplitude = phasor.abs()
    if steady_temperature is None:
        bias = torch.zeros_like(amplitude)
    else:
        if steady_temperature.is_complex():
            raise ValueError("steady_temperature must be real")
        bias = thermal_contact_pair_power(model,steady_temperature)
    if not torch.isfinite(amplitude).all() or not torch.isfinite(bias).all():
        raise ValueError("contact powers must be finite")
    magnitude = bias.abs()
    reverses = amplitude > magnitude
    # Half-width of the smaller lobe. Retain A-|bias| close to tangency.
    gap = torch.where(reverses,amplitude-magnitude,0.)
    if phasor.is_complex():
        # Near a coordinate axis, rounding hypot(x,y) can hide a shallow
        # crossing. Recover hypot(x,y)-x as y*(y/x)/(hypot(x,y)/x+1).
        major = torch.maximum(phasor.real.abs(), phasor.imag.abs())
        minor = torch.minimum(phasor.real.abs(), phasor.imag.abs())
        scale = torch.where(major > 0, major, 1.)
        ratio = minor/scale
        near_axis = (major > 0) & (ratio <= math.sqrt(8*torch.finfo(major.dtype).eps))
        correction = minor*(ratio/(amplitude/scale+1))
        resolved_gap = (major-magnitude)+correction
        gap = torch.where(near_axis, resolved_gap.clamp_min(0.), gap)
        reverses = gap > 0
    fraction = gap/torch.where(reverses,amplitude,1.)
    angle = 2*torch.asin(torch.sqrt(torch.where(reverses,.5*fraction,.5)))
    if phasor.is_complex():
        # When q(0)=0, the imaginary component gives a nonzero slope even
        # if hypot(real, imag) rounds to abs(real) at a shallow crossing.
        origin = (bias == -phasor.real) & (phasor.imag != 0)
        angle = torch.where(origin, torch.atan2(phasor.imag.abs(), phasor.real.abs()), angle)
        reverses = reverses | origin
    return phasor, bias, reverses, angle


def thermal_contact_pair_harmonic_reversal_times(model: ThermalModel, temperature: torch.Tensor,
                                                angular_frequency: float, *, steady_temperature=None):
    """Times (s) of entry into forward and reverse flow within one period.

    Phase uses Re(T*exp(iωt)) at t=0. Returns two tensors in pair order,
    with broadcast batch axes. Both entries are NaN for constant-sign power,
    tangencies and zero power. A crossing at the period boundary is at zero.
    """
    phasor, bias, reverses, half = _contact_pair_cycle_parameters(
        model, temperature, angular_frequency, steady_temperature)
    phase = torch.angle(phasor)
    # Relative to the phasor peak, reverse flow begins on the falling branch.
    falling = torch.where(bias > 0, math.pi-half, half)
    falling = torch.where(bias == 0, math.pi/2, falling)
    def cycle_time(angle):
        wrapped = torch.remainder(angle, 2*math.pi)
        wrapped = torch.where(wrapped >= 2*math.pi, 0., wrapped)
        return torch.where(reverses, wrapped/angular_frequency, float("nan"))
    rising_time, falling_time = cycle_time(-falling-phase), cycle_time(falling-phase)
    if phasor.is_complex():
        # Factor q(t) at the known root. The other root satisfies
        # tan(ωt/2) = -imag/real. Use the narrow lobe to avoid cancellation.
        origin = (bias == -phasor.real) & (phasor.imag != 0)
        short_arc = ((phasor.real < 0) & (phasor.imag > 0)) | ((phasor.real >= 0) & (phasor.imag < 0))
        other = torch.where(short_arc, 2*half, 2*math.pi-2*half)/angular_frequency
        rising_time = torch.where(origin, torch.where(phasor.imag < 0, 0., other), rising_time)
        falling_time = torch.where(origin, torch.where(phasor.imag > 0, 0., other), falling_time)
    return rising_time, falling_time


def thermal_contact_pair_harmonic_cycle_duration(model: ThermalModel, temperature: torch.Tensor,
                                                 angular_frequency: float, *, steady_temperature=None):
    """Return seconds of forward and reverse flow per lumped contact over a cycle.

    Supply peak nodal phasors and optional real steady temperatures, as for
    thermal_contact_pair_harmonic_cycle_heat. A contact at zero power throughout
    contributes zero in both directions. An isolated zero costs no duration.
    """
    phasor, bias, reverses, angle = _contact_pair_cycle_parameters(
        model, temperature, angular_frequency, steady_temperature)
    amplitude = phasor.abs()
    minor = torch.where(reverses, 2*angle/angular_frequency, 0.)
    period = 2*math.pi/angular_frequency
    major = torch.where((amplitude > 0) | (bias != 0), period-minor, 0.)
    return torch.where(bias >= 0, major, minor), torch.where(bias >= 0, minor, major)


def thermal_contact_pair_harmonic_cycle_heat(model: ThermalModel, temperature: torch.Tensor,
                                             angular_frequency: float, *, steady_temperature=None):
    """Return forward and reverse heat magnitudes (J) per lumped contact and cycle.

    Supply peak nodal phasors for Re(T*exp(iωt)) and, optionally, real steady
    nodal temperatures. Leading batch dimensions may broadcast. Positive
    transfer follows contact_pairs order. A common absolute temperature offset
    can be omitted from the steady field.
    """
    phasor, bias, reverses, angle = _contact_pair_cycle_parameters(
        model, temperature, angular_frequency, steady_temperature)
    amplitude = phasor.abs()
    square = angle.square()
    # sin(a)-a*cos(a) loses its leading cubic term to roundoff for small a.
    series = angle*square*(1/3+square*(-1/30+square*(1/840-square/45360)))
    lobe = torch.where(angle < .05,series,torch.sin(angle)-angle*torch.cos(angle))
    minor = torch.where(reverses,amplitude*lobe,0.)*(2/angular_frequency)
    period = 2*math.pi/angular_frequency
    return minor+period*bias.clamp_min(0), minor+period*(-bias).clamp_min(0)


def thermal_contact_pair_heat_steps(model: ThermalModel, result: ThermalTransientResult) -> torch.Tensor:
    """Heat in J per time interval and lumped contact, shaped (nt-1, n_pairs).

    Positive heat leaves the first node in each pair. The integrator's
    stage-weighted temperature accounts for nonuniform steps and SDIRK2 stages.
    """
    power = thermal_contact_pair_power(model, result.interval_temperature)
    return result.time.diff().to(power)[:, None]*power


def _contact_pair_interval_parts(model, time, temperature):
    if time.is_complex() or time.ndim != 1 or len(time) < 2 or not torch.isfinite(time).all():
        raise ValueError("time must be a finite real vector with at least two entries")
    if temperature.is_complex() or temperature.shape != (len(time),len(model.nodes)):
        raise ValueError("temperature must be a real (nt, n_nodes) history")
    power = thermal_contact_pair_power(model,temperature)
    dt = time.diff().to(power)[:, None]
    if not torch.isfinite(power).all() or not torch.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("require finite contact powers and positive finite time steps")
    left,right = power[:-1],power[1:]
    # Normalize before forming the zero-crossing fractions, including for
    # weak signals. A zero-power interval contributes zero to both integrals.
    scale = torch.maximum(left.abs(),right.abs())
    scale = torch.where(scale > 0,scale,torch.ones_like(scale))
    a,b = left.abs()/scale,right.abs()/scale
    total = a+b
    total = torch.where(total > 0,total,torch.ones_like(total))
    crossing = ((left > 0) & (right < 0)) | ((left < 0) & (right > 0))
    return dt, left, right, crossing, a/total, b/total


def thermal_contact_pair_directional_duration(model: ThermalModel, time: torch.Tensor,
                                             temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Seconds of forward and reverse power per sampling interval and contact pair.

    Interpolate power linearly between samples. Positive power follows pair
    order. An interval at zero power contributes to neither duration. Multiple
    reversals within an interval require intermediate samples.
    """
    dt, left, right, crossing, left_part, right_part = _contact_pair_interval_parts(model, time, temperature)
    forward = torch.where(crossing, torch.where(left > 0, left_part, right_part), (left > 0) | (right > 0))
    reverse = torch.where(crossing, torch.where(left < 0, left_part, right_part), (left < 0) | (right < 0))
    return dt*forward, dt*reverse


def thermal_contact_pair_directional_heat(model: ThermalModel, time: torch.Tensor,
                                          temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and reverse heat magnitudes (J) from piecewise-linear temperature histories.

    Both outputs have shape (nt-1, n_pairs). Forward follows contact_pairs
    order. Within a sampling interval a linear sign change contributes a
    triangle to each direction. Their difference equals the trapezoidal net
    heat, and thus the stage-integrated net heat for Crank--Nicolson. For
    other time integrators thermal_contact_pair_heat_steps gives the solver's
    own net heat. Multiple reversals require intermediate temperature samples.
    """
    dt, left, right, crossing, left_part, right_part = _contact_pair_interval_parts(model, time, temperature)
    left_weight = torch.where(crossing,left_part,1.)*.5
    right_weight = torch.where(crossing,right_part,1.)*.5
    forward = dt*(left_weight*left.clamp_min(0)+right_weight*right.clamp_min(0))
    reverse = dt*(left_weight*(-left).clamp_min(0)+right_weight*(-right).clamp_min(0))
    return forward,reverse


def thermal_contact_edge_power(model: ThermalModel, temperature: torch.Tensor, *,
                               start: float = 0., end: float = 1.) -> torch.Tensor:
    """Integrated W per Q4 contact edge, positive from the first trace to the second.

    Accept real temperatures or complex phasors with arbitrary leading axes.
    start and end are fractions along each edge in the supplied endpoint order.
    Both the temperature jump and conductance are linear in this coordinate.
    """
    if not 0. <= start <= end <= 1.:
        raise ValueError("contact edge fractions must satisfy 0 <= start <= end <= 1")
    _,g,jump = _contact_edge_jump(model, temperature)
    # The product of two linear interpolants is quadratic. Integrate about
    # the subinterval midpoint, whose central second moment is width²/12.
    width = end-start
    mid = start+.5*width
    dg = g[:,1]-g[:,0]
    dj = jump[...,1]-jump[...,0]
    return width*((g[:,0]+mid*dg)*(jump[...,0]+mid*dj) + (width**2/12)*dg*dj)


def thermal_contact_edge_directional_power(model: ThermalModel, temperature: torch.Tensor, *,
                                           start: float = 0., end: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and reverse power magnitudes (W) along each Q4 contact edge.

    Use real instantaneous nodal temperatures, optionally with leading batch
    dimensions. Forward is from the first trace to the second. Integrals
    cover the supplied edge fractions and split at a zero of the linear jump.
    Conductance may vary linearly along the edge.
    """
    if not 0. <= start <= end <= 1.:
        raise ValueError("contact edge fractions must satisfy 0 <= start <= end <= 1")
    if temperature.is_complex():
        raise ValueError("directional power requires real instantaneous temperatures")
    _,g,jump = _contact_edge_jump(model,temperature)
    if not torch.isfinite(jump).all():
        raise ValueError("contact temperature jumps must be finite")
    def endpoints(values):
        left,right = values.unbind(-1)
        return (1-start)*left+start*right, (1-end)*left+end*right
    left,right = endpoints(jump)
    gl,gr = endpoints(g)
    return _edge_directional_integrals(left, right, gl, gr, end-start)


def _edge_directional_integrals(left, right, gl, gr, width):
    crossing = ((left>0)&(right<0)) | ((left<0)&(right>0))
    scale = torch.maximum(left.abs(),right.abs())
    scale = torch.where(crossing,scale,1.)
    a,b = left.abs()/scale,right.abs()/scale
    total = torch.where(crossing,a+b,1.)
    # Keep both subedge lengths independently for a zero close to an endpoint.
    fl,fr = torch.where(crossing,a/total,0.),torch.where(crossing,b/total,0.)
    gz = fr*gl+fl*gr
    # Integrating g times each endpoint hat gives these weights. On a
    # crossing subedge the jump is zero at its internal endpoint.
    wl = torch.where(crossing,fl*(gl/3+gz/6),gl/3+gr/6)
    wr = torch.where(crossing,fr*(gr/3+gz/6),gr/3+gl/6)
    forward = width*(wl*left.clamp_min(0)+wr*right.clamp_min(0))
    reverse = width*(wl*(-left).clamp_min(0)+wr*(-right).clamp_min(0))
    return forward,reverse


def thermal_contact_edge_harmonic_cycle_heat(model: ThermalModel, temperature: torch.Tensor,
                                             angular_frequency: float, *, start: float = 0.,
                                             end: float = 1., quadrature_order: int = 16) -> torch.Tensor:
    """Heat in J flowing in each direction during one zero-mean harmonic cycle.

    For peak nodal phasors with exp(iωt), the forward and reverse magnitudes
    are equal. Return that common value per Q4 contact edge and leading batch.
    Integrate the local positive half-cycle (2*g*abs(jump)/ω) over the edge
    fraction using Gauss-Legendre quadrature. Increase quadrature_order to
    resolve sharply varying phasor magnitudes near a phase reversal.
    """
    from numpy.polynomial.legendre import leggauss

    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    if not 0. <= start <= end <= 1.:
        raise ValueError("contact edge fractions must satisfy 0 <= start <= end <= 1")
    if type(quadrature_order) is not int or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    _,g,jump = _contact_edge_jump(model,temperature)
    if not torch.isfinite(jump).all():
        raise ValueError("contact temperature jumps must be finite")
    left,right = jump.unbind(-1)
    delta = right-left
    length = delta.abs()
    safe_length = torch.where(length > 0,length,1.)
    # Split at the nearest approach to zero, including a true spatial reversal.
    projection = -(left*(delta/safe_length).conj()).real
    split = (projection.clamp_min(0).minimum(length)/safe_length).clamp(start,end)
    points,weights = leggauss(quadrature_order)
    nodes = g.new_tensor((1+points)/2)
    weights = g.new_tensor(weights/2)
    def integrate(lo,hi):
        width = hi-lo
        s = lo + width*nodes
        conductance = (1-s)*g[:,0:1]+s*g[:,1:2]
        magnitude = ((1-s)*left[...,None]+s*right[...,None]).abs()
        return width.squeeze(-1)*(weights*conductance*magnitude).sum(dim=-1)
    middle = split[...,None]
    return 2*(integrate(start,middle)+integrate(middle,end))/angular_frequency


def thermal_contact_edge_biased_cycle_heat(model: ThermalModel, temperature: torch.Tensor,
                                          angular_frequency: float, *, steady_temperature: torch.Tensor,
                                          start: float = 0., end: float = 1., quadrature_order: int = 32):
    """Forward and reverse heat magnitudes (J) per Q4 contact edge per period.

    Supply peak phasors and real steady nodal temperatures with broadcastable
    batch dimensions. Positive transfer follows contact_edge_pairs order.
    Spatial integration is exact at each sampled phase. Gauss-Legendre time
    quadrature splits at zero crossings of either subedge endpoint.
    """
    from numpy.polynomial.legendre import leggauss

    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    if not 0 <= start <= end <= 1 or type(quadrature_order) is not int or quadrature_order < 1:
        raise ValueError("require ordered edge fractions in [0,1] and positive integer quadrature_order")
    if steady_temperature.is_complex():
        raise ValueError("steady_temperature must be real")
    _, g, phasor = _contact_edge_jump(model, temperature)
    _, _, bias = _contact_edge_jump(model, steady_temperature)
    if not torch.isfinite(phasor).all() or not torch.isfinite(bias).all():
        raise ValueError("contact temperature jumps must be finite")

    def endpoints(values):
        left, right = values.unbind(-1)
        return (1-start)*left + start*right, (1-end)*left + end*right

    def crossings(ac, dc):
        amplitude = ac.abs()
        crosses = amplitude > dc.abs()
        ratio = -dc/torch.where(crosses, amplitude, 1.)
        angle = torch.acos(ratio.clamp(-1., 1.))
        return [torch.where(crosses, (sign*angle-ac.angle()) % (2*math.pi), 0.)
                for sign in (-1, 1)]

    al, ar = endpoints(phasor)
    bl, br = endpoints(bias)
    gl, gr = endpoints(g)
    roots = crossings(al, bl) + crossings(ar, br)
    zero = torch.zeros_like(roots[0])
    cuts = torch.stack([zero, *roots, zero+2*math.pi], dim=-1).sort().values
    lo = cuts[..., :-1, None]
    span = cuts.diff(dim=-1)[..., None]
    points, weights = leggauss(quadrature_order)
    phase = torch.exp(1j*(lo + span*g.new_tensor((1+points)/2)))
    left = bl[..., None, None] + (al[..., None, None]*phase).real
    right = br[..., None, None] + (ar[..., None, None]*phase).real
    powers = _edge_directional_integrals(left, right, gl[:, None, None], gr[:, None, None], end-start)
    time_weights = span*g.new_tensor(weights/2)/angular_frequency
    return tuple((p*time_weights).sum(dim=(-2, -1)) for p in powers)


def thermal_contact_edge_heat_steps(model: ThermalModel, result: ThermalTransientResult, *,
                                    start: float = 0., end: float = 1.) -> torch.Tensor:
    """Heat crossing each Q4 contact edge per step in J, shape (nt-1, n_edges).

    Positive heat leaves the first trace. The edge fractions follow
    thermal_contact_edge_power. Use the integrator's weighted stage field so
    that full-edge exchange balances the discrete energy equations.
    """
    power = thermal_contact_edge_power(model, result.interval_temperature, start=start, end=end)
    return result.time.diff().to(power)[:, None]*power


def assemble_thermal(model: ThermalModel, *, sparse: bool = False, complex_loads: bool = False) -> ThermalMatrices:
    """Assemble conductivity ``K``, capacity ``C`` and heat vector ``f``."""
    x = model.nodes
    load_like = x
    if complex_loads:
        dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        load_like = x.new_empty(0, dtype=dtype)
    conn = model.elements.long()
    ne, nen = conn.shape
    ke, ce, fe, _ = _thermal_element_matrices(model, load_like)
    nd = len(x)
    rows = conn[:,:,None].expand(-1,-1,nen).reshape(-1)
    cols = conn[:,None,:].expand(-1,nen,-1).reshape(-1)
    def matrix(entries, row, col):
        if sparse:
            return torch.sparse_coo_tensor(torch.stack((row, col)), entries.reshape(-1),
                                           (nd, nd)).coalesce()
        return x.new_zeros((nd, nd)).index_put((row, col), entries.reshape(-1), accumulate=True)

    K = matrix(ke, rows, cols)
    C = matrix(ce, rows, cols)
    if model.nodal_heat_capacity is not None:
        lump = torch.as_tensor(model.nodal_heat_capacity, dtype=x.dtype, device=x.device)
        if lump.shape != (nd,) or not torch.isfinite(lump).all() or torch.any(lump < 0):
            raise ValueError("nodal_heat_capacity needs one finite nonnegative capacity per node")
        indices = torch.arange(nd, device=x.device)
        C = (C + matrix(lump, indices, indices)).coalesce() if sparse else C + torch.diag(lump)
    f = load_like.new_zeros(nd).index_put((conn.reshape(-1),),fe.reshape(-1),accumulate=True)
    if model.nodal_heat is not None:
        if model.nodal_heat.shape != (nd,):
            raise ValueError("nodal_heat must have shape [n_nodes]")
        f = f + model.nodal_heat.to(load_like)
    if model.convection_edges is not None:
        edges = model.convection_edges.long().to(x.device)
        ec = x[edges]
        length = torch.linalg.vector_norm(ec[:,1]-ec[:,0],dim=1)
        h = _expand(model.convection_coefficient,len(edges),x)
        ta = torch.as_tensor(model.ambient_temperature,dtype=load_like.dtype,device=x.device)
        # In 1-D an endpoint is represented as [node,node] and has unit area.
        length = torch.where(length > 0,length,torch.ones_like(length))
        # ``thickness`` is cross-sectional area for LINE2 and out-of-plane
        # thickness for Q4 boundary edges.
        if model.convection_area is None:
            boundary_measure = length*_boundary_thickness(model,edges)
        else:
            area = torch.as_tensor(model.convection_area, dtype=x.dtype, device=x.device)
            if area.numel() not in (1,len(edges)) or not torch.isfinite(area).all() or (area < 0).any():
                raise ValueError("convection_area needs finite nonnegative areas per edge")
            boundary_measure = area.reshape(-1).expand(len(edges))
        edge_mass = boundary_measure[:,None,None]*x.new_tensor([[2.,1.],[1.,2.]])/6
        hk = h[:,None,None]*edge_mass
        if ta.shape == (len(edges), 2):
            hf = torch.einsum("eij,ej->ei", hk.to(ta.dtype), ta)
        else:
            ta = _expand(ta,len(edges),load_like)
            hf = (h*ta*boundary_measure/2)[:,None]*x.new_ones((len(edges),2))
        er = edges[:,:,None].expand(-1,-1,2).reshape(-1)
        ecidx = edges[:,None,:].expand(-1,2,-1).reshape(-1)
        if model.wall_areal_heat_capacity is not None:
            cw = torch.as_tensor(model.wall_areal_heat_capacity, dtype=x.dtype, device=x.device)
            if cw.numel() not in (1,len(edges)) or not torch.isfinite(cw).all() or (cw < 0).any():
                raise ValueError("wall_areal_heat_capacity needs finite nonnegative values per edge")
            C = C + matrix(cw.reshape(-1,1,1)*edge_mass, er, ecidx)
            if sparse:
                C = C.coalesce()
        if sparse:
            K = (K + matrix(hk, er, ecidx)).coalesce()
        else:
            K = K.index_put((er,ecidx),hk.reshape(-1),accumulate=True)
        f = f.index_put((edges.reshape(-1),),hf.reshape(-1),accumulate=True)
    if model.contact_pairs is not None:
        pairs,g = _contact_pair_data(model)
        left, right = pairs.unbind(dim=1)
        indices = (torch.cat((left, right, left, right)), torch.cat((left, right, right, left)))
        entries = torch.cat((g, g, -g, -g))
        if sparse:
            K = (K + matrix(entries, *indices)).coalesce()
        else:
            K = K.index_put(indices, entries, accumulate=True)
    if model.contact_edge_pairs is not None:
        pairs,mass = _contact_edge_matrix(model)
        ids = pairs.reshape(-1,4)
        top = torch.cat((mass,-mass),dim=2)
        blocks = torch.cat((top,-top),dim=1)
        row = ids[:,:,None].expand_as(blocks).reshape(-1)
        col = ids[:,None,:].expand_as(blocks).reshape(-1)
        K = K + matrix(blocks,row,col)
        if sparse:
            K = K.coalesce()
    return ThermalMatrices(K,C,f)


def _partition(model):
    fixed = model.fixed_nodes.long().to(model.nodes.device)
    if fixed.numel() and torch.unique(fixed).numel() != fixed.numel():
        raise ValueError("fixed_nodes must be unique")
    mask = torch.ones(len(model.nodes),dtype=torch.bool,device=model.nodes.device)
    mask[fixed] = False
    return fixed, torch.arange(len(model.nodes),device=model.nodes.device)[mask]


def _solve_sparse_dirichlet(operator, load, fixed, free, prescribed):
    from scipy.sparse.linalg import splu

    state = load.copy()
    fixed_ids, free_ids = fixed.numpy(), free.numpy()
    state[fixed_ids] = prescribed.resolve_conj().numpy()
    if free_ids.size:
        rhs = load[free_ids] - operator[free_ids][:,fixed_ids] @ state[fixed_ids]
        state[free_ids] = splu(operator[free_ids][:,free_ids].tocsc()).solve(rhs)
    return state


def apply_thermal_temperature_jump(model: ThermalModel, temperature: torch.Tensor,
                                   fixed_temperatures: torch.Tensor, *, sparse: bool=False,
                                   return_element_impulse: bool=False):
    """Apply an instantaneous change of Dirichlet temperature with consistent capacity.

    Return the right-limit temperature and inward nodal impulse heat (J).
    Pass the left-limit field and new temperatures in model.fixed_nodes order.
    Integrating C*Tdot across the jump gives (C*delta_T)[free]=0; the fixed
    rows give the reaction heat. Include these impulses when accumulating a
    heat budget across segments. With return_element_impulse=True, append the
    bulk consistent-capacity impulse in J at each element node (ne, nen).
    Additional nodal or boundary storage is included in the global reaction.
    The finite-element right-limit field can
    overshoot near the wall even if both prescribed temperatures are positive.
    """
    fixed,free = _partition(model)
    old = torch.as_tensor(temperature,dtype=model.nodes.dtype,device=model.nodes.device)
    target = torch.as_tensor(fixed_temperatures,dtype=model.nodes.dtype,device=model.nodes.device)
    if old.shape!=(len(model.nodes),) or target.shape!=fixed.shape:
        raise ValueError("provide one temperature per node and one target per fixed node")
    if not torch.isfinite(old).all() or not torch.isfinite(target).all():
        raise ValueError("jump temperatures must be finite")
    if not fixed.numel():
        unchanged = (old.clone(), torch.zeros_like(old))
        if return_element_impulse:
            return (*unchanged, old.new_zeros(model.elements.shape))
        return unchanged
    capacity = assemble_thermal(model,sparse=sparse).capacity
    change = target-old[fixed]
    if sparse:
        if capacity.device.type!="cpu" or any(t.requires_grad for t in (capacity,old,target)):
            raise ValueError("sparse temperature jump requires CPU tensors without autograd")
        delta = torch.from_numpy(_solve_sparse_dirichlet(_scipy_csc(capacity),old.new_zeros(old.shape).numpy(),
                                                         fixed,free,change))
    else:
        delta = torch.zeros_like(old).index_put((fixed,),change)
        if free.numel() and fixed.numel():
            rhs = -capacity[free][:,fixed] @ change
            delta = delta.index_put((free,),torch.linalg.solve(capacity[free][:,free],rhs))
    result = ((old+delta).index_put((fixed,),target), capacity @ delta)
    if return_element_impulse:
        _, ce, _, _ = _thermal_element_matrices(model, old)
        local = torch.einsum("eij,ej->ei", ce, delta[model.elements.long()])
        return (*result, local)
    return result


def solve_steady_thermal(model: ThermalModel, *, sparse: bool = False) -> torch.Tensor:
    """Solve steady conduction. sparse=True uses CPU SciPy LU without autograd."""
    matrices = assemble_thermal(model, sparse=sparse)
    fixed, free = _partition(model)
    prescribed = model.fixed_temperatures.to(dtype=model.nodes.dtype,device=model.nodes.device)
    if sparse:
        if any(t.device.type != "cpu" or t.requires_grad for t in
               (matrices.heat, matrices.conductivity, prescribed)):
            raise ValueError("sparse steady solution requires CPU tensors without autograd")
        state = _solve_sparse_dirichlet(_scipy_csc(matrices.conductivity),
                                        matrices.heat.numpy(), fixed, free, prescribed)
        return torch.from_numpy(state)
    temperature = model.nodes.new_zeros(len(model.nodes)).index_put((fixed,),prescribed)
    if free.numel():
        rhs = matrices.heat[free]-matrices.conductivity[free][:,fixed]@prescribed
        temperature[free] = torch.linalg.solve(matrices.conductivity[free][:,free],rhs)
    return temperature


@dataclass(frozen=True)
class ThermalHarmonicResult:
    temperature: torch.Tensor
    reaction_power: torch.Tensor
    mean_temperature: torch.Tensor | None = None
    mean_heating_rate: torch.Tensor | None = None


def solve_harmonic_thermal(model: ThermalModel, angular_frequency: float, *,
                           sparse: bool = False, separate_mean: bool = False,
                           return_mean_rate: bool = False) -> ThermalHarmonicResult:
    """Solve (K+iωC)T=f using the exp(iωt) convention, with ω in rad/s.

    Temperatures, nodal and volumetric loads, and convective ambient temperatures
    in the model are phasors. Material data remain real. Positive reaction
    power enters through prescribed temperatures; free-node entries are residuals.
    sparse=True uses CPU SciPy sparse LU for inputs without autograd.
    For fully insulated, lossless models, separate_mean=True returns the
    relative nodal temperature plus a separate mean_temperature per node.
    Each connected component has its own capacity-weighted mean. Pass both
    fields to thermal_element_nodal_power to retain precision at low frequency.
    Alternatively return_mean_rate=True gives the relative temperature and
    nodal mean_heating_rate (K/s), including at zero frequency for steady drift.
    """
    if separate_mean and return_mean_rate:
        raise ValueError("choose either separate_mean or return_mean_rate")
    if not math.isfinite(angular_frequency) or angular_frequency < 0 or (angular_frequency == 0 and not return_mean_rate):
        raise ValueError("angular_frequency must be finite and positive, or zero with return_mean_rate")
    m = assemble_thermal(model, sparse=sparse, complex_loads=True)
    fixed, free = _partition(model)
    prescribed = model.fixed_temperatures.to(m.heat)
    if not torch.isfinite(m.heat).all() or not torch.isfinite(prescribed).all():
        raise ValueError("thermal phasor loads and prescribed temperatures must be finite")
    if sparse and any(t.device.type != "cpu" or t.requires_grad for t in
                      (m.heat, m.conductivity, m.capacity, prescribed)):
        raise ValueError("sparse harmonic solution requires CPU tensors without autograd")
    if separate_mean or return_mean_rate:
        if fixed.numel() or any(v is not None and torch.any(torch.as_tensor(v) != 0)
                                for v in (model.convection_coefficient, model.heat_loss_coefficient)):
            raise ValueError("separate_mean or return_mean_rate requires an insulated model without temperature constraints or heat loss")
        return _solve_insulated_thermal(m, sparse, angular_frequency, return_mean_rate)
    if sparse:
        operator = _scipy_csc(m.conductivity) + (1j*angular_frequency)*_scipy_csc(m.capacity)
        load = m.heat.resolve_conj().numpy()
        state = _solve_sparse_dirichlet(operator, load, fixed, free, prescribed)
        return ThermalHarmonicResult(torch.from_numpy(state), torch.from_numpy(operator @ state - load))
    operator = m.conductivity + (1j*angular_frequency)*m.capacity
    temperature = torch.zeros_like(m.heat).index_put((fixed,), prescribed)
    if free.numel():
        rhs = m.heat[free] - operator[free][:,fixed] @ prescribed
        temperature = temperature.index_put((free,), torch.linalg.solve(operator[free][:,free], rhs))
    return ThermalHarmonicResult(temperature, operator @ temperature - m.heat)


@dataclass(frozen=True)
class ThermalLowFrequencyResult:
    steady: ThermalHarmonicResult
    first_order: ThermalHarmonicResult


def _solve_insulated_thermal(m, sparse, omega=None, return_mean_rate=False):
    """Solve on the zero-stored-energy subspace of each insulated component."""
    # Only the connectivity is discrete. Capacity weights and both dense
    # solutions remain in the autograd graph.
    k, c, f = m.conductivity, m.capacity, m.heat
    if sparse:
        from scipy import sparse as sp
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.linalg import splu
        import numpy as np

        k = _scipy_csc(k).astype(f.resolve_conj().numpy().dtype)
        k.eliminate_zeros()
        count, labels = connected_components(k,directed=False)
        c = _scipy_csc(c)
        basis = sp.csc_matrix((np.ones(len(labels),dtype=c.dtype),(np.arange(len(labels)),labels)),
                              shape=(len(labels),count))
        weights = c @ basis
        if np.any(np.asarray(weights.sum(axis=0)) <= 0):
            raise ValueError("each insulated component requires positive heat capacity")
        operator = k if omega is None else k+1j*omega*c
        border = sp.bmat([[operator,weights],[weights.T,None]],format="csc")
        groups = torch.from_numpy(labels).long()
    else:
        # Flood each connected component on the dense conductivity graph,
        # without requiring SciPy for the differentiable backend.
        graph = k.detach() != 0
        groups = torch.full((len(f),),-1,device=f.device,dtype=torch.long)
        count = 0
        while (groups < 0).any():
            active = torch.zeros_like(groups,dtype=torch.bool)
            active[torch.where(groups < 0)[0][0]] = True
            while active.any():
                groups[active] = count
                active = graph[active].any(0) & (groups < 0)
            count += 1
        basis = torch.nn.functional.one_hot(groups,count).to(c)
        weights = c @ basis
        if torch.any(weights.sum(0) <= 0):
            raise ValueError("each insulated component requires positive heat capacity")
        operator = k if omega is None else k+1j*omega*c
        border = torch.cat((torch.cat((operator,weights),1),
                            torch.cat((weights.T,k.new_zeros(count,count)),1)),0).to(f)

    net = f.new_zeros(count).index_add(0,groups,f)
    if omega is not None:
        if sparse:
            rate = net.resolve_conj().numpy()/np.asarray(weights.sum(axis=0)).ravel()
            rhs = f.resolve_conj().numpy()-weights @ rate
            u = splu(border).solve(np.r_[rhs,np.zeros(count,dtype=rhs.dtype)])[:len(f)]
            temperature = torch.from_numpy(u)
            residual = torch.from_numpy(operator @ u-rhs)
            nodal_rate = torch.from_numpy(rate[labels])
        else:
            rate = net/weights.sum(0)
            rhs = f-weights.to(f) @ rate
            temperature = torch.linalg.solve(border,torch.cat((rhs,f.new_zeros(count))))[:len(f)]
            residual = operator.to(f) @ temperature-rhs
            nodal_rate = rate[groups]
        if return_mean_rate:
            return ThermalHarmonicResult(temperature, residual, mean_heating_rate=nodal_rate)
        return ThermalHarmonicResult(temperature, residual, mean_temperature=nodal_rate/(1j*omega))
    scale = f.real.new_zeros(count).index_add(0,groups,f.abs())
    if torch.any(net.abs() > 64*torch.finfo(scale.dtype).eps*scale):
        raise ValueError("each insulated component requires zero net heat input")
    if sparse:
        load = f.resolve_conj().numpy()
        factor = splu(border)
        def solve(rhs):
            return factor.solve(np.r_[rhs,np.zeros(count,dtype=rhs.dtype)])[:len(f)]
        u = solve(load)
        lag = solve(-(c @ u))
        return ThermalLowFrequencyResult(
            ThermalHarmonicResult(torch.from_numpy(u),torch.from_numpy(k @ u-load)),
            ThermalHarmonicResult(torch.from_numpy(lag),torch.from_numpy(k @ lag+c @ u)))
    factors = torch.linalg.lu_factor(border)
    def solve(rhs):
        return torch.linalg.lu_solve(*factors,torch.cat((rhs,f.new_zeros(count)))[:,None])[:len(f),0]
    k,c = k.to(f),c.to(f)
    u = solve(f)
    lag = solve(-(c @ u))
    return ThermalLowFrequencyResult(ThermalHarmonicResult(u,k @ u-f),
                                     ThermalHarmonicResult(lag,k @ lag+c @ u))


def solve_low_frequency_thermal(model: ThermalModel, *, sparse: bool=False) -> ThermalLowFrequencyResult:
    """Return constant and i*omega coefficients of temperature and reaction power.

    Assume frequency-independent loads and prescribed-temperature phasors.
    For fully insulated, lossless bodies each connected component must have
    balanced heat input and positive heat capacity. Both coefficients then
    have zero capacity-weighted mean on each component (the harmonic limit).
    Otherwise require a nonsingular steady free-node matrix. With exp(+i*omega*t),
    K*T0=f and K*T1=-C*T0 on free nodes. Prescribed nodes have zero T1.
    First-order temperature and reaction power are in K s and W s, respectively.
    """
    m = assemble_thermal(model,sparse=sparse,complex_loads=True)
    fixed,free = _partition(model)
    prescribed = model.fixed_temperatures.to(m.heat)
    if not torch.isfinite(m.heat).all() or not torch.isfinite(prescribed).all():
        raise ValueError("thermal phasor loads and prescribed temperatures must be finite")
    if sparse and any(t.device.type!="cpu" or t.requires_grad
                      for t in (m.conductivity,m.capacity,m.heat,prescribed)):
        raise ValueError("sparse low-frequency solution requires CPU tensors without autograd")
    if not fixed.numel() and all(value is None or not torch.any(torch.as_tensor(value) != 0)
                                 for value in (model.convection_coefficient,model.heat_loss_coefficient)):
        return _solve_insulated_thermal(m,sparse)
    if sparse:
        from scipy.sparse.linalg import splu
        f = m.heat.resolve_conj().numpy()
        k = _scipy_csc(m.conductivity).astype(f.dtype)
        c = _scipy_csc(m.capacity)
        u = f.copy()
        u[fixed.numpy()] = prescribed.resolve_conj().numpy()
        lag = f*0.
        ids = free.numpy()
        if ids.size:
            factor = splu(k[ids][:,ids].tocsc())
            u[ids] = factor.solve(f[ids]-k[ids][:,fixed.numpy()] @ u[fixed.numpy()])
            lag[ids] = factor.solve(-(c @ u)[ids])
        return ThermalLowFrequencyResult(
            ThermalHarmonicResult(torch.from_numpy(u),torch.from_numpy(k @ u-f)),
            ThermalHarmonicResult(torch.from_numpy(lag),torch.from_numpy(k @ lag+c @ u)))
    k,c = m.conductivity.to(m.heat.dtype),m.capacity.to(m.heat.dtype)
    u = torch.zeros_like(m.heat).index_put((fixed,),prescribed)
    lag = torch.zeros_like(u)
    if free.numel():
        factors = torch.linalg.lu_factor(k[free][:,free])
        rhs = m.heat[free]-k[free][:,fixed] @ prescribed
        u = u.index_put((free,),torch.linalg.lu_solve(*factors,rhs[:,None])[:,0])
        rhs = -(c @ u)[free]
        lag = lag.index_put((free,),torch.linalg.lu_solve(*factors,rhs[:,None])[:,0])
    return ThermalLowFrequencyResult(ThermalHarmonicResult(u,k @ u-m.heat),
                                     ThermalHarmonicResult(lag,k @ lag+c @ u))


def thermal_contact_edge_low_frequency_power(model: ThermalModel, result: ThermalLowFrequencyResult,
                                           *, start: float = 0., end: float = 1.):
    """Return the steady (W) and iω (W s) contact power coefficients per Q4 edge.

    Positive power enters the second trace from the first. Each coefficient
    integrates the same fractional edge interval for fixed contact conductance.
    """
    fields = torch.stack((result.steady.temperature, result.first_order.temperature))
    return thermal_contact_edge_power(model, fields, start=start, end=end).unbind(0)


def thermal_element_low_frequency_power(model: ThermalModel, result: ThermalLowFrequencyResult):
    """Return steady and first-order inward nodal powers for each element (W, W s).

    Both arrays have shape (elements, nodes per element). Element volume
    loads enter the steady balance. Bulk heat loss acts at both orders,
    and consistent material capacity acts on the steady temperature at first
    order. Surface films, contacts and lumped loads remain external exchanges.
    """
    u = result.steady.temperature
    lag = result.first_order.temperature
    if u.shape!=(len(model.nodes),) or lag.shape!=u.shape:
        raise ValueError("both temperature coefficients must have one value per node")
    conn = model.elements.long()
    k,c,f,_ = _thermal_element_matrices(model,u)
    k = k.to(u)
    t0 = u[conn].unsqueeze(-1)
    t1 = lag[conn].unsqueeze(-1)
    return (k @ t0).squeeze(-1)-f,(k @ t1+c.to(u) @ t0).squeeze(-1)


def solve_transient_thermal(model: ThermalModel, time: torch.Tensor,
                            initial_temperature: torch.Tensor, theta: float=1.0, *,
                            sparse: bool=False, fixed_temperature_history: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None=None,
                            nodal_heat_history: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None=None,
                            nodal_step_heat: torch.Tensor | None=None, method: str="theta") -> ThermalTransientResult:
    """Integrate ``C Tdot + K T = f`` using the theta method.

    ``theta >= 0.5`` is required to make the stable public path fail closed.
    Backward Euler (1.0) and Crank--Nicolson (0.5) are supported. sparse=True
    uses CPU SciPy sparse LU and returns torch COO matrices. Use the default
    dense path for autograd or GPU execution. fixed_temperature_history can
    prescribe one temperature per time and fixed node, in model.fixed_nodes
    order. Its first row sets the initial constrained temperatures. A callable
    accepts scalar time and returns one temperature per fixed node. SDIRK2
    samples it at the internal stages as well as at output times.
    nodal_heat_history adds inward applied powers in W, shaped (nt, nn),
    to the static assembled loads. It may also be a function of scalar time
    returning one power per node. SDIRK2 evaluates such functions at its two
    stage times; theta integration samples the output times.
    nodal_step_heat additionally supplies integrated inward energy in joules
    for each interval and node, shape (nt-1, nn). method="sdirk2" uses the
    L-stable two-stage second-order scheme, linearly interpolating endpoint
    powers and prescribed temperatures and distributing step heat uniformly
    over its interval. interval_temperature returns the stage-weighted field
    used for integrated Fourier fluxes. theta selects the theta method weight.

    """
    if method not in ("theta","sdirk2"):
        raise ValueError("method must be theta or sdirk2")
    if method=="sdirk2":
        # The RK quadrature integrates a linear input at its midpoint on average.
        theta = .5
    if not 0.5 <= theta <= 1.0:
        raise ValueError("theta must be in [0.5, 1] for unconditional stability")
    if time.ndim != 1 or time.numel() < 2 or torch.any(time[1:] <= time[:-1]):
        raise ValueError("time must be strictly increasing")
    if initial_temperature.shape != (len(model.nodes),):
        raise ValueError("initial_temperature must have shape [n_nodes]")
    m = assemble_thermal(model, sparse=sparse); fixed, free = _partition(model)
    prescribed = model.fixed_temperatures.to(dtype=model.nodes.dtype,device=model.nodes.device)
    def sample_history(fn,t,count,name):
        value = torch.as_tensor(fn(t)).to(model.nodes)
        if value.shape!=(count,) or not torch.isfinite(value).all():
            raise ValueError(f"{name} callable must return {count} finite nodal values")
        return value

    fixed_stage_history = None
    if method=="sdirk2":
        gamma = 1-1/math.sqrt(2)
        first_times = time[:-1]+gamma*time.diff()
    if fixed_temperature_history is not None:
        if callable(fixed_temperature_history):
            def temperatures(t):
                return sample_history(fixed_temperature_history,t,fixed.numel(),"fixed_temperature_history")
            prescribed = torch.stack([temperatures(t) for t in time])
            if method=="sdirk2":
                fixed_stage_history = torch.stack([temperatures(t) for t in first_times])
        else:
            prescribed = torch.as_tensor(fixed_temperature_history).to(model.nodes)
        if prescribed.shape != (len(time), fixed.numel()) or not torch.isfinite(prescribed).all():
            raise ValueError("fixed_temperature_history needs finite values at every time and fixed node")
    stage_power = first_offset = None
    if nodal_heat_history is not None:
        def sample_power(t):
            return sample_history(nodal_heat_history,t,len(model.nodes),"nodal_heat_history")

        if callable(nodal_heat_history) and method=="sdirk2":
            first = torch.stack([sample_power(t) for t in first_times])
            end = torch.stack([sample_power(t) for t in time[1:]])
            stage_power = (1-gamma)*first+gamma*end
            first_offset = first-stage_power
        else:
            powers = (torch.stack([sample_power(t) for t in time]) if callable(nodal_heat_history)
                      else torch.as_tensor(nodal_heat_history).to(model.nodes))
            if powers.shape != (len(time), len(model.nodes)) or not torch.isfinite(powers).all():
                raise ValueError("nodal_heat_history needs finite powers at every time and node")
            stage_power = (1-theta)*powers[:-1] + theta*powers[1:]
            if method=="sdirk2":
                first_offset = (.5-1/math.sqrt(2))*powers.diff(dim=0)
    if nodal_step_heat is not None:
        energy = torch.as_tensor(nodal_step_heat).to(model.nodes)
        if energy.shape != (len(time)-1,len(model.nodes)) or not torch.isfinite(energy).all():
            raise ValueError("nodal_step_heat needs finite joules per interval and node")
        mean_power = energy/time.diff().to(energy)[:,None]
        stage_power = mean_power if stage_power is None else stage_power + mean_power
    out = model.nodes.new_zeros((len(time),len(model.nodes)))
    out[0] = initial_temperature.to(dtype=model.nodes.dtype,device=model.nodes.device)
    out[:,fixed] = prescribed
    if sparse:
        inputs = (time, out, m.heat, m.conductivity, m.capacity, stage_power, fixed_stage_history, first_offset)
        if any(v.device.type != "cpu" or v.requires_grad for v in inputs if v is not None):
            raise ValueError("sparse transient integration requires CPU tensors without autograd")
    if method=="sdirk2":
        means = _advance_sdirk2(m,time,out,fixed,free,stage_power,first_offset,fixed_stage_history,sparse)
        return ThermalTransientResult(time,out,m,theta,stage_power,means)
    if sparse:
        if free.numel():
            _advance_sparse_thermal(m, time, out, fixed, free, prescribed, theta, stage_power)
        return ThermalTransientResult(time, out, m, theta, stage_power)
    Kff=m.conductivity[free][:,free]; Cff=m.capacity[free][:,free]
    Kfc=m.conductivity[free][:,fixed]
    source = m.heat[free]
    if prescribed.ndim == 1:
        source = source - Kfc @ prescribed
    else:
        Cfc = m.capacity[free][:,fixed]
    for i,dt in enumerate(time[1:]-time[:-1]):
        A=Cff/dt+theta*Kff
        rhs=(Cff/dt-(1-theta)*Kff)@out[i,free]+source
        if stage_power is not None:
            rhs = rhs + stage_power[i, free]
        if prescribed.ndim == 2:
            change = prescribed[i+1] - prescribed[i]
            rhs = rhs - Kfc @ (prescribed[i] + theta*change) - Cfc @ (change/dt)
        out[i+1,free]=torch.linalg.solve(A,rhs)
    return ThermalTransientResult(time, out, m, theta, stage_power)


def _scipy_csc(tensor):
    from scipy.sparse import coo_matrix

    row, col = tensor.indices().numpy()
    return coo_matrix((tensor.values().numpy(), (row, col)), shape=tensor.shape).tocsc()


def _advance_sdirk2(m,time,out,fixed,free,average_power,first_offset,fixed_stage_history,sparse):
    """Solve both stages using the same constrained operator on each interval."""
    gamma = 1-1/math.sqrt(2)
    if not free.numel():
        means = .5*(out[:-1]+out[1:])
        if fixed_stage_history is not None:
            means[:,fixed] = (1-gamma)*fixed_stage_history+gamma*out[1:,fixed]
        return means
    if sparse:
        from scipy.sparse.linalg import splu
        fixed,free = fixed.numpy(),free.numpy()
        if fixed_stage_history is not None:
            fixed_stage_history = fixed_stage_history.numpy()
        state = out.numpy()
        c = _scipy_csc(m.capacity)[free]
        k = _scipy_csc(m.conductivity)[free]
        source = m.heat.numpy()[free]
        steps = time.diff().numpy()
        if average_power is not None:
            average_power = average_power.numpy()
        if first_offset is not None:
            first_offset = first_offset.numpy()
    else:
        state,c,k,source,steps = out,m.capacity[free],m.conductivity[free],m.heat[free],time.diff()
    cff,kff,cfc,kfc = c[:,free],k[:,free],c[:,fixed],k[:,fixed]
    means = torch.empty_like(out[:-1])
    previous = None
    for i,dt in enumerate(steps):
        shift = gamma*dt
        if sparse:
            if dt!=previous:
                factor = splu((cff+shift*kff).tocsc())
                previous = dt
            solve = factor.solve
        else:
            lu,piv = torch.linalg.lu_factor(cff+shift*kff)
            def solve(rhs):
                return torch.linalg.lu_solve(lu,piv,rhs[:,None]).squeeze(1)
        old = state[i,free]
        fixed_start,fixed_end = state[i,fixed],state[i+1,fixed]
        fixed_change = fixed_end-fixed_start
        fixed_stage = fixed_start+gamma*fixed_change if fixed_stage_history is None else fixed_stage_history[i]
        fixed_mean = (1-gamma)*fixed_stage+gamma*fixed_end
        load = source if average_power is None else source+average_power[i,free]
        first_load = load if first_offset is None else load+first_offset[i,free]
        stored = cff@old
        boundary_storage = cfc@fixed_change
        first = solve(stored-cfc@(fixed_stage-fixed_start)+shift*(first_load-kfc@fixed_stage))
        end = solve(stored-boundary_storage+dt*(load-(1-gamma)*(kff@first)-kfc@fixed_mean))
        state[i+1,free] = end
        means[i,free] = torch.as_tensor((1-gamma)*first+gamma*end)
        means[i,fixed] = torch.as_tensor(fixed_mean)
    return means


def _advance_sparse_thermal(matrices, time, out, fixed, free, prescribed, theta, stage_power):
    try:
        from scipy.sparse.linalg import splu
    except ImportError as exc:
        raise RuntimeError("sparse thermal integration requires SciPy") from exc

    free = free.numpy()
    k_rows = _scipy_csc(matrices.conductivity)[free]
    kfc = k_rows[:, fixed.numpy()]
    c_rows = _scipy_csc(matrices.capacity)[free]
    source = matrices.heat.numpy()[free]
    prescribed = prescribed.numpy()
    if prescribed.ndim == 1:
        source = source - kfc @ prescribed
    else:
        cfc = c_rows[:, fixed.numpy()]
    k = k_rows[:, free]
    c = c_rows[:, free]
    if stage_power is not None:
        stage_power = stage_power.numpy()
    history = out.numpy()
    previous_dt = None
    for i, dt in enumerate(time.diff().numpy()):
        if dt != previous_dt:
            factor = splu((c + (theta*dt)*k).tocsc())
            previous_dt = dt
        state = history[i, free]
        rhs = c @ state + dt*(source - (1-theta)*(k @ state))
        if stage_power is not None:
            rhs += dt*stage_power[i, free]
        if prescribed.ndim == 2:
            change = prescribed[i+1] - prescribed[i]
            rhs -= dt*(kfc @ (prescribed[i] + theta*change)) + cfc @ change
        history[i+1, free] = factor.solve(rhs)


@dataclass(frozen=True)
class ThermoelasticBarResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    axial_force: torch.Tensor


def solve_thermoelastic_bar(nodes: torch.Tensor, elements: torch.Tensor,
                            young_modulus, area, expansion_coefficient,
                            temperature: torch.Tensor, reference_temperature: float,
                            fixed_nodes: torch.Tensor) -> ThermoelasticBarResult:
    """Sequential 1-D thermoelastic solve using a supplied thermal field."""
    conn=elements.long(); ne=len(conn); x=nodes[:,0]; lengths=x[conn[:,1]]-x[conn[:,0]]
    if torch.any(lengths <= 0): raise ValueError("bar element must have positive length")
    E=_expand(young_modulus,ne,nodes); A=_expand(area,ne,nodes); alpha=_expand(expansion_coefficient,ne,nodes)
    ke=(E*A/lengths)[:,None,None]*nodes.new_tensor([[1.,-1.],[-1.,1.]])
    dT=temperature[conn].mean(1)-reference_temperature
    fe=(E*A*alpha*dT)[:,None]*nodes.new_tensor([-1.,1.])
    rows=conn[:,:,None].expand(-1,-1,2).reshape(-1); cols=conn[:,None,:].expand(-1,2,-1).reshape(-1)
    K=nodes.new_zeros((len(nodes),len(nodes))).index_put((rows,cols),ke.reshape(-1),accumulate=True)
    f=nodes.new_zeros(len(nodes)).index_put((conn.reshape(-1),),fe.reshape(-1),accumulate=True)
    fixed=fixed_nodes.long(); mask=torch.ones(len(nodes),dtype=torch.bool,device=nodes.device); mask[fixed]=False
    free=torch.arange(len(nodes),device=nodes.device)[mask]; u=nodes.new_zeros(len(nodes))
    if free.numel(): u[free]=torch.linalg.solve(K[free][:,free],f[free])
    reaction=K@u-f
    strain=(u[conn[:,1]]-u[conn[:,0]])/lengths
    force=E*A*(strain-alpha*dT)
    return ThermoelasticBarResult(u,reaction,force)
