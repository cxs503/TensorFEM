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
            ke = ke+(thick*length/12)[:,None,None]*entries.reshape(ne,2,2)
        elif loss is not None:
            ke = ke+loss[:,None,None]*measure
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
        g = 3.0**-0.5
        for xi, eta in ((-g,-g),(g,-g),(g,g),(-g,g)):
            n, grad, det = _q4_geometry(coords, xi, eta)
            weight = det*thick
            weighted_grad = grad @ k if k.ndim == 3 else grad * k[:, None, None]
            ke = ke + (weighted_grad @ grad.transpose(1, 2)) * weight[:, None, None]
            measure = weight[:,None,None]*(n[None,:,None]*n[None,None,:])
            ce = ce+(rho*cp)[:,None,None]*measure
            if loss is not None and not variable_loss:
                ke = ke+loss[:,None,None]*measure
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
                        ke = ke+(weight*(loss @ n))[:,None,None]*torch.outer(n,n)
                    if quadratic is not None:
                        # With a bilinear map, N_i*q(x,y)*det(J) has degree at
                        # most four in either parent coordinate.
                        px, py = (coords.transpose(1, 2) @ n).unbind(1)
                        a, b, c = quadratic.unbind(1)
                        power = weight*(a*px.square() + b*px*py + c*py.square())
                        fe = fe + power[:, None]*n
    return ke, ce, fe


def thermal_element_nodal_power(model: ThermalModel, temperature: torch.Tensor,
                                angular_frequency: float = 0., *, element_nodal_heat=None) -> torch.Tensor:
    """Inward power phasors (W) at LINE2 or Q4 element nodes under exp(+iωt).

    Zero frequency recovers steady powers. The return dtype is complex.
    Includes the consistent storage term and subtracts the model's element
    volume loads. Supply any additional element-resolved nodal loads (ne, nen)
    in W to subtract them before assembly. Global point loads, films and
    contacts otherwise appear as exchanges. Q4 entries are weak-form nodal powers.
    """
    if not math.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValueError("angular_frequency must be finite and nonnegative")
    if temperature.shape != (len(model.nodes),):
        raise ValueError("temperature must have one value per node")
    dtype = torch.promote_types(model.nodes.dtype, torch.promote_types(temperature.dtype, torch.complex64))
    values = temperature.to(device=model.nodes.device, dtype=dtype)[model.elements.long()]
    ke, ce, fe = _thermal_element_matrices(model, values)
    power = torch.einsum("eij,ej->ei", ke.to(dtype) + 1j*angular_frequency*ce.to(dtype), values) - fe
    if element_nodal_heat is not None:
        load = torch.as_tensor(element_nodal_heat, dtype=dtype, device=values.device)
        if load.shape != values.shape or not torch.isfinite(load).all():
            raise ValueError("element_nodal_heat needs finite powers at every element node")
        power = power - load
    return power


def thermal_element_nodal_heat_steps(model: ThermalModel, result: ThermalTransientResult, *,
                                    element_nodal_heat_history: torch.Tensor | None = None,
                                    element_nodal_step_heat: torch.Tensor | None = None) -> torch.Tensor:
    """Inward energy at each element node, shaped (nt-1, ne, nen), in J.

    Uses local consistent capacity, conductivity and static volumetric sources.
    An optional (nt, ne, nen) array assigns additional applied powers (W) to
    individual element nodes, before global assembly. element_nodal_step_heat
    additionally assigns integrated joules of shape (nt-1, ne, nen). Supply each distributed
    load once. Global point loads, films and contacts appear as exchanges unless
    explicitly allocated to element nodes. For LINE2, negate the right entry
    for energy leaving the right end. Q4 entries are nodal weak-form quantities.
    """
    ke, ce, fe = _thermal_element_matrices(model, model.nodes)
    t = result.temperature[:, model.elements.long()]
    stage = result.interval_temperature[:, model.elements.long()]
    heat = torch.einsum("eij,tej->tei", ce, t.diff(dim=0))
    power = torch.einsum("eij,tej->tei", ke, stage) - fe
    if element_nodal_heat_history is not None:
        load = torch.as_tensor(element_nodal_heat_history).to(t)
        if load.shape != t.shape or not torch.isfinite(load).all():
            raise ValueError("element_nodal_heat_history needs finite powers at every time and element node")
        power = power - ((1-result.theta)*load[:-1] + result.theta*load[1:])
    if element_nodal_step_heat is not None:
        supplied = torch.as_tensor(element_nodal_step_heat).to(t)
        if supplied.shape != heat.shape or not torch.isfinite(supplied).all():
            raise ValueError("element_nodal_step_heat needs finite joules per interval and element node")
        heat = heat - supplied
    return heat + result.time.diff().to(t)[:, None, None]*power


def thermal_element_section_power(model: ThermalModel, nodal_power: torch.Tensor, fractions, *,
                                  net_line_source_gradient=None, net_line_source_curvature=None) -> torch.Tensor:
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


def assemble_thermal(model: ThermalModel, *, sparse: bool = False, complex_loads: bool = False) -> ThermalMatrices:
    """Assemble conductivity ``K``, capacity ``C`` and heat vector ``f``."""
    x = model.nodes
    load_like = x
    if complex_loads:
        dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        load_like = x.new_empty(0, dtype=dtype)
    conn = model.elements.long()
    ne, nen = conn.shape
    ke, ce, fe = _thermal_element_matrices(model, load_like)
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
        pairs = model.contact_pairs.to(device=x.device, dtype=torch.long)
        g = torch.as_tensor(model.contact_conductance, dtype=x.dtype, device=x.device)
        if g.numel() not in (1, len(pairs)) or not torch.isfinite(g).all() or (g < 0).any():
            raise ValueError("contact_conductance must be finite, nonnegative, scalar or one per pair")
        g = g.reshape(-1).expand(len(pairs))
        left, right = pairs.unbind(dim=1)
        indices = (torch.cat((left, right, left, right)), torch.cat((left, right, right, left)))
        entries = torch.cat((g, g, -g, -g))
        if sparse:
            K = (K + matrix(entries, *indices)).coalesce()
        else:
            K = K.index_put(indices, entries, accumulate=True)
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
                                   fixed_temperatures: torch.Tensor, *, sparse: bool=False):
    """Apply an instantaneous change of Dirichlet temperature with consistent capacity.

    Return the right-limit temperature and inward nodal impulse heat (J).
    Pass the left-limit field and new temperatures in model.fixed_nodes order.
    Integrating C*Tdot across the jump gives (C*delta_T)[free]=0; the fixed
    rows give the reaction heat. Include these impulses when accumulating a
    heat budget across segments. The finite-element right-limit field can
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
        return old.clone(),torch.zeros_like(old)
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
    return (old+delta).index_put((fixed,),target),capacity @ delta


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


def solve_harmonic_thermal(model: ThermalModel, angular_frequency: float, *,
                           sparse: bool = False) -> ThermalHarmonicResult:
    """Solve (K+iωC)T=f using the exp(iωt) convention, with ω in rad/s.

    Temperatures, nodal and volumetric loads, and convective ambient temperatures
    in the model are phasors. Material data remain real. Positive reaction
    power enters through prescribed temperatures; free-node entries are residuals.
    sparse=True uses CPU SciPy sparse LU for inputs without autograd.
    """
    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    m = assemble_thermal(model, sparse=sparse, complex_loads=True)
    fixed, free = _partition(model)
    prescribed = model.fixed_temperatures.to(m.heat)
    if not torch.isfinite(m.heat).all() or not torch.isfinite(prescribed).all():
        raise ValueError("thermal phasor loads and prescribed temperatures must be finite")
    if sparse:
        if any(t.device.type != "cpu" or t.requires_grad for t in
               (m.heat, m.conductivity, m.capacity, prescribed)):
            raise ValueError("sparse harmonic solution requires CPU tensors without autograd")
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


def solve_low_frequency_thermal(model: ThermalModel, *, sparse: bool=False) -> ThermalLowFrequencyResult:
    """Return constant and i*omega coefficients of temperature and reaction power.

    Assume frequency-independent loads and prescribed-temperature phasors, and
    a nonsingular steady free-node conductivity matrix. With exp(+i*omega*t),
    K*T0=f and K*T1=-C*T0 on free nodes. Prescribed nodes have zero T1.
    First-order temperature and reaction power are in K s and W s, respectively.
    """
    m = assemble_thermal(model,sparse=sparse,complex_loads=True)
    fixed,free = _partition(model)
    prescribed = model.fixed_temperatures.to(m.heat)
    if not torch.isfinite(m.heat).all() or not torch.isfinite(prescribed).all():
        raise ValueError("thermal phasor loads and prescribed temperatures must be finite")
    if sparse:
        from scipy.sparse.linalg import splu
        if any(t.device.type!="cpu" or t.requires_grad for t in (m.conductivity,m.capacity,m.heat,prescribed)):
            raise ValueError("sparse low-frequency solution requires CPU tensors without autograd")
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
    k,c,f = _thermal_element_matrices(model,u)
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
