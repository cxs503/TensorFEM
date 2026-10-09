"""Small, differentiable heat-conduction and sequential thermoelastic kernels.

The implementation intentionally mirrors the dense reference solvers used by
TensorFEM's verified element kernels.  It supports LINE2 and Q4 conduction,
consistent heat capacity, prescribed temperature, nodal heat input, boundary
convection and element volumetric heat generation.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThermalModel:
    nodes: torch.Tensor
    elements: torch.Tensor
    conductivity: torch.Tensor
    density: torch.Tensor
    specific_heat: torch.Tensor
    fixed_nodes: torch.Tensor
    fixed_temperatures: torch.Tensor
    nodal_heat: torch.Tensor | None = None
    heat_source: torch.Tensor | None = None
    thickness: torch.Tensor | None = None
    convection_edges: torch.Tensor | None = None
    convection_coefficient: torch.Tensor | None = None
    ambient_temperature: torch.Tensor | None = None
    element_type: str = "line2"

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


def _expand(value, ne, like, default=1.0):
    if value is None:
        return like.new_full((ne,), default)
    return torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(-1).expand(ne)


def assemble_thermal(model: ThermalModel) -> ThermalMatrices:
    """Assemble conductivity ``K``, capacity ``C`` and heat vector ``f``."""
    x = model.nodes
    conn = model.elements.long()
    ne, nen = conn.shape
    k = _expand(model.conductivity, ne, x)
    rho = _expand(model.density, ne, x)
    cp = _expand(model.specific_heat, ne, x)
    thick = _expand(model.thickness, ne, x)
    source = _expand(model.heat_source, ne, x, 0.0)
    ke = x.new_zeros((ne, nen, nen))
    ce = torch.zeros_like(ke)
    fe = x.new_zeros((ne, nen))
    coords = x[conn]
    if model.element_type.lower() == "line2":
        length = coords[:, 1, 0] - coords[:, 0, 0]
        if torch.any(length <= 0):
            raise ValueError("LINE2 element must have positive length")
        base_k = x.new_tensor([[1., -1.], [-1., 1.]])
        base_c = x.new_tensor([[2., 1.], [1., 2.]])
        ke = k[:, None, None] * thick[:, None, None] / length[:, None, None] * base_k
        ce = (rho*cp*thick*length/6)[:, None, None] * base_c
        fe = (source*thick*length/2)[:, None] * x.new_ones((ne, 2))
    else:
        g = 3.0**-0.5
        for xi, eta in ((-g,-g),(g,-g),(g,g),(-g,g)):
            n = x.new_tensor([(1-xi)*(1-eta),(1+xi)*(1-eta),
                              (1+xi)*(1+eta),(1-xi)*(1+eta)]) / 4
            dn = x.new_tensor([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],
                               [(1+eta),(1+xi)],[-(1+eta),(1-xi)]]) / 4
            jac = torch.einsum("eia,ib->eab", coords, dn)
            det = torch.linalg.det(jac)
            if torch.any(det <= 0):
                raise ValueError("Q4 element has non-positive Jacobian")
            grad = torch.einsum("ib,ebc->eic", dn, torch.linalg.inv(jac))
            weight = det*thick
            ke = ke + k[:,None,None]*(grad @ grad.transpose(1,2))*weight[:,None,None]
            ce = ce + (rho*cp*weight)[:,None,None]*(n[None,:,None]*n[None,None,:])
            fe = fe + (source*weight)[:,None]*n[None,:]
    nd = len(x)
    rows = conn[:,:,None].expand(-1,-1,nen).reshape(-1)
    cols = conn[:,None,:].expand(-1,nen,-1).reshape(-1)
    K = x.new_zeros((nd,nd)).index_put((rows,cols),ke.reshape(-1),accumulate=True)
    C = x.new_zeros((nd,nd)).index_put((rows,cols),ce.reshape(-1),accumulate=True)
    f = x.new_zeros(nd).index_put((conn.reshape(-1),),fe.reshape(-1),accumulate=True)
    if model.nodal_heat is not None:
        if model.nodal_heat.shape != (nd,):
            raise ValueError("nodal_heat must have shape [n_nodes]")
        f = f + model.nodal_heat.to(dtype=x.dtype,device=x.device)
    if model.convection_edges is not None:
        edges = model.convection_edges.long().to(x.device)
        ec = x[edges]
        length = torch.linalg.vector_norm(ec[:,1]-ec[:,0],dim=1)
        h = _expand(model.convection_coefficient,len(edges),x)
        ta = _expand(model.ambient_temperature,len(edges),x)
        # In 1-D an endpoint is represented as [node,node] and has unit area.
        length = torch.where(length > 0,length,torch.ones_like(length))
        # ``thickness`` is cross-sectional area for LINE2 and out-of-plane
        # thickness for Q4 boundary edges.
        boundary_measure = length*_expand(model.thickness,len(edges),x)
        hk = (h*boundary_measure)[:,None,None]*x.new_tensor([[2.,1.],[1.,2.]])/6
        hf = (h*ta*boundary_measure/2)[:,None]*x.new_ones((len(edges),2))
        er = edges[:,:,None].expand(-1,-1,2).reshape(-1)
        ecidx = edges[:,None,:].expand(-1,2,-1).reshape(-1)
        K = K.index_put((er,ecidx),hk.reshape(-1),accumulate=True)
        f = f.index_put((edges.reshape(-1),),hf.reshape(-1),accumulate=True)
    return ThermalMatrices(K,C,f)


def _partition(model):
    fixed = model.fixed_nodes.long().to(model.nodes.device)
    if fixed.numel() and torch.unique(fixed).numel() != fixed.numel():
        raise ValueError("fixed_nodes must be unique")
    mask = torch.ones(len(model.nodes),dtype=torch.bool,device=model.nodes.device)
    mask[fixed] = False
    return fixed, torch.arange(len(model.nodes),device=model.nodes.device)[mask]


def solve_steady_thermal(model: ThermalModel) -> torch.Tensor:
    matrices = assemble_thermal(model)
    fixed, free = _partition(model)
    prescribed = model.fixed_temperatures.to(dtype=model.nodes.dtype,device=model.nodes.device)
    temperature = model.nodes.new_zeros(len(model.nodes)).index_put((fixed,),prescribed)
    if free.numel():
        rhs = matrices.heat[free]-matrices.conductivity[free][:,fixed]@prescribed
        temperature[free] = torch.linalg.solve(matrices.conductivity[free][:,free],rhs)
    return temperature


def solve_transient_thermal(model: ThermalModel, time: torch.Tensor,
                            initial_temperature: torch.Tensor, theta: float=1.0) -> ThermalTransientResult:
    """Integrate ``C Tdot + K T = f`` using the theta method.

    ``theta >= 0.5`` is required to make the stable public path fail closed.
    Backward Euler (1.0) and Crank--Nicolson (0.5) are supported.
    """
    if not 0.5 <= theta <= 1.0:
        raise ValueError("theta must be in [0.5, 1] for unconditional stability")
    if time.ndim != 1 or time.numel() < 2 or torch.any(time[1:] <= time[:-1]):
        raise ValueError("time must be strictly increasing")
    if initial_temperature.shape != (len(model.nodes),):
        raise ValueError("initial_temperature must have shape [n_nodes]")
    m = assemble_thermal(model); fixed, free = _partition(model)
    prescribed = model.fixed_temperatures.to(dtype=model.nodes.dtype,device=model.nodes.device)
    out = model.nodes.new_zeros((len(time),len(model.nodes)))
    out[0] = initial_temperature.to(dtype=model.nodes.dtype,device=model.nodes.device)
    out[:,fixed] = prescribed
    Kff=m.conductivity[free][:,free]; Cff=m.capacity[free][:,free]
    Kfc=m.conductivity[free][:,fixed]
    rhs_const=m.heat[free]-Kfc@prescribed
    for i,dt in enumerate(time[1:]-time[:-1]):
        A=Cff/dt+theta*Kff
        rhs=(Cff/dt-(1-theta)*Kff)@out[i,free]+rhs_const
        out[i+1,free]=torch.linalg.solve(A,rhs)
    return ThermalTransientResult(time,out,m)


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
