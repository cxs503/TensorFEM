"""True 3-D two-solid spherical-cap Hertz contact path."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .finite_strain_contact3d import (
    build_curved_nonmatching_two_block_contact,solve_finite_strain_contact_path)
from .hertz_3d_qualification import deformable_hertz_reference

@dataclass(frozen=True)
class Hertz3DFEResult:
    cells: int
    force: float
    reference_force: float
    relative_error: float
    residual_norm: float
    minimum_jacobian: float
    force_imbalance: float
    contact_radius_reference: float
    domain_radii: float
    maximum_tet_edge_ratio: float

def solve_hertz_cap_block(*,cells:int,approach:float=.005,radius:float=8.,
    young:float=1.e3,poisson:float=.3,normal_penalty:float=1.e4,
    lateral_size:float=1.,block_depth:float=1.,center_grading:float=0.,
    vertical_cells:int=1,vertical_grading:float=1.,maximum_edge_ratio:float=12.):
    """Solve imposed approach of two real TET4 solids with a faceted cap."""
    if approach<=0: raise ValueError("approach must be positive")
    model=build_curved_nonmatching_two_block_contact(master_cells=cells,
        slave_cells=cells+1,clearance=-approach,radius=radius,young=young,
        poisson=poisson,normal_penalty=normal_penalty,lateral_size=lateral_size,
        block_depth=block_depth,center_grading=center_grading,
        vertical_cells=vertical_cells,vertical_grading=vertical_grading)
    tet=model.solid.reference_nodes[model.solid.elements]
    pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    lengths=torch.stack(tuple(torch.linalg.vector_norm(tet[:,i]-tet[:,j],dim=1)
                              for i,j in pairs),1)
    edge_ratio=float(torch.max(lengths.max(1).values/lengths.min(1).values))
    if edge_ratio>maximum_edge_ratio:
        raise ValueError(f"Hertz mesh edge ratio {edge_ratio:.6g} exceeds quality gate {maximum_edge_ratio}")
    zero=torch.zeros(model.solid.n_dofs,dtype=torch.float64)
    step=solve_finite_strain_contact_path(model,[zero],tolerance=1.e-9,
        max_iterations=40)[0]
    force=float(torch.linalg.vector_norm(step.contact.slave_forces.sum(0)))
    # The paraboloid is the local spherical expansion; two elastic bodies use
    # the standard reduced modulus and prescribed mutual approach.
    ref=deformable_hertz_reference(force=1.,radius=radius,
        young_sphere=young,poisson_sphere=poisson,
        young_halfspace=young,poisson_halfspace=poisson)
    reference_force=4*ref.effective_modulus*math.sqrt(radius)*approach**1.5/3
    imbalance=float(torch.linalg.vector_norm(
        step.contact.slave_forces.sum(0)+step.contact.master_forces.sum(0)))
    contact_radius=math.sqrt(radius*approach)
    return Hertz3DFEResult(cells,force,reference_force,
        abs(force/reference_force-1),step.residual_norm,
        float(torch.min(step.jacobian)),imbalance,contact_radius,
        lateral_size/(2*contact_radius),edge_ratio)
