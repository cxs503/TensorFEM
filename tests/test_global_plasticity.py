import pytest
import torch

from tensorfem.global_plasticity import solve_tet4_j2_path
from tensorfem.nonlinear_step import NonlinearConvergenceError
from tensorfem.solid_plasticity import Tet4J2Model

D = torch.float64


def bar_model():
    # Two cubes along x, each split into six conforming tetrahedra.
    nodes=[]
    for x in (0., .5, 1.):
        for z in (0., 1.):
            for y in (0., 1.): nodes.append((x,y,z))
    nodes=torch.tensor(nodes,dtype=D)
    elements=[]
    # local ordering: 000,010,001,011,100,110,101,111 due loops above
    pattern=((0,6,4,7),(0,2,6,7),(0,3,2,7),(0,1,3,7),(0,5,1,7),(0,4,5,7))
    for slab in (0,1):
        left=4*slab; mapping=(left,left+1,left+2,left+3,left+4,left+5,left+6,left+7)
        elements.extend(tuple(mapping[i] for i in tet) for tet in pattern)
    # Symmetry planes x=0, y=0, z=0 permit exact homogeneous uniaxial stress.
    fixed=[]
    for i,(x,y,z) in enumerate(nodes.tolist()):
        if x==0.: fixed.append(3*i)
        if y==0.: fixed.append(3*i+1)
        if z==0.: fixed.append(3*i+2)
    return Tet4J2Model(nodes,torch.tensor(elements,dtype=torch.long),200000.,.3,
                       250.,10000.,torch.tensor(sorted(set(fixed)),dtype=torch.long))


def end_traction(model, stress=400.):
    f=torch.zeros(model.n_dofs,dtype=D)
    # Consistent nodal resultants on the two triangles of the x=1 unit face.
    face_nodes=torch.nonzero(model.nodes[:,0]==1.,as_tuple=False)[:,0]
    # The surface diagonal is local 4--7: two triangle-consistent loads.
    f[3*face_nodes]=stress*torch.tensor([1/3,1/6,1/6,1/3],dtype=D)
    return f


def test_multi_element_uniaxial_elastoplastic_analytical_solution():
    m=bar_model(); results=solve_tet4_j2_path(m,end_traction(m),[1.],initial_increment=.08)
    r=results[-1]; exact=400./m.young+(400.-m.yield_stress)/m.hardening
    tip=r.displacement[torch.nonzero(m.nodes[:,0]==1.)[:,0]*3]
    assert torch.max(torch.abs(tip-exact))/exact < 1e-9
    assert torch.max(torch.abs(r.stress[:,0]-400.))/400. < 1e-9
    assert len(r.material_state.points)==m.elements.shape[0]==12
    assert len(r.increments)>1


def test_elastic_plastic_transition_unload_and_restart_are_path_equivalent():
    m=bar_model(); f=end_traction(m)
    full=solve_tet4_j2_path(m,f,[.5,.625,1.,0.],initial_increment=.1)
    checkpoint=solve_tet4_j2_path(m,f,[.5,.625],initial_increment=.1)[-1]
    restarted=solve_tet4_j2_path(m,f,[1.,0.],initial=checkpoint,initial_increment=.1)
    assert torch.equal(full[-1].displacement,restarted[-1].displacement)
    for a,b in zip(full[-1].material_state.points,restarted[-1].material_state.points):
        assert torch.equal(a.plastic_strain,b.plastic_strain)
        assert torch.equal(a.alpha,b.alpha)
    expected=(400.-250.)/m.hardening
    tip=full[-1].displacement[torch.nonzero(m.nodes[:,0]==1.)[:,0]*3]
    assert torch.max(torch.abs(tip-expected))/expected < 1e-9
    assert torch.max(torch.abs(full[-1].stress)) < 1e-8


def test_failure_does_not_mutate_restart_state():
    m=bar_model(); f=end_traction(m)
    checkpoint=solve_tet4_j2_path(m,f,[.5])[-1]
    before=tuple((p.plastic_strain.clone(),p.alpha.clone()) for p in checkpoint.material_state.points)
    with pytest.raises(NonlinearConvergenceError):
        solve_tet4_j2_path(m,f,[1.],initial=checkpoint,max_iterations=1,
                           initial_increment=.1,minimum_increment=.02)
    for p,(strain,alpha) in zip(checkpoint.material_state.points,before):
        assert torch.equal(p.plastic_strain,strain) and torch.equal(p.alpha,alpha)


def test_invalid_topology_fails_closed():
    m=bar_model()
    bad=Tet4J2Model(m.nodes,m.elements.to(torch.int32),m.young,m.poisson,
                    m.yield_stress,m.hardening,m.fixed_dofs)
    with pytest.raises(TypeError,match="torch.long"):
        solve_tet4_j2_path(bad,end_traction(m),[1.])
