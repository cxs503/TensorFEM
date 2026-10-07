import math
import pytest
import torch

from tensorfem.finite_strain_elasticity import (FiniteStrainTet4Model,
    assemble_finite_tet4,neo_hookean_response,solve_finite_elastic_path)
from tensorfem.nonlinear_step import NonlinearConvergenceError

D=torch.float64


def rotation_z(angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=D)


def test_large_rotation_objectivity_and_rigid_rotation_gate():
    R=rotation_z(1.37); U=torch.diag(torch.tensor([1.3,.85,1.1],dtype=D))
    e0,P0,s0=neo_hookean_response(U,1200.,.29)
    e1,P1,s1=neo_hookean_response(R@U,1200.,.29)
    assert abs(float(e1-e0)) < 1e-12
    assert torch.linalg.vector_norm(s1-R@s0@R.T)/torch.linalg.vector_norm(s0) < 1e-12
    er,Pr,sr=neo_hookean_response(R,1200.,.29)
    assert abs(float(er))<1e-12 and torch.linalg.vector_norm(Pr)<1e-12


def bar_model():
    nodes=torch.tensor([(x,y,z) for x in (0.,.5,1.) for z in (0.,1.) for y in (0.,1.)],dtype=D)
    pat=((0,6,4,7),(0,2,6,7),(0,3,2,7),(0,1,3,7),(0,5,1,7),(0,4,5,7)); elements=[]
    for s in (0,1):
        mp=tuple(4*s+i for i in range(8)); elements += [tuple(mp[i] for i in t) for t in pat]
    # Uniaxial-strain reference: transverse DOFs fixed; x fixed at left end.
    fixed=[]
    for i,(x,_,_) in enumerate(nodes.tolist()):
        if x==0: fixed.append(3*i)
        fixed.extend((3*i+1,3*i+2))
    return FiniteStrainTet4Model(nodes,torch.tensor(elements),1000.,.3,
                                 torch.tensor(sorted(set(fixed))))


def traction_for_stretch(model,stretch):
    F=torch.diag(torch.tensor([stretch,1.,1.],dtype=D)); _,P,_=neo_hookean_response(F,model.young,model.poisson)
    f=torch.zeros(model.n_dofs,dtype=D); face=torch.nonzero(model.reference_nodes[:,0]==1.)[:,0]
    f[3*face]=P[0,0]*torch.tensor([1/3,1/6,1/6,1/3],dtype=D)
    return f,float(P[0,0])


def test_finite_extension_matches_independent_nominal_stress_reference_and_restart():
    m=bar_model(); f,nominal=traction_for_stretch(m,1.5)
    direct=solve_finite_elastic_path(m,f,[1.],initial_increment=.08)[-1]
    first=solve_finite_elastic_path(m,f,[.4],initial_increment=.08)[-1]
    restarted=solve_finite_elastic_path(m,f,[1.],initial=first,initial_increment=.08)[-1]
    tip=direct.displacement[3*torch.nonzero(m.reference_nodes[:,0]==1.)[:,0]]
    assert torch.max(torch.abs(tip-.5))/.5 < 1e-9
    assert torch.max(torch.abs(direct.first_piola[:,0,0]-nominal))/abs(nominal)<1e-9
    restart_error=torch.linalg.vector_norm(direct.displacement-restarted.displacement)/torch.linalg.vector_norm(direct.displacement)
    assert restart_error < 1e-10
    assert torch.min(direct.jacobian)>.0


def test_exact_tangent_matches_force_difference():
    m=bar_model(); u=torch.zeros(m.n_dofs,dtype=D)
    u[3*torch.nonzero(m.reference_nodes[:,0]>.0)[:,0]]=.08*m.reference_nodes[m.reference_nodes[:,0]>.0,0]
    force,k,_,_,_=assemble_finite_tet4(m,u)
    j=3*8; h=1e-6; up=u.clone(); um=u.clone(); up[j]+=h; um[j]-=h
    fp=assemble_finite_tet4(m,up,tangent=False)[0]; fm=assemble_finite_tet4(m,um,tangent=False)[0]
    assert torch.linalg.vector_norm((fp-fm)/(2*h)-k[:,j])/torch.linalg.vector_norm(k[:,j])<1e-8


def test_inverted_deformation_fails_closed():
    with pytest.raises(ValueError,match="inverted"):
        neo_hookean_response(torch.diag(torch.tensor([-1.,1.,1.],dtype=D)),1000.,.3)


def test_failed_restart_is_transactional():
    m=bar_model(); f,_=traction_for_stretch(m,1.5)
    checkpoint=solve_finite_elastic_path(m,f,[.2])[-1]
    before=checkpoint.displacement.clone()
    with pytest.raises(NonlinearConvergenceError):
        solve_finite_elastic_path(m,f,[1.],initial=checkpoint,max_iterations=1,
                                  initial_increment=.1,minimum_increment=.02)
    assert torch.equal(checkpoint.displacement,before)
