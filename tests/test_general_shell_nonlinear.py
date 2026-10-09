import math
import torch

from tensorfem.general_shell_nonlinear import (GeneralShellMesh,
    general_shell_energy,general_shell_force_tangent,solve_general_shell)
from tensorfem.spherical_shell import hemisphere_with_hole
from tensorfem.corotational_shell import axis_angle

D=torch.float64


def _panel():
    # Two adjacent facets sampled from a sphere: normals vary in both directions.
    phi=torch.tensor([.45,.75],dtype=D);theta=torch.tensor([0.,.35,.7],dtype=D)
    gp,gt=torch.meshgrid(phi,theta,indexing="ij")
    x=5*torch.stack((torch.sin(gp)*torch.cos(gt),torch.sin(gp)*torch.sin(gt),torch.cos(gp)),-1).reshape(-1,3)
    return GeneralShellMesh(x,torch.tensor([[0,3,4,1],[1,4,5,2]]),2e7,.25,.03)


def _problem():
    mesh=_panel();loads=torch.zeros(36,dtype=D);normal=mesh.nodes[4]/5;loads[24:27]=-.05*normal
    fixed={}
    for node in (0,1,2):
        for k in range(6):fixed[6*node+k]=0.
    # Suppress drilling mechanisms at the loaded edge while retaining bending.
    fixed[6*3+5]=fixed[6*4+5]=fixed[6*5+5]=0.
    return mesh,loads,fixed,24


def test_general_shell_tangent_matches_finite_difference():
    mesh=_panel();ref=mesh.nodes[mesh.elements[0]];q=torch.linspace(-2e-4,2e-4,24,dtype=D)
    _,_,K=general_shell_force_tangent(ref,q,mesh.young,mesh.poisson,mesh.thickness)
    d=torch.linspace(-1.,1.,24,dtype=D);d/=torch.linalg.vector_norm(d);h=1e-7
    qp=(q+h*d).requires_grad_();qm=(q-h*d).requires_grad_()
    fp=torch.autograd.grad(general_shell_energy(ref,qp,mesh.young,mesh.poisson,mesh.thickness),qp)[0]
    fm=torch.autograd.grad(general_shell_energy(ref,qm,mesh.young,mesh.poisson,mesh.thickness),qm)[0]
    assert torch.linalg.vector_norm((fp-fm)/(2*h)-K@d)/torch.linalg.vector_norm(K@d)<3e-6


def test_doubly_curved_global_path_and_restart(tmp_path):
    mesh,loads,fixed,probe=_problem();cp=tmp_path/"general-shell.json"
    direct=solve_general_shell(mesh,loads,fixed,probe_dof=probe,increment=.25,checkpoint=cp,tolerance=1e-6)
    assert direct.converged and len(direct.path)>=3
    values=[abs(v) for _,v in direct.path];assert all(b>a for a,b in zip(values,values[1:]))
    restarted=solve_general_shell(mesh,loads,fixed,probe_dof=probe,restart=cp,tolerance=1e-6)
    assert restarted.converged and torch.allclose(restarted.dofs,direct.dofs)


def test_multifacet_large_rigid_rotation_is_objective():
    mesh=_panel();axis=torch.tensor([.2,-.4,1.],dtype=D);axis/=torch.linalg.vector_norm(axis);angle=1.4
    R=axis_angle(axis,angle);moved=mesh.nodes@R.T+torch.tensor([3.,-8.,2.],dtype=D)
    q=torch.zeros((len(mesh.nodes),6),dtype=D);q[:,:3]=moved-mesh.nodes;q[:,3:]=angle*axis
    energy=sum(general_shell_energy(mesh.nodes[e],q[e].reshape(-1),mesh.young,mesh.poisson,mesh.thickness)
               for e in mesh.elements)
    scale=mesh.young*mesh.thickness
    assert abs(float(energy))/scale<1e-25


def test_failed_increment_rolls_back():
    mesh,loads,fixed,probe=_problem()
    result=solve_general_shell(mesh,loads,fixed,probe_dof=probe,increment=.25,
                               minimum_increment=.2,max_iterations=1,tolerance=1e-14)
    assert not result.converged and result.load_factor==0.
    assert torch.count_nonzero(result.dofs)==0


def test_hemisphere_spatial_convergence_stays_below_three_percent():
    results=[hemisphere_with_hole(n) for n in (20,24,28)]
    assert max(r.relative_error for r in results)<.03
    assert abs(results[-1].displacement/results[-2].displacement-1.)<.03
