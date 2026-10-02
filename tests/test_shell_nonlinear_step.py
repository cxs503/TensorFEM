import torch

from tensorfem.shell_nonlinear_step import (CylindricalShellMesh, assemble_shell,
                                             solve_shell_step)
from tensorfem.corotational_shell import axis_angle, cylindrical_nodes

D=torch.float64


def _strip():
    xs=(0.,1.,2.); ts=(-.1,.1)
    p=torch.tensor([(x,t) for x in xs for t in ts],dtype=D)
    e=torch.tensor([[0,2,3,1],[2,4,5,3]],dtype=torch.long)
    return CylindricalShellMesh(p,e,5.,2e8,0.,.02)


def _axial_problem(force=1000.):
    mesh=_strip(); ndof=6*len(mesh.parameters); loads=torch.zeros(ndof,dtype=D)
    loads[6*4]=force/2; loads[6*5]=force/2
    prescribed={}
    for node in range(6):
        for component in range(1,6): prescribed[6*node+component]=0.
    prescribed[0]=0.; prescribed[6]=0.
    return mesh,loads,prescribed


def test_two_element_strip_matches_axial_bar_below_three_percent(tmp_path):
    mesh,loads,prescribed=_axial_problem(); cp=tmp_path/"shell-checkpoint.json"
    result=solve_shell_step(mesh,loads,prescribed,initial_increment=.5,checkpoint=cp)
    assert result.converged and result.load_factor==1 and cp.exists()
    area=mesh.thickness*mesh.radius*.2
    exact=1000.*2./(mesh.young*area)
    numeric=float(result.dofs.reshape(-1,6)[4:,0].mean())
    assert abs(numeric/exact-1.) < .03
    assert abs(float(result.reaction[[0,6]].sum())+1000.) < 1e-6


def test_checkpoint_restart_reaches_same_solution(tmp_path):
    mesh,loads,prescribed=_axial_problem(); cp=tmp_path/"restart.json"
    first=solve_shell_step(mesh,.5*loads,prescribed,initial_increment=.5,checkpoint=cp)
    assert first.converged
    # Rewrite the proportional half-load solution as factor .5 of full loading.
    from tensorfem.shell_nonlinear_step import write_shell_checkpoint
    write_shell_checkpoint(cp,first.dofs,.5)
    resumed=solve_shell_step(mesh,loads,prescribed,restart=cp,initial_increment=.5)
    direct=solve_shell_step(mesh,loads,prescribed,initial_increment=.5)
    assert resumed.converged and torch.allclose(resumed.dofs,direct.dofs,rtol=1e-9,atol=1e-12)


def test_failed_increment_rolls_back_and_fails_closed():
    mesh,loads,prescribed=_axial_problem()
    result=solve_shell_step(mesh,loads,prescribed,initial_increment=.25,
                            minimum_increment=.2,max_iterations=1)
    assert not result.converged and result.load_factor==0.
    assert torch.count_nonzero(result.dofs)==0


def test_global_mesh_energy_is_objective_under_finite_rigid_motion():
    mesh=_strip(); X=cylindrical_nodes(mesh.parameters,mesh.radius)
    R=axis_angle(torch.tensor([.3,-.2,1.],dtype=D),1.1); moved=X@R.T+torch.tensor([4.,-7.,2.],dtype=D)
    # Build exact global displacement/rotation-vector data for the rigid state.
    axis=torch.tensor([.3,-.2,1.],dtype=D); axis=axis/torch.linalg.vector_norm(axis)
    q=torch.zeros((len(X),6),dtype=D); q[:,:3]=moved-X; q[:,3:]=1.1*axis
    energy,internal,_=assemble_shell(mesh,q.reshape(-1),tangent=False)
    scale=mesh.young*mesh.thickness*2.*mesh.radius*.2
    assert abs(float(energy))/scale < 1e-26
    assert float(torch.linalg.vector_norm(internal))/scale < 2e-12
