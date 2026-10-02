import math,torch,pytest
from tensorfem.adaptive_global_plasticity import AdaptiveGlobalState,assemble_adaptive_global,solve_adaptive_global_path
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonproportional_plasticity import virgin_adaptive_state,integrate_path,fixed_substep_reference
from tensorfem.nonlinear_step import NonlinearConvergenceError
D=torch.float64; MAT=(1000.,.3,35.,60.)

def complex_path():
    S=torch.tensor([[1.,.24,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
    L=torch.diag(torch.tensor([1.18,1/1.18,1.],dtype=D))@S
    R=torch.tensor([[1.,-.18,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)@L
    return S,L,R,torch.eye(3,dtype=D)

def test_shear_stretch_reverse_cycle_matches_ultrafine_reference():
    path=complex_path(); adaptive=integrate_path(path,virgin_adaptive_state(),*MAT,rtol=2e-5)
    state=virgin_adaptive_state(); refs=[]
    for F in path:
        r=fixed_substep_reference(F,state,*MAT,nsteps=500); refs.append(r); state=r.state
    fp=torch.linalg.vector_norm(adaptive[-1].state.plastic.plastic_gradient-refs[-1].state.plastic.plastic_gradient)/torch.linalg.vector_norm(refs[-1].state.plastic.plastic_gradient)
    stress=torch.linalg.vector_norm(adaptive[-1].first_piola-refs[-1].first_piola)/torch.linalg.vector_norm(refs[-1].first_piola)
    assert fp<.003 and stress<.01
    assert sum(float(x.dissipation) for x in adaptive)>=0
    assert abs(float(torch.linalg.det(adaptive[-1].state.plastic.plastic_gradient))-1)<1e-10

def model():
    x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
    return FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3]]),1000.,.3,
      torch.tensor([0,1,2,4,5,6,7,8,9,10,11]))

def test_global_cyclic_newton_restart_dissipation_and_rollback():
    m=model(); f=torch.zeros(m.n_dofs,dtype=D); f[3]=16.
    targets=(.7,1.,-.35,.55,0.)
    direct=solve_adaptive_global_path(m,f,targets,yield_stress=35.,hardening=60.,material_rtol=3e-5)
    checkpoint=solve_adaptive_global_path(m,f,targets[:2],yield_stress=35.,hardening=60.,material_rtol=3e-5)[-1]
    restart=solve_adaptive_global_path(m,f,targets[2:],yield_stress=35.,hardening=60.,initial=checkpoint,material_rtol=3e-5)
    assert abs(float(direct[-1].displacement[3]-restart[-1].displacement[3]))<2e-7
    assert all(float(torch.min(r.increment_dissipation))>=-1e-12 for r in direct)
    assert abs(float(torch.linalg.det(direct[-1].state.points[0].plastic.plastic_gradient))-1)<1e-10
    assert max(len(i.iterations) for r in direct for i in r.increments)<=10
    before=checkpoint.state.points[0].plastic.plastic_gradient.clone()
    with pytest.raises(NonlinearConvergenceError):
        solve_adaptive_global_path(m,f,[1.5],yield_stress=35.,hardening=60.,initial=checkpoint,
          max_iterations=1,initial_increment=.1,minimum_increment=.02,material_rtol=3e-5)
    assert torch.equal(checkpoint.state.points[0].plastic.plastic_gradient,before)

def test_global_richardson_tangent_independent_difference():
    m=model(); state=AdaptiveGlobalState.virgin(m); u=torch.zeros(m.n_dofs,dtype=D); u[3]=.08
    fi,k,_,_,_,evaluations=assemble_adaptive_global(m,u,state,yield_stress=35.,hardening=60.,material_rtol=3e-5)
    h=4e-6; up=u.clone(); um=u.clone(); up[3]+=h; um[3]-=h
    fp=assemble_adaptive_global(m,up,state,yield_stress=35.,hardening=60.,material_rtol=3e-5,tangent=False)[0]
    fm=assemble_adaptive_global(m,um,state,yield_stress=35.,hardening=60.,material_rtol=3e-5,tangent=False)[0]
    assert torch.linalg.vector_norm((fp-fm)/(2*h)-k[:,3])/torch.linalg.vector_norm(k[:,3])<3e-3
    assert evaluations==49
