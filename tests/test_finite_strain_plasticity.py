import math,pytest,torch
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.finite_strain_plasticity import (MultiplicativeJ2State,FinitePlasticState,
 update_multiplicative_j2,assemble_finite_plastic,solve_finite_plastic_path)
from tensorfem.nonlinear_step import NonlinearConvergenceError
D=torch.float64

def virgin(): return MultiplicativeJ2State(torch.eye(3,dtype=D),torch.zeros((),dtype=D))

def test_objectivity_isochoric_flow_and_nonnegative_dissipation():
    e=.18; F=torch.diag(torch.tensor([math.exp(e),math.exp(-e/2),math.exp(-e/2)],dtype=D))
    a=1.1; R=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]],dtype=D)
    P,s,d,y=update_multiplicative_j2(F,1000.,.3,40.,80.,virgin())
    Pr,sr,dr,yr=update_multiplicative_j2(R@F,1000.,.3,40.,80.,virgin())
    assert torch.linalg.vector_norm(Pr-R@P)/torch.linalg.vector_norm(P)<1e-11
    assert abs(float(torch.linalg.det(s.plastic_gradient))-1)<1e-12
    assert float(d)>=0 and abs(float(d-dr))<1e-12 and abs(float(y))<1e-10

def test_finite_uniaxial_isochoric_closed_form_and_unload_residual():
    E,nu,sy,H=1000.,.3,40.,80.; G=E/(2*(1+nu)); strain=.18
    F=torch.diag(torch.tensor([math.exp(strain),math.exp(-strain/2),math.exp(-strain/2)],dtype=D))
    P,s,d,y=update_multiplicative_j2(F,E,nu,sy,H,virgin())
    dg=(3*G*strain-sy)/(3*G+H); exact_q=sy+H*dg
    tau=P@F.T; q=torch.sqrt(1.5*torch.sum((tau-torch.trace(tau)/3*torch.eye(3,dtype=D))**2))
    assert abs(float(q)-exact_q)/exact_q<1e-10
    assert abs(float(s.alpha)-dg)/dg<1e-10
    P0,_,_,_=update_multiplicative_j2(torch.eye(3,dtype=D),E,nu,sy,H,s)
    assert torch.linalg.vector_norm(P0)>0 # residual stress after total unloading

def one_tet():
    nodes=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
    return FiniteStrainTet4Model(nodes,torch.tensor([[0,1,2,3]]),1000.,.3,
      torch.tensor([0,1,2,4,5,6,7,8,9,10,11]))

def test_global_newton_restart_and_algorithmic_tangent():
    m=one_tet(); sy,H=40.,80.; force=torch.zeros(m.n_dofs,dtype=D); force[3]=18.
    direct=solve_finite_plastic_path(m,force,[1.],yield_stress=sy,hardening=H,initial_increment=.05)[-1]
    half=solve_finite_plastic_path(m,force,[.5],yield_stress=sy,hardening=H)[-1]
    restart=solve_finite_plastic_path(m,force,[1.],yield_stress=sy,hardening=H,initial=half)[-1]
    assert abs(float(direct.displacement[3]-restart.displacement[3]))/abs(float(direct.displacement[3]))<1e-6
    assert all(float(x)>=0 for x in direct.dissipation)
    assert max(len(i.iterations) for i in direct.increments)<=8
    # Independently recheck the numerical tangent against a second step size.
    state=FinitePlasticState.virgin(m); u=direct.displacement*.7
    f,k,_,_,_=assemble_finite_plastic(m,u,state,yield_stress=sy,hardening=H,difference_step=2e-7)
    h=8e-7; up=u.clone(); um=u.clone(); up[3]+=h; um[3]-=h
    fp=assemble_finite_plastic(m,up,state,yield_stress=sy,hardening=H,tangent=False)[0]
    fm=assemble_finite_plastic(m,um,state,yield_stress=sy,hardening=H,tangent=False)[0]
    assert torch.linalg.vector_norm((fp-fm)/(2*h)-k[:,3])/torch.linalg.vector_norm(k[:,3])<1e-5

def test_inversion_fails_closed():
    with pytest.raises(ValueError,match="inverted"):
        update_multiplicative_j2(torch.diag(torch.tensor([-1.,1.,1.],dtype=D)),1000.,.3,40.,80.,virgin())

def test_failed_increment_rolls_back_restart_state():
    m=one_tet(); force=torch.zeros(m.n_dofs,dtype=D); force[3]=18.
    checkpoint=solve_finite_plastic_path(m,force,[.4],yield_stress=40.,hardening=80.)[-1]
    before=checkpoint.state.points[0].plastic_gradient.clone()
    with pytest.raises(NonlinearConvergenceError):
        solve_finite_plastic_path(m,force,[1.],yield_stress=40.,hardening=80.,initial=checkpoint,
                                  max_iterations=1,initial_increment=.1,minimum_increment=.02)
    assert torch.equal(checkpoint.state.points[0].plastic_gradient,before)

def test_multi_element_global_assembly_has_independent_point_states():
    base=one_tet(); model=FiniteStrainTet4Model(base.reference_nodes,
        torch.tensor([[0,1,2,3],[0,1,2,3]]),base.young,base.poisson,base.fixed_dofs)
    state=FinitePlasticState.virgin(model); u=torch.zeros(model.n_dofs,dtype=D); u[3]=.12
    force,k,stress,trial,diss=assemble_finite_plastic(model,u,state,yield_stress=40.,hardening=80.)
    single=assemble_finite_plastic(base,u,FinitePlasticState.virgin(base),yield_stress=40.,hardening=80.)[0]
    assert torch.allclose(force,2*single,rtol=1e-10,atol=1e-10)
    assert len(trial.points)==2 and stress.shape==(2,3,3) and diss.shape==(2,)
