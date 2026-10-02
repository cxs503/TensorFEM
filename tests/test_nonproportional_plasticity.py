import math,torch,pytest
from tensorfem.nonproportional_plasticity import (virgin_adaptive_state,integrate_adaptive,
 integrate_path,richardson_algorithmic_tangent,fixed_substep_reference)
D=torch.float64; MAT=(1000.,.3,35.,60.)

def history():
    shear=torch.tensor([[1.,.28,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
    a=.63; c,s=math.cos(a),math.sin(a); R=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=D)
    stretch=R@torch.diag(torch.tensor([1.22,1/1.22,1.],dtype=D))@R.T
    return shear,stretch@shear

def test_noncoaxial_adaptive_matches_independent_fine_reference():
    F1,F2=history(); adaptive=integrate_path((F1,F2),virgin_adaptive_state(),*MAT,rtol=2e-5)
    s=virgin_adaptive_state(); r1=fixed_substep_reference(F1,s,*MAT,nsteps=400)
    ref=fixed_substep_reference(F2,r1.state,*MAT,nsteps=400)
    fp_err=torch.linalg.vector_norm(adaptive[-1].state.plastic.plastic_gradient-ref.state.plastic.plastic_gradient)/torch.linalg.vector_norm(ref.state.plastic.plastic_gradient)
    p_err=torch.linalg.vector_norm(adaptive[-1].first_piola-ref.first_piola)/torch.linalg.vector_norm(ref.first_piola)
    assert fp_err<3e-3 and p_err<3e-3 # well inside 3 percent
    assert abs(float(torch.linalg.det(adaptive[-1].state.plastic.plastic_gradient))-1)<1e-10
    assert all(float(x.dissipation)>=0 for x in adaptive)
    assert abs(float(adaptive[-1].yield_value))<2e-3

def test_objectivity_path_dependence_cycle_and_restart():
    F1,F2=history(); base=integrate_path((F1,F2,torch.eye(3,dtype=D)),virgin_adaptive_state(),*MAT,rtol=2e-5)
    a=.9; R=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]],dtype=D)
    rotated_initial=virgin_adaptive_state(); rotated_initial=type(rotated_initial)(rotated_initial.plastic,R)
    rotated=integrate_path(tuple(R@F for F in (F1,F2,torch.eye(3,dtype=D))),rotated_initial,*MAT,rtol=2e-5)
    assert torch.linalg.vector_norm(rotated[-1].first_piola-R@base[-1].first_piola)/torch.linalg.vector_norm(base[-1].first_piola)<1e-8
    first=integrate_path((F1,),virgin_adaptive_state(),*MAT,rtol=2e-5)[-1]
    restart=integrate_path((F2,torch.eye(3,dtype=D)),first.state,*MAT,rtol=2e-5)
    assert torch.allclose(restart[-1].state.plastic.plastic_gradient,base[-1].state.plastic.plastic_gradient,rtol=1e-12,atol=1e-12)
    reverse=integrate_path((F2,F1,torch.eye(3,dtype=D)),virgin_adaptive_state(),*MAT,rtol=2e-5)
    assert torch.linalg.vector_norm(reverse[-1].state.plastic.plastic_gradient-base[-1].state.plastic.plastic_gradient)>1e-3
    assert torch.linalg.vector_norm(base[-1].first_piola)>0 # residual stress after cycle

def test_richardson_tangent_and_material_newton():
    F1,F2=history(); committed=integrate_adaptive(F1,virgin_adaptive_state(),*MAT,rtol=2e-5).state
    C=richardson_algorithmic_tangent(F2,committed,*MAT,step=3e-5,rtol=2e-5)
    # Independent smaller central difference for one shear column.
    h=4e-6; plus=F2.clone(); minus=F2.clone(); plus[0,1]+=h; minus[0,1]-=h
    pp=integrate_adaptive(plus,committed,*MAT,rtol=2e-5).first_piola
    pm=integrate_adaptive(minus,committed,*MAT,rtol=2e-5).first_piola
    ref=(pp-pm)/(2*h)
    assert torch.linalg.vector_norm(C[:,:,0,1]-ref)/torch.linalg.vector_norm(ref)<2e-3
    target=float(integrate_adaptive(F2,committed,*MAT,rtol=2e-5).first_piola[0,1]); gamma=float(F2[0,1])*.8
    for iteration in range(8):
        F=F2.clone(); F[0,1]=gamma
        result=integrate_adaptive(F,committed,*MAT,rtol=2e-5)
        Ct=richardson_algorithmic_tangent(F,committed,*MAT,step=3e-5,rtol=2e-5)
        residual=float(result.first_piola[0,1])-target
        if abs(residual)<1e-8: break
        gamma-=residual/float(Ct[0,1,0,1])
    assert abs(residual)<1e-8 and iteration<7

def test_substep_failure_is_transactional():
    F1,_=history(); state=virgin_adaptive_state(); before=state.plastic.plastic_gradient.clone()
    with pytest.raises(RuntimeError,match="tolerance"):
        integrate_adaptive(F1,state,*MAT,rtol=1e-16,max_depth=0)
    assert torch.equal(state.plastic.plastic_gradient,before)
