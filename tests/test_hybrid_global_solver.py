import torch,pytest
from tensorfem.hybrid_global_solver import solve_hybrid_global_path
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonlinear_step import NonlinearConvergenceError
D=torch.float64
def model():
 # Two coincident TET4 integration domains isolate multi-point state/assembly
 # while retaining one free global axial DOF and fast deterministic CI.
 x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
 return FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3],[0,1,2,3]]),1000.,.3,
  torch.tensor([0,1,2,4,5,6,8,9,10,11]))
def load(m):f=torch.zeros(m.n_dofs,dtype=D);f[3]=32.;f[7]=1.;return f
def test_two_element_cycle_hybrid_matches_ad_and_richardson():
 m=model();f=load(m);path=(.2,)
 # Preload with the branch-robust hybrid tangent; pure spectral AD is then
 # compared away from its documented exactly-repeated virgin spectrum.
 cp=solve_hybrid_global_path(m,f,(.1,),yield_stress=35.,hardening=60.,strategy='hybrid')[-1]
 h=solve_hybrid_global_path(m,f,path,yield_stress=35.,hardening=60.,strategy='hybrid',initial=cp)
 a=solve_hybrid_global_path(m,f,path,yield_stress=35.,hardening=60.,strategy='ad',initial=cp)
 r=solve_hybrid_global_path(m,f,path,yield_stress=35.,hardening=60.,strategy='richardson',initial=cp)
 scale=max(float(torch.linalg.vector_norm(r[-1].displacement)),1e-12)
 assert float(torch.linalg.vector_norm(h[-1].displacement-r[-1].displacement))/scale<.03
 assert float(torch.linalg.vector_norm(a[-1].displacement-r[-1].displacement))/scale<.03
 assert len(h[-1].state.points)==2 and all(float(x)>=-1e-12 for z in h for x in z.increment_dissipation)
 assert sum(z.material_paths for z in h)<sum(z.material_paths for z in r)
 assert max(len(i.iterations) for z in h for i in z.increments)<=max(len(i.iterations) for z in r for i in z.increments)
def test_restart_and_failed_increment_do_not_pollute_two_point_history():
 m=model();f=load(m);direct=solve_hybrid_global_path(m,f,(.3,-.1,0.),yield_stress=35.,hardening=60.)
 cp=solve_hybrid_global_path(m,f,(.3,),yield_stress=35.,hardening=60.)[-1]
 resumed=solve_hybrid_global_path(m,f,(-.1,0.),yield_stress=35.,hardening=60.,initial=cp)
 assert torch.linalg.vector_norm(direct[-1].displacement-resumed[-1].displacement)<2e-7
 before=tuple(p.plastic.plastic_gradient.clone() for p in cp.state.points)
 with pytest.raises(NonlinearConvergenceError):
  solve_hybrid_global_path(m,f,(1.5,),yield_stress=35.,hardening=60.,initial=cp,max_iterations=1,initial_increment=.1,minimum_increment=.02)
 assert all(torch.equal(p.plastic.plastic_gradient,b) for p,b in zip(cp.state.points,before))
