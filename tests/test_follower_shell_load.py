import torch
from tensorfem.follower_shell_load import follower_pressure_force,follower_pressure_force_tangent
from tensorfem.corotational_shell import axis_angle
from tensorfem.arc_length import ArcLengthProblem,solve_arc_length

D=torch.float64
X=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]],dtype=D)


def test_planar_pressure_has_exact_resultant_and_moment():
    f=follower_pressure_force(X,torch.zeros(24,dtype=D),2.).reshape(4,6)
    resultant=f[:,:3].sum(0);moment=torch.linalg.cross(X,f[:,:3],dim=1).sum(0)
    assert torch.allclose(resultant,torch.tensor([0.,0.,2.],dtype=D),atol=1e-14)
    assert torch.allclose(moment,torch.tensor([1.,-1.,0.],dtype=D),atol=1e-14)


def test_pressure_force_rotates_covariantly_with_current_surface():
    R=axis_angle(torch.tensor([.2,-.4,1.],dtype=D),1.1);moved=X@R.T+torch.tensor([4.,-2.,7.],dtype=D)
    q=torch.zeros((4,6),dtype=D);q[:,:3]=moved-X
    base=follower_pressure_force(X,torch.zeros(24,dtype=D)).reshape(4,6)[:,:3]
    rotated=follower_pressure_force(X,q.reshape(-1)).reshape(4,6)[:,:3]
    assert torch.allclose(rotated,base@R.T,rtol=1e-12,atol=1e-12)


def test_consistent_pressure_tangent_matches_finite_difference():
    q=torch.linspace(-.03,.02,24,dtype=D);force,K=follower_pressure_force_tangent(X,q,1.7)
    direction=torch.linspace(-1.,1.,24,dtype=D);direction/=torch.linalg.vector_norm(direction);h=1e-6
    fd=(follower_pressure_force(X,q+h*direction,1.7)-follower_pressure_force(X,q-h*direction,1.7))/(2*h)
    error=torch.linalg.vector_norm(fd-K@direction)/torch.linalg.vector_norm(K@direction)
    assert error<3e-6


def test_load_dependent_arc_length_crosses_analytic_limit_point():
    # An independently auditable one-DOF path qualifies the load-dependent
    # augmented column.  This is an algorithm gate, not a shell benchmark.
    def residual_tangent(u,load):
        external=1+.2*u
        return u-u**3-load*external,torch.diag(1-3*u**2-.2*load),-external
    problem=ArcLengthProblem(lambda _u: (_u,_u.reshape(1,1)),torch.zeros(1,dtype=D),
                             residual_tangent)
    result=solve_arc_length(problem,torch.zeros(1,dtype=D),steps=35,step_size=.04,
                            maximum_step=.04,load_scale=.3,tolerance=1e-11)
    loads=torch.tensor([p.load_factor for p in result.points],dtype=D)
    assert result.converged and len(loads)==35
    assert bool(torch.any(loads[1:]<loads[:-1]))
    for point in result.points:
        u=point.displacement
        assert abs(float((u-u**3-point.load_factor*(1+.2*u))[0]))<1e-9
