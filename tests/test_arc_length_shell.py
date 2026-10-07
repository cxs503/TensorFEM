import torch
from tensorfem.arc_length import ArcLengthProblem,general_shell_arc_problem,solve_arc_length
from tensorfem.von_mises_arch import arch_limit_reference,trace_von_mises_arch
from tensorfem.general_shell_nonlinear import GeneralShellMesh


def _peak(result): return max(result.points,key=lambda p:p.load_factor)


def test_arc_length_crosses_limit_point_and_matches_exact_peak():
    result=trace_von_mises_arch(.005,70,maximum_step=.005)
    assert result.converged
    loads=[p.load_factor for p in result.points]; peak=_peak(result);ref=arch_limit_reference()
    assert any(loads[i+1]<loads[i] for i in range(loads.index(max(loads)),len(loads)-1))
    assert abs(peak.load_factor/ref.load-1.)<.03
    assert abs(float(peak.displacement[0])/ref.displacement-1.)<.03


def test_arc_length_step_convergence_and_restart(tmp_path):
    coarse=trace_von_mises_arch(.01,25,maximum_step=.01);fine=trace_von_mises_arch(.005,50,maximum_step=.005)
    assert abs(_peak(coarse).load_factor/_peak(fine).load_factor-1.)<.03
    cp=tmp_path/"arc.json";first=trace_von_mises_arch(.01,12,maximum_step=.01,checkpoint=cp)
    resumed=trace_von_mises_arch(.01,23,maximum_step=.01,restart=cp)
    direct=trace_von_mises_arch(.01,35,maximum_step=.01)
    assert first.converged and resumed.converged
    assert torch.allclose(resumed.points[-1].displacement,direct.points[-1].displacement,rtol=1e-8,atol=1e-10)
    assert abs(resumed.points[-1].load_factor-direct.points[-1].load_factor)<1e-8


def test_failed_arc_step_rolls_back_fail_closed():
    def response(u): return torch.stack((u[0]+u[0]**3,)),torch.stack((1+3*u[0]**2,)).reshape(1,1)
    problem=ArcLengthProblem(response,torch.ones(1,dtype=torch.float64))
    result=solve_arc_length(problem,torch.zeros(1,dtype=torch.float64),steps=1,
                            step_size=.2,load_scale=1.,max_iterations=1,minimum_step=.15)
    assert not result.converged and not result.points


def test_general_shell_adapter_exposes_reduced_consistent_tangent():
    nodes=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,.1],[0.,1.,0.]],dtype=torch.float64)
    mesh=GeneralShellMesh(nodes,torch.tensor([[0,1,2,3]]),1e6,.25,.05)
    load=torch.zeros(24,dtype=torch.float64);load[14]=-1.
    problem,free=general_shell_arc_problem(mesh,load,range(12))
    internal,K=problem.internal_tangent(torch.zeros(len(free),dtype=torch.float64))
    assert internal.shape==(len(free),) and K.shape==(len(free),len(free))
    assert torch.allclose(K,K.T,rtol=1e-10,atol=1e-8)
