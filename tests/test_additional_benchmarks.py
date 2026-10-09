import copy
import pytest
import torch
from tensorfem.additional_benchmark_fields import mindlin_fields,cook_fields
from tensorfem.sphere_pressure_benchmark import lame_reference,solve_sphere_pressure,verify_sphere_pressure,PROBLEM
from tensorfem.continuum_benchmarks import cook_membrane


def test_lame_satisfies_inner_outer_traction_and_radial_equilibrium():
    r=torch.tensor([8.,9.,10.],dtype=torch.float64,requires_grad=True)
    x=torch.stack((r,r*0,r*0),1)
    _,_,sr,st,_=lame_reference(x)
    assert abs(float(sr[0].detach())+1e6)<1e-8 and abs(float(sr[-1].detach()))<1e-8
    derivative=torch.autograd.grad(sr.sum(),r)[0]
    assert torch.max(abs(derivative+2*(sr-st)/r))<1e-9


def test_coarse_sphere_does_not_pass_by_displacement_alone_and_tamper_rejected():
    c=solve_sphere_pressure(4,1)
    assert not c['verification']['passed']
    assert verify_sphere_pressure(c)==c['verification']
    bad=copy.deepcopy(c);bad['element_stress'][0][0]+=1000
    with pytest.raises(ValueError,match='not recovered'):verify_sphere_pressure(bad)


def test_mindlin_stress_and_shear_converge_against_independent_exact_fields():
    a,b=mindlin_fields(4),mindlin_fields(8)
    assert b['verification']['passed']
    assert all(b['verification']['errors'][k]<a['verification']['errors'][k] for k in a['verification']['errors'])


def test_cook_reference_probe_is_loaded_edge_midpoint_and_no_stress_claim():
    model,probe=cook_membrane(8)
    assert torch.equal(model.nodes[probe//2],torch.tensor([48.,52.],dtype=torch.float64))
    c=cook_fields(8)
    assert c['stress_accuracy_qualified'] is False
    assert c['verification']['response_relative_error']>.03


def test_report_reproduction_handles_json_metadata_and_rejects_false_pass():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'scripts/validate_additional_benchmark_reports.py'
    spec=importlib.util.spec_from_file_location('additional_gate',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.compare({'metadata':['ux','uy'],'path':[{'value':1.}], 'passed':True},
                   {'metadata':['ux','uy'],'path':({'value':1.},), 'passed':True})
    with pytest.raises(ValueError,match='stale qualification'):
        module.compare({'passed':True},{'passed':False})


def test_pointwise_stress_gate_retains_coarse_failures_and_requires_fine_pass():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for folder,filename in [('sphere-pressure','results.json'),('hull-girder','hull-girder-fe.json')]:
        report=json.loads((root/'docs/assets/benchmark-clouds'/folder/filename).read_text())
        assert report['cases'][-1]['verification']['passed']
        assert any(not c['verification']['passed'] for c in report['cases'][:-1])
        key='radial_stress_pointwise_max' if folder=='sphere-pressure' else 'moment_and_sigma_x_pointwise_max'
        assert report['cases'][-1]['verification']['errors'][key]<.03
        assert any(c['verification']['errors'][key]>.03 for c in report['cases'][:-1])


def test_independent_report_manifest_does_not_capture_other_case_assets():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'docs/benchmarks/tutorials/evidence-manifest.json').read_text())
    prefix='docs/assets/benchmark-clouds/'
    for path in manifest['files']:
        if path.startswith(prefix):
            assert path[len(prefix):].split('/')[0] in {'cantilever-fe','hull-girder','stiffened-panel'}


def test_sphere_csr_jacobi_solver_preserves_gauge_constrained_fe_solution():
    coarse=solve_sphere_pressure(4,1,preconditioned=False)
    optimized=solve_sphere_pressure(4,1,preconditioned=True)
    a=torch.tensor(coarse['displacement'],dtype=torch.float64)
    b=torch.tensor(optimized['displacement'],dtype=torch.float64)
    assert float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a))<1e-9
    assert optimized['verification']['free_residual_relative']<1e-8
    assert optimized['verification']['gauge_reaction_relative']<1e-8
