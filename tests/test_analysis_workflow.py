import json
from pathlib import Path
import torch

from tensorfem.analysis_workflow import (
    AnalysisPlan, KernelRegistry, PlanStep, default_registry, deterministic_plan_id,
    migrate_plan_dict, plan_from_dict, plan_to_dict, run_analysis,
)
from tensorfem.modeldb import (
    BoundaryCondition, ConcentratedLoad, ElementBlock, Material, ModelDB, Section,
)


def model():
    return ModelDB(nodes={10:(0.,0.),20:(1.,0.),30:(2.,0.)},
        elements=[ElementBlock("T2D2",[101,102],[[10,20],[20,30]],"bars")],
        node_sets={"ALL":{10,20,30},"LEFT":{10},"RIGHT":{30}},element_sets={"BARS":{101,102}},
        materials={"STEEL":Material("steel",(200.,.3))},
        sections=[Section("BARS","STEEL","truss",(2.,))],
        boundaries=[BoundaryCondition("LEFT",1,1),BoundaryCondition("ALL",2,2)],
        loads=[ConcentratedLoad("RIGHT",1,40.)],
        metadata={"owner":{"purpose":"golden","tags":["lossless",7]},
                  "thermal":{"conductivity":5.,"density":1.,"specific_heat":1.,"thickness":1.,
                             "fixed_temperature":{"10":100.,"30":20.},"nodal_heat":{}}})


def plan():
    return AnalysisPlan("bar multiphysics",model(),(
        PlanStep("structure","modeldb.linear_static"),
        PlanStep("heat","modeldb.thermal_steady"),),
        {"length":"m","force":"N","stress":"Pa","temperature":"K","heat_rate":"W"},
        {"trace":{"requirement":"UNIFIED-001"}})


def test_golden_schema_lossless_id_and_two_physics(tmp_path):
    p=plan();raw=plan_to_dict(p)
    golden=Path(__file__).parent/"golden"/"analysis_plan_v2.json"
    assert raw==json.loads(golden.read_text())
    restored=plan_from_dict(json.loads(json.dumps(raw)))
    assert plan_to_dict(restored)==raw
    assert deterministic_plan_id(restored)==deterministic_plan_id(p)
    execution,directory=run_analysis(restored,tmp_path)
    assert execution.completed and [x.kind for x in execution.steps]==["modeldb.linear_static","modeldb.thermal_steady"]
    assert torch.allclose(execution.steps[0].fields["displacement"][:,0],torch.tensor([0.,.1,.2],dtype=torch.float64))
    assert torch.allclose(execution.steps[1].fields["temperature"][:,0],torch.tensor([100.,60.,20.],dtype=torch.float64))
    summary=json.loads((directory/"results.json").read_text())
    assert summary["schema"]=="tensorfem.result-db.v2" and summary["counts"]["steps"]==2
    assert json.loads((directory/"job.json").read_text())["status"]=="completed"


def test_v1_migration_is_explicit_and_preserves_payload():
    current=plan_to_dict(AnalysisPlan("legacy",model(),(PlanStep("static","modeldb.linear_static"),),{"length":"m"},{"x":[1,2]}))
    old={"schema":"tensorfem.analysis-plan.v1","name":current["name"],"model":current["model"],
         "step":{"name":"static","kind":"modeldb.linear_static"},"units":current["units"],"metadata":current["metadata"]}
    migrated=migrate_plan_dict(old)
    assert migrated["metadata"]==old["metadata"] and migrated["model"]==old["model"]
    assert plan_from_dict(migrated).steps[0].inputs=={}


def test_unknown_types_and_fields_fail_closed_and_are_recorded(tmp_path):
    raw=plan_to_dict(plan());raw["python"]="raise SystemExit"
    try: plan_from_dict(raw)
    except ValueError as exc: assert "unknown" in str(exc)
    else: raise AssertionError("unknown field accepted")
    bad=AnalysisPlan("bad",model(),(PlanStep("x","evil.import"),),{"length":"m"})
    execution,directory=run_analysis(bad,tmp_path)
    assert not execution.completed and "unsupported step kind" in execution.steps[-1].diagnostic
    checkpoint=json.loads((directory/"checkpoint.json").read_text())
    assert checkpoint["steps"][-1]["success"] is False
    assert json.loads((directory/"job.json").read_text())["status"]=="failed"


def test_failure_rollback_and_resume_reuses_only_committed_step(tmp_path):
    p=AnalysisPlan("resume",model(),(PlanStep("first","test.first"),PlanStep("second","test.second")),{"length":"m"})
    calls=[];registry=KernelRegistry()
    registry.register("test.first",lambda db,inputs:(calls.append("first") or {"temperature":torch.zeros((3,1))}))
    registry.register("test.second",lambda db,inputs:(_ for _ in ()).throw(RuntimeError("worker lost")))
    failed,directory=run_analysis(p,tmp_path,registry=registry)
    assert not failed.completed and calls==["first"]
    repaired=KernelRegistry()
    repaired.register("test.first",lambda db,inputs:(_ for _ in ()).throw(AssertionError("must not rerun")))
    repaired.register("test.second",lambda db,inputs:{"temperature":torch.ones((3,1))})
    resumed,_=run_analysis(p,tmp_path,resume=True,registry=repaired)
    assert resumed.completed and resumed.resumed_steps==1
    assert torch.equal(resumed.steps[1].fields["temperature"],torch.ones((3,1)))


def test_resume_precondition_failure_is_recorded(tmp_path):
    p=AnalysisPlan("missing-checkpoint",model(),(PlanStep("first","modeldb.linear_static"),),{"length":"m"})
    try: run_analysis(p,tmp_path,resume=True)
    except ValueError as exc: assert "checkpoint" in str(exc)
    else: raise AssertionError("resume without checkpoint accepted")
    directory=tmp_path/deterministic_plan_id(p)
    job=json.loads((directory/"job.json").read_text())
    assert job["status"]=="failed" and job["error_type"]=="ValueError"


def test_capability_discovery_and_dependency_validation():
    assert default_registry().capabilities()==("modeldb.linear_static","modeldb.thermal_steady")
    bad=AnalysisPlan("bad-dependency",model(),(PlanStep("a","modeldb.linear_static",dependencies={"x":"later.y"}),),{"length":"m"})
    try: plan_to_dict(bad)
    except ValueError as exc: assert "earlier" in str(exc)
    else: raise AssertionError("forward dependency accepted")
