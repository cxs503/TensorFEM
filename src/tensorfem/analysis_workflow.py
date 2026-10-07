"""Unified ModelDB -> AnalysisPlan -> Job -> ResultDB v2 workflow.

This module is deliberately a vertical slice over the existing ModelDB,
StepExecutor and ResultDB v2 implementations.  It is not another solver input
model: plans select registered adapters which consume one immutable ModelDB.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, os, time
from pathlib import Path
from typing import Any, Callable, Mapping
import torch

from .modeldb import ModelDB, modeldb_from_dict, modeldb_to_dict, to_truss_model
from .result_db_v2 import FieldSpec, ResultDBv2, ResultFrame, ResultStep, write_result_db_v2
from .step_executor import ExecutionResult, StepExecutor, StepSpec

SCHEMA = "tensorfem.analysis-plan.v2"
LEGACY_SCHEMA = "tensorfem.analysis-plan.v1"


@dataclass(frozen=True)
class PlanStep:
    name: str
    kind: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    dependencies: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisPlan:
    name: str
    model: ModelDB
    steps: tuple[PlanStep, ...]
    units: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan_to_dict(plan: AnalysisPlan) -> dict[str, Any]:
    validate_plan(plan)
    return {"schema": SCHEMA, "name": plan.name, "model": modeldb_to_dict(plan.model),
            "steps": [{"name": x.name, "kind": x.kind, "inputs": dict(x.inputs),
                       "dependencies": dict(x.dependencies)} for x in plan.steps],
            "units": dict(plan.units), "metadata": dict(plan.metadata)}


def deterministic_plan_id(plan: AnalysisPlan) -> str:
    return hashlib.sha256(_canonical(plan_to_dict(plan)).encode()).hexdigest()[:20]


def migrate_plan_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the only historical plan form without discarding fields."""
    raw = dict(raw)
    if raw.get("schema") == SCHEMA: return raw
    if raw.get("schema") != LEGACY_SCHEMA:
        raise ValueError("unsupported analysis plan schema")
    expected = {"schema", "name", "model", "step", "units", "metadata"}
    if set(raw) != expected: raise ValueError("unknown or missing legacy plan fields")
    step = dict(raw.pop("step")); step.setdefault("inputs", {}); step.setdefault("dependencies", {})
    raw["steps"] = [step]; raw["schema"] = SCHEMA
    return raw


def plan_from_dict(source: Mapping[str, Any]) -> AnalysisPlan:
    raw = migrate_plan_dict(source)
    if set(raw) != {"schema", "name", "model", "steps", "units", "metadata"}:
        raise ValueError("unknown or missing analysis plan fields")
    steps = []
    for item in raw["steps"]:
        if set(item) != {"name", "kind", "inputs", "dependencies"}:
            raise ValueError("unknown or missing plan step fields")
        steps.append(PlanStep(item["name"], item["kind"], dict(item["inputs"]),
                              dict(item["dependencies"])))
    plan = AnalysisPlan(raw["name"], modeldb_from_dict(raw["model"]), tuple(steps),
                        dict(raw["units"]), dict(raw["metadata"]), raw["schema"])
    validate_plan(plan); return plan


def read_plan(path: str | Path) -> AnalysisPlan:
    return plan_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_plan(path: str | Path, plan: AnalysisPlan) -> None:
    _atomic_json(Path(path), plan_to_dict(plan))


def validate_plan(plan: AnalysisPlan) -> None:
    if plan.schema != SCHEMA: raise ValueError("unsupported analysis plan schema")
    if not plan.name or not plan.steps: raise ValueError("plan name and steps are required")
    plan.model.validate()
    names = [x.name for x in plan.steps]
    if len(names) != len(set(names)) or any(not x for x in names):
        raise ValueError("step names must be unique and nonempty")
    previous = set()
    for step in plan.steps:
        for source in step.dependencies.values():
            parent = source.split(".", 1)[0]
            if parent not in previous: raise ValueError(f"dependency must reference an earlier step: {source}")
        previous.add(step.name)
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in plan.units.items()):
        raise ValueError("units must be string mappings")


Kernel = Callable[[ModelDB, Mapping[str, Any]], Mapping[str, torch.Tensor]]


class KernelRegistry:
    """Explicit allow-list with introspectable capabilities."""
    def __init__(self): self._kernels: dict[str, Kernel] = {}
    def register(self, kind: str, kernel: Kernel) -> None:
        if not kind or kind in self._kernels: raise ValueError(f"duplicate kernel {kind}")
        self._kernels[kind] = kernel
    def resolve(self, kind: str) -> Kernel:
        try: return self._kernels[kind]
        except KeyError as exc: raise KeyError(f"unsupported step kind {kind}") from exc
    def capabilities(self) -> tuple[str, ...]: return tuple(sorted(self._kernels))


def _linear(db: ModelDB, inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if inputs: raise ValueError(f"unknown linear inputs {sorted(inputs)}")
    from .solvers import solve_linear_static
    model, _ = to_truss_model(db); r = solve_linear_static(model)
    n = len(db.nodes)
    return {"displacement": r.displacement.reshape(n, -1), "reaction": r.reaction.reshape(n, -1),
            "axial_stress": r.axial_stress.reshape(-1, 1),
            "axial_strain": r.axial_strain.reshape(-1, 1)}


def _thermal(db: ModelDB, inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if inputs: raise ValueError(f"unknown thermal inputs {sorted(inputs)}")
    from .thermal import ThermalModel, assemble_thermal, solve_steady_thermal
    cfg = db.metadata.get("thermal")
    if not isinstance(cfg, dict) or set(cfg) != {"conductivity", "density", "specific_heat", "thickness",
                                                "fixed_temperature", "nodal_heat"}:
        raise ValueError("ModelDB metadata.thermal is missing required fields")
    ids = sorted(db.nodes); index = {n: i for i, n in enumerate(ids)}
    blocks = [b for b in db.elements if b.element_type.upper() == "T2D2"]
    if len(blocks) != 1: raise ValueError("thermal line adapter requires one T2D2 block")
    conn = torch.tensor([[index[n] for n in c] for c in blocks[0].connectivity], dtype=torch.long)
    # The qualified line kernel uses its curvilinear abscissa.  A 2-D T2D2
    # ModelDB remains the source of truth; cumulative physical lengths preserve
    # geometry instead of silently selecting one global coordinate.
    points = torch.tensor([db.nodes[n] for n in ids], dtype=torch.float64)
    xyz = torch.zeros((len(ids), 1), dtype=torch.float64)
    for i in range(1, len(ids)): xyz[i, 0] = xyz[i-1, 0] + torch.linalg.vector_norm(points[i]-points[i-1])
    def per_element(name):
        value = cfg[name]; return torch.full((len(conn),), float(value), dtype=torch.float64) if isinstance(value, (int, float)) else torch.tensor(value, dtype=torch.float64)
    fixed = sorted((index[int(k)], float(v)) for k, v in cfg["fixed_temperature"].items())
    heat = torch.zeros(len(ids), dtype=torch.float64)
    for key, value in cfg["nodal_heat"].items(): heat[index[int(key)]] += float(value)
    model = ThermalModel(xyz, conn, per_element("conductivity"), per_element("density"),
                         per_element("specific_heat"), torch.tensor([x[0] for x in fixed]),
                         torch.tensor([x[1] for x in fixed], dtype=torch.float64), nodal_heat=heat,
                         thickness=per_element("thickness"), element_type="line2")
    temperature = solve_steady_thermal(model); flux = assemble_thermal(model).conductivity @ temperature - assemble_thermal(model).heat
    return {"temperature": temperature.reshape(-1, 1), "heat_residual": flux.reshape(-1, 1)}


def default_registry() -> KernelRegistry:
    registry = KernelRegistry(); registry.register("modeldb.linear_static", _linear)
    registry.register("modeldb.thermal_steady", _thermal); return registry


def _topology(db: ModelDB):
    node_ids = sorted(db.nodes); blocks = db.elements
    ids = [eid for b in blocks for eid in b.ids]; conn = [c for b in blocks for c in b.connectivity]
    widths = {len(c) for c in conn}
    if len(widths) > 1: raise ValueError("ResultDB v2 requires uniform connectivity width in this workflow")
    return torch.tensor(node_ids), torch.tensor(ids), torch.tensor(conn, dtype=torch.long)


def execution_result_db_v2(plan: AnalysisPlan, execution: ExecutionResult, job_id: str) -> ResultDBv2:
    node_ids, element_ids, connectivity = _topology(plan.model); specs = {}; result_steps = []
    node_names = {"displacement", "reaction", "temperature", "heat_residual"}
    default_units = {"displacement": plan.units.get("length", "1"), "reaction": plan.units.get("force", "1"),
                     "axial_stress": plan.units.get("stress", "1"), "axial_strain": "1",
                     "temperature": plan.units.get("temperature", "1"), "heat_residual": plan.units.get("heat_rate", "1")}
    for step in execution.steps:
        if not step.success: continue
        grouped = {"node": {}, "element": {}}
        for name, value in step.fields.items():
            if not isinstance(value, torch.Tensor): continue
            location = "node" if name in node_names else "element"; grouped[location][name] = value
            ncomp = 1 if value.ndim == 1 else value.shape[-1]
            components = (name,) if ncomp == 1 else tuple(f"{name}{i+1}" for i in range(ncomp))
            candidate = FieldSpec(location, default_units.get(name, "1"), components)
            if name in specs and specs[name] != candidate: raise ValueError(f"incompatible field schema {name}")
            specs[name] = candidate
        result_steps.append(ResultStep(step.name, (ResultFrame(0, 0., 1., {k: v for k, v in grouped.items() if v}),)))
    db = ResultDBv2(node_ids, element_ids, connectivity, specs, tuple(result_steps), {},
                    {"plan_id": deterministic_plan_id(plan), "name": plan.name, **dict(plan.metadata)},
                    {"job_id": job_id, "completed": execution.completed, "resumed_steps": execution.resumed_steps,
                     "steps": [{"name": x.name, "kind": x.kind, "success": x.success,
                                "diagnostic": x.diagnostic} for x in execution.steps]})
    return db.validate()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_analysis(plan: AnalysisPlan, root: str | Path, *, resume: bool = False,
                 registry: KernelRegistry | None = None) -> tuple[ExecutionResult, Path]:
    """Run or resume a deterministic job and atomically record every outcome."""
    validate_plan(plan); registry = registry or default_registry(); job_id = deterministic_plan_id(plan)
    directory = Path(root) / job_id; directory.mkdir(parents=True, exist_ok=True)
    project = directory / "analysis-plan.json"; current = plan_to_dict(plan)
    if project.exists() and _canonical(json.loads(project.read_text())) != _canonical(current):
        raise RuntimeError("deterministic plan ID collision")
    _atomic_json(project, current); checkpoint = directory / "checkpoint.json"; job = directory / "job.json"
    _atomic_json(job, {"schema": SCHEMA, "job_id": job_id, "status": "running", "updated_unix": time.time()})
    kernels = {}
    for kind in registry.capabilities():
        kernel = registry.resolve(kind); kernels[kind] = lambda inputs, k=kernel: k(inputs.pop("__modeldb__"), inputs)
    specs = [StepSpec(x.name, x.kind, {"__modeldb__": plan.model, **dict(x.inputs)}, x.dependencies) for x in plan.steps]
    try:
        execution = StepExecutor(kernels).execute(specs, checkpoint=checkpoint, resume=resume)
        status = "completed" if execution.completed else "failed"
        result_path = directory / "results"
        if execution.completed: write_result_db_v2(result_path, execution_result_db_v2(plan, execution, job_id))
        _atomic_json(job, {"schema": SCHEMA, "job_id": job_id, "status": status,
                           "capabilities": list(registry.capabilities()), "resumed_steps": execution.resumed_steps,
                           "steps": [{"name": x.name, "kind": x.kind, "success": x.success,
                                      "diagnostic": x.diagnostic} for x in execution.steps],
                           "updated_unix": time.time()})
        return execution, directory
    except Exception as exc:
        _atomic_json(job, {"schema": SCHEMA, "job_id": job_id, "status": "failed",
                           "error_type": type(exc).__name__, "error": str(exc),
                           "capabilities": list(registry.capabilities()), "updated_unix": time.time()})
        raise
