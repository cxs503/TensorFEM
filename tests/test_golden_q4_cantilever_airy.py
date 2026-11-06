"""Golden regression gate: Q4 cantilever vs exact Airy field.

The scenario is the rectangular Q4 plane-stress cantilever of
``tests/test_q4_cantilever_airy.py`` (L = 1 m, H = 0.1 m, t = 12 mm,
E = 210 GPa, nu = 0.3, end shear -100 N, exact Timoshenko & Goodier
tractions).  That test gates *accuracy* against the closed-form solution;
this gate locks *reproducibility*: the identical problem is re-solved and
every number recorded in ``tests/golden/q4_cantilever_airy.json`` must
reproduce, so a silent change of the solver, the mesh generator or the
exact-field module is caught even when the error metrics keep their shape.

The golden was generated on the qualification machine by::

    PYTHONPATH=src python3 scripts/generate_golden_q4_cantilever_airy.py

(see ``generated.command`` inside the file).  Regenerate deliberately after a
reviewed solver change; never edit the data by hand.
"""

import json
from pathlib import Path

import torch

from scripts.audit_q4_cantilever_airy import errors, solve
from tensorfem.airy_cantilever import AiryCantilever

GOLDEN = Path(__file__).parent / "golden" / "q4_cantilever_airy.json"
SCHEMA = "tensorfem.golden/q4-cantilever-airy/1.0"
#: The solve is a direct float64 factorisation; on the same machine it is
#: bit-stable, so a tight tolerance still leaves headroom for platform drift.
REL_TOL = 1e-9
ABS_TOL = 1e-12


def _flat(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.reshape(-1)]


def _compare(path: str, golden, actual) -> None:
    if isinstance(golden, dict):
        assert isinstance(actual, dict), f"{path}: expected object, got {type(actual).__name__}"
        missing = set(golden) - set(actual)
        assert not missing, f"{path}: keys missing from fresh solve: {sorted(missing)}"
        for key, value in golden.items():
            _compare(f"{path}.{key}", value, actual[key])
    elif isinstance(golden, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(golden) == len(actual), f"{path}: length {len(actual)} != golden {len(golden)}"
        for i, (g, a) in enumerate(zip(golden, actual)):
            _compare(f"{path}[{i}]", g, a)
    else:
        assert isinstance(actual, (int, float)), f"{path}: expected number, got {actual!r}"
        ok = torch.allclose(
            torch.tensor(golden, dtype=torch.float64),
            torch.tensor(actual, dtype=torch.float64),
            rtol=REL_TOL,
            atol=ABS_TOL,
        )
        assert ok, f"{path}: golden={golden!r} fresh={actual!r}"


def test_golden_metadata_matches_current_model_definition():
    golden = json.loads(GOLDEN.read_text())
    model = AiryCantilever()
    assert golden["schema"] == SCHEMA
    assert golden["case_id"] == "rectangular-q4-cantilever-airy"
    assert golden["mesh"] == {"nx": 40, "ny": 8}
    assert golden["generated"]["command"].endswith("scripts/generate_golden_q4_cantilever_airy.py")
    problem = golden["problem"]
    assert problem["length_m"] == model.length
    assert problem["height_m"] == model.height
    assert problem["thickness_m"] == model.thickness
    assert problem["young_modulus_pa"] == model.young_modulus
    assert problem["poisson"] == model.poisson
    assert problem["end_shear_n"] == model.end_shear
    assert problem["second_moment_m4"] == model.second_moment


def test_fresh_solve_reproduces_golden_metrics():
    golden = json.loads(GOLDEN.read_text())
    model = AiryCantilever()
    mesh = (golden["mesh"]["nx"], golden["mesh"]["ny"])
    _compare("data", golden["data"], errors(mesh, model))


def test_fresh_solve_reproduces_golden_fields():
    golden = json.loads(GOLDEN.read_text())
    model = AiryCantilever()
    mesh = (golden["mesh"]["nx"], golden["mesh"]["ny"])
    nodes, elements, result = solve(mesh, model)
    centres = nodes[elements].mean(1)
    fields = {
        "element_centres": _flat(centres),
        "node_displacement_fe": _flat(result.displacement),
        "node_displacement_exact": _flat(model.displacement(nodes)),
        "element_stress_fe": _flat(result.stress),
        "element_stress_exact": _flat(model.stress(centres)),
    }
    _compare("fields", golden["fields"], fields)


def test_golden_data_is_not_empty():
    golden = json.loads(GOLDEN.read_text())
    data = golden["data"]
    assert data["mesh"] == {"nx": 40, "ny": 8, "nodes": 369, "elements": 320}
    assert 0 < data["interior_element_fraction"] < 1
    fields = golden["fields"]
    assert len(fields["element_centres"]) == 320 * 2
    assert len(fields["node_displacement_fe"]) == 369 * 2
    assert len(fields["element_stress_fe"]) == 320 * 3
    assert len(fields["node_displacement_exact"]) == 369 * 2
    assert len(fields["element_stress_exact"]) == 320 * 3
