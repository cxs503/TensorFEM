from pathlib import Path
import json
import pytest
from tensorfem.engineering_cases import load_case,run_case

CASES=Path(__file__).parents[1]/"examples"/"engineering_cases"


@pytest.mark.parametrize("name",["pressurized_component.json","thin_wall_panel.json","contact_connector.json"])
def test_case_package_solve_resultdb_report_and_replay(tmp_path,name):
    first=run_case(CASES/name,tmp_path)
    assert first["passed"] and first["relative_error"]<.03
    directory=tmp_path/first["job_id"]
    assert all((directory/x).exists() for x in ("input.json","job.json","checkpoint.json","validation.json","validation.md","results.json","results.vtk"))
    assert json.loads((directory/"validation.json").read_text())["passed"]
    replay=run_case(CASES/name,tmp_path)
    assert replay["execution"]["replayed"] and replay["computed"]==first["computed"]


def test_checkpoint_restart_reconstructs_same_report(tmp_path):
    staged=run_case(CASES/"contact_connector.json",tmp_path,replay=False,checkpoint_only=True)
    resumed=run_case(CASES/"contact_connector.json",tmp_path,replay=False,restart=True)
    assert staged["checkpointed"] and resumed["execution"]["restarted"] and resumed["passed"]


def test_unknown_case_fails_closed(tmp_path):
    raw=json.loads((CASES/"thin_wall_panel.json").read_text());raw["kind"]="unsupported_pressure_shell_contact"
    path=tmp_path/"bad.json";path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="unsupported"):load_case(path)


def test_corrupt_checkpoint_fails_closed(tmp_path):
    staged=run_case(CASES/"contact_connector.json",tmp_path,replay=False,checkpoint_only=True)
    path=tmp_path/staged["job_id"]/"checkpoint.json";raw=json.loads(path.read_text());raw["computed"]+=1;path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError,match="checksum"):run_case(CASES/"contact_connector.json",tmp_path,replay=False,restart=True)
