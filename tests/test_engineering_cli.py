import json
from pathlib import Path

import pytest

from tensorfem.cli import main


PROJECT = Path(__file__).parents[1] / "examples" / "projects" / "truss_project.json"


def test_validate_and_run_mesh_project(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["tensorfem", "validate", str(PROJECT)])
    main()
    assert json.loads(capsys.readouterr().out)["valid"]
    monkeypatch.setattr("sys.argv", [
        "tensorfem", "run", str(PROJECT), "--runs", str(tmp_path / "runs")
    ])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["displacement"][2] == pytest.approx(0.2)
    job_dir = tmp_path / "runs" / result["job_id"]
    monkeypatch.setattr("sys.argv", ["tensorfem", "status", str(job_dir)])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_inspect_result_inventory(monkeypatch, capsys, tmp_path):
    index = tmp_path / "result.json"
    index.write_text(json.dumps({
        "schema": "tensorfem.result-db.v2",
        "database_sha256": "abc",
        "body": {"available": False},
        "steps": [{"name": "Static"}],
        "inventory": {"fields": []},
    }))
    monkeypatch.setattr("sys.argv", ["tensorfem", "inspect", str(index)])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "tensorfem.result-db.v2"
    assert payload["steps"][0]["name"] == "Static"
