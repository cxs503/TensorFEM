import hashlib
import json

import pytest
import torch

from tensorfem.marine_panel_execution import (
    CHUNKED_SCHEMA, ENERGY_DEFINITION, migrate_panel_checkpoint_strategy,
)


def _source(tmp_path):
    binary = tmp_path / "chunked-old-g000003.pt"
    saved = {
        "displacement": torch.tensor([1., 2.], dtype=torch.float64),
        "state": {"alpha": torch.tensor([.1], dtype=torch.float64)},
        "previous_increment": torch.tensor([.2, .3]),
        "load_factor": 42., "step_size": .01,
        "reference_recoverable_energy": 3.,
        "cumulative_external_work": 5.,
        "cumulative_plastic_dissipation": 1.,
        "energy_definition": ENERGY_DEFINITION,
        "energy_prefix_complete": True,
    }
    torch.save(saved, binary)
    manifest = {
        "schema": CHUNKED_SCHEMA, "energy_definition": ENERGY_DEFINITION,
        "divisions": 2, "normalized_arc_step": .02,
        "arc_metric": "dimensionally_scaled", "solver_maximum_step": .0002,
        "relative_equilibrium_tolerance": 1e-6, "accepted_points": 3,
        "job_key": "old", "checkpoint_file": binary.name,
        "checkpoint_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "point_history": [{"state_sha256": "a"*64}], "status": "executed",
    }
    path = tmp_path / "chunked-old.json"
    path.write_text(json.dumps(manifest))
    return path, saved


def test_strategy_migration_preserves_state_and_creates_independent_identity(tmp_path):
    source, saved = _source(tmp_path)
    result = migrate_panel_checkpoint_strategy(source, tmp_path/"target")
    target = json.loads(open(result["target_manifest"]).read())
    migrated = torch.load(
        (tmp_path/"target"/target["checkpoint_file"]), weights_only=False)
    assert target["job_key"] != "old"
    assert target["line_search"] == "backtracking"
    assert target["strategy_migration"]["source_state_sha256"] == "a"*64
    assert torch.equal(migrated["displacement"], saved["displacement"])
    assert torch.equal(migrated["state"]["alpha"], saved["state"]["alpha"])
    assert migrated["cumulative_external_work"] == saved["cumulative_external_work"]
    assert result["mechanical_state_exact"] and result["energy_ledger_exact"]
    assert len(result["migration_sha256"]) == 64
    with pytest.raises(FileExistsError, match="immutable generations"):
        migrate_panel_checkpoint_strategy(source, tmp_path/"target")


def test_strategy_migration_rejects_tampered_source(tmp_path):
    source, _ = _source(tmp_path)
    manifest = json.loads(source.read_text())
    manifest["checkpoint_sha256"] = "0"*64
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity mismatch"):
        migrate_panel_checkpoint_strategy(source, tmp_path/"target")
