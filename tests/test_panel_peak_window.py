import hashlib
import json

import pytest
import torch

import tensorfem.panel_peak_window as window
from tensorfem.panel_peak_window import (
    PeakTargetEstimate, PeakWindowPolicy, estimate_peak_window_target,
    retune_next_arc_step, select_peak_window,
)
from tensorfem.panel_path_evidence import ENERGY_DEFINITION


def test_unfinished_4x4_uses_conservative_mechanics_anchor_not_false_peak():
    manifest = {"point_history": [{"force_n": 100_000.}, {"force_n": 200_000.}],
                "peak_confirmed": False, "peak_force_n": 200_000.}
    target = estimate_peak_window_target(manifest)
    assert target.force_n > 200_000.
    assert not target.is_confirmed_peak
    assert target.source_points == 2


def test_confirmed_4x4_peak_drives_three_nested_windows():
    target = estimate_peak_window_target({
        "point_history": [{"force_n": 1_000_000.}],
        "peak_confirmed": True, "peak_force_n": 1_000_000.,
    })
    policy = PeakWindowPolicy()
    assert target.is_confirmed_peak
    assert select_peak_window(700_000., target, policy) == ("coarse", .10)
    assert select_peak_window(800_000., target, policy) == ("approach", .05)
    assert select_peak_window(950_000., target, policy) == ("peak", .02)


def test_running_manifest_preserves_peak_after_sustained_accepted_descent():
    forces = [100_000., 300_000., 500_000., 510_000., 509_800., 509_400.,
              509_000., 508_700., 508_500., 508_400.]
    target = estimate_peak_window_target({
        "status": "running",
        "point_history": [{"force_n": force} for force in forces],
    })
    assert target.force_n == 510_000.
    assert target.is_confirmed_peak
    assert target.source == "observed_4x4_peak_from_running_history"


def test_checkpoint_retune_preserves_state_and_renews_hashes(tmp_path):
    checkpoint = tmp_path / "path.pt"
    manifest_path = tmp_path / "path.json"
    state = {"step_size": .001, "displacement": torch.tensor([1., 2.]),
             "load_factor": 12.}
    torch.save(state, checkpoint)
    manifest = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "accepted_points": 3, "evidence_sha256": "old"}
    manifest_path.write_text(json.dumps(manifest))
    retune_next_arc_step(manifest_path, checkpoint,
                         normalized_arc_step=.02,
                         characteristic_displacement_m=.01)
    saved = torch.load(checkpoint, weights_only=False)
    renewed = json.loads(manifest_path.read_text())
    assert saved["step_size"] == pytest.approx(.0002)
    assert torch.equal(saved["displacement"], state["displacement"])
    assert saved["load_factor"] == state["load_factor"]
    assert renewed["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert renewed["evidence_sha256"] != "old"


def test_scheduler_checkpoint_identity_tracks_energy_definition(tmp_path):
    manifest, _ = window._checkpoint_paths(tmp_path, .1, 1e-6, divisions=8)
    identity = {"schema": window.CHUNKED_SCHEMA,
                "energy_definition": ENERGY_DEFINITION,
                "divisions": 8, "normalized_arc_step": .1,
                "relative_equilibrium_tolerance": 1e-6}
    expected = hashlib.sha256(window._canonical(identity).encode()).hexdigest()[:20]
    assert manifest.name == f"chunked-{expected}.json"


def test_executor_advances_only_one_recoverable_point_per_call(tmp_path, monkeypatch):
    accepted = 0
    calls = []

    def fake_execute(divisions, normalized_arc_step, *, steps, **kwargs):
        nonlocal accepted
        calls.append((divisions, normalized_arc_step, steps, kwargs["chunk_size"]))
        accepted = steps
        return {"accepted_points": accepted, "point_history": [
            {"force_n": 100_000. * number} for number in range(1, accepted + 1)
        ]}

    monkeypatch.setattr(window, "execute_panel_chunked_job", fake_execute)
    result = window.execute_8x8_peak_window(cache_dir=tmp_path, target_steps=3)
    assert calls == [(8, .1, 1, 1), (8, .1, 2, 1), (8, .1, 3, 1)]
    assert result["path"]["accepted_points"] == 3
    assert [item["zone"] for item in result["schedule"]] == ["coarse"] * 3


def test_scheduler_does_not_replay_when_disk_manifest_lags_returned_result(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "lagging.json"
    checkpoint_path = tmp_path / "lagging.pt"
    manifest_path.write_text(json.dumps({"accepted_points": 0,
                                         "point_history": []}))
    calls = []

    monkeypatch.setattr(
        window, "_checkpoint_paths",
        lambda *args, **kwargs: (manifest_path, checkpoint_path),
    )

    def fake_execute(divisions, normalized_arc_step, *, steps, **kwargs):
        calls.append(steps)
        # Deliberately do not update the on-disk manifest: this models a stale
        # path lookup after the executor has returned a newer accepted prefix.
        return {"accepted_points": steps, "point_history": [
            {"force_n": 100_000. * number} for number in range(1, steps + 1)
        ]}

    monkeypatch.setattr(window, "execute_panel_chunked_job", fake_execute)
    result = window.execute_8x8_peak_window(cache_dir=tmp_path, target_steps=2)
    assert calls == [1, 2]
    assert result["path"]["accepted_points"] == 2


def test_generic_executor_preserves_requested_mesh_identity(tmp_path, monkeypatch):
    calls = []

    def fake_execute(divisions, normalized_arc_step, *, steps, **kwargs):
        calls.append((divisions, normalized_arc_step, steps))
        return {"accepted_points": steps, "point_history": [{"force_n": 1.}]}

    monkeypatch.setattr(window, "execute_panel_chunked_job", fake_execute)
    result = window.execute_panel_peak_window(
        divisions=12, cache_dir=tmp_path, target_steps=1,
    )
    assert calls == [(12, .1, 1)]
    assert result["mesh_divisions"] == 12


@pytest.mark.parametrize("divisions", [2, 5])
def test_generic_executor_rejects_unsupported_meshes(tmp_path, divisions):
    with pytest.raises(ValueError, match="even integer >= 4"):
        window.execute_panel_peak_window(
            divisions=divisions, cache_dir=tmp_path, target_steps=1,
        )
