import hashlib
import json

import pytest
import torch

import tensorfem.panel_peak_window as window
from tensorfem.panel_peak_window import (
    PeakTargetEstimate, PeakWindowPolicy, estimate_peak_window_target,
    retune_next_arc_step, select_peak_window,
)


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
