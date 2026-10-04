from types import SimpleNamespace

import torch

import tensorfem.marine_panel_execution as execution


def test_energy_payload_uses_relaxed_baseline_and_enriches_each_point(monkeypatch):
    relaxed_u = torch.tensor([.125])
    relaxed_state = object()
    accepted = [object()]
    path = SimpleNamespace(
        points=accepted,
        relaxed_initial_displacement=relaxed_u,
        relaxed_initial_state=relaxed_state,
    )
    case = SimpleNamespace(model=object(), reference_load=object())
    captured = {}
    point = SimpleNamespace(
        external_work=12., recoverable_energy=8., plastic_dissipation=3.,
        internal_energy=11., energy_residual=1., relative_energy_residual=.08,
        failure_mode="interactive_buckling_yielding", is_peak=False,
        is_post_peak=True,
    )

    def fake_evaluate(model, load, points, **kwargs):
        captured.update(model=model, load=load, points=points, **kwargs)
        return SimpleNamespace(
            points=(point,), failure_mode=point.failure_mode,
            post_peak_confirmed=True,
        )

    monkeypatch.setattr(execution, "evaluate_panel_path", fake_evaluate)
    history = [{"step": 1}]
    summary = execution._accepted_path_energy_payload(case, path, history)

    assert captured["initial_displacement"] is relaxed_u
    assert captured["initial_state"] is relaxed_state
    assert captured["points"] is accepted
    assert history[0]["external_work_j"] == 12.
    assert history[0]["plastic_dissipation_j"] == 3.
    assert history[0]["relative_energy_residual"] == .08
    assert history[0]["failure_mode"] == "interactive_buckling_yielding"
    assert summary["recoverable_energy_j"] == 8.
    assert summary["internal_energy_j"] == 11.
    assert summary["energy_post_peak_confirmed"] is True


def test_energy_payload_fails_closed_without_relaxed_baseline():
    path = SimpleNamespace(points=[], relaxed_initial_displacement=None,
                           relaxed_initial_state=None)
    payload = execution._accepted_path_energy_payload(
        SimpleNamespace(), path, [],
    )
    assert payload["energy_evidence_available"] is False
    assert payload["external_work_j"] is None
    assert payload["failure_mode"] == "undetermined"
