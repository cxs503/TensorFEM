import torch

from tensorfem.nonlinear_controller import recommend_nonlinear_controls
from tensorfem.von_mises_arch import arch_limit_reference, trace_von_mises_arch


def _peak(path):
    return max(path.points, key=lambda point: point.load_factor)


def test_von_mises_arch_step_policy_preserves_peak_and_restart_path(tmp_path):
    """Step-only opt-in cannot alter a path while its upper bound is active."""
    fixed = trace_von_mises_arch(.005, 70, maximum_step=.005)
    checkpoint = tmp_path / "arch.json"
    prefix = trace_von_mises_arch(.005, 30, maximum_step=.005,
                                  checkpoint=checkpoint)
    restarted = trace_von_mises_arch(.005, 40, maximum_step=.005,
                                     restart=checkpoint)
    assert fixed.converged and prefix.converged and restarted.converged
    assert torch.allclose(restarted.points[-1].displacement,
                          fixed.points[-1].displacement, rtol=1e-8, atol=1e-10)
    assert abs(restarted.points[-1].load_factor-fixed.points[-1].load_factor) < 1e-8
    assert abs(_peak(fixed).load_factor/arch_limit_reference().load-1.) < .03

    history = [
        {"force_n": float(point.load_factor),
         "edge_shortening_m": float(point.displacement[0]),
         "equilibrium_relative_norm": 1e-12,
         "energy_balance_gate": {"passed": True}}
        for point in fixed.points[:3]
    ]
    diagnostics = [{"attempt": 1, "reason": "accepted", "iterations": [
        {"residual_relative": 1e-2}, {"residual_relative": 1e-4},
        {"residual_relative": 1e-8},
    ]}]
    decision = recommend_nonlinear_controls(
        history, diagnostics, automatic_step_control=True,
        current_step_size=.005, minimum_step_size=.005/128,
        maximum_step_size=.005,
    )
    assert decision["step_application"]["applied_step_size"] == .005
    assert decision["automatic_algorithm_switching_enabled"] is False

