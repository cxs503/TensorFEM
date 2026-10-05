import json

import pytest
import torch

from tensorfem.von_mises_arch import arch_limit_reference, trace_von_mises_arch


def _difficult_arch_checkpoint(tmp_path):
    checkpoint = tmp_path / "arch-prefix.json"
    prefix = trace_von_mises_arch(
        .005, 24, maximum_step=.005, checkpoint=checkpoint,
    )
    saved = json.loads(checkpoint.read_text())
    saved["step_size"] = .4
    checkpoint.write_text(json.dumps(saved))
    return prefix, checkpoint


def test_backtracking_closes_real_arch_failure_and_preserves_path(tmp_path):
    prefix, checkpoint = _difficult_arch_checkpoint(tmp_path)
    fixed_diagnostics = []
    fixed = trace_von_mises_arch(
        .4, 1, maximum_step=.4, minimum_step=.396, max_iterations=6,
        restart=checkpoint, diagnostics=fixed_diagnostics,
    )
    line_diagnostics = []
    line = trace_von_mises_arch(
        .4, 1, maximum_step=.4, minimum_step=.396, max_iterations=6,
        restart=checkpoint, line_search="backtracking",
        diagnostics=line_diagnostics,
    )
    reference = trace_von_mises_arch(
        .4, 1, maximum_step=.4, minimum_step=.396, max_iterations=20,
        restart=checkpoint,
    )
    assert fixed.converged is False and fixed_diagnostics[0]["reason"] == "rejected"
    assert line.converged is True and line_diagnostics[0]["reason"] == "accepted"
    assert min(item["line_search_alpha"]
               for item in line_diagnostics[0]["iterations"]) == .5
    assert abs(line.points[0].load_factor/reference.points[0].load_factor-1) < 1e-12
    assert torch.allclose(line.points[0].displacement,
                          reference.points[0].displacement, rtol=1e-12, atol=1e-12)

    loads = [point.load_factor for point in prefix.points]
    first_peak = next(loads[i] for i in range(1, len(loads)-1)
                      if loads[i] > loads[i-1] and loads[i] > loads[i+1])
    assert abs(first_peak/arch_limit_reference().load-1) < .01


def test_backtracking_restart_is_exact_and_identity_is_fail_closed(tmp_path):
    _, checkpoint = _difficult_arch_checkpoint(tmp_path)
    direct = trace_von_mises_arch(
        .4, 3, maximum_step=.4, minimum_step=.396, max_iterations=6,
        restart=checkpoint, line_search="backtracking",
    )
    line_checkpoint = tmp_path / "line-search.json"
    first = trace_von_mises_arch(
        .4, 1, maximum_step=.4, minimum_step=.396, max_iterations=6,
        restart=checkpoint, checkpoint=line_checkpoint,
        line_search="backtracking",
    )
    resumed = trace_von_mises_arch(
        .4, 2, maximum_step=.4, minimum_step=.396, max_iterations=6,
        restart=line_checkpoint, line_search="backtracking",
    )
    assert direct.converged and first.converged and resumed.converged
    assert torch.allclose(resumed.points[-1].displacement,
                          direct.points[-1].displacement, rtol=1e-12, atol=1e-12)
    assert resumed.points[-1].load_factor == pytest.approx(
        direct.points[-1].load_factor, rel=1e-12,
    )
    with pytest.raises(ValueError, match="line-search identity mismatch"):
        trace_von_mises_arch(.4, 1, restart=line_checkpoint)


def test_line_search_is_explicit_and_validated():
    baseline = trace_von_mises_arch(.005, 5, maximum_step=.005)
    explicit = trace_von_mises_arch(.005, 5, maximum_step=.005,
                                    line_search=None)
    assert torch.equal(baseline.points[-1].displacement,
                       explicit.points[-1].displacement)
    assert baseline.points[-1].load_factor == explicit.points[-1].load_factor
    with pytest.raises(ValueError, match="line_search"):
        trace_von_mises_arch(.005, 1, line_search="automatic")

