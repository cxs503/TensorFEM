"""Fail-closed regression for the experimental assumed-shear shell."""
import math
import torch

from tensorfem.shell_mitc_benchmarks import scordelis_lo_roof_experimental


def test_scordelis_lo_assumed_shear_remains_unqualified():
    result = scordelis_lo_roof_experimental(4, 4)
    assert math.isfinite(result.probe_displacement)
    error = abs(result.probe_displacement / -0.3024 - 1.0)
    assert error > 0.03


def test_scordelis_lo_assumed_shear_load_balance():
    result = scordelis_lo_roof_experimental(4, 4)
    reaction = result.reaction.reshape(-1, 6)[:, 2].sum()
    exact = 90.0 * 50.0 * 25.0 * torch.deg2rad(torch.tensor(80.0, dtype=torch.float64))
    assert abs(float(reaction / exact) - 1.0) < 0.03
