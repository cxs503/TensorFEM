"""Regression tests for curved-shell benchmark infrastructure.

Scordelis--Lo is intentionally *not* a qualifying accuracy test: the current
flat Shell4 formulation misses its published reference by much more than 3%.
The test prevents that known limitation from being silently advertised.
"""
import torch

from tensorfem.shell_benchmarks import scordelis_lo_roof


D = torch.float64


def test_cylindrical_mesh_is_genuinely_faceted():
    result = scordelis_lo_roof(2, 2, dtype=D)
    e0, e1 = result.elements[0], result.elements[1]
    x0, x1 = result.nodes[e0], result.nodes[e1]
    n0 = torch.linalg.cross(x0[1]-x0[0], x0[3]-x0[0])
    n1 = torch.linalg.cross(x1[1]-x1[0], x1[3]-x1[0])
    n0 /= torch.linalg.vector_norm(n0); n1 /= torch.linalg.vector_norm(n1)
    assert abs(float(n0@n1)) < 1.0 - 1e-3


def test_scordelis_lo_load_reaction_balance():
    result = scordelis_lo_roof(4, 4, dtype=D)
    reaction_z = result.reaction.reshape(-1, 6)[:, 2].sum()
    # Exact shell area times the prescribed vertical surface load; the faceted
    # area approaches this value from below.
    exact_load = 90.0*50.0*25.0*torch.deg2rad(torch.tensor(80.0, dtype=D))
    assert abs(float(reaction_z/exact_load)-1.0) < 0.03


def test_scordelis_lo_remains_explicitly_unqualified():
    result = scordelis_lo_roof(4, 4, dtype=D)
    published_reference = -0.3024
    relative_error = abs(result.probe_displacement/published_reference-1.0)
    # A fail-closed guard: this experimental case must not be promoted into the
    # <3% benchmark registry until the shell formulation itself is improved.
    assert relative_error > 0.03
