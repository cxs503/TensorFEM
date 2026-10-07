import pytest

from tensorfem.marine_imperfect_strip import (
    elastic_strip_oracle, imperfect_plastic_strip_path,
    run_imperfect_strip_qualification,
)


def test_elastic_path_matches_independent_continuum_cubic_oracle():
    path = imperfect_plastic_strip_path(maximum_shortening=0.002, steps=40,
                                        fibres=64, yield_stress=10.0)
    q, force = elastic_strip_oracle(shortening=0.002)
    assert path[-1].amplitude == pytest.approx(q, rel=1e-12)
    assert path[-1].membrane_force == pytest.approx(force, rel=1e-12)


def test_auditable_plastic_state_obeys_invariants_and_evolves():
    path = imperfect_plastic_strip_path(maximum_shortening=0.025, steps=160, fibres=64)
    assert path[0].amplitude == pytest.approx(0.02)
    assert path[-1].amplitude > path[0].amplitude
    assert path[-1].yielded_fraction > 0.0
    assert max(p.equilibrium_residual for p in path) < 1e-10
    assert max(p.maximum_yield_residual for p in path) < 1e-12
    assert path[-1].energy_residual / path[-1].external_work < 0.03
    assert all(path[i].plastic_dissipation <= path[i + 1].plastic_dissipation
               for i in range(len(path) - 1))


def test_qualification_has_monotone_convergence_and_three_percent_gate():
    report = run_imperfect_strip_qualification()
    errors = report["response_errors"]
    assert report["passed"] is True
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < report["tolerance"]
    assert report["analytic_elastic_oracle_error"] < 1e-10
    assert "not a shell FE" in report["scope"]


@pytest.mark.parametrize("kwargs, exception", [
    ({"steps": True}, TypeError), ({"steps": 3}, ValueError),
    ({"fibres": 9}, ValueError), ({"young": float("nan")}, ValueError),
    ({"yield_stress": 0.0}, ValueError),
    ({"residual_stress": 1.0}, ValueError),
])
def test_strict_inputs(kwargs, exception):
    values = dict(maximum_shortening=0.02, steps=20, fibres=16)
    values.update(kwargs)
    with pytest.raises(exception):
        imperfect_plastic_strip_path(**values)
