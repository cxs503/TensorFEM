"""Round-off-scaled acceptance of the plane-stress sigma_zz condensation.

The layered Shell4 material point enforces ``sigma_zz = 0`` by a scalar Newton
iteration on the out-of-plane strain.  The acceptance thresholds used to be
fixed absolute numbers (``1e-15*E`` on ``sigma_zz`` and ``1e-17`` on the Newton
step).  As soon as the trial strain is large -- which happens at every
non-converged Newton iterate of an arc-length continuation step -- the residual
is a difference of O(E*strain) stresses, so no ``eps_zz`` can satisfy the
absolute bound and the Newton step stalls inside one unit in the last place.
The element then raised ``plane-stress local condensation did not converge``
and aborted a solve that only needed to keep a *non-converged* iterate finite
(issue #8; previously recorded as machine-dependent ``xfail`` in
``test_shell4_arc_line_search.py`` and ``test_v1_evidence_aggregator.py``).

These tests pin the fix at material-point level, independently of the arc-length
solver: the condensed in-plane stress is compared against an independent
bisection root of ``sigma_zz(eps_zz) = 0`` computed from the same return map.
"""
import pytest
import torch

from tensorfem.layered_shell4_plasticity import (LayeredShell4Model,
                                                 LayeredShell4State,
                                                 _tensor_from_inplane,
                                                 plane_stress_j2_update)
from tensorfem.plasticity import update_j2

D = torch.float64


def _model(*, hardening=1400.):
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [1., 1., 0.], [0., 1., 0.]], dtype=D)
    return LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3, .08,
                              250., hardening, layers=3)


def _virgin(model):
    return LayeredShell4State.virgin(model).points[0][0][0]


def _sigma_zz(zz, value, model, committed):
    full = _tensor_from_inplane(value, torch.as_tensor(float(zz), dtype=D))
    sigma, _, _ = update_j2(full, model.young, model.poisson,
                            model.yield_stress, model.hardening, committed)
    return float(sigma[2, 2]), sigma


def _bisection_root(value, model, committed, *, iterations=200):
    """Independent bracketing root of sigma_zz(eps_zz) = 0 (monotone in eps_zz)."""
    poisson = model.poisson
    centre = -poisson/(1-poisson)*(value[0]+value[1]).item()
    lo = hi = centre
    f_lo = f_hi = _sigma_zz(centre, value, model, committed)[0]
    step = max(1., .5*abs(centre))
    for _ in range(200):
        if not f_lo < 0.:
            lo -= step
            f_lo = _sigma_zz(lo, value, model, committed)[0]
        if not f_hi > 0.:
            hi += step
            f_hi = _sigma_zz(hi, value, model, committed)[0]
        if f_lo < 0. < f_hi:
            break
        step *= 2.
    else:
        raise AssertionError('could not bracket the sigma_zz root')
    for _ in range(iterations):
        mid = .5*(lo+hi)
        if _sigma_zz(mid, value, model, committed)[0] < 0.:
            lo = mid
        else:
            hi = mid
    return .5*(lo+hi)


# The exact strain at which the arc-length cantilever (test_shell4_arc_line_search)
# stalled: a non-converged Newton iterate with 1000% membrane strain.
STALLED_STRAIN = [3.368958579694938, 10.018737002848292, 1.115051018219515]


@pytest.mark.parametrize('hardening', [0., 1400.])
@pytest.mark.parametrize('strains', [
    STALLED_STRAIN,
    [0.002, 0.001, 0.0005],
    [0.05, 0.20, 0.01],
    [1.0, 5.0, -2.0],
    [-8.0, 0.0, 0.0],
])
def test_condensation_closes_sigma_zz_to_working_precision(strains, hardening):
    model = _model(hardening=hardening)
    committed = _virgin(model)
    value = torch.tensor(strains, dtype=D)
    stress, tangent, trial = plane_stress_j2_update(value, model, committed)
    assert bool(torch.all(torch.isfinite(stress)))
    assert tangent.shape == (3, 3)

    root = _bisection_root(value, model, committed)
    f_root, sigma = _sigma_zz(root, value, model, committed)
    reference = torch.stack((sigma[0, 0], sigma[1, 1], sigma[0, 1]))
    scale = max(float(reference.abs().max()), 1.)
    # The condensed stress must be the plane-stress point, not an off-plane one.
    assert float((stress-reference).abs().max()) <= 1e-9*scale
    # ... and the local root itself only carries round-off level sigma_zz.
    assert abs(f_root) <= 1e-9*scale
    assert bool(torch.all(torch.isfinite(tangent)))
    assert float(trial.alpha) >= 0.


def test_stalled_arc_length_iterate_is_not_reported_as_non_convergence():
    """Regression for issue #8: this strain used to raise RuntimeError."""
    model = _model()
    committed = _virgin(model)
    value = torch.tensor(STALLED_STRAIN, dtype=D)
    stress, _, _ = plane_stress_j2_update(value, model, committed)
    assert bool(torch.all(torch.isfinite(stress)))
    # Stresses are O(1e4) here, so the former absolute 2e-10 bound on sigma_zz
    # was not reachable in double precision: the iterate oscillated by one ulp.
    assert float(stress.abs().max()) > 1e4
    root = _bisection_root(value, model, committed)
    sigma = _sigma_zz(root, value, model, committed)[1]
    reference = torch.stack((sigma[0, 0], sigma[1, 1], sigma[0, 1]))
    assert float((stress-reference).abs().max()) <= 1e-9*float(reference.abs().max())


def test_non_finite_tangent_still_fails_closed():
    model = _model()
    committed = _virgin(model)
    with pytest.raises(RuntimeError, match='plane-stress local condensation'):
        plane_stress_j2_update(torch.tensor([1e200, -1e200, 1e200], dtype=D),
                               model, committed)


def test_plane_stress_path_is_continuous_across_yield():
    """Yielding changes the slope, not the stress: no jump at first yield."""
    model = _model(hardening=0.)
    committed = _virgin(model)
    previous = None
    increment = 1e-5
    # Elastic stress change of one increment: E*(1+nu)/(1-nu^2)*deps < 4e2*increment
    bound = 1.5*model.young*increment*(1+model.poisson)/(1-model.poisson**2)
    for step in range(200):
        value = torch.tensor([increment*step, -2*increment*step, 0.], dtype=D)
        stress, _, _ = plane_stress_j2_update(value, model, committed)
        if previous is not None:
            assert float((stress-previous).abs().max()) <= bound
        previous = stress
    # The walk must actually reach the yield surface (sigma_y = 250): a purely
    # elastic path would stop at about 90 here.
    assert float(previous.abs().max()) > 200.