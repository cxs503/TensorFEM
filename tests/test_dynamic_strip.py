import math

import pytest
import torch

from tensorfem.dynamic_strip import DynamicStrip2D


def strip(n):
    return DynamicStrip2D(length=.09, thickness=.004, young=3.5e6, density=1161.54, elements=n)


def test_interface_preserves_work_resultant_and_moment_on_deformed_strip():
    s = strip(5)
    w = s.station_weights([0., .017, .065, .09])
    s.q[1:, 1] = torch.linspace(.001, .03, 5)
    s.velocity[1:, 0] = torch.linspace(.1, .5, 5)
    s.velocity[1:, 1] = .7
    f = torch.tensor([[1., -2.], [3., 2.], [-5., .4], [6., 7.]], dtype=torch.float64)
    points, vel = s.sample(w)
    nodes, v_nodes = s.sample(torch.eye(6, dtype=torch.float64))
    nodal = w.T @ f
    assert torch.allclose(nodal.sum(0), f.sum(0))
    assert torch.allclose((vel * f).sum(), (v_nodes * nodal).sum())
    def moment(x, force):
        return (x[:, 0] * force[:, 1] - x[:, 1] * force[:, 0]).sum()
    assert torch.allclose(moment(points, f), moment(nodes, nodal))
    with pytest.raises(ValueError):
        s.station_weights([.10])


@pytest.mark.parametrize("n", [4, 8])
def test_free_vibration_period_against_dry_cantilever(n):
    s = strip(n)
    tip = s.station_weights([s.length])
    # Small-amplitude first mode in the transverse direction, initially at rest.
    beta = 1.8751040687
    x = s.reference[:, 0] / s.length
    ratio = (math.cosh(beta) + math.cos(beta)) / (math.sinh(beta) + math.sin(beta))
    s.q[:, 1] = 1.e-6 * (torch.cosh(beta*x) - torch.cos(beta*x)
                          - ratio * (torch.sinh(beta*x) - torch.sin(beta*x)))
    s.q[:, 2] = 1.e-6 * beta / s.length * (
        torch.sinh(beta*x) + torch.sin(beta*x) - ratio*(torch.cosh(beta*x)-torch.cos(beta*x)))
    period = 2*math.pi / (beta**2 * math.sqrt(s.ei/(1161.54*.004)) / s.length**2)
    force = torch.zeros((1, 2), dtype=torch.float64)
    last = s.q[-1, 1].item()
    crossings = []
    dt = period/180
    for k in range(220):
        s.advance(tip, force, dt)
        current = s.q[-1, 1].item()
        if current * last < 0:
            crossings.append((k + last/(last-current))*dt)
        last = current
    assert len(crossings) == 2
    assert abs(2*(crossings[1]-crossings[0])/period-1) < .04
    assert torch.count_nonzero(s.q[0]) == 0


def test_integrated_station_load_drives_dynamic_dofs_with_correct_axes():
    s = strip(4)
    tip = s.station_weights([s.length])
    force = torch.tensor([[.02, -.03]], dtype=torch.float64)
    dt = s.max_dt / 4
    s.advance(tip, force, dt)
    _, velocity = s.sample(tip)
    assert torch.allclose(velocity[0], force[0]*dt/s.mass[-1, 0], atol=1.e-14, rtol=1.e-10)
