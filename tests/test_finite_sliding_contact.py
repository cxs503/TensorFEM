import math

import torch

from tensorfem.finite_sliding_contact import (
    initial_friction_state, project_point_to_polyline,
    update_node_polyline_contact,
)

D = torch.float64


def test_large_motion_researches_projection_and_preserves_gap():
    surface = torch.tensor([[0., 0.], [1., 0.], [2., .5]], dtype=D)
    p0 = torch.tensor([.1, .2], dtype=D)
    state = initial_friction_state(p0, surface)
    # Motion is many times the first segment length and ends over segment 1.
    p1 = torch.tensor([1.6, .5], dtype=D)
    projection = project_point_to_polyline(p1, surface)
    exact_xi = ((p1 - surface[1]) @ (surface[2] - surface[1])
                / torch.sum((surface[2] - surface[1])**2))
    assert projection.segment == 1
    assert abs(float(projection.coordinate - exact_xi)) < 1e-14
    update = update_node_polyline_contact(
        p1, surface, state, normal_penalty=1e5,
        tangential_penalty=1e4, friction=.3,
        relative_tangential_increment=0.,
    )
    # Point is on the open side: contact correctly opens after re-search.
    assert not update.state.active
    assert torch.equal(update.traction, torch.zeros(2, dtype=D))


def test_inclined_plane_stick_slip_matches_coulomb_oracle():
    angle = math.radians(30.)
    t = torch.tensor([math.cos(angle), math.sin(angle)], dtype=D)
    n = torch.tensor([-math.sin(angle), math.cos(angle)], dtype=D)
    surface = torch.stack((-2*t, 2*t))
    kn, kt, mu, penetration = 2e5, 4e4, .25, .002
    point = .2*t - penetration*n
    state = initial_friction_state(.2*t + .001*n, surface)
    stick = update_node_polyline_contact(
        point, surface, state, normal_penalty=kn,
        tangential_penalty=kt, friction=mu,
        relative_tangential_increment=.001,
    )
    exact_normal = kn*penetration
    exact_stick = -kt*.001
    assert abs(float(stick.normal_traction)-exact_normal)/exact_normal < .03
    assert abs(float(stick.tangential_traction)-exact_stick)/abs(exact_stick) < .03
    assert stick.state.sticking
    slide = update_node_polyline_contact(
        point, surface, stick.state, normal_penalty=kn,
        tangential_penalty=kt, friction=mu,
        relative_tangential_increment=.02,
    )
    exact_limit = mu*exact_normal
    assert not slide.state.sticking
    assert abs(float(abs(slide.tangential_traction))-exact_limit)/exact_limit < .03
    assert slide.dissipation_increment > 0
    assert abs(float(slide.traction@n-slide.normal_traction)) < 1e-11
    assert abs(float(slide.traction@t-slide.tangential_traction)) < 1e-11


def test_contact_open_close_and_incremental_energy_balance():
    surface = torch.tensor([[-1., 0.], [1., 0.]], dtype=D)
    state = initial_friction_state(torch.tensor([0., .1], dtype=D), surface)
    closed = update_node_polyline_contact(
        torch.tensor([0., -.001], dtype=D), surface, state,
        normal_penalty=1e5, tangential_penalty=2e4, friction=.4,
        relative_tangential_increment=.01,
    )
    assert closed.state.active and not closed.state.sticking
    # Contact work splits into recoverable tangential energy plus dissipation.
    # Work is the area below the elastic-then-perfectly-plastic traction path.
    limit = abs(float(closed.tangential_traction))
    yield_slip = limit/2e4
    tangential_work = .5*limit*yield_slip + limit*(.01-yield_slip)
    tangential_stored = .5*2e4*float(closed.state.elastic_slip**2)
    assert abs(tangential_work-(tangential_stored+float(closed.dissipation_increment))) < 1e-12
    opened = update_node_polyline_contact(
        torch.tensor([0., .01], dtype=D), surface, closed.state,
        normal_penalty=1e5, tangential_penalty=2e4, friction=.4,
        relative_tangential_increment=0.,
    )
    assert not opened.state.active
    assert float(opened.normal_traction) == 0.
    assert float(opened.tangential_traction) == 0.
    assert float(opened.state.elastic_slip) == 0.


def test_invalid_geometry_and_parameters_are_rejected():
    p = torch.tensor([0., 0.], dtype=D)
    bad = torch.tensor([[0., 0.], [0., 0.]], dtype=D)
    try:
        project_point_to_polyline(p, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate segment must be rejected")
    surface = torch.tensor([[-1., 0.], [1., 0.]], dtype=D)
    state = initial_friction_state(p, surface)
    try:
        update_node_polyline_contact(p, surface, state, normal_penalty=-1.,
                                     tangential_penalty=1., friction=.3)
    except ValueError:
        pass
    else:
        raise AssertionError("negative penalty must be rejected")
