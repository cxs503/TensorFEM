import math
import torch

from tensorfem.nonlinear_truss import (
    NonlinearTrussModel, internal_force_and_tangent, solve_load_control,
    two_bar_shallow_arch_reaction,
)
from tensorfem.plasticity import Plastic1DState, J2State, update_bilinear_1d, update_j2


D = torch.float64


def test_total_lagrangian_tangent_by_finite_difference():
    model = NonlinearTrussModel(torch.tensor([[0., 0.], [1.2, .4]], dtype=D),
        torch.tensor([[0, 1]]), torch.tensor([210.], dtype=D),
        torch.tensor([.03], dtype=D), torch.tensor([0, 1]))
    u = torch.tensor([0., 0., .08, -.03], dtype=D)
    f, k, _ = internal_force_and_tangent(model, u)
    eps = 1e-7
    for j in (2, 3):
        up, um = u.clone(), u.clone()
        up[j] += eps; um[j] -= eps
        fp = internal_force_and_tangent(model, up)[0]
        fm = internal_force_and_tangent(model, um)[0]
        rel = torch.linalg.vector_norm((fp-fm)/(2*eps)-k[:, j]) / torch.linalg.vector_norm(k[:, j])
        assert float(rel) < 1e-7


def test_incremental_newton_single_bar_exact_solution():
    model = NonlinearTrussModel(torch.tensor([[0., 0.], [1., 0.]], dtype=D),
        torch.tensor([[0, 1]]), torch.tensor([200.], dtype=D), torch.tensor([2.], dtype=D),
        torch.tensor([0, 1, 3]))
    load = torch.tensor([0., 0., 46.2, 0.], dtype=D)
    steps = solve_load_control(model, load, torch.linspace(.1, 1., 10, dtype=D))
    # TL equilibrium: P=EA/2*((1+u)^3-(1+u)); u=.1 gives P=46.2.
    assert abs(float(steps[-1].displacement[2]) - .1) / .1 < 1e-9
    assert max(s.iterations for s in steps) <= 5


def test_two_bar_shallow_arch_exact_curve_and_limit_point():
    a, h, young, area = 1.0, .2, 2000.0, .01
    v = torch.linspace(0., .35, 10001, dtype=D)
    p = two_bar_shallow_arch_reaction(a, h, young, area, v)
    # d[(h^2-y^2)y]/dy=h^2-3y^2=0; first peak y=h/sqrt(3).
    vcrit = h * (1.0 - 1.0/math.sqrt(3.0))
    pcrit = young*area * (h*h-(h/math.sqrt(3))**2)*(h/math.sqrt(3)) / (a*a+h*h)**1.5
    i = int(torch.argmax(p))
    assert abs(float(v[i])-vcrit)/vcrit < 3e-4
    assert abs(float(p[i])-pcrit)/pcrit < 1e-7


def test_bilinear_return_mapping_yield_and_tangent():
    E, sy, H = 200000., 250., 10000.
    state = Plastic1DState(torch.tensor(0., dtype=D), torch.tensor(0., dtype=D))
    stress, tangent, state = update_bilinear_1d(torch.tensor(.005, dtype=D), E, sy, H, state)
    exact = sy + E*H/(E+H) * (.005-sy/E)
    assert abs(float(stress)-exact)/exact < 1e-12
    assert abs(float(tangent)-E*H/(E+H))/(E*H/(E+H)) < 1e-12
    assert abs(float(stress)-(sy+H*float(state.alpha))) < 1e-10


def test_perfect_plastic_cyclic_dissipation():
    E, sy, amplitude = 200000., 250., .005
    state = Plastic1DState(torch.tensor(0., dtype=D), torch.tensor(0., dtype=D))
    _, _, state = update_bilinear_1d(torch.tensor(amplitude, dtype=D), E, sy, 0., state)
    history = torch.cat((torch.linspace(amplitude, -amplitude, 1601, dtype=D),
                         torch.linspace(-amplitude, amplitude, 1601, dtype=D)[1:]))
    stresses = []
    for strain in history:
        stress, _, state = update_bilinear_1d(strain, E, sy, 0., state)
        stresses.append(stress)
    stresses = torch.stack(stresses)
    work = torch.sum(.5*(stresses[:-1]+stresses[1:])*(history[1:]-history[:-1]))
    exact = 4.*sy*(amplitude-sy/E)
    assert abs(float(work)-exact)/exact < 1e-10


def test_j2_radial_return_satisfies_yield_surface():
    E, nu, sy, H = 210000., .3, 300., 2000.
    z = torch.zeros((3, 3), dtype=D)
    state = J2State(z.clone(), torch.tensor(0., dtype=D))
    strain = torch.diag(torch.tensor([.004, -.002, -.002], dtype=D))
    stress, state, dg = update_j2(strain, E, nu, sy, H, state)
    s = stress - torch.trace(stress)/3 * torch.eye(3, dtype=D)
    seq = torch.sqrt(1.5*torch.sum(s*s))
    radius = sy + H*state.alpha
    assert float(dg) > 0
    assert abs(float(seq-radius))/float(radius) < 1e-12
