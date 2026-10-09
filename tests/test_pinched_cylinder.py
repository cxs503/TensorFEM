from tensorfem.pinched_cylinder import (solve_pinched_cylinder_linear,
                                         solve_pinched_cylinder_nonlinear)


def test_pinched_cylinder_fine_mesh_qualifies_and_converges():
    results=[solve_pinched_cylinder_linear(n) for n in (6,8,12)]
    errors=[r.relative_error for r in results]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < .03


def test_pinched_cylinder_global_nonlinear_step_below_three_percent():
    result=solve_pinched_cylinder_nonlinear(3)
    assert result.relative_error < .03
