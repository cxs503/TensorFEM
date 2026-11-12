from tensorfem.pinched_cylinder import (solve_pinched_cylinder_linear,
                                         solve_pinched_cylinder_sparse,
                                         solve_pinched_cylinder_nonlinear)


def test_pinched_cylinder_fine_mesh_qualifies_and_converges():
    results=[solve_pinched_cylinder_linear(n) for n in (6,8,12)]
    errors=[r.relative_error for r in results]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < .03


def test_pinched_cylinder_global_nonlinear_step_below_three_percent():
    result=solve_pinched_cylinder_nonlinear(3)
    assert result.relative_error < .03


def test_sparse_path_reproduces_the_production_dense_solution():
    """CSR/SuperLU and dense elimination are two storage paths for one operator."""
    dense=solve_pinched_cylinder_linear(6).displacement
    sparse=solve_pinched_cylinder_sparse(6).displacement
    assert abs(sparse/dense-1.) < 1e-10


def test_case_pool_pinched_cylinder_16x16_within_five_percent():
    """CASE-FEM-PINCHED-CYLINDER acceptance: 16x16 octant mesh within 5%."""
    result=solve_pinched_cylinder_sparse(16)
    assert result.relative_error <= .05
    assert result.displacement < 0.


def test_axial_end_clamp_stiffens_the_diaphragm_model():
    """A rigid diaphragm leaves the axial translation free; clamping it is 15% stiffer."""
    free_response=solve_pinched_cylinder_sparse(8).displacement
    clamped=solve_pinched_cylinder_sparse(8,axial_u_clamped=True).displacement
    assert abs(clamped) < .95*abs(free_response)


def test_transverse_shear_correction_is_not_a_fit_parameter():
    """R/t = 100 bending response must be insensitive to the shear factor."""
    k_56=solve_pinched_cylinder_sparse(8,shear_factor=5./6.).displacement
    k_1=solve_pinched_cylinder_sparse(8,shear_factor=1.).displacement
    assert abs(k_1/k_56-1.) < .01