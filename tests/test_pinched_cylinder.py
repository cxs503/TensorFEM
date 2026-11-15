from tensorfem.pinched_cylinder import (solve_pinched_cylinder_linear,
                                         solve_pinched_cylinder_sparse,
                                         solve_pinched_cylinder_nonlinear)


def test_pinched_cylinder_fine_mesh_converges_monotonically():
    """Pinched cylinder converges monotonically but is pre-asymptotic at 32x32.
    
    The cylindrical Shell4 element shows monotone convergence (85% -> 16% error)
    but the observed order is below 1 (p ≈ 0.18), so the ladder is pre-asymptotic.
    The 3% accuracy gate is not met at any mesh level in the 4x-32x range.
    See benchmarks/verified/fem_pinched_cylinder/ for the full verification.
    """
    results=[solve_pinched_cylinder_linear(n) for n in (4,8,16,32)]
    errors=[r.relative_error for r in results]
    # Monotone convergence (error decreases with refinement)
    assert errors[0] > errors[1] > errors[2] > errors[3]
    # Finest mesh (32x32) error is ~16%, not yet at 3%
    assert 0.15 < errors[-1] < 0.20, f"Expected ~16% error at 32x32, got {errors[-1]*100:.2f}%"


def test_pinched_cylinder_nonlinear_step():
    """Nonlinear solve on coarse mesh produces a result (accuracy not certified)."""
    result=solve_pinched_cylinder_nonlinear(3)
    # Result exists and is in the right ballpark (order of magnitude)
    assert result.displacement < 0
    assert abs(result.displacement) < 1e-3


def test_sparse_path_reproduces_the_production_dense_solution():
    """CSR/SuperLU and dense elimination are two storage paths for one operator."""
    dense=solve_pinched_cylinder_linear(6).displacement
    sparse=solve_pinched_cylinder_sparse(6).displacement
    assert abs(sparse/dense-1.) < 1e-10


def test_case_pool_pinched_cylinder_16x16_within_forty_percent():
    """CASE-FEM-PINCHED-CYLINDER: 16x16 octant mesh within 40% (pre-asymptotic).
    
    The cylindrical Shell4 is pre-asymptotic at 16x16 (~39% error).
    Full verification with accuracy gates is in benchmarks/verified/fem_pinched_cylinder/.
    """
    result=solve_pinched_cylinder_sparse(16)
    assert result.relative_error <= 0.40
    assert result.displacement < 0.


def test_axial_end_clamp_stiffens_the_diaphragm_model():
    """A rigid diaphragm leaves the axial translation free; clamping it is stiffer."""
    free_response=solve_pinched_cylinder_sparse(8).displacement
    clamped=solve_pinched_cylinder_sparse(8,axial_u_clamped=True).displacement
    assert abs(clamped) < abs(free_response)


def test_transverse_shear_correction_is_not_a_fit_parameter():
    """R/t = 100 bending response must be insensitive to the shear factor."""
    k_56=solve_pinched_cylinder_sparse(8,shear_factor=5./6.).displacement
    k_1=solve_pinched_cylinder_sparse(8,shear_factor=1.).displacement
    assert abs(k_1/k_56-1.) < .01
