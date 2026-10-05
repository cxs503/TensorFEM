from tensorfem.hertz_3d_fe import solve_hertz_cap_block

def test_true_tet4_two_solid_hertz_level_one_closure():
    result=solve_hertz_cap_block(cells=2)
    assert result.force>0 and result.reference_force>0
    assert result.residual_norm<1e-7
    assert result.minimum_jacobian>0
    assert result.force_imbalance<1e-9
