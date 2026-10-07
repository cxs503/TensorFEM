import math

from tensorfem.large_rotation_shell_benchmark import pure_bending_shell


def test_quarter_circle_shell_path_and_mesh_convergence_below_three_percent():
    results=[pure_bending_shell(n) for n in (1,2,4)]
    errors=[r.tip_error for r in results]
    assert errors[0]>errors[1]>errors[2]
    assert errors[-1]<.03
    assert results[-1].moment_error<.03
    # Rotation-controlled accepted states follow the analytical circular path.
    for factor,uz in results[-1].result.path:
        theta=factor*math.pi/2; exact=10/theta*(1-math.cos(theta))
        assert abs(uz/exact-1.)<.03
