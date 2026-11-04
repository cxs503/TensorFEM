"""Verification tests for the rotating-disk (centrifugal body force) benchmark.

Two layers:

* the *oracle* must be proved before it grades anything -- free rims, radial
  equilibrium, compatibility, global hoop-force balance, solid-disk limit, and a
  negative control showing that the equilibrium check rejects a wrong field;
* the *merged HEX8 + sparse CG path* is then graded against that oracle, with
  scaling and null-load controls so a wrong unit/leak cannot pass as agreement.
"""
import math

import pytest
import torch

from tensorfem.rotating_disk import (
    RotatingDisk,
    centrifugal_load,
    cylindrical_stress,
    quarter_disk_mesh,
    rotating_disk_run,
    solid_disk_reference,
)


@pytest.fixture(scope="module")
def disk():
    return RotatingDisk()


# ---------------------------------------------------------------- oracle ----
def test_free_rims(disk):
    inner, outer = disk.rim_tractions()
    peak = disk.sigma_peak()
    assert abs(inner) <= 1e-12 * peak
    assert abs(outer) <= 1e-12 * peak


def test_equilibrium_residual_vanishes(disk):
    r = torch.tensor([0.5, 0.55, 0.7, 0.85, 0.95, 1.0], dtype=torch.float64)
    residual = disk.equilibrium_residual(r)
    assert float(residual.abs().max()) < 1e-8


def test_compatibility_residual_vanishes(disk):
    r = torch.tensor([0.5, 0.6, 0.75, 0.9, 1.0], dtype=torch.float64)
    assert float(disk.compatibility_residual(r).abs().max()) < 1e-10


def test_hoop_force_balance(disk):
    lhs, rhs = disk.hoop_force_balance()
    assert abs(lhs - rhs) / abs(rhs) < 1e-6


def test_solid_disk_limit_matches(disk):
    solid = RotatingDisk(inner=0.0, outer=disk.outer, young_modulus=disk.young_modulus,
                         poisson_ratio=disk.poisson_ratio, density=disk.density,
                         omega=disk.omega)
    r = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float64)
    reference = solid_disk_reference(solid, r)
    assert torch.allclose(solid.sigma_r(r), reference["sigma_r"], rtol=1e-12)
    assert torch.allclose(solid.sigma_theta(r), reference["sigma_theta"], rtol=1e-12)
    assert torch.allclose(solid.u_r(r), reference["u_r"], rtol=1e-12)


def test_equilibrium_check_has_teeth(disk):
    """A deliberately wrong hoop stress must be rejected by the same check."""

    class Broken(RotatingDisk):
        def sigma_theta(self, r):
            return super().sigma_theta(r) * 1.01

    broken = Broken()
    r = torch.tensor([0.6, 0.8], dtype=torch.float64)
    assert float(broken.equilibrium_residual(r).abs().max()) > 1e-2
    assert float(disk.equilibrium_residual(r).abs().max()) < 1e-8


@pytest.mark.parametrize("bad", [
    {"inner": 1.0, "outer": 1.0},
    {"poisson_ratio": 0.5},
    {"thickness": 0.0},
    {"omega": -1.0},
])
def test_parameter_validation(bad):
    with pytest.raises(ValueError):
        RotatingDisk(**bad)


# ---------------------------------------------------------------- mesh -----
def test_quarter_mesh_is_conforming(disk):
    nodes, hexes = quarter_disk_mesh(disk, 4, 3, 2)
    assert nodes.shape == ((4 + 1) * (3 + 1) * (2 + 1), 3)
    assert hexes.shape == (4 * 3 * 2, 8)
    assert int(hexes.max()) == len(nodes) - 1
    assert int(hexes.min()) == 0
    # every Jacobian positive: the element refuses anything else
    from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness

    D = elasticity_matrix_3d(nodes.new_full((len(hexes),), disk.young_modulus),
                             nodes.new_full((len(hexes),), disk.poisson_ratio))
    hex8_stiffness(nodes[hexes], D)


def test_centrifugal_load_resultant_converges(disk):
    """The consistent load reproduces the analytic resultant up to the mesh geometry.

    A polar block of HEX8 elements is straight sided, so its volume (and therefore
    the integrated centrifugal resultant) is the *inscribed* polygonal value; the
    difference must decay like ``1/angular**2``, which is what is checked here
    rather than a machine-precision identity that the mesh cannot deliver.
    """
    expected = (disk.density * disk.omega ** 2
                * (disk.outer ** 3 - disk.inner ** 3) / 3.0 * 0.5 * disk.thickness)
    errors = []
    for angular in (4, 8, 16):
        nodes, hexes = quarter_disk_mesh(disk, 4 * angular, angular, 1)
        total = centrifugal_load(disk, nodes, hexes).reshape(-1, 3).sum(dim=0)
        assert abs(float(total[0]) - float(total[1])) < 1e-9 * expected
        assert abs(float(total[2])) < 1e-6 * expected
        errors.append(abs(float(total[0]) - expected) / expected)
    assert errors[0] < 5e-2
    assert errors[2] < 3e-3
    assert errors[0] / errors[2] > 12.0


# ---------------------------------------------------------------- solver ---
def test_coarse_run_is_accurate(disk):
    result = rotating_disk_run(disk, radial=16, angular=8, layers=2, rtol=1e-10)
    assert result.u_error < 1.2e-2
    assert result.hoop_error < 2e-2
    assert result.radial_error < 4e-2
    assert result.plane_stress_ratio < 1e-2
    assert result.extra["reaction_closure"] < 1e-10


def test_h_convergence(disk):
    coarse = rotating_disk_run(disk, radial=8, angular=4, layers=1, rtol=1e-10)
    fine = rotating_disk_run(disk, radial=32, angular=16, layers=2, rtol=1e-10)
    assert fine.hoop_error < coarse.hoop_error / 8.0
    assert fine.u_error < coarse.u_error / 5.0
    assert fine.hoop_ring_error < 1e-2


def test_zero_speed_gives_zero_displacement(disk):
    still = RotatingDisk(omega=0.0, outer=disk.outer, inner=disk.inner,
                         young_modulus=disk.young_modulus,
                         poisson_ratio=disk.poisson_ratio,
                         density=disk.density, thickness=disk.thickness)
    result = rotating_disk_run(still, radial=8, angular=4, layers=1, rtol=1e-12)
    assert max(abs(v) for row in result.extra["displacement_fe"] for v in row) < 1e-18


def test_displacement_scales_with_omega_squared(disk):
    fast = RotatingDisk(omega=2.0 * disk.omega, outer=disk.outer, inner=disk.inner,
                        young_modulus=disk.young_modulus,
                        poisson_ratio=disk.poisson_ratio,
                        density=disk.density, thickness=disk.thickness)
    base = rotating_disk_run(disk, radial=8, angular=4, layers=1, rtol=1e-10)
    scaled = rotating_disk_run(fast, radial=8, angular=4, layers=1, rtol=1e-10)
    a = torch.tensor(base.extra["displacement_fe"], dtype=torch.float64)
    b = torch.tensor(scaled.extra["displacement_fe"], dtype=torch.float64)
    assert float(torch.linalg.vector_norm(b - 4.0 * a) / torch.linalg.vector_norm(4.0 * a)) < 1e-8
    assert abs(scaled.hoop_peak_got / base.hoop_peak_got - 4.0) < 1e-8


def test_thickness_insensitivity(disk):
    """Halving the thickness changes the plane-stress answer by well under 0.1%."""
    thin = RotatingDisk(thickness=0.5 * disk.thickness, outer=disk.outer,
                        inner=disk.inner, young_modulus=disk.young_modulus,
                        poisson_ratio=disk.poisson_ratio, density=disk.density,
                        omega=disk.omega)
    a = rotating_disk_run(disk, radial=16, angular=8, layers=2, rtol=1e-10)
    b = rotating_disk_run(thin, radial=16, angular=8, layers=2, rtol=1e-10)
    assert abs(a.hoop_error - b.hoop_error) < 1e-3
    assert abs(a.hoop_peak_got - b.hoop_peak_got) / a.hoop_peak_got < 1e-3


def test_cylindrical_rotation_is_consistent(disk):
    centres = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                            [0.5 / math.sqrt(2), 0.5 / math.sqrt(2), 0.0]],
                           dtype=torch.float64)
    sigma = torch.zeros((3, 6), dtype=torch.float64)
    sigma[:, 0] = 1.0            # sigma_xx = 1, everything else zero
    srr, stt, srt, szz = cylindrical_stress(sigma, centres)
    assert abs(float(srr[0]) - 1.0) < 1e-12 and abs(float(stt[0])) < 1e-12
    assert abs(float(stt[1]) - 1.0) < 1e-12 and abs(float(srr[1])) < 1e-12
    assert abs(float(srr[2]) - 0.5) < 1e-12 and abs(float(stt[2]) - 0.5) < 1e-12
    assert abs(float(srt[2]) + 0.5) < 1e-12
    assert float(szz.abs().max()) < 1e-12