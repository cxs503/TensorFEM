"""Verification anchors for the surface stress recovery and the on-boundary peak reference.

The tests are deliberately self-validating: the extrapolation operator is checked
against polynomials it is claimed to reproduce (and against one it is claimed to
fail), the recovery is checked by a patch test on a distorted mesh, and the new
torsion boundary reference is checked against the interior limit of the very
series it differentiates, against the Timoshenko coefficient and against the
documented monotone-from-below behaviour of the interior grid scan.  Only the
last test solves a finite-element problem, and it is the smallest useful mesh.
"""
import math

import pytest
import torch

from tensorfem.solid3d import elasticity_matrix_3d
from tensorfem.solid3d import structured_hex_mesh
from tensorfem.hex20 import structured_hex20_mesh
from tensorfem.surface_stress_recovery import (
    extrapolation_matrix,
    extrapolation_weights_at,
    gauss_points_1d,
    natural_coordinates,
    recover_nodal_stress,
    surface_traction,
    transverse_shear_magnitude,
    voigt_stress_to_tensor,
)
from tensorfem.torsion_bar import (
    boundary_shear_peak,
    max_shear_stress,
    rectangular_torsion_constant,
    solve_torsion_bar,
    torsion_shear_stress,
)

DTYPE = torch.float64
CATALAN = 0.9159655941772190


# --------------------------------------------------------------------------- #
# the extrapolation operator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("element,natural", [("hex8", 8), ("hex20", 20)])
def test_extrapolation_matrix_partition_of_unity(element, natural):
    matrix = extrapolation_matrix(element)
    assert matrix.shape == (natural, len(gauss_points_1d(element)) ** 3)
    assert torch.allclose(matrix.sum(1), torch.ones(natural, dtype=DTYPE), atol=1e-14)


def test_natural_coordinates_match_the_element_ordering():
    assert natural_coordinates("hex8").shape == (8, 3)
    hex20 = natural_coordinates("hex20")
    assert hex20.shape == (20, 3)
    # corners are the +-1 cube, edge midpoints carry exactly one zero
    assert torch.allclose(hex20[:8].abs(), torch.ones(8, 3, dtype=DTYPE))
    assert torch.allclose((hex20[8:] == 0).sum(1).to(DTYPE), torch.ones(12, dtype=DTYPE))


def _monomial(point, degree):
    return point[0] ** degree * point[1] ** degree * point[2] ** degree


def _gauss_values(element, degree):
    points = gauss_points_1d(element)
    return torch.tensor([_monomial((r, s, t), degree) for r in points for s in points for t in points],
                        dtype=DTYPE)


def _reproduction_error(element, degree, count=32, seed=11):
    generator = torch.Generator().manual_seed(seed)
    targets = (torch.rand(count, 3, generator=generator, dtype=DTYPE) * 2.0 - 1.0).tolist()
    weights = extrapolation_weights_at(element, targets)
    values = _gauss_values(element, degree)
    return max(abs(float(weights[i] @ values) - _monomial(target, degree))
               for i, target in enumerate(targets))


def test_operator_reproduces_its_stated_order():
    assert _reproduction_error("hex8", 1) < 1e-12
    assert _reproduction_error("hex20", 1) < 1e-12
    assert _reproduction_error("hex20", 2) < 1e-12


def test_two_point_gauss_extrapolation_control_cannot_reproduce_a_quadratic():
    # negative control: if this ever passes, the operator's stated order is wrong
    assert _reproduction_error("hex8", 2) > 1e-3


def test_weights_produced_at_arbitrary_points_interpolate_the_gauss_values():
    # at a Gauss abscissa the weight vector must pick that Gauss point only
    q = gauss_points_1d("hex8")[0]
    weights = extrapolation_weights_at("hex8", [[q, q, q]])[0]
    assert weights.shape == (8,)
    assert float(weights.max()) == pytest.approx(1.0, abs=1e-14)
    assert int((weights.abs() > 1e-14).sum()) == 1


# --------------------------------------------------------------------------- #
# the recovery: patch test on distorted meshes
# --------------------------------------------------------------------------- #
def _patch(element, nx, ny, nz, seed=3):
    if element == "hex20":
        nodes, cells = structured_hex20_mesh(1.0, 2.0, 3.0, nx, ny, nz)
    else:
        nodes, cells = structured_hex_mesh(1.0, 2.0, 3.0, nx, ny, nz)
    generator = torch.Generator().manual_seed(seed)
    # a distorted but still positive-Jacobian mesh: the patch test must not need regularity
    nodes = nodes + 0.02 * torch.rand(nodes.shape, generator=generator, dtype=DTYPE)
    young = nodes.new_full((len(cells),), 2.0e11)
    nu = nodes.new_full((len(cells),), 0.3)
    modulus = elasticity_matrix_3d(young, nu)
    strain = torch.tensor([1.0e-3, -4.0e-4, 2.5e-4, 6.0e-4, -3.0e-4, 8.0e-4], dtype=DTYPE)
    u = torch.zeros(3 * len(nodes), dtype=DTYPE)
    u[0::3] = strain[0] * nodes[:, 0] + 0.5 * strain[3] * nodes[:, 1] + 0.5 * strain[5] * nodes[:, 2]
    u[1::3] = 0.5 * strain[3] * nodes[:, 0] + strain[1] * nodes[:, 1] + 0.5 * strain[4] * nodes[:, 2]
    u[2::3] = 0.5 * strain[5] * nodes[:, 0] + 0.5 * strain[4] * nodes[:, 1] + strain[2] * nodes[:, 2]
    recovered = recover_nodal_stress(nodes, cells, u, modulus, element)
    exact = torch.einsum("ij,j->i", modulus[0], strain)
    return recovered, exact


@pytest.mark.parametrize("element", ["hex8", "hex20"])
def test_recovery_patch_test_reproduces_a_uniform_strain_on_a_distorted_mesh(element):
    recovered, exact = _patch(element, 2, 2, 2)
    error = float((recovered.stress - exact).abs().max() / exact.abs().max())
    assert error < 1e-12
    assert bool((recovered.counts >= 1).all())


def test_recovery_counts_report_the_number_of_incident_elements():
    recovered, _ = _patch("hex8", 1, 1, 1)
    # a single element: every node is touched exactly once
    assert int(recovered.counts.min()) == 1
    assert int(recovered.counts.max()) == 1


@pytest.mark.parametrize("element", ["hex20"])
def test_recovery_reproduces_a_linearly_varying_stress_field(element):
    """A quadratic displacement gives a linear stress field, which both operators reproduce.

    The quadratic *element* is the only one that could follow a stress field varying
    faster than linear; that limit is asserted at the operator level in
    ``test_two_point_gauss_extrapolation_control_cannot_reproduce_a_quadratic``,
    because neither brick spans a cubic displacement: the linear element cannot hold the
    quadratic field that generates a linear stress at all (its own strain is piecewise
    constant), so the assertion is made on the element that can.
    """
    nodes, cells = (structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1) if element == "hex20"
                    else structured_hex_mesh(1.0, 1.0, 1.0, 1, 1, 1))
    modulus = elasticity_matrix_3d(nodes.new_full((len(cells),), 1.0e9),
                                   nodes.new_full((len(cells),), 0.3))
    u = torch.zeros(3 * len(nodes), dtype=DTYPE)
    u[0::3] = 0.5 * nodes[:, 0] ** 2          # eps_xx = x  ->  sigma_xx linear in x
    zero = torch.zeros_like(nodes[:, 0])
    strain_nodes = torch.stack((nodes[:, 0], zero, zero, zero, zero, zero), 1)
    exact = torch.einsum("ij,nj->ni", modulus[0], strain_nodes)
    recovered = recover_nodal_stress(nodes, cells, u, modulus, element)
    error = float((recovered.stress[:, 0] - exact[:, 0]).abs().max() / exact[:, 0].abs().max())
    assert error < 1e-10


# --------------------------------------------------------------------------- #
# reading a stress peak off a surface
# --------------------------------------------------------------------------- #
def test_voigt_stress_to_tensor_round_trip():
    voigt = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=DTYPE)
    tensor = voigt_stress_to_tensor(voigt)
    assert torch.allclose(tensor[0], tensor[0].transpose(0, 1))
    assert float(tensor[0, 0, 1]) == 4.0
    assert float(tensor[0, 1, 2]) == 5.0
    assert float(tensor[0, 0, 2]) == 6.0


def test_surface_traction_splits_normal_and_tangential_parts():
    # uniaxial stress along x: the traction on a face normal to x has no tangential part
    voigt = torch.tensor([[7.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    result = surface_traction(voigt, (1.0, 0.0, 0.0))
    assert float(result["normal"][0]) == pytest.approx(7.0)
    assert float(result["tangential_magnitude"][0]) == pytest.approx(0.0, abs=1e-14)
    # pure tau_xz: the tangential part of the traction on the x face is tau_xz itself
    voigt = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.5]], dtype=DTYPE)
    result = surface_traction(voigt, (2.0, 0.0, 0.0))   # unnormalised normal is accepted
    assert float(result["tangential_magnitude"][0]) == pytest.approx(2.5)


def test_transverse_shear_magnitude_reads_the_section_shear_not_the_traction():
    # the state that a free lateral face carries: tau_xz and tau_zy non-zero, its traction zero
    voigt = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0, 4.0]], dtype=DTYPE)
    assert float(transverse_shear_magnitude(voigt)[0]) == pytest.approx(5.0)
    traction = surface_traction(voigt, (0.0, 1.0, 0.0))
    assert float(traction["traction"][0, 2]) == pytest.approx(3.0)   # tau_zy on the y face
    assert float(traction["normal"][0]) == pytest.approx(0.0, abs=1e-14)


# --------------------------------------------------------------------------- #
# the new on-boundary reference
# --------------------------------------------------------------------------- #
def test_boundary_peak_agrees_with_the_interior_limit_of_the_same_series():
    side = 1.0
    peak = boundary_shear_peak(side, side, shear_modulus=1.0, twist_rate=1.0)["tau_max"]
    values = []
    for eps in (1e-3, 1e-4, 1e-5):
        tx, ty = torsion_shear_stress(torch.tensor([0.0]), torch.tensor([side / 2 - eps]),
                                     side, side, twist_rate=1.0, shear_modulus=1.0)
        values.append(float(torch.hypot(tx, ty)))
    assert abs(values[2] - peak) / peak < 1e-4
    assert values[0] < values[1] < values[2] < peak   # monotone approach from inside


def test_boundary_peak_matches_the_timoshenko_coefficient_table():
    coefficient = boundary_shear_peak(1.0, 1.0)["coefficient"]
    assert coefficient == pytest.approx(0.675, abs=1e-3)
    # tau_max = T/(alpha a b^2) with alpha = 0.208 must give the same value to table rounding
    j = rectangular_torsion_constant(1.0, 1.0)
    tau = 1.0 / (0.208 * 1.0 ** 3)
    assert abs(tau - coefficient / j) / tau < 5e-3


def test_grid_scan_is_a_strict_lower_bound_of_the_boundary_peak():
    peak = boundary_shear_peak(1.0, 1.0, shear_modulus=1.0, twist_rate=1.0)["tau_max"]
    previous = 0.0
    for samples in (51, 101, 201):
        value, _ = max_shear_stress(1.0, 1.0, torque=1.0, samples=samples)
        # the scan returns |grad psi| scaled by T/J = G theta, so undo that to compare with G theta = 1
        scaled = value * rectangular_torsion_constant(1.0, 1.0)
        assert scaled > previous
        assert scaled < peak
        previous = scaled


def test_boundary_peak_reaches_the_thin_strip_limit():
    strip = boundary_shear_peak(50.0, 1.0, terms=5000)["coefficient"]
    assert strip == pytest.approx((8.0 / math.pi ** 2) * CATALAN, rel=2e-3)


def test_boundary_peak_rejects_a_swapped_section():
    with pytest.raises(ValueError):
        boundary_shear_peak(1.0, 2.0)


# --------------------------------------------------------------------------- #
# end to end on the smallest useful torsion mesh
# --------------------------------------------------------------------------- #
def test_recovered_surface_peak_beats_the_centroid_readout_on_a_small_torsion_mesh():
    side, length, young, poisson, torque = 1.0, 4.0, 1.0e9, 0.3, 1.0
    shear = young / (2.0 * (1.0 + poisson))
    result = solve_torsion_bar(element="hex20", segments=2, side=side, length=length,
                               young=young, poisson=poisson, torque=torque)
    exact = boundary_shear_peak(side, side, shear_modulus=shear, twist_rate=result.twist_rate)["tau_max"]
    recovered = recover_nodal_stress(result.nodes, result.cells, result.displacement,
                                     result.modulus, "hex20")
    xc = result.nodes[:, 0] - side / 2
    yc = result.nodes[:, 1] - side / 2
    mid = (result.nodes[:, 2] - length / 2).abs() < 1e-12
    boundary = ((xc.abs() > side / 2 - 1e-12) | (yc.abs() > side / 2 - 1e-12)) & mid
    peak = float(transverse_shear_magnitude(recovered.stress)[boundary].max())
    recovered_error = abs(peak - exact) / exact
    centroid_error = abs(result.shear_peak_fe - exact) / exact
    assert recovered_error < 0.20                     # measured 16.8% on this mesh
    assert centroid_error > 0.40                      # measured 62.7%: why it was withheld
    assert recovered_error < centroid_error