import torch

from tensorfem.solid3d import structured_hex_mesh, hex_to_tet_mesh
from tensorfem.solid3d_edgecases import (
    assemble_tet4_consistent_mass, free_body_rigid_modes,
    tet4_axial_frequency_benchmark, tet4_transient_axial_benchmark,
    hex8_near_incompressible_locking_diagnostic,
)

D = torch.float64


def test_tet4_consistent_mass_exact_total_and_symmetry():
    nodes, hexes = structured_hex_mesh(2., 3., 4., 1, 1, 1, dtype=D)
    nodes, tets = hex_to_tet_mesh(nodes, hexes)
    mass = assemble_tet4_consistent_mass(nodes, tets, 5.)
    assert torch.allclose(mass, mass.T)
    for c in range(3):
        assert abs(float(mass[c::3, c::3].sum()) - 120.) < 1e-10


def test_free_solid_has_exactly_six_rigid_modes():
    for element_type in ("tet4", "hex8"):
        ev, ratio = free_body_rigid_modes(element_type)
        assert float(ratio) < 1e-10
        assert float(ev[6]/ev[-1]) > 1e-5


def test_tet4_axial_frequency_under_three_percent():
    _, _, error = tet4_axial_frequency_benchmark()
    assert error < .03


def test_tet4_transient_first_mode_under_three_percent():
    error, *_ = tet4_transient_axial_benchmark()
    assert error < .03


def test_hex8_near_incompressible_locking_is_explicitly_detected():
    _, _, error = hex8_near_incompressible_locking_diagnostic()
    assert error > .03
