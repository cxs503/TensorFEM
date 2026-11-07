"""Mortar segment integration: consistency, patch test and rejection gates."""

import math

import pytest
import torch

from tensorfem.mortar_segment_contact3d import (
    assemble_mortar_segment_contact,
    sample_mortar_segment_contact,
)
from tensorfem.mortar_segment_transfer import mortar_transfer_accuracy
from tensorfem.mortar_two_block import build_two_block, interface_metrics, solve_two_block

torch.set_default_dtype(torch.float64)


def plate(n, z, *, flip=False):
    """Structured QUAD4 plate over [0,1]^2 with outward normal +z (or -z)."""
    x = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
    nodes = torch.tensor(
        [(float(x[i]), float(x[j]), float(z)) for j in range(n + 1) for i in range(n + 1)],
        dtype=torch.float64,
    )
    faces = []
    for j in range(n):
        for i in range(n):
            k = j * (n + 1) + i
            face = (k, k + 1, k + n + 2, k + n + 1)
            faces.append(tuple(reversed(face)) if flip else face)
    return nodes, torch.tensor(faces, dtype=torch.long)


def test_uniform_penetration_transfer_is_exact():
    """Non-matching plates: area and transferred force are exact."""
    penalty, gap = 2.0e5, 1.0e-6
    master, master_faces = plate(3, 0.0)
    slave, slave_faces = plate(2, 0.0)
    displacement = torch.tensor([0.0, 0.0, -gap]).expand_as(slave)
    assembly = assemble_mortar_segment_contact(
        slave,
        slave_faces,
        master,
        master_faces,
        displacement,
        torch.zeros_like(master),
        normal_penalty=penalty,
    )
    assert assembly.segments > 0
    assert float(assembly.integrated_area) == pytest.approx(1.0, abs=1e-12)
    assert float(assembly.penalty_energy) == pytest.approx(0.5 * penalty * gap**2, rel=1e-12)
    assert float(assembly.slave_forces.sum()) == pytest.approx(penalty * gap, rel=1e-12)
    assert float(assembly.master_forces.sum()) == pytest.approx(-penalty * gap, rel=1e-12)
    assert float(assembly.maximum_penetration) == pytest.approx(gap, rel=1e-12)
    # interior slave node carries a quarter of the patch, corners a sixteenth
    nz = assembly.slave_forces[:, 2].reshape(3, 3)
    assert float(nz[1, 1]) == pytest.approx(penalty * gap * 0.25, rel=1e-10)
    assert float(nz[0, 0]) == pytest.approx(penalty * gap * 0.0625, rel=1e-10)
    # no tangential force on a frictionless aligned interface
    assert float(assembly.slave_forces[:, :2].abs().max()) < 1e-12


def test_open_interface_carries_no_force():
    penalty = 1.0e5
    master, master_faces = plate(2, 0.0)
    slave, slave_faces = plate(3, 0.0)
    displacement = torch.tensor([0.0, 0.0, 1e-5]).expand_as(slave)
    assembly = assemble_mortar_segment_contact(
        slave,
        slave_faces,
        master,
        master_faces,
        displacement,
        torch.zeros_like(master),
        normal_penalty=penalty,
    )
    assert float(assembly.penalty_energy) == 0.0
    assert float(assembly.slave_forces.abs().max()) == 0.0
    assert float(assembly.maximum_penetration) == 0.0


def test_role_interchange_preserves_traction():
    """Swapping slave and master (with the master normal flipped) is symmetric."""
    penalty, gap = 1.0e5, 2.0e-6
    upper, upper_faces = plate(2, 0.0)
    _, upper_faces_down = plate(2, 0.0, flip=True)
    lower, lower_faces = plate(3, 0.0)
    forward = assemble_mortar_segment_contact(
        upper,
        upper_faces,
        lower,
        lower_faces,
        torch.tensor([0.0, 0.0, -gap]).expand_as(upper),
        torch.zeros_like(lower),
        normal_penalty=penalty,
    )
    # reversed roles: the former slave becomes the master with a downward normal
    backward = assemble_mortar_segment_contact(
        lower,
        lower_faces,
        upper,
        upper_faces_down,
        torch.zeros_like(lower),
        torch.tensor([0.0, 0.0, -gap]).expand_as(upper),
        normal_penalty=penalty,
        check_initial_penetration=False,
    )
    # the same physical body must receive the same force whichever side it plays
    assert float(backward.master_forces.sum()) == pytest.approx(
        float(forward.slave_forces.sum()), rel=1e-10
    )
    assert float(backward.slave_forces.sum()) == pytest.approx(
        float(forward.master_forces.sum()), rel=1e-10
    )
    assert float(forward.integrated_area) == pytest.approx(
        float(backward.integrated_area), rel=1e-12
    )


def test_frozen_geometry_sample_matches_assembly():
    penalty, gap = 5.0e4, 1.0e-6
    master, master_faces = plate(2, 0.0)
    slave, slave_faces = plate(3, 0.0)
    displacement = torch.tensor([0.0, 0.0, -gap]).expand_as(slave)
    assembly = assemble_mortar_segment_contact(
        slave,
        slave_faces,
        master,
        master_faces,
        displacement,
        torch.zeros_like(master),
        normal_penalty=penalty,
    )
    samples = sample_mortar_segment_contact(
        slave + displacement,
        slave_faces,
        master,
        master_faces,
        normal_penalty=penalty,
        slave_reference=slave,
        master_reference=master,
    )
    assert float(samples.total_normal_force) == pytest.approx(
        float(assembly.slave_forces.sum()), rel=1e-12
    )
    assert float(samples.integrated_area) == pytest.approx(
        float(assembly.integrated_area), rel=1e-12
    )


def test_negative_penalty_and_bad_search_rejected():
    master, master_faces = plate(2, 0.0)
    slave, slave_faces = plate(2, 0.0)
    zero = torch.zeros_like(slave)
    with pytest.raises(ValueError, match="normal_penalty"):
        assemble_mortar_segment_contact(
            slave,
            slave_faces,
            master,
            master_faces,
            zero,
            torch.zeros_like(master),
            normal_penalty=0.0,
        )
    with pytest.raises(ValueError, match="normal_penalty"):
        sample_mortar_segment_contact(slave, slave_faces, master, master_faces, normal_penalty=-1.0)
    with pytest.raises(ValueError, match="search_distance"):
        assemble_mortar_segment_contact(
            slave,
            slave_faces,
            master,
            master_faces,
            zero,
            torch.zeros_like(master),
            normal_penalty=1e5,
            search_distance=-1.0,
        )


def test_initial_interpenetration_rejected():
    """A reference configuration that already overlaps is a modelling error."""
    penalty = 1.0e5
    master, master_faces = plate(2, 0.0)
    slave, slave_faces = plate(3, 0.0)
    overlap = slave.clone()
    overlap[:, 2] = -1e-3
    with pytest.raises(ValueError, match="initial interpenetration"):
        assemble_mortar_segment_contact(
            overlap,
            slave_faces,
            master,
            master_faces,
            torch.zeros_like(slave),
            torch.zeros_like(master),
            normal_penalty=penalty,
            search_distance=1e-2,
        )
    allowed = assemble_mortar_segment_contact(
        overlap,
        slave_faces,
        master,
        master_faces,
        torch.zeros_like(slave),
        torch.zeros_like(master),
        normal_penalty=penalty,
        search_distance=1e-2,
        check_initial_penetration=False,
    )
    assert allowed.segments > 0


def test_invalid_connectivity_rejected():
    master, master_faces = plate(2, 0.0)
    slave, slave_faces = plate(2, 0.0)
    zero = torch.zeros_like(slave)
    with pytest.raises(ValueError, match="connectivity"):
        assemble_mortar_segment_contact(
            slave,
            slave_faces.to(torch.int32),
            master,
            master_faces,
            zero,
            torch.zeros_like(master),
            normal_penalty=1e5,
        )
    bad = slave_faces.clone()
    bad[0, 0] = 10**4
    with pytest.raises(ValueError, match="undefined nodes"):
        assemble_mortar_segment_contact(
            slave, bad, master, master_faces, zero, torch.zeros_like(master), normal_penalty=1e5
        )
    with pytest.raises(ValueError, match="must match"):
        assemble_mortar_segment_contact(
            slave, slave_faces, master, master_faces, zero, torch.zeros(1, 3), normal_penalty=1e5
        )


def test_two_block_uniform_pressure_patch_test():
    """Identical blocks: interface traction equals the analytic uniform pressure."""
    pressure = 10.0
    model = build_two_block(n_upper=2, n_base=3, nz=2, pressure=pressure)
    solution = solve_two_block(model, penalty=1e9)
    metrics = interface_metrics(model, solution)
    assert metrics["band_l2_error"] < 1e-8
    assert metrics["full_l2_error"] < 1e-8
    assert metrics["band_mean_traction"] == pytest.approx(pressure, rel=1e-8)
    assert metrics["area_error"] < 1e-12
    assert metrics["equilibrium_error"] < 1e-8
    assert metrics["maximum_penetration"] <= pressure / 1e9 * 1.001
    assert metrics["zero_tension"]


def test_transfer_accuracy_second_order_and_rejects_bad_input():
    coarse = mortar_transfer_accuracy(2, 3)
    fine = mortar_transfer_accuracy(6, 7)
    assert fine["l2_error"] < coarse["l2_error"]
    rate = math.log(coarse["l2_error"] / fine["l2_error"]) / math.log(3.0)
    assert rate > 1.4
    assert fine["l2_error"] < 0.03
    assert fine["total_force_error"] < 0.03
    with pytest.raises(ValueError, match="positive"):
        mortar_transfer_accuracy(2, 3, amplitude=0.0)
    with pytest.raises(ValueError, match="positive"):
        mortar_transfer_accuracy(0, 3)
