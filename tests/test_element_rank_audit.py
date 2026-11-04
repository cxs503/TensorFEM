"""Tests for the zero-energy (rank-deficiency) audit of the element library."""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.element_rank_audit import (
    audit_element,
    audit_library,
    audit_plate_mesh_scan,
    audit_shell_mesh_scan,
    classify_null_space,
    rigid_body_basis_plate,
    rigid_body_basis_shell,
    rigid_body_basis_solid3d,
    summary_table,
)
from tensorfem.plate import q4_mindlin_stiffness
from tensorfem.plate_vibration import (
    kirchhoff_angular_frequency,
    q4_plate_mass_matrix,
    structured_rectangular_mesh,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)  # this host is shared; thread oversubscription dominates runtime

YOUNG = 200.0e9
POISSON = 0.3
THICKNESS = 0.01
DENSITY = 7800.0

SQUARE = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
CUBE = torch.tensor(
    [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ]
)


def _plate_mesh(n: int, treatment: str):
    nodes, cells = structured_rectangular_mesh(n, n, 1.0, 1.0)
    ndof = 3 * len(nodes)
    k = torch.zeros((ndof, ndof))
    m = torch.zeros((ndof, ndof))
    for cell in cells.tolist():
        ke = q4_mindlin_stiffness(
            nodes[cell], YOUNG, POISSON, THICKNESS, shear_treatment=treatment
        )
        me = q4_plate_mass_matrix(nodes[cell], DENSITY, THICKNESS)
        idx = torch.tensor([3 * i + a for i in cell for a in range(3)])
        k[idx[:, None], idx] = k[idx[:, None], idx] + ke
        m[idx[:, None], idx] = m[idx[:, None], idx] + me
    return nodes, k, m


def _frequencies(k, m, free):
    kr, mr = k[free][:, free], m[free][:, free]
    lower = torch.linalg.cholesky(mr)
    inverse = torch.linalg.inv(lower)
    a = inverse @ kr @ inverse.transpose(0, 1)
    a = 0.5 * (a + a.transpose(0, 1))
    return torch.linalg.eigvalsh(a).clamp(min=0.0).sqrt()


def test_solids_and_membranes_are_rank_sufficient():
    """Every solid and plane element must have exactly the rigid-body null space."""
    records = {r.element: r for r in audit_library(young=YOUNG, poisson=POISSON, thickness=THICKNESS)}
    clean = [
        "hex8 (2x2x2 integration)",
        "hex8 (distorted)",
        "tet4",
        "hex8 B-bar (unit cube)",
        "hex8 B-bar (distorted)",
        "hex20 (3x3x3 integration)",
        "q4 membrane (2x2 integration)",
        "cst membrane",
    ]
    for name in clean:
        record = records[name]
        assert record.spurious == 0, f"{name} has {record.spurious} spurious zero-energy modes"
        assert record.null_dim == record.expected_rigid_dim
    assert records["hex8 (2x2x2 integration)"].null_dim == 6
    assert records["q4 membrane (2x2 integration)"].null_dim == 3


def test_rigid_body_basis_is_annihilated_by_rigid_invariance():
    """The analytic rigid-body span must be reproduced exactly, no tolerance fudge."""
    from tensorfem.continuum import elasticity_matrix, q4_stiffness
    from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness

    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    k3, _ = hex8_stiffness(CUBE.unsqueeze(0), d3)
    report = classify_null_space(k3[0], rigid_body_basis_solid3d(CUBE))
    assert report["null_dim"] == 6
    assert report["explained"] == 6

    d2 = elasticity_matrix(torch.tensor([YOUNG]), torch.tensor([POISSON]), "stress")
    k2, _ = q4_stiffness(SQUARE.unsqueeze(0), d2, torch.tensor([THICKNESS]))
    report = classify_null_space(k2[0], rigid_body_basis_plane2d_local(SQUARE))
    assert report["null_dim"] == 3
    assert report["explained"] == 3


def rigid_body_basis_plane2d_local(nodes: torch.Tensor) -> torch.Tensor:
    from tensorfem.element_rank_audit import rigid_body_basis_plane2d

    return rigid_body_basis_plane2d(nodes)


def test_plate_sri_has_two_spurious_mechanisms():
    """Fail-closed record of the documented defect of the one-point shear rule."""
    k = q4_mindlin_stiffness(SQUARE, YOUNG, POISSON, THICKNESS, shear_treatment="sri")
    report = classify_null_space(k, rigid_body_basis_plate(SQUARE))
    assert report["expected_rigid_dim"] == 3
    assert report["null_dim"] == 5
    assert report["spurious"] == 2


def test_plate_assumed_shear_strain_removes_the_mechanisms():
    for points in (SQUARE, torch.tensor([[0.0, 0.0], [2.0, 0.3], [1.5, 1.2], [0.2, 1.0]])):
        k = q4_mindlin_stiffness(points, YOUNG, POISSON, THICKNESS, shear_treatment="mitc")
        report = classify_null_space(k, rigid_body_basis_plate(points))
        assert report["null_dim"] == 3
        assert report["spurious"] == 0


def test_shell_facets_inherit_and_lose_the_plate_mechanisms():
    xyz = torch.cat((SQUARE, torch.zeros(4, 1)), 1)
    basis = rigid_body_basis_shell(xyz)
    records = {r.element: r for r in audit_library(young=YOUNG, poisson=POISSON, thickness=THICKNESS)}
    assert records["shell4 facet, plate shear SRI (square)"].spurious == 2
    assert records["shell4m facet, plate shear SRI (square)"].spurious == 2
    assert records["shell4 facet, plate shear MITC (square)"].spurious == 0
    assert records["shell4m facet, plate shear MITC (square)"].spurious == 0
    for name in ("shell4 facet, plate shear MITC (square)", "shell4m facet, plate shear MITC (square)"):
        assert records[name].null_dim == records[name].expected_rigid_dim == 7
    assert basis.shape[1] == 7


def test_rejects_unknown_shear_treatment():
    with pytest.raises(ValueError):
        q4_mindlin_stiffness(SQUARE, YOUNG, POISSON, THICKNESS, shear_treatment="reduced")


def test_free_free_modal_analysis_mode_count():
    """The SRI rule produces a fourth zero-frequency mode; the MITC field does not."""
    nodes, k, m = _plate_mesh(8, "sri")
    omega = _frequencies(k, m, torch.arange(k.shape[0]))
    assert int((omega[:8] < 1.0).sum()) == 4

    nodes, k, m = _plate_mesh(8, "mitc")
    omega = _frequencies(k, m, torch.arange(k.shape[0]))
    assert int((omega[:8] < 1.0).sum()) == 3

    reference = kirchhoff_angular_frequency(
        length=1.0, width=1.0, density=DENSITY, young=YOUNG, poisson=POISSON, thickness=THICKNESS
    )
    free_free = float(omega[3])
    rigidity = YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2))
    leissa = 13.49 * math.sqrt(rigidity / (DENSITY * THICKNESS))
    assert abs(free_free - leissa) / leissa < 0.01
    assert free_free != pytest.approx(reference)


def test_simply_supported_frequency_is_unaffected_by_the_shear_field():
    """Accuracy guard: the cure must not change the verified frequency response."""
    reference = kirchhoff_angular_frequency(
        length=1.0, width=1.0, density=DENSITY, young=YOUNG, poisson=POISSON, thickness=THICKNESS
    )
    values = {}
    for treatment in ("sri", "mitc"):
        nodes, k, m = _plate_mesh(8, treatment)
        fixed = []
        for i, (x, y) in enumerate(zip(nodes[:, 0].tolist(), nodes[:, 1].tolist())):
            on_x = abs(x) < 1e-10 or abs(x - 1.0) < 1e-10
            on_y = abs(y) < 1e-10 or abs(y - 1.0) < 1e-10
            if on_x or on_y:
                fixed.append(3 * i)
            if on_x:
                fixed.append(3 * i + 2)
            if on_y:
                fixed.append(3 * i + 1)
        free = torch.tensor([i for i in range(k.shape[0]) if i not in set(fixed)])
        omega = _frequencies(k, m, free)
        values[treatment] = float(omega[0])
        # no spurious mode below the exact fundamental
        assert int((omega[:10] < 0.5 * reference).sum()) == 0
        assert abs(values[treatment] - reference) / reference < 0.02
    assert abs(values["sri"] - values["mitc"]) / values["sri"] < 5e-4


def test_free_mesh_scan_keeps_one_spurious_mode_with_sri_only():
    sri = audit_plate_mesh_scan(sizes=(2, 4), young=YOUNG, poisson=POISSON, thickness=THICKNESS)
    mitc = audit_plate_mesh_scan(
        sizes=(2, 4), young=YOUNG, poisson=POISSON, thickness=THICKNESS, treatment="mitc"
    )
    assert [r.spurious for r in sri] == [1, 1]
    assert [r.spurious for r in mitc] == [0, 0]
    # the surviving mode is soft and gets softer with refinement
    assert sri[1].first_positive_ratio < sri[0].first_positive_ratio


def test_shell_mesh_scan_reports_no_spurious_mode_with_assumed_shear():
    sri = audit_shell_mesh_scan(sizes=(1, 2), formulation="sri", young=YOUNG, poisson=POISSON, thickness=THICKNESS)
    mitc = audit_shell_mesh_scan(
        sizes=(1, 2), formulation="sri", young=YOUNG, poisson=POISSON, thickness=THICKNESS, plate_shear="mitc"
    )
    assert [r.spurious for r in sri] == [2, 1]
    assert [r.spurious for r in mitc] == [0, 0]


def test_summary_table_renders():
    records = audit_library(young=YOUNG, poisson=POISSON, thickness=THICKNESS)
    table = summary_table(records)
    assert table.startswith("| element |")
    assert table.count("\n") == len(records) + 2