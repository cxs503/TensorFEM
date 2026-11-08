"""Regression tests for the HEX20 inertia operator (``tensorfem.hex20``).

The mass matrix is the one kernel where "symmetric, right total mass" proves
nothing: a shared Jacobian/node-ordering error cancels in the sum.  Every test
below therefore compares against a *second* code path or a closed form, and the
numbers asserted are the ones the shipped evidence bundle
(``benchmarks/hex20_mass_audit.json``, ``docs/hex20-consistent-mass.md``) quotes:

* straight brick: mass matrix vs an independent NumPy serendipity oracle
  (own node table, own Jacobian) to <=1e-8 Frobenius,
* kinetic energy of translation *and* rigid rotation vs the analytic cuboid
  inertia -> pins the spatial distribution, not just the sum,
* four Gauss orders: order 3 is exact, order 2 is 13.17 % off in the *matrix*
  while the *total mass* stays exact (the trap this operator replaces),
* geometry scope: affine shear stays exact, curved/non-planar faces do not,
* row-sum lumping is indefinite at the corners (-rho V / 8): a documented trap,
  HRZ is the default,
* eigenfrequency evidence: 1-D constrained rod converges at the optimal O(h^4),
  full 3-D rod reproduces sqrt(E/rho) axial modes (lateral contraction resolved),
  and the assembled (K, M) pair has six rigid-body modes at 1e-9 relative.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.explicit_dynamics import critical_time_step
from tensorfem.hex20 import (
    assemble_hex20_mass,
    hex20_lumped_mass,
    hex20_mass,
    hex20_mass_totals,
    hex20_stiffness,
    hex_gauss_rule,
    structured_hex20_mesh,
)
from tensorfem.hex20_mass_audit import (
    axial_modes,
    box_inertia,
    build_evidence,
    constrained_rod_modes,
    curved_element_residual,
    distortion_scope,
    free_free_modes,
    lumping_report,
    reference_mass,
    reference_volume,
    rigid_rotation_energy,
    rigid_velocity_energy,
)
from tensorfem.solid3d import elasticity_matrix_3d

torch.set_num_threads(4)
DENSITY = 7850.0
BOX = (1.0, 2.0, 3.0)


def _box_element(lengths=BOX, density=DENSITY):
    nodes, cells = structured_hex20_mesh(*lengths, 1, 1, 1)
    return nodes, cells, nodes[cells]


def test_consistent_mass_shape_symmetry_and_per_element_density():
    _, _, x = _box_element()
    m = hex20_mass(x, DENSITY)
    assert m.shape == (1, 60, 60)
    assert torch.allclose(m, m.transpose(-1, -2), atol=1e-9)
    # per-element density: doubling rho doubles the operator, linearly
    m2 = hex20_mass(torch.cat((x, x)), torch.tensor([DENSITY, 2 * DENSITY], dtype=x.dtype))
    assert torch.allclose(m2[0], m[0], rtol=1e-12)
    assert torch.allclose(m2[1], 2.0 * m[0], rtol=1e-12)
    # positive definite (three identical scalar blocks)
    lam = torch.linalg.eigvalsh(m[0])
    assert float(lam.min()) > 0.0
    block = m.reshape(20, 3, 20, 3)[:, 0, :, 0]
    assert torch.allclose(block, m.reshape(20, 3, 20, 3)[:, 2, :, 2], atol=0.0)


def test_total_mass_matches_exact_volume_and_reference_oracle():
    _, _, x = _box_element()
    volume = float(reference_volume(x)[0])
    mass = float(hex20_mass_totals(x, DENSITY).sum())
    assert mass == pytest.approx(DENSITY * volume, rel=1e-10)
    assert mass == pytest.approx(DENSITY * 6.0, rel=1e-10)  # analytic box volume
    block = hex20_mass(x, DENSITY).reshape(20, 3, 20, 3)[:, 0, :, 0]
    ref = torch.as_tensor(reference_mass(x, DENSITY)[0], dtype=x.dtype)
    assert float((block - ref).norm() / ref.norm()) < 1e-8


def test_kinetic_energy_translation_and_rigid_rotation_match_closed_form():
    _, _, x = _box_element()
    full = hex20_mass(x, DENSITY)
    volume = float(reference_volume(x)[0])
    v = (0.7, -1.1, 2.3)
    expected = 0.5 * DENSITY * volume * sum(c * c for c in v)
    assert rigid_velocity_energy(full, v) == pytest.approx(expected, rel=1e-9)
    inertia = box_inertia(DENSITY * volume, BOX)
    centre = torch.tensor([0.5, 1.0, 1.5], dtype=x.dtype)
    for axis, omega in enumerate(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.5))):
        got = rigid_rotation_energy(full, x, omega, about=centre)
        want = 0.5 * float(inertia[axis, axis]) * sum(c * c for c in omega)
        assert got == pytest.approx(want, rel=1e-8), f"axis {axis}"


def test_rule_order_two_is_wrong_in_the_matrix_but_right_in_total_mass():
    """The trap: total mass is exact for every order, the matrix is not."""
    _, _, x = _box_element()
    ref = torch.as_tensor(reference_mass(x, DENSITY)[0], dtype=x.dtype)

    def block(order):
        m = hex20_mass(x, DENSITY, rule=hex_gauss_rule(order, dtype=x.dtype))
        return m.reshape(20, 3, 20, 3)[:, 0, :, 0]

    assert float((block(3) - ref).norm() / ref.norm()) < 1e-8
    assert float((block(4) - ref).norm() / ref.norm()) < 1e-8
    err2 = float((block(2) - ref).norm() / ref.norm())
    assert 0.10 < err2 < 0.16, err2  # 13.17 % measured
    for order in (1, 2, 3):
        total = float(hex20_mass_totals(x, DENSITY, rule=hex_gauss_rule(order, dtype=x.dtype)).sum())
        assert total == pytest.approx(DENSITY * 6.0, rel=1e-10)


def test_geometry_scope_affine_exact_nonplanar_and_curved_quantified():
    scope = distortion_scope()
    assert scope["affine_shear_rule3_reference_rel"] < 1e-8  # det J constant
    assert 1e-5 < scope["nonplanar_rule3_reference_rel"] < 1e-2  # ~7.3e-4
    assert scope["nonplanar_rule4_reference_rel"] < 1e-8
    curved = curved_element_residual()
    assert 0.001 < curved["order3"]["rel_frobenius"] < 0.005  # 0.182 %
    assert curved["order4"]["rel_frobenius"] < 1e-7


def test_hrz_lumping_preserves_mass_and_stays_positive():
    report = lumping_report()
    assert report["hrz_min"] > 0.0
    assert report["hrz_component_total_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert report["row_negative_component_count"] == 24  # 8 corners x 3 components


def test_row_sum_lumping_is_indefinite_counterexample():
    """Negative control: the naive lumping must fail, and it must fail loudly."""
    _, _, x = _box_element()
    report = lumping_report()
    assert report["row_corner_over_element_mass"] == pytest.approx(-1.0 / 8.0, rel=1e-12)
    assert report["row_edge_over_element_mass"] == pytest.approx(1.0 / 6.0, rel=1e-12)
    row = hex20_lumped_mass(x, DENSITY, scheme="row")[0]
    diagonal = torch.diag(row)
    assert float(torch.linalg.eigvalsh(diagonal).min()) < 0.0
    hrz = hex20_lumped_mass(x, DENSITY, scheme="hrz")[0]
    assert float(hrz.min()) > 0.0


def test_assembled_mass_total_and_explicit_dynamics_time_step():
    nodes, cells = structured_hex20_mesh(1.0, 1.0, 1.0, 2, 1, 1)
    consistent = assemble_hex20_mass(nodes, cells, DENSITY)
    lumped = assemble_hex20_mass(nodes, cells, DENSITY, lumping="hrz")
    n_dofs = 3 * len(nodes)
    assert consistent.shape == (n_dofs, n_dofs)
    # one displacement component carries exactly the total mass
    assert float(lumped[0::3, 0::3].sum()) == pytest.approx(DENSITY, rel=1e-12)
    assert float(consistent[0::3, 0::3].sum()) == pytest.approx(DENSITY, rel=1e-11)
    d = elasticity_matrix_3d(nodes.new_full((len(cells),), 200e9),
                             nodes.new_full((len(cells),), 0.3))
    ke, _ = hex20_stiffness(nodes[cells], d)
    k_element = ke[0]
    dt = critical_time_step(torch.diag(hex20_lumped_mass(nodes[cells], DENSITY)[0]), k_element)
    assert math.isfinite(dt) and dt > 0.0


def test_constrained_rod_converges_at_optimal_quartic_rate():
    coarse = constrained_rod_modes(4)
    fine = constrained_rod_modes(8)
    finer = constrained_rod_modes(16)
    e4 = abs(coarse["rows"][0]["rel_error"])
    e8 = abs(fine["rows"][0]["rel_error"])
    e16 = abs(finer["rows"][0]["rel_error"])
    assert e16 < e8 < e4
    assert e16 < 1e-4  # 6.4e-6 measured
    assert math.log2(e8 / e16) > 3.5  # observed order 4.00


def test_full_3d_rod_axial_mode_matches_sqrt_E_over_rho():
    result = axial_modes(8)
    row = result["rows"][0]
    assert row["axial_fraction"] > 0.99
    assert abs(row["rel_error"]) < 5e-3  # 4.37e-3 measured
    # using the constrained modulus instead of E would be a +16 % bias
    from tensorfem.hex20_mass_audit import constrained_rod_frequencies

    ratio = constrained_rod_frequencies(1, young=200e9, poisson=0.3,
                                       density=DENSITY, length=10.0) / row["omega_exact"]
    assert ratio == pytest.approx(math.sqrt((1 - 0.3) / (1.3 * 0.4)), rel=1e-12)  # 1.1602
    assert ratio - 1.0 > 0.15  # the +16 % modulus trap


def test_free_free_pair_has_six_zero_rigid_body_modes():
    report = free_free_modes(2)
    assert report["rigid_max_over_max"] < 1e-8  # 6.45e-10 measured
    assert report["first_elastic_over_max"] > 1e-3


def test_guards_reject_bad_rule_bad_scheme_and_degenerate_element():
    _, _, x = _box_element()
    with pytest.raises(ValueError):
        hex20_mass(x, DENSITY, rule=(torch.zeros(4, 2), torch.ones(4)))
    with pytest.raises(ValueError):
        hex20_mass(x, DENSITY, rule=(torch.zeros(4, 3), torch.ones(3)))
    with pytest.raises(ValueError):
        hex20_lumped_mass(x, DENSITY, scheme="lumped")
    with pytest.raises(ValueError):
        hex_gauss_rule(7)
    degenerate = x.clone()
    degenerate[0, 6] = degenerate[0, 2]  # collapse a corner -> det J <= 0
    with pytest.raises(ValueError):
        hex20_mass(degenerate, DENSITY)


def test_evidence_bundle_is_reproducible_and_self_consistent():
    ev = build_evidence()
    assert ev["schema"] == "tensorfem-hex20-mass-evidence/1"
    assert ev["straight_element"]["volume_rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert all(abs(v) < 1e-9 for v in ev["kinetic_energy"]["rotation_rel_error"])
    assert ev["rules"][2]["rel_frobenius"] < 1e-8  # order 3 exact
    assert ev["lumping"]["hrz_component_total_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert ev["free_free"]["rigid_max_over_max"] < 1e-8