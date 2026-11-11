"""Verification of the 3-D zero-thickness cohesive interface element.

The suite is deliberately organised as *oracles first*: every assertion is a
closed-form statement about the cohesive law or about the exact quadrature of
the element, never a snapshot of a previously computed number.

Reference chain
---------------
``tensorfem.mixed_mode_cohesive.BenzeggaghKenaneLaw`` (2-D, already merged and
verified) is the parent of :class:`tensorfem.cohesive3d.BKCohesive3DLaw`.  The
first test pins the 3-D law to that parent on the ``delta_s2 = 0`` plane, so any
regression in the mixed-mode algebra is caught against an independent
implementation rather than against itself.

Geometry oracles
----------------
A flat 1x1 m interface has unit area; a zero-thickness element must therefore
reproduce ``resultant = t * A`` exactly under a uniform opening, be invariant
under rigid translation and rigid rotation, and be mesh-size independent (a
1x1 patch, a 2x2 patch and a 4x4 patch of the same interface must give the same
resultant).  All of those are checked to ~1e-12 relative.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.cohesive3d import (
    BKCohesive3DLaw,
    assemble_cohesive_interface,
    cohesive_hex8_response,
    cohesive_shape_functions,
    interface_area,
    mid_surface_frame,
)
from tensorfem.mixed_mode_cohesive import BenzeggaghKenaneLaw

D = torch.float64
torch.set_default_dtype(D)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def law(reference: bool = False):
    """Typical structural-adhesive parameters (glass/epoxy-like, SI)."""
    if reference:
        return BenzeggaghKenaneLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)
    return BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)


def flat_element(size: float = 1.0, gap: float = 0.1, shift=(0.0, 0.0, 0.0)):
    """Unit-square interface element in the z=0 plane with a +z gap."""
    s, g = size, gap
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [s, s, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, g],
            [s, 0.0, g],
            [s, s, g],
            [0.0, s, g],
        ],
        dtype=D,
    )
    return coords + torch.tensor(shift, dtype=D)


def patch_mesh(n: int, law_obj: BKCohesive3DLaw | None = None):
    """Pre-meshed n x n interface patch; returns coords, connectivity, area."""
    xs = torch.linspace(0.0, 1.0, n + 1, dtype=D)
    bottom = torch.stack(
        [torch.stack([x, y, torch.zeros((), dtype=D)]) for y in xs for x in xs]
    )
    top = bottom.clone()
    top[:, 2] = 0.1
    coords = torch.cat([bottom, top])
    nn = (n + 1) ** 2

    def idx(i, j):
        return j * (n + 1) + i

    conn = []
    for j in range(n):
        for i in range(n):
            conn.append(
                [
                    idx(i, j),
                    idx(i + 1, j),
                    idx(i + 1, j + 1),
                    idx(i, j + 1),
                    idx(i, j) + nn,
                    idx(i + 1, j) + nn,
                    idx(i + 1, j + 1) + nn,
                    idx(i, j + 1) + nn,
                ]
            )
    return coords, torch.tensor(conn, dtype=torch.long), 1.0


# --------------------------------------------------------------------------- #
# Constitutive law
# --------------------------------------------------------------------------- #


def test_3d_law_reduces_to_the_verified_2d_bk_law():
    """On delta_s2 = 0 the 3-D law must equal the merged 2-D BK reference."""
    three = law()
    two = law(reference=True)
    worst = 0.0
    for amplitude in (1e-7, 1e-6, 4e-6, 6e-6, 2e-5, 1e-4, 1.9e-4, 2.5e-4):
        for ratio in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (3.0, 4.0), (1.0, 3.0)):
            direction = torch.tensor([ratio[0], ratio[1], 0.0], dtype=D)
            direction = direction / torch.linalg.vector_norm(direction)
            t3 = three.envelope_traction(amplitude * direction)
            t2 = two.proportional_response(
                torch.tensor([amplitude], dtype=D), torch.tensor(ratio, dtype=D)
            ).traction[0]
            for component in range(2):
                a3, a2 = float(t3[component]), float(t2[component])
                scale = max(abs(a2), 1e-3 * float(three.tn0))
                worst = max(worst, abs(a3 - a2) / scale)
    assert worst < 1e-12, f"3-D cohesive law deviates from the 2-D BK reference: {worst:.3e}"


@pytest.mark.parametrize(
    "direction,expected_energy",
    [
        ((1.0, 0.0, 0.0), 500.0),  # mode I  -> G_Ic
        ((0.0, 1.0, 0.0), 1200.0),  # mode II -> G_IIc
        ((0.0, 0.0, 1.0), 1200.0),  # mode III behaves as mode II
    ],
)
def test_pure_mode_dissipated_work_equals_fracture_energy(direction, expected_energy):
    law_obj = law()
    d = torch.tensor(direction, dtype=D)
    failure = float(law_obj.failure_amplitude(d))
    amplitudes = torch.linspace(0.0, failure, 40001, dtype=D)
    traction = law_obj.envelope_traction(amplitudes[:, None] * d[None, :])
    work = float(torch.trapezoid(traction @ d, amplitudes))
    assert abs(work - expected_energy) / expected_energy < 1e-6
    assert abs(float(law_obj.critical_energy(d)) - expected_energy) < 1e-9


def test_mixed_mode_dissipated_work_equals_bk_energy():
    law_obj = law()
    for direction in ((1.0, 1.0, 0.0), (3.0, 4.0, 0.0), (1.0, 1.0, 1.0), (2.0, 0.0, 1.0)):
        d = torch.tensor(direction, dtype=D)
        gc = float(law_obj.critical_energy(d))
        failure = float(law_obj.failure_amplitude(d))
        amplitudes = torch.linspace(0.0, failure, 40001, dtype=D)
        traction = law_obj.envelope_traction(amplitudes[:, None] * d[None, :])
        work = float(torch.trapezoid(traction @ d, amplitudes))
        assert abs(work - gc) / gc < 1e-6, (direction, work, gc)


def test_quadratic_initiation_criterion_is_exact():
    """At the onset amplitude the nominal-stress quadratic criterion is 1."""
    law_obj = law()
    for direction in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)):
        d = torch.tensor(direction, dtype=D) / torch.linalg.vector_norm(
            torch.tensor(direction, dtype=D)
        )
        onset = float(law_obj.onset_amplitude(d))
        peak = law_obj.envelope_traction(onset * d[None, :])[0]
        criterion = (peak[0] / law_obj.tn0) ** 2 + (
            (peak[1] ** 2 + peak[2] ** 2) ** 0.5 / law_obj.ts0
        ) ** 2
        assert abs(float(criterion) - 1.0) < 1e-12


def test_envelope_is_zero_and_failure_amplitude_ends_the_softening_branch():
    law_obj = law()
    d = torch.tensor([1.0, 0.0, 0.0], dtype=D)
    onset = float(law_obj.onset_amplitude(d))
    failure = float(law_obj.failure_amplitude(d))
    assert failure > onset > 0.0
    # Peak traction is reached exactly at the onset amplitude.
    t_onset = law_obj.envelope_traction(torch.tensor([[onset, 0.0, 0.0]], dtype=D))[0, 0]
    assert abs(float(t_onset) - law_obj.tn0) / law_obj.tn0 < 1e-12
    # Zero traction at and beyond the failure amplitude.
    for amplitude in (failure, failure * 1.01, failure * 3.0):
        t = law_obj.envelope_traction(torch.tensor([[amplitude, 0.0, 0.0]], dtype=D))
        assert float(t.abs().max()) < 1e-9


def test_consistent_tangent_matches_automatic_differentiation():
    """The analytic tangent must be the derivative of the discrete law."""
    law_obj = law()
    torch.manual_seed(20261111)
    worst = 0.0
    for _ in range(64):
        separation = torch.randn(3, dtype=D) * torch.tensor([3e-5, 2e-5, 2e-5], dtype=D)
        separation[0] = abs(float(separation[0])) * 0.7 - 5e-6  # include compression
        response = law_obj.update(separation)
        jacobian = torch.autograd.functional.jacobian(
            lambda x: law_obj.update(x).traction, separation
        )
        scale = max(float(jacobian.abs().max()), 1.0)
        worst = max(worst, float((response.tangent - jacobian).abs().max()) / scale)
    assert worst < 1e-10, f"consistent tangent deviates from autograd: {worst:.3e}"


def test_unloading_is_secant_and_history_is_irreversible():
    law_obj = law()
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=D)
    onset = float(law_obj.onset_amplitude(direction))
    peak_open = 4.0 * onset
    first = law_obj.update(torch.tensor([peak_open, 0.0, 0.0], dtype=D))
    # Unload to half the peak opening: secant stiffness to the origin.
    unloaded = law_obj.update(torch.tensor([peak_open / 2, 0.0, 0.0], dtype=D), first.history)
    secant = float(first.traction[0]) / peak_open
    assert abs(float(unloaded.traction[0]) - secant * peak_open / 2) < 1e-6 * float(
        first.traction[0]
    )
    assert float(unloaded.history) == pytest.approx(peak_open, rel=1e-14)
    assert float(unloaded.damage) == pytest.approx(float(first.damage), rel=1e-12)
    # Reload to the same opening: same traction (no healing, no extra damage).
    reloaded = law_obj.update(torch.tensor([peak_open, 0.0, 0.0], dtype=D), unloaded.history)
    assert float(reloaded.traction[0]) == pytest.approx(float(first.traction[0]), rel=1e-12)
    assert float(reloaded.damage) == pytest.approx(float(first.damage), rel=1e-12)


def test_compression_uses_undamaged_penalty_and_never_grows_history():
    law_obj = law()
    response = law_obj.update(torch.tensor([-1e-5, 0.0, 0.0], dtype=D))
    assert float(response.traction[0]) == pytest.approx(-1e-5 * law_obj.kn, rel=1e-14)
    assert abs(float(response.damage)) < 1e-15
    assert abs(float(response.history)) < 1e-15
    # Shear under compression still damages, normal response stays elastic.
    shear_damaged = law_obj.update(torch.tensor([-1e-5, 4e-5, 0.0], dtype=D))
    assert float(shear_damaged.traction[0]) == pytest.approx(-1e-5 * law_obj.kn, rel=1e-14)
    assert float(shear_damaged.damage) > 0.0


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        BKCohesive3DLaw(-1.0, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)
    with pytest.raises(ValueError):
        # G_Ic too small: 2G/t0 = 1e-7 m < t0/kn = 5e-6 m
        BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 0.25, 1200.0, 1.6)
    with pytest.raises(ValueError):
        law().update(torch.zeros(2, dtype=D))


# --------------------------------------------------------------------------- #
# Element geometry and quadrature
# --------------------------------------------------------------------------- #


def test_shape_functions_partition_of_unity_and_gradient_sums_to_zero():
    for xi, eta in ((-1.0, -1.0), (0.0, 0.3), (0.577, -0.577)):
        n, dn_dxi, dn_deta = cohesive_shape_functions(xi, eta, dtype=D)
        assert float(n.sum()) == pytest.approx(1.0, abs=1e-14)
        assert float(dn_dxi.sum()) == pytest.approx(0.0, abs=1e-14)
        assert float(dn_deta.sum()) == pytest.approx(0.0, abs=1e-14)


def test_interface_area_of_unit_square_and_skewed_patch():
    assert float(interface_area(flat_element(1.0))) == pytest.approx(1.0, rel=1e-14)
    assert float(interface_area(flat_element(2.0))) == pytest.approx(4.0, rel=1e-14)
    # Skewed parallelogram 1 x 1 with 30 degrees shear -> area sin(30) = 0.5
    coords = flat_element(1.0)
    theta = math.radians(30.0)
    shear = torch.tensor([[1.0, 0.0, 0.0], [math.cos(theta), math.sin(theta), 0.0], [0.0, 0.0, 1.0]], dtype=D)
    coords = coords @ shear.T
    assert float(interface_area(coords)) == pytest.approx(0.5, rel=1e-12)


def test_mid_surface_frame_is_orthonormal_and_right_handed():
    frames, weights, mid = mid_surface_frame(flat_element(1.0, 0.1))
    assert len(frames) == 4
    assert float(weights.sum()) == pytest.approx(4.0, rel=1e-14)
    for normal, e1, e2, jac, _ in frames:
        triad = torch.stack((normal, e1, e2))
        assert torch.allclose(triad @ triad.T, torch.eye(3, dtype=D), atol=1e-14)
        assert float(torch.linalg.det(triad)) == pytest.approx(1.0, rel=1e-14)
        assert float(normal[2]) == pytest.approx(1.0, rel=1e-14)
        assert float(jac) == pytest.approx(0.25, rel=1e-14)


def test_degenerate_element_and_bad_history_are_rejected():
    collinear = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [2.0, 0.0, 0.1],
            [3.0, 0.0, 0.1],
        ],
        dtype=D,
    )
    with pytest.raises(ValueError):
        interface_area(collinear)
    with pytest.raises(ValueError):
        cohesive_hex8_response(flat_element(), torch.zeros(8, 3, dtype=D), law(), torch.zeros(3, dtype=D))


# --------------------------------------------------------------------------- #
# Element behaviour under exact states
# --------------------------------------------------------------------------- #


def test_patch_test_uniform_opening_reproduces_traction_times_area():
    law_obj = law()
    coords = flat_element(1.0)
    for opening in (1e-6, 2.5e-6, 5e-6, 2e-5, 1.5e-4, 2.5e-4):
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = opening
        force, _, _, diag = cohesive_hex8_response(coords, u, law_obj)
        expected = float(law_obj.envelope_traction(torch.tensor([[opening, 0.0, 0.0]], dtype=D))[0, 0])
        assert float(force[4:, 2].sum()) == pytest.approx(expected * 1.0, rel=1e-12)
        # Every Gauss point sees the same separation (uniform field).
        separations = diag["separation"]
        assert float(separations[:, 0].std()) < 1e-18
        assert float(separations[0, 0]) == pytest.approx(opening, rel=1e-12)
        # Action-reaction: the nodal force vector sums to zero.
        assert float(force.sum(dim=0).abs().max()) < 1e-6 * max(1.0, float(force.abs().max()))


def test_patch_test_under_shear_and_mixed_opening():
    law_obj = law()
    coords = flat_element(1.0)
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 0] = 6e-6
    u[4:, 2] = 4e-6
    force, _, _, diag = cohesive_hex8_response(coords, u, law_obj)
    expected = law_obj.envelope_traction(torch.tensor([[4e-6, 6e-6, 0.0]], dtype=D))[0]
    assert float(force[4:, 2].sum()) == pytest.approx(float(expected[0]), rel=1e-12)
    assert float(force[4:, 0].sum()) == pytest.approx(float(expected[1]), rel=1e-12)
    assert float(force[4:, 1].sum()) == pytest.approx(float(expected[2]), abs=1e-6)


def test_rigid_translation_gives_zero_traction():
    law_obj = law()
    coords = flat_element(1.0)
    u0 = torch.zeros(8, 3, dtype=D)
    u0[4:, 2] = 3e-5
    force0, _, _, _ = cohesive_hex8_response(coords, u0, law_obj)
    shift = torch.tensor([1e-3, -2e-3, 5e-4], dtype=D)
    force, _, _, _ = cohesive_hex8_response(coords, u0 + shift, law_obj)
    scale = max(float(force0.abs().max()), 1.0)
    assert float((force - force0).abs().max()) / scale < 1e-12


@pytest.mark.parametrize("angle", [0.3, 0.7, 1.4])
def test_rigid_rotation_frame_indifference(angle):
    """Rotating both the geometry and the displacement rotates the force."""

    def rotation(axis, angle):
        axis = torch.tensor(axis, dtype=D)
        axis = axis / torch.linalg.vector_norm(axis)
        k = torch.tensor(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=D,
        )
        return torch.eye(3, dtype=D) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)

    law_obj = law()
    coords = flat_element(1.0)
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 3e-5
    u[4:, 0] = 1e-5
    force0, _, hist0, _ = cohesive_hex8_response(coords, u, law_obj)

    r = rotation((0.3, -0.5, 0.8), angle)
    force_rot, _, hist_rot, diag_rot = cohesive_hex8_response(coords @ r.T, u @ r.T, law_obj)
    scale = float(force0.abs().max())
    assert float((force_rot - force0 @ r.T).abs().max()) / scale < 1e-12
    assert torch.allclose(hist_rot, hist0, rtol=1e-14, atol=0.0)
    # Traction magnitude is invariant (the normal is carried by the rotation).
    assert float(diag_rot["traction"].abs().max()) == pytest.approx(
        float(cohesive_hex8_response(coords, u, law_obj)[3]["traction"].abs().max()), rel=1e-13
    )


def test_mesh_objectivity_of_uniform_opening():
    """1x1, 2x2 and 4x4 patches of the same interface give the same resultant."""
    law_obj = law()
    opening = 2e-5
    resultants = []
    for n in (1, 2, 4, 8):
        coords, conn, _ = patch_mesh(n)
        u = torch.zeros(coords.shape[0], 3, dtype=D)
        top = torch.zeros(coords.shape[0], dtype=torch.bool)
        top[(n + 1) ** 2 :] = True
        u[top, 2] = opening
        force, _, _ = assemble_cohesive_interface(coords.to(D), conn, u, law_obj)
        total = float(force.reshape(-1, 3)[top, 2].sum())
        resultants.append(total)
    expected = float(law_obj.envelope_traction(torch.tensor([[opening, 0.0, 0.0]], dtype=D))[0, 0])
    for total in resultants:
        assert total == pytest.approx(expected, rel=1e-11), (resultants, expected)


def test_element_tangent_is_symmetric_and_positive_definite_in_elastic_regime():
    law_obj = law()
    coords = flat_element(1.0)
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 1e-6  # well below onset
    _, tangent, _, _ = cohesive_hex8_response(coords, u, law_obj)
    assert torch.allclose(tangent, tangent.T, atol=1e-6 * float(tangent.abs().max()))
    eigenvalues = torch.linalg.eigvalsh(0.5 * (tangent + tangent.T))
    assert float(eigenvalues.min()) > -1e-9 * float(eigenvalues.max())
    # Rigid-translation modes are in the null space (6 modes).
    for axis in range(3):
        mode = torch.zeros((8, 3), dtype=D)
        mode[:, axis] = 1.0
        assert float((tangent @ mode.reshape(-1)).abs().max()) < 1e-6 * float(
            tangent.abs().max()
        )


def test_element_dissipated_work_equals_fracture_energy_times_area():
    """Quasi-static element loading: int F dW = G_Ic * A."""
    law_obj = law()
    coords = flat_element(1.0)
    failure = float(law_obj.failure_amplitude(torch.tensor([1.0, 0.0, 0.0], dtype=D)))
    openings = torch.linspace(0.0, failure * 1.05, 4001, dtype=D)
    forces = []
    history = None
    for opening in openings:
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = opening
        force, _, history, _ = cohesive_hex8_response(coords, u, law_obj, history)
        forces.append(float(force[4:, 2].sum()))
    forces = torch.tensor(forces, dtype=D)
    work = float(torch.trapezoid(forces, openings))
    assert work == pytest.approx(500.0 * 1.0, rel=2e-3)


def test_history_shape_and_assembly_consistency():
    law_obj = law()
    coords, conn, _ = patch_mesh(2)
    u = torch.zeros(coords.shape[0], 3, dtype=D)
    u[(coords.shape[0] // 2) :, 2] = 3e-6
    force, tangent, histories = assemble_cohesive_interface(coords, conn, u, law_obj)
    assert histories.shape == (4, 4)
    # Assembly equals the sum of the element contributions.
    manual_force = torch.zeros_like(force)
    manual_tangent = torch.zeros_like(tangent)
    for e, conn_e in enumerate(conn):
        dofs = (3 * conn_e[:, None] + torch.arange(3)[None, :]).reshape(-1)
        f_e, k_e, _, _ = cohesive_hex8_response(coords[conn_e], u[conn_e], law_obj, histories[e])
        manual_force.index_add_(0, dofs, f_e.reshape(-1))
        rows = dofs[:, None].expand(-1, dofs.numel())
        cols = dofs[None, :].expand(dofs.numel(), -1)
        manual_tangent.index_put_((rows, cols), k_e, accumulate=True)
    assert torch.allclose(force, manual_force, atol=1e-9 * float(manual_force.abs().max()))
    assert torch.allclose(tangent, manual_tangent, atol=1e-9 * float(manual_tangent.abs().max()))


def test_warped_interface_keeps_uniform_normal_opening_exact():
    """A non-planar mid-surface still transmits the exact uniform normal traction."""
    law_obj = law()
    coords = flat_element(1.0)
    # Warp the mid-surface: both faces move together, so the jump is unchanged.
    warp = torch.zeros(8, 3, dtype=D)
    warp[:4, 2] = torch.tensor([0.0, 0.02, -0.03, 0.01], dtype=D)
    warp[4:, 2] = warp[:4, 2]
    coords = coords + warp
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 2e-5
    force, _, _, diag = cohesive_hex8_response(coords, u, law_obj)
    area = float(interface_area(coords))
    assert area > 1.0  # warping increases the mid-surface area
    # The opening is exactly 2e-5 along the *local normal* at each Gauss point.
    for gp in range(4):
        normal = diag["normal"][gp]
        local = float(diag["separation"][gp][0])
        assert local == pytest.approx(2e-5 * float(normal[2]), rel=1e-12)
    expected = float(
        law_obj.envelope_traction(torch.tensor([[2e-5, 0.0, 0.0]], dtype=D))[0, 0]
    )
    # The resultant follows the *warped* normal, so it matches the flat-patch
    # traction only to O(warp slope^2); 1e-3 covers that geometric residual and
    # the softening slope at this opening.
    assert float(force[4:, 2].sum()) == pytest.approx(expected * area, rel=1e-3)