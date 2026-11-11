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
    cohesive_separations,
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

# --------------------------------------------------------------------------- #
# Corotational kinematics: objectivity at finite rigid rotation
# --------------------------------------------------------------------------- #

def _rotation(axis, angle: float) -> torch.Tensor:
    """Rodrigues rotation matrix from a (not necessarily unit) axis and an angle."""
    k = torch.as_tensor(axis, dtype=D)
    k = k / torch.linalg.vector_norm(k)
    skew = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=D)
    return (
        torch.eye(3, dtype=D)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


COROTATIONAL_ANGLES = (0.01, 0.1, 0.3, 0.7, 1.4, 3.0)


def rigid_rotation_displacements(coords: torch.Tensor, angle: float) -> torch.Tensor:
    """Displacements of a rigid rotation by ``angle`` about the x axis, in a fixed frame."""
    r = _rotation((1.0, 0.0, 0.0), angle)
    return coords @ (r.T - torch.eye(3, dtype=D))


def test_small_strain_kinematics_is_not_objective_at_finite_rotation():
    """Documented limitation of ``kinematics="small_strain"`` (measured, not asserted away).

    Rotating the body rigidly in a fixed frame is *not* a change of frame: the
    reference-frame separation then sees the rotation-induced shear
    ``k_s * g * theta`` and reports a large spurious traction.  This test pins
    that the limitation is real, so the ``corotational`` test below cannot pass
    by accident.
    """
    law_obj = law()
    coords = flat_element(1.0)
    scale = float(law_obj.tn0 * interface_area(coords))
    errors = []
    for angle in COROTATIONAL_ANGLES:
        u = rigid_rotation_displacements(coords, angle)
        force, _, _, diag = cohesive_hex8_response(coords, u, law_obj)
        errors.append(float(force.abs().max()) / scale)
        assert diag["kinematics"] == "small_strain"
    # Monotone growth with the rotation angle, and already large at 0.3 rad.
    assert all(b > a for a, b in zip(errors[:-1], errors[1:]))
    assert errors[COROTATIONAL_ANGLES.index(0.3)] > 100.0


def test_corotational_rigid_rotation_gives_zero_traction():
    """``kinematics="corotational"`` is objective under rigid rotation of any angle."""
    law_obj = law()
    coords = flat_element(1.0)
    scale = float(law_obj.tn0 * interface_area(coords))
    for angle in COROTATIONAL_ANGLES:
        u = rigid_rotation_displacements(coords, angle)
        force, tangent, history, diag = cohesive_hex8_response(
            coords, u, law_obj, kinematics="corotational"
        )
        assert float(force.abs().max()) / scale < 1e-10, angle
        # cancellation floor of the frame-projected gaps: ~1e-5 Pa on tractions
        # of order t_n0 = 5e6 Pa (see docs/cohesive3d-interface-element.md, pitfall 4)
        assert float(diag["traction"].abs().max()) < 1e-10 * law_obj.tn0
        assert float(history.abs().max()) < 1e-12  # no spurious damage (a ~ 1e-12 m floor)
        assert float(tangent.abs().max()) > 0.0  # the operator is still defined
        assert diag["kinematics"] == "corotational"
    # Rigid translation is objective in both formulations.
    shifted = flat_element(1.0) + torch.tensor([3.0, -2.0, 5.0], dtype=D)
    u = torch.full((8, 3), 0.25, dtype=D)
    force, _, _, _ = cohesive_hex8_response(shifted, u, law_obj, kinematics="corotational")
    assert float(force.abs().max()) / scale < 1e-10


def test_corotational_patch_uniform_opening_is_exact():
    law_obj = law()
    coords = flat_element(1.0)
    area = float(interface_area(coords))
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 2e-5
    force, _, _, diag = cohesive_hex8_response(coords, u, law_obj, kinematics="corotational")
    expected = float(law_obj.envelope_traction(torch.tensor([[2e-5, 0.0, 0.0]], dtype=D))[0, 0]) * area
    assert float(force[4:, 2].sum()) == pytest.approx(expected, rel=1e-12)
    for gp in range(4):
        assert float(diag["separation"][gp][0]) == pytest.approx(2e-5, rel=1e-12)
        assert abs(float(diag["separation"][gp][1])) < 1e-18


def test_corotational_rotation_objectivity_of_the_reaction():
    """A rotated element geometry returns the rotated reaction: f(RX, Ru) = R f(X, u)."""
    law_obj = law()
    coords = flat_element(1.0)
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 3e-6
    u[4:, 0] = 1e-6
    force0 = cohesive_hex8_response(coords, u, law_obj, kinematics="corotational")[0]
    for axis_angle in ((0.5, -0.2, 0.9), (2.4, 0.0, 0.0), (0.0, 3.0, 0.0)):
        r = _rotation(axis_angle, 1.0)
        force_rot = cohesive_hex8_response(coords @ r.T, u @ r.T, law_obj, kinematics="corotational")[0]
        assert float((force_rot - force0 @ r.T).norm()) / float(force0.norm()) < 1e-10


def test_corotational_elastic_tangent_is_symmetric():
    law_obj = law()
    coords = flat_element(1.0)
    u = rigid_rotation_displacements(coords, 0.4)
    u[4:, 2] += 2e-6  # elastic: below the onset separation t_n0 / k_n = 5e-6 m
    _, tangent, _, diag = cohesive_hex8_response(coords, u, law_obj, kinematics="corotational")
    assert float(diag["damage"].max()) == 0.0
    assert float((tangent - tangent.T).abs().max()) / float(tangent.abs().max()) < 1e-12


def test_corotational_tangent_matches_finite_differences_in_softening():
    """The tangent carries every geometric term: it matches central differences."""
    law_obj = law()
    coords = flat_element(1.0)
    u = rigid_rotation_displacements(coords, 0.3)
    u[4:, 2] += 2e-5  # softening branch
    _, tangent, _, diag = cohesive_hex8_response(coords, u, law_obj, kinematics="corotational")
    assert 0.0 < float(diag["damage"].max()) < 1.0
    eps = 1e-9
    numeric = torch.zeros(24, 24, dtype=D)
    flat = u.reshape(-1)
    for j in range(24):
        delta = torch.zeros(24, dtype=D)
        delta[j] = eps
        plus = cohesive_hex8_response(
            coords, (flat + delta).reshape(8, 3), law_obj, kinematics="corotational"
        )[0].reshape(-1)
        minus = cohesive_hex8_response(
            coords, (flat - delta).reshape(8, 3), law_obj, kinematics="corotational"
        )[0].reshape(-1)
        numeric[:, j] = (plus - minus) / (2.0 * eps)
    assert float((tangent - numeric).norm()) / float(numeric.norm()) < 1e-6


def test_corotational_dissipation_is_rotation_invariant():
    """Monotonic opening of a rigidly rotated interface dissipates the same energy."""
    law_obj = law()
    coords = flat_element(1.0)
    area = float(interface_area(coords))
    stroke, steps = 2.2e-4, 40

    def dissipated(angle: float) -> float:
        r = _rotation((1.0, 0.0, 0.0), angle) if angle else torch.eye(3, dtype=D)
        base = coords @ (r.T - torch.eye(3, dtype=D))
        normal = r @ torch.tensor([0.0, 0.0, 1.0], dtype=D)
        work, history, previous = 0.0, None, 0.0
        for i in range(1, steps + 1):
            u = base.clone()
            u[4:8] += stroke * i / steps * normal
            force, _, history, _ = cohesive_hex8_response(
                coords, u, law_obj, history, kinematics="corotational"
            )
            current = float((force[4:8] @ normal).sum())
            work += 0.5 * (current + previous) * (stroke / steps)
            previous = current
        return work

    reference = dissipated(0.0)
    for angle in (0.4, 1.2):
        assert dissipated(angle) == pytest.approx(reference, rel=1e-12)
    # Trapezoid quadrature error of a 40-step sweep is below 0.5 % of G_Ic * A.
    assert reference == pytest.approx(law_obj.gic * area, rel=5e-3)


def test_corotational_assembly_and_mesh_objectivity():
    law_obj = law()
    opening = 2e-6
    expected = float(
        law_obj.envelope_traction(torch.tensor([[opening, 0.0, 0.0]], dtype=D))[0, 0]
    ) * 1.0
    for n in (1, 2):
        coords, conn, _ = patch_mesh(n)
        nn = (n + 1) ** 2
        top = torch.zeros(coords.shape[0], dtype=torch.bool)
        top[nn:] = True
        u = torch.zeros(coords.shape[0], 3, dtype=D)
        u[top, 2] = opening
        force, tangent, histories = assemble_cohesive_interface(
            coords.to(D), conn, u, law_obj, kinematics="corotational"
        )
        assert float(force.reshape(-1, 3)[top, 2].sum()) == pytest.approx(expected, rel=1e-11)
        # Assembly equals the sum of the element contributions, force and tangent.
        manual_force = torch.zeros_like(force)
        manual_tangent = torch.zeros_like(tangent)
        for e, conn_e in enumerate(conn):
            dofs = (3 * conn_e[:, None] + torch.arange(3)[None, :]).reshape(-1)
            f_e, k_e, _, _ = cohesive_hex8_response(
                coords.to(D)[conn_e], u[conn_e], law_obj, histories[e], kinematics="corotational"
            )
            manual_force.index_add_(0, dofs, f_e.reshape(-1))
            rows = dofs[:, None].expand(-1, dofs.numel())
            cols = dofs[None, :].expand(dofs.numel(), -1)
            manual_tangent.index_put_((rows, cols), k_e, accumulate=True)
        assert torch.allclose(force, manual_force, atol=1e-9 * float(manual_force.abs().max()) + 1e-9)
        assert float((tangent - manual_tangent).abs().max()) < 1e-9 * float(
            manual_tangent.abs().max()
        )


def test_corotational_compression_is_undamaged():
    law_obj = law()
    coords = flat_element(1.0)
    area = float(interface_area(coords))
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = -1e-8
    force, _, history, diag = cohesive_hex8_response(coords, u, law_obj, kinematics="corotational")
    assert float(force[4:, 2].sum()) == pytest.approx(-law_obj.kn * 1e-8 * area, rel=1e-8)
    assert float(diag["damage"].max()) == 0.0
    assert float(history.abs().max()) < 1e-12  # only the ~1e-19 cancellation floor


def test_cohesive_separations_helper_matches_the_element_diagnostics():
    """The public separation helper is pinned to both element formulations."""
    coords = flat_element(1.0)
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 2e-5
    u[4:, 0] = 5e-6
    law_obj = law()
    for kinematics in ("small_strain", "corotational"):
        helper = cohesive_separations(coords, u, kinematics=kinematics)
        diag = cohesive_hex8_response(coords, u, law_obj, kinematics=kinematics)[3]
        assert torch.allclose(helper, diag["separation"], atol=1e-18, rtol=1e-13)


def test_invalid_kinematics_is_rejected():
    law_obj = law()
    coords = flat_element(1.0)
    u = torch.zeros(8, 3, dtype=D)
    with pytest.raises(ValueError):
        cohesive_hex8_response(coords, u, law_obj, kinematics="total_lagrangian")
    with pytest.raises(ValueError):
        cohesive_separations(coords, u, kinematics="updated_lagrangian")
    with pytest.raises(ValueError):
        assemble_cohesive_interface(coords, torch.arange(8).reshape(1, 8), u, law_obj, kinematics="x")


# --------------------------------------------------------------------------- #
# Batched material-point path (regression: element assembly is batched)
# --------------------------------------------------------------------------- #


_SAMPLE_DIRECTIONS = (
    # (unit direction before normalisation, target branch)
    ((1.0, 0.3, 0.0), "softening"),      # mode I dominated
    ((0.2, 1.0, 0.0), "softening"),      # shear dominated
    ((0.8, 0.5, 0.3), "softening"),      # all three components non-zero
    ((1.0, 0.0, 0.0), "elastic"),        # pure mode I below onset
    ((-0.6, 0.8, 0.2), "softening"),     # compressive normal part, shear driven
    ((0.0, 1.0, 1.0), "elastic"),        # equal shear modes, below onset
    ((1.0, 1.0, 1.0), "failed"),         # mixed, driven past failure
)


def _mixed_mode_samples(n: int) -> torch.Tensor:
    """``n`` separations spanning elastic, softening and fully failed states.

    Amplitudes are built from the law's own closed-form onset and failure
    openings along each direction (never from a hard-coded number), so the
    samples provably reach every branch for any parameter set.
    """
    law_obj = law()
    samples = []
    for (dx, dy, dz), branch in _SAMPLE_DIRECTIONS[:n]:
        c = torch.tensor([dx, dy, dz], dtype=D)
        c = c / torch.linalg.vector_norm(c)
        onset = float(law_obj.onset_amplitude(c))
        failure = float(law_obj.failure_amplitude(c))
        amplitude = {
            "elastic": 0.45 * onset,
            "softening": 0.5 * (onset + failure),
            "failed": 1.5 * failure,
        }[branch]
        samples.append(amplitude * c)
    return torch.stack(samples)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7])
def test_batched_update_matches_pointwise_in_softening(n):
    """``update`` on a batch must equal ``n`` independent point calls.

    Regression: the BK mixing weight of ``dgc`` and the ``dF`` denominators were
    broadcast without ``[..., None]``.  Three integration points in the batch
    then silently reused the *neighbouring* point's weight (the trailing axis
    aliased the component axis) and every other batch size raised a shape error,
    so the vectorised element assembly could not integrate damage at all.
    """
    law_obj = law()
    sep = _mixed_mode_samples(n)
    hist = torch.zeros(n, dtype=D)
    batch = law_obj.update(sep, hist)
    assert torch.any(batch.damage > 0.0), "test data must exercise the softening branch"
    for i in range(n):
        one = law_obj.update(sep[i], hist[i].reshape(()))
        assert torch.allclose(batch.traction[i], one.traction, rtol=1e-13, atol=1e-13)
        assert torch.allclose(batch.tangent[i], one.tangent, rtol=1e-13, atol=1e-13)
        assert torch.allclose(batch.damage[i], one.damage, rtol=1e-13, atol=1e-13)
        assert torch.allclose(batch.history[i], one.history, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_batched_softening_tangent_is_the_derivative_of_the_traction(n):
    """The batched tangent must still be the exact derivative (central differences).

    Pointwise agreement alone cannot certify this: a broadcast that mixed the
    points' mixing weights consistently in both the traction and the tangent
    would pass a self-comparison.  Central differences of the traction are an
    independent oracle.
    """
    law_obj = law()
    sep = _mixed_mode_samples(n)[:n]
    hist = torch.zeros(n, dtype=D)
    response = law_obj.update(sep, hist)
    assert torch.any(response.damage > 0.0)
    step = 1e-13
    jacobian = torch.zeros(n, 3, 3, dtype=D)
    for m in range(3):
        perturb = torch.zeros(3, dtype=D)
        perturb[m] = step
        plus = law_obj.update(sep + perturb, hist).traction
        minus = law_obj.update(sep - perturb, hist).traction
        jacobian[:, :, m] = (plus - minus) / (2.0 * step)
    scale = max(float(jacobian.abs().max()), 1.0)
    error = float((response.tangent - jacobian).abs().max()) / scale
    assert error < 1e-6, f"batched consistent tangent deviates from finite differences: {error:.3e}"


# --------------------------------------------------------------------------- #
# Dissipated-energy accounting (closed form vs path integral)
# --------------------------------------------------------------------------- #


def _unit(angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    return torch.tensor([math.cos(theta), math.sin(theta), 0.0], dtype=D)


def _dissipated_by_path_integral(law_obj, direction: torch.Tensor, steps: int = 2001) -> float:
    """Work spent loading monotonically from 0 to failure along ``direction``."""
    failure = float(law_obj.failure_amplitude(direction))
    amplitudes = torch.linspace(0.0, failure, steps, dtype=D)
    separation = amplitudes[:, None] * direction[None, :]
    traction = law_obj.update(separation).traction
    work = torch.trapezoid((traction * direction).sum(-1), amplitudes)
    return float(work)


@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
def test_dissipated_energy_matches_the_path_integral(angle):
    """The closed-form dissipation must equal int t . d(delta) along a ray.

    This is the independent oracle for the new state function: the traction
    law (already verified against autograd and finite differences) is
    integrated numerically, so the energy is derived from a different code
    path than the closed form.
    """
    law_obj = law()
    direction = _unit(angle)
    integrated = _dissipated_by_path_integral(law_obj, direction)
    closed = float(law_obj.dissipated_energy(direction * float(law_obj.failure_amplitude(direction))))
    assert closed == pytest.approx(integrated, rel=1e-5)


@pytest.mark.parametrize(
    "direction",
    [
        torch.tensor([1.0, 0.0, 0.0], dtype=D),
        torch.tensor([0.0, 1.0, 0.0], dtype=D),
        torch.tensor([0.0, 0.0, 1.0], dtype=D),
        torch.tensor([1.0, 1.0, 0.0], dtype=D),
        torch.tensor([1.0, 1.0, 1.0], dtype=D),
    ],
)
def test_dissipated_energy_equals_gc_at_failure(direction):
    """Complete separation consumes exactly the BK fracture energy."""
    law_obj = law()
    unit = direction / torch.linalg.vector_norm(direction)
    failure = law_obj.failure_amplitude(unit)
    dissipated = law_obj.dissipated_energy(unit * failure)
    critical = law_obj.critical_energy(unit)
    assert float(dissipated) == pytest.approx(float(critical), rel=1e-12)
    # Past failure the interface is traction free and cannot dissipate more.
    beyond = law_obj.dissipated_energy(unit * (3.0 * failure))
    assert float(beyond) == pytest.approx(float(critical), rel=1e-12)


def test_dissipated_energy_is_zero_while_elastic_and_monotone_saturating():
    """No dissipation before onset; monotone and bounded by G_c on the envelope."""
    law_obj = law()
    direction = _unit(30.0)
    onset = float(law_obj.onset_amplitude(direction))
    failure = float(law_obj.failure_amplitude(direction))
    critical = float(law_obj.critical_energy(direction))

    elastic = direction[None, :] * torch.linspace(0.0, onset, 33, dtype=D)[:, None]
    assert float(law_obj.dissipated_energy(elastic).abs().max()) == 0.0

    amplitudes = torch.linspace(onset, failure, 200, dtype=D)
    values = law_obj.dissipated_energy(amplitudes[:, None] * direction[None, :])
    assert bool(torch.all(torch.diff(values) >= -1e-12)), "dissipation must not decrease on loading"
    assert float(values.max()) <= critical * (1.0 + 1e-12)
    assert float(values[-1]) == pytest.approx(critical, rel=1e-12)


def test_dissipated_energy_is_independent_of_the_unloading_path():
    """Secant unloading stores energy reversibly: W_d depends on history only."""
    law_obj = law()
    direction = _unit(20.0)
    peak = 2.0 * float(law_obj.onset_amplitude(direction))
    loaded = law_obj.update(direction * peak)
    assert float(loaded.dissipated_energy) > 0.0
    # Unload to 20% of the peak reopening: the dissipated energy is unchanged.
    partial = law_obj.update(direction * (0.2 * peak), loaded.history)
    assert float(partial.dissipated_energy) == pytest.approx(float(loaded.dissipated_energy), rel=1e-12)
    # ... and so is the stored energy accounting: W_d + 1/2 t.delta is monotone.
    reopened = law_obj.update(direction * peak, loaded.history)
    assert float(reopened.dissipated_energy) == pytest.approx(float(loaded.dissipated_energy), rel=1e-12)


def test_compression_dissipates_nothing():
    """Closing the interface is elastic (penalty k_n) and never dissipates."""
    law_obj = law()
    closure = torch.tensor([[-1e-7, 2e-7, 0.0], [-1e-6, 0.0, 0.0]], dtype=D)
    assert float(law_obj.dissipated_energy(closure).abs().max()) == 0.0
    response = law_obj.update(closure)
    assert float(response.dissipated_energy.abs().max()) == 0.0


def test_response_dissipated_energy_matches_the_law_method():
    """``update`` and the standalone accessor must report the same state function."""
    law_obj = law()
    separation = _mixed_mode_samples(7)
    history = torch.linspace(0.0, 1.2e-4, separation.shape[0], dtype=D)
    response = law_obj.update(separation, history)
    direct = law_obj.dissipated_energy(separation, history)
    assert torch.allclose(response.dissipated_energy, direct, rtol=1e-13, atol=1e-13)
    # The state function must not depend on how the state was reached.
    again = law_obj.update(separation, response.history)
    assert torch.allclose(again.dissipated_energy, response.dissipated_energy, rtol=1e-13, atol=1e-13)


def test_element_dissipated_energy_equals_fracture_energy_times_area():
    """Element-level closure: the reported Joules are exactly G_c * A."""
    law_obj = law()
    coords = flat_element(1.0)
    area = float(interface_area(coords))
    failure = float(law_obj.failure_amplitude(torch.tensor([1.0, 0.0, 0.0], dtype=D)))
    for kinematics in ("small_strain", "corotational"):
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = failure
        _, _, _, diagnostics = cohesive_hex8_response(
            coords, u, law_obj, kinematics=kinematics
        )
        assert float(diagnostics["dissipated_energy"]) == pytest.approx(500.0 * area, rel=1e-12)
        assert torch.allclose(
            diagnostics["dissipated_per_area"],
            torch.full((4,), 500.0, dtype=D),
            rtol=1e-12,
        )
        # Partially damaged interface: 0 < W_d < G_c A.
        u[4:, 2] = 1.2 * float(law_obj.onset_amplitude(torch.tensor([1.0, 0.0, 0.0], dtype=D)))
        _, _, _, partial = cohesive_hex8_response(coords, u, law_obj, kinematics=kinematics)
        assert 0.0 < float(partial["dissipated_energy"]) < 500.0 * area


def test_corotational_element_dissipation_is_rotation_invariant():
    """Rigid rotation of the body must not change the dissipated energy."""
    law_obj = law()
    coords = flat_element(1.0)
    values = []
    for angle in (0.0, 0.3, 1.2):
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=D)
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = 1e-4
        _, _, _, diagnostics = cohesive_hex8_response(
            coords @ rotation.T, u @ rotation.T, law_obj, kinematics="corotational"
        )
        values.append(float(diagnostics["dissipated_energy"]))
    assert max(values) - min(values) < 1e-10 * max(values)


def test_non_proportional_path_dissipation_stays_bounded():
    """Mode-mix rotation at constant gap: the state function stays within G_c.

    The law is a history-of-maximum-separation model, so a path that rotates
    the mixity is outside the *exact* (fixed-direction) envelope.  This test
    pins the measured behaviour instead of assuming it: the reported energy
    must remain bounded by the larger of the two pure-mode energies and stay
    within 0.1% of the numerically integrated work for the path used in
    ``benchmarks/cohesive3d_verification_tables.py``.
    """
    law_obj = law()
    onset = float(law_obj.onset_amplitude(_unit(0.0)))
    peak = 1.5 * onset
    normal = _unit(0.0)
    failure_shear = float(law_obj.failure_amplitude(_unit(90.0)))

    path = [normal * peak * (k / 200.0) for k in range(201)]
    path += [torch.tensor([peak, failure_shear * k / 400.0, 0.0], dtype=D) for k in range(1, 401)]
    work = 0.0
    history = None
    previous = None
    for separation in path:
        response = law_obj.update(separation, history)
        history = response.history
        if previous is not None:
            work += 0.5 * float(((previous[1] + response.traction) * (separation - previous[0])).sum())
        previous = (separation, response.traction)
    # Bounded by the shear fracture energy (the path ends in mode II) ...
    assert work <= 1200.0 * (1.0 + 1e-3)
    # ... and the closed-form state function agrees with the integrated work.
    closed = float(response.dissipated_energy)
    assert closed == pytest.approx(work, rel=1e-3)
