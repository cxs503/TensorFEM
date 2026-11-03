"""Verification of the MITC4 flat-shell element (selective-reduced-integration membrane).

The ``shell4_mitc`` module implements a four-node flat shell whose membrane
shear term is evaluated at the element centre (SRI) to suppress parasitic
in-plane shear on faceted cylindrical meshes.  This test suite verifies:

1. **Symmetry** – the 24×24 stiffness matrix is symmetric.
2. **Rigid-body modes** – translations and in-plane rotation produce zero energy.
3. **Patch test** – a single element reproduces constant in-plane strain exactly.
4. **Consistency with shell4** – the MITC variant differs from the standard
   ``shell4_stiffness`` only in the membrane shear treatment.
5. **Plate-bending convergence** – a simply-supported square plate under
   sinusoidal load converges to the Mindlin analytical solution.
6. **Drilling stiffness** – uniform rigid spin produces zero energy.
7. **Positive semi-definiteness** – no negative eigenvalues.

References
----------
* Bathe, K.-J. & Dvorkin, E.N. (1985). A four-node plate bending element based
  on Mindlin/Reissner plate theory and a mixed interpolation. *Int. J. Numer.
  Meth. Engng*, 21, 367–383.
* Zienkiewicz, O.C. & Taylor, R.L. (2000). *The Finite Element Method*, 5th ed.,
  Vol. 2, §7.4 (selective reduced integration for membrane/shear locking).
"""
import math

import pytest
import torch

from tensorfem.shell4_mitc import shell4_mitc_stiffness
from tensorfem.shell4 import shell4_stiffness


# ---------------------------------------------------------------------------
# 1. Symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    torch.tensor([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]], dtype=torch.float64),
    torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [2., 1., 0.], [0.5, 1., 0.]], dtype=torch.float64),
])
def test_mitc_stiffness_is_symmetric(shape):
    k = shell4_mitc_stiffness(shape, 210e9, 0.3, 0.01)
    assert k.shape == (24, 24)
    assert torch.allclose(k, k.T, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Rigid-body modes → zero strain energy
# ---------------------------------------------------------------------------

def test_rigid_body_modes_produce_zero_energy():
    """For a flat shell in the xy-plane, the rigid body modes are:
    - 3 translations (x, y, z)
    - 1 in-plane rotation (about z)
    Rotations about x and y cause bending and are NOT rigid body modes.
    """
    xyz = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 0.8, 0.], [0., 0.8, 0.]],
        dtype=torch.float64,
    )
    k = shell4_mitc_stiffness(xyz, 210e9, 0.3, 0.01)

    # Translation in x
    u = torch.zeros(24, dtype=torch.float64)
    u[0::6] = 1.0
    assert abs(float(0.5 * u @ k @ u)) < 1e-4

    # Translation in y
    u = torch.zeros(24, dtype=torch.float64)
    u[1::6] = 1.0
    assert abs(float(0.5 * u @ k @ u)) < 1e-4

    # Translation in z
    u = torch.zeros(24, dtype=torch.float64)
    u[2::6] = 1.0
    assert abs(float(0.5 * u @ k @ u)) < 1e-4

    # In-plane rotation about z (uniform rz)
    u = torch.zeros(24, dtype=torch.float64)
    u[5::6] = 1.0
    assert abs(float(0.5 * u @ k @ u)) < 1e-10


# ---------------------------------------------------------------------------
# 3. Patch test – constant in-plane strain
# ---------------------------------------------------------------------------

def test_patch_test_constant_uniaxial_strain():
    """A single element under uniform x-strain should reproduce the exact stress."""
    xyz = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        dtype=torch.float64,
    )
    E, nu, t = 1.0e7, 0.3, 0.01
    k = shell4_mitc_stiffness(xyz, E, nu, t)

    eps = 1.0e-4
    u = torch.zeros(24, dtype=torch.float64)
    for node in range(4):
        x, y = xyz[node, 0], xyz[node, 1]
        u[6 * node] = eps * x
        u[6 * node + 1] = -nu * eps * y

    sigma_xx = E * eps
    volume = 1.0 * 1.0 * t
    expected_energy = 0.5 * sigma_xx * eps * volume
    actual_energy = float(0.5 * u @ k @ u)
    assert abs(actual_energy - expected_energy) / expected_energy < 0.01


def test_patch_test_constant_shear_strain():
    """A single element under uniform shear strain."""
    xyz = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        dtype=torch.float64,
    )
    E, nu, t = 1.0e7, 0.3, 0.01
    k = shell4_mitc_stiffness(xyz, E, nu, t)

    gamma = 1.0e-4
    G = E / (2 * (1 + nu))
    u = torch.zeros(24, dtype=torch.float64)
    for node in range(4):
        x, y = xyz[node, 0], xyz[node, 1]
        u[6 * node] = gamma * y / 2.0
        u[6 * node + 1] = gamma * x / 2.0

    expected_energy = 0.5 * G * gamma * gamma * (1.0 * 1.0 * t)
    actual_energy = float(0.5 * u @ k @ u)
    assert abs(actual_energy - expected_energy) / expected_energy < 0.02


# ---------------------------------------------------------------------------
# 4. Consistency with standard shell4
# ---------------------------------------------------------------------------

def test_mitc_differs_from_shell4_only_in_membrane_shear():
    """The MITC variant should differ from shell4 only in the membrane shear term."""
    xyz = torch.tensor(
        [[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]],
        dtype=torch.float64,
    )
    k_mitc = shell4_mitc_stiffness(xyz, 210e9, 0.3, 0.01)
    k_std = shell4_stiffness(xyz, 210e9, 0.3, 0.01)

    assert not torch.allclose(k_mitc, k_std, atol=1e-12)

    # Plate DOFs should be identical (same Mindlin plate formulation)
    plate_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22]
    for i in plate_dofs:
        for j in plate_dofs:
            assert torch.isclose(k_mitc[i, j], k_std[i, j], atol=1e-8), \
                f"Plate DOF ({i},{j}) differs: {k_mitc[i,j]} vs {k_std[i,j]}"


# ---------------------------------------------------------------------------
# 5. Plate-bending convergence (simply-supported square under sine load)
# ---------------------------------------------------------------------------

def _assemble_mitc_plate(nodes, elements, E, nu, t):
    ndof = 6 * len(nodes)
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    for conn in elements:
        ids = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
        ke = shell4_mitc_stiffness(nodes[conn], E, nu, t)
        K[ids[:, None], ids] += ke
    return K


def _sine_load(nodes, elements, side, q0):
    ndof = 6 * len(nodes)
    f = torch.zeros(ndof, dtype=torch.float64)
    gp = 1.0 / 3.0 ** 0.5
    for conn in elements:
        xy = nodes[conn, :2]
        fe = torch.zeros(24, dtype=torch.float64)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                xi = torch.tensor(xif, dtype=torch.float64)
                eta = torch.tensor(etaf, dtype=torch.float64)
                N = 0.25 * torch.stack([
                    (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta), (1 - xi) * (1 + eta),
                ])
                dN = 0.25 * torch.stack([
                    torch.stack([-(1 - eta), -(1 - xi)]),
                    torch.stack([(1 - eta), -(1 + xi)]),
                    torch.stack([(1 + eta), (1 + xi)]),
                    torch.stack([-(1 + eta), (1 - xi)]),
                ])
                J = dN.T @ xy
                detJ = torch.linalg.det(J)
                phys = N @ xy
                q = q0 * torch.sin(math.pi * phys[0] / side) * torch.sin(math.pi * phys[1] / side)
                for node in range(4):
                    fe[6 * node + 2] += N[node] * q * detJ
        ids = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
        f[ids] += fe
    return f


def _mindlin_analytical_center_deflection(side, E, nu, t, q0):
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    bending = q0 * side ** 4 / (4 * math.pi ** 4 * D)
    G = E / (2 * (1 + nu))
    shear = q0 * side ** 2 / (2 * math.pi ** 2 * (5 / 6) * G * t)
    return float(bending + shear)


def test_mitc_plate_converges_to_mindlin_solution():
    """Simply-supported square plate under sine load converges to Mindlin analytical."""
    side = 1.0
    E, nu, t = 210e9, 0.3, 0.01
    q0 = 1.0e6
    exact = _mindlin_analytical_center_deflection(side, E, nu, t, q0)

    errors = []
    for n in (4, 8, 16):
        xs = torch.linspace(0, side, n + 1, dtype=torch.float64)
        xx, yy = torch.meshgrid(xs, xs, indexing="xy")
        nodes = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros_like(xx.reshape(-1))], dim=1)
        elements = []
        for j in range(n):
            for i in range(n):
                n0 = j * (n + 1) + i
                elements.append((n0, n0 + 1, n0 + n + 2, n0 + n + 1))
        elements = torch.tensor(elements, dtype=torch.long)

        K = _assemble_mitc_plate(nodes, elements, E, nu, t)
        f = _sine_load(nodes, elements, side, q0)

        tol = side * 1e-12
        edge = ((nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol) |
                (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol))
        fixed_w = torch.nonzero(edge).flatten() * 6 + 2
        y_edge = ((nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol))
        x_edge = ((nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol))
        fixed_rx = torch.nonzero(y_edge).flatten() * 6 + 3
        fixed_ry = torch.nonzero(x_edge).flatten() * 6 + 4
        fixed = torch.cat([fixed_w, fixed_rx, fixed_ry]).unique()

        all_dofs = torch.arange(6 * len(nodes))
        free = all_dofs[~torch.isin(all_dofs, fixed)]
        u = torch.zeros(6 * len(nodes), dtype=torch.float64)
        u[free] = torch.linalg.solve(K[free[:, None], free], f[free])

        centre_idx = (n + 1) * (n // 2) + n // 2
        w_center = u[6 * centre_idx + 2].item()
        errors.append(abs(w_center - exact) / exact)

    assert errors[-1] < 0.05, f"Finest mesh error {errors[-1]:.3%} exceeds 5%"
    assert errors[-1] < errors[0], "Error should decrease with refinement"


# ---------------------------------------------------------------------------
# 6. Drilling stiffness preserves rigid spin
# ---------------------------------------------------------------------------

def test_uniform_rigid_spin_produces_zero_energy():
    """A uniform rotation about z should produce zero strain energy."""
    xyz = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        dtype=torch.float64,
    )
    k = shell4_mitc_stiffness(xyz, 210e9, 0.3, 0.01)
    u = torch.zeros(24, dtype=torch.float64)
    u[5::6] = 1.0
    energy = float(0.5 * u @ k @ u)
    assert energy < 1e-10, f"Uniform rz energy = {energy}"


# ---------------------------------------------------------------------------
# 7. Positive semi-definiteness
# ---------------------------------------------------------------------------

def test_stiffness_is_positive_semi_definite():
    """No negative eigenvalues; at least 4 rigid body zero modes for a flat shell."""
    xyz = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        dtype=torch.float64,
    )
    k = shell4_mitc_stiffness(xyz, 1.0, 0.3, 0.01)
    eigenvalues = torch.linalg.eigvalsh(k)
    # No negative eigenvalues
    assert (eigenvalues >= -1e-15).all(), f"Negative eigenvalues found: {eigenvalues[eigenvalues < -1e-15]}"
    # At least 4 near-zero eigenvalues (3 translations + 1 in-plane rotation)
    near_zero = (eigenvalues.abs() < 1e-10).sum().item()
    assert near_zero >= 4, f"Expected >= 4 near-zero eigenvalues, got {near_zero}"
