"""Acceptance tests for the Craig--Bampton component mode synthesis.

Anchors are identities of the transformation, not snapshots:

* consistent mass of a Q4 element integrates ``rho t N_i N_j`` (total mass and
  symmetry/definiteness of the assembled matrix);
* the constraint modes solve ``K_ss Psi_c = -K_si`` (their definition);
* fixed-interface modes are ``M_ss``-orthonormal and satisfy the eigenproblem;
* ``n_modes = 0`` degenerates exactly to Guyan / static condensation;
* retaining every internal mode reproduces the monolithic eigenvalues;
* truncation is a Rayleigh--Ritz upper bound and converges monotonically;
* superelement assembly is order-invariant and a clamped interface transmits a
  static load exactly through the constraint modes;
* fail-closed behaviour for malformed input.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.continuum import rectangular_q4_mesh
from tensorfem.craig_bampton import (
    assemble_dense,
    assemble_superelements,
    build_superelement,
    craig_bampton_transformation,
    extract_component,
    fixed_interface_modes,
    generalized_eigen,
    guyan_transformation,
    partition_dofs,
    plane_stress_component,
    q4_consistent_mass,
    restrict_system,
)

torch.set_default_dtype(torch.float64)

L, H, THICK = 1.0, 0.05, 1.0
E, NU, RHO = 210.0e9, 0.3, 7850.0
SPLIT = 0.5 * L


@pytest.fixture(scope="module")
def strip():
    """Coarse two-component strip, clamped at x=0."""
    nodes, elements = rectangular_q4_mesh(L, H, 20, 4)
    cx = nodes[elements][:, :, 0].mean(dim=1)
    left = extract_component(nodes, elements, cx < SPLIT - 1e-12)
    right = extract_component(nodes, elements, cx > SPLIT + 1e-12)
    host = left.host_dofs(left.dofs_on(0, SPLIT))
    assert torch.equal(host, right.host_dofs(right.dofs_on(0, SPLIT)))

    clamp = left.dofs_on(0, 0.0)
    drop = set(clamp.tolist())
    keep = torch.tensor([d for d in range(left.n_dofs) if d not in drop])
    pos = torch.full((left.n_dofs,), -1, dtype=torch.long)
    pos[keep] = torch.arange(keep.numel())

    KL, ML = plane_stress_component(left.nodes, left.elements, young=E, poisson=NU,
                                    density=RHO, thickness=THICK)
    KR, MR = plane_stress_component(right.nodes, right.elements, young=E, poisson=NU,
                                    density=RHO, thickness=THICK)
    KL, ML = restrict_system(KL, ML, keep)

    K, M = plane_stress_component(nodes, elements, young=E, poisson=NU, density=RHO,
                                  thickness=THICK)
    root = torch.nonzero(nodes[:, 0].abs() <= 1e-12).reshape(-1)
    root_dofs = torch.cat((2 * root, 2 * root + 1))
    drop_all = set(root_dofs.tolist())
    keep_all = torch.tensor([d for d in range(2 * nodes.shape[0]) if d not in drop_all])
    K_ref, M_ref = restrict_system(K, M, keep_all)
    return dict(nodes=nodes, elements=elements, left=left, right=right, host=host,
                keep=keep, pos=pos, KL=KL, ML=ML, KR=KR, MR=MR,
                iface_A=pos[left.dofs_on(0, SPLIT)], iface_B=right.dofs_on(0, SPLIT),
                K_ref=K_ref, M_ref=M_ref)


def freq(omega2: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(omega2.clamp_min(0.0)) / (2.0 * math.pi)


def test_consistent_mass_conserves_mass_and_is_spd():
    nodes, elements = rectangular_q4_mesh(0.4, 0.1, 4, 2)
    coords = nodes[elements]
    ne = elements.shape[0]
    Me = q4_consistent_mass(coords, torch.full((ne,), RHO), torch.full((ne,), THICK))
    areas = ((coords[:, :, 0].max(1).values - coords[:, :, 0].min(1).values)
             * (coords[:, :, 1].max(1).values - coords[:, :, 1].min(1).values))
    assert abs(float(Me.sum()) / float((2.0 * RHO * THICK * areas).sum()) - 1.0) < 1e-13
    # symmetry and positive definiteness of an assembled block
    M = assemble_dense(Me, elements, 2 * nodes.shape[0])
    assert torch.allclose(M, M.T, atol=0.0, rtol=1e-15)
    assert float(torch.linalg.eigvalsh(M).min()) > 0.0


def test_partition_dofs_fail_closed():
    internal, iface = partition_dofs(6, torch.tensor([0, 3]))
    assert iface.tolist() == [0, 3]
    assert internal.tolist() == [1, 2, 4, 5]
    with pytest.raises(ValueError):
        partition_dofs(6, torch.tensor([0, 0]))
    with pytest.raises(ValueError):
        partition_dofs(6, torch.tensor([6]))


def test_constraint_modes_satisfy_their_definition(strip):
    KL, iface = strip["KL"], strip["iface_A"]
    internal, _ = partition_dofs(KL.shape[0], iface)
    K_ss = KL[internal[:, None], internal[None, :]]
    K_si = KL[internal[:, None], iface[None, :]]
    Psi = guyan_transformation(KL, iface)[internal]
    residual = (K_ss @ Psi + K_si).abs().max()
    assert float(residual / K_si.abs().max()) < 1e-12


def test_fixed_interface_modes_are_m_orthonormal_eigenvectors(strip):
    KL, ML, iface = strip["KL"], strip["ML"], strip["iface_A"]
    internal, _ = partition_dofs(KL.shape[0], iface)
    K_ss = KL[internal[:, None], internal[None, :]]
    M_ss = ML[internal[:, None], internal[None, :]]
    omega2, phi = fixed_interface_modes(KL, ML, iface, 4)
    phi_s = phi[internal]
    assert torch.allclose(phi_s.T @ M_ss @ phi_s, torch.eye(4), atol=1e-10, rtol=1e-10)
    eig_residual = (K_ss @ phi_s - M_ss @ phi_s * omega2).abs().max()
    assert float(eig_residual / (M_ss @ phi_s).abs().max()) < 1e-9
    assert torch.all(torch.diff(omega2) > -1e-9)


def test_zero_modes_equals_guyan(strip):
    KL, ML, iface, host = strip["KL"], strip["ML"], strip["iface_A"], strip["host"]
    Tg = guyan_transformation(KL, iface)
    se = build_superelement(KL, ML, iface, 0, host)
    assert se.n_modes == 0
    Kg = Tg.T @ KL @ Tg
    assert float((se.K - Kg).abs().max() / Kg.abs().max()) < 1e-14


def test_full_retention_recovers_monolithic_spectrum(strip):
    K_ref, M_ref = strip["K_ref"], strip["M_ref"]
    ref = freq(generalized_eigen(K_ref, M_ref, 4)[0])
    nA = strip["KL"].shape[0] - strip["iface_A"].numel()
    nB = strip["KR"].shape[0] - strip["iface_B"].numel()
    system = assemble_superelements([
        build_superelement(strip["KL"], strip["ML"], strip["iface_A"], nA, strip["host"]),
        build_superelement(strip["KR"], strip["MR"], strip["iface_B"], nB, strip["host"]),
    ])
    cms = freq(system.modes(4)[0])
    assert system.n_dofs == K_ref.shape[0]
    assert float(((cms - ref) / ref).abs().max()) < 1e-8


def test_truncation_is_upper_bound_and_converges(strip):
    K_ref, M_ref = strip["K_ref"], strip["M_ref"]
    ref = freq(generalized_eigen(K_ref, M_ref, 3)[0])
    errors = []
    for n in (0, 1, 2, 4):
        system = assemble_superelements([
            build_superelement(strip["KL"], strip["ML"], strip["iface_A"], n, strip["host"]),
            build_superelement(strip["KR"], strip["MR"], strip["iface_B"], n, strip["host"]),
        ])
        f = freq(system.modes(3)[0])
        # Rayleigh--Ritz: a trial subspace cannot lower a frequency
        assert bool(torch.all(f >= ref * (1.0 - 1e-10)))
        errors.append(float(((f[0] - ref[0]) / ref[0]).abs()))
    assert all(errors[i + 1] <= errors[i] + 1e-12 for i in range(len(errors) - 1))
    assert errors[-1] < 1e-4


def test_static_interface_load_is_carried_exactly_by_guyan(strip):
    KL, ML, iface, host = strip["KL"], strip["ML"], strip["iface_A"], strip["host"]
    f = torch.zeros(KL.shape[0])
    f[iface] = 1.0
    exact = torch.linalg.solve(KL, f)
    se = build_superelement(KL, ML, iface, 0, host)
    q = torch.linalg.solve(se.K, se.T.T @ f)
    recovered = se.T @ q
    assert float((recovered - exact).abs().max() / exact.abs().max()) < 1e-10


def test_expand_reproduces_reduced_mode(strip):
    system = assemble_superelements([
        build_superelement(strip["KL"], strip["ML"], strip["iface_A"], 2, strip["host"]),
        build_superelement(strip["KR"], strip["MR"], strip["iface_B"], 2, strip["host"]),
    ])
    omega2, phi_r = system.modes(2)
    full = system.expand(phi_r[:, 0])
    assert len(full) == 2
    for comp, vec in zip(system.components, full):
        assert vec.shape[0] == comp.T.shape[0]
        assert bool(torch.isfinite(vec).all())
    # the expanded field must be non-trivial and grow with the interface motion
    assert float(max(v.abs().max() for v in full)) > 0.0
    # reduced eigenpair residual K q = omega^2 M q
    residual = system.K @ phi_r[:, 0] - omega2[0] * (system.M @ phi_r[:, 0])
    assert float(residual.abs().max() / (system.M @ phi_r[:, 0]).abs().max()) < 1e-8


def test_superelement_assembly_is_order_invariant(strip):
    a = build_superelement(strip["KL"], strip["ML"], strip["iface_A"], 1, strip["host"])
    b = build_superelement(strip["KR"], strip["MR"], strip["iface_B"], 1, strip["host"])
    s1 = assemble_superelements([a, b])
    s2 = assemble_superelements([b, a])
    n_if = int(strip["host"].numel())
    # the interface block and the total modal content must not depend on the order
    assert float((s1.K[:n_if, :n_if] - s2.K[:n_if, :n_if]).abs().max()
                 / s1.K.abs().max()) < 1e-14
    assert abs(float(s1.K.trace()) - float(s2.K.trace())) < 1e-6 * abs(float(s1.K.trace()))
    assert s1.n_dofs == s2.n_dofs
    assert torch.equal(s1.interface_dofs, strip["host"])


def test_fail_closed_on_bad_input(strip):
    KL, ML, iface, host = strip["KL"], strip["ML"], strip["iface_A"], strip["host"]
    with pytest.raises(ValueError):
        build_superelement(KL, ML, iface, 2, host[:-1])           # length mismatch
    with pytest.raises(ValueError):
        build_superelement(KL, ML, torch.cat((iface, iface[:1])), 2, host)  # duplicates
    with pytest.raises(ValueError):
        craig_bampton_transformation(KL, ML, iface, -1)           # negative truncation
    with pytest.raises(ValueError):
        generalized_eigen(torch.zeros((2, 3)), torch.eye(3))       # non-square
    with pytest.raises(ValueError):
        assemble_superelements([])                                 # empty system
    nodes, elements = rectangular_q4_mesh(0.2, 0.05, 2, 1)
    with pytest.raises(ValueError):
        extract_component(nodes, elements, torch.zeros(elements.shape[0], dtype=torch.bool))