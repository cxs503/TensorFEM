"""Gates for ``CASE-THERMAL-BAR``: constrained thermal stress ``sigma = E alpha dT``.

These tests re-derive the reference and recompute the gates on small meshes, so
they fail if the thermal-strain load path, the restraint bookkeeping or the
stress recovery regresses.  Tolerance policy (predeclared in
:mod:`tensorfem.constrained_thermal_bar`):

* ``TASK_REL_TOL = 3%`` -- the benchmark-pool acceptance criterion;
* ``ROUNDOFF_REL_TOL = 1e-8`` -- used for configurations whose exact solution is
  representable by the discretisation (uniform ``dT``, restrained), where
  anything above float64 round-off means a bug;
* ``ORDER_TOL = 0.35`` -- observed h-convergence order of the quadratic-``dT``
  variant versus the theoretical ``2``.
"""
import math

import pytest
import torch

from tensorfem import constrained_thermal_bar as ctb

T_REF = 20.0
DT = 100.0


@pytest.fixture(scope="module")
def steel():
    return ctb.constant_steel("EH36", T_ref=T_REF)


# ---------------------------------------------------------------------------
# reference / derivation
# ---------------------------------------------------------------------------
def test_closed_form_value(steel):
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    sigma = ctb.restrained_uniaxial_stress(E, alpha, DT)
    # sign: heating against restraint -> compression
    assert sigma < 0.0
    assert sigma == pytest.approx(-float(E) * float(alpha) * DT, rel=0.0, abs=0.0)
    assert ctb.free_expansion_strain(alpha, DT) > 0.0


def test_derivation_residual_is_machine_zero(steel):
    for restraint in ("uniaxial", "biaxial", "triaxial"):
        r = ctb.derivation_residual(steel.E(T_REF), steel.alpha(T_REF), DT,
                                    steel.nu, restraint)
        assert abs(r) < 1e-15, (restraint, r)


def test_biaxial_and_triaxial_are_stiffer_than_uniaxial(steel):
    E, alpha, nu = steel.E(T_REF), steel.alpha(T_REF), steel.nu
    uni = abs(ctb.restrained_uniaxial_stress(E, alpha, DT))
    bia = abs(ctb.restrained_biaxial_stress(E, alpha, DT, nu))
    tri = abs(ctb.restrained_triaxial_stress(E, alpha, DT, nu))
    assert uni < bia < tri
    assert bia == pytest.approx(uni / (1.0 - nu), rel=1e-14)
    assert tri == pytest.approx(uni / (1.0 - 2.0 * nu), rel=1e-14)


# ---------------------------------------------------------------------------
# 1-D path (reuse of tensorfem.thermal.solve_thermoelastic_bar)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_elem", [1, 2, 4, 8, 16])
def test_bar_uniform_dT_matches_closed_form(steel, n_elem):
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    ref = ctb.restrained_uniaxial_stress(E, alpha, DT)
    res = ctb.solve_bar(n_elem, E, alpha, DT, T_ref=T_REF, length=1.0)
    rel = ctb.rel_error(float(res.sigma.mean()), ref)
    assert abs(rel) < ctb.TASK_REL_TOL, rel
    assert abs(rel) < ctb.ROUNDOFF_REL_TOL, rel
    # the restrained bar must not move, and the stress must be element-uniform
    assert float(res.displacement.abs().max()) < 1e-15
    assert float(res.sigma.max() - res.sigma.min()) < 1e-6


def test_bar_reactions_are_self_equilibrated(steel):
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    res = ctb.solve_bar(4, E, alpha, DT, T_ref=T_REF, length=1.0)
    assert abs(float(res.reaction.sum())) < 1e-6 * abs(float(res.sigma.mean()))


def test_bar_free_expansion_is_stress_free(steel):
    """Control: with one end released the same thermal load produces no stress."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    res = ctb.solve_bar(4, E, alpha, DT, T_ref=T_REF, length=1.0, fixed="one")
    assert float(res.sigma.abs().max()) < 1e-6 * abs(
        ctb.restrained_uniaxial_stress(E, alpha, DT))
    assert float(res.displacement[-1]) == pytest.approx(
        ctb.free_expansion_strain(alpha, DT) * 1.0, rel=1e-12)


def test_bar_quadratic_dT_converges_second_order(steel):
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    ref = ctb.restrained_uniaxial_stress(E, alpha, DT) / 3.0
    errors, hs = [], []
    for n in (2, 4, 8, 16, 32):
        res = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=1.0,
                            temperature="quadratic")
        errors.append(abs(ctb.rel_error(float(res.sigma.mean()), ref)))
        hs.append(1.0 / n)
    order = ctb.observed_order(errors, hs)
    assert abs(order - 2.0) <= ctb.ORDER_TOL, order
    assert errors[-1] < errors[0]


# ---------------------------------------------------------------------------
# 3-D path (reuse of the hex8 thermo-elastic assembly)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("restraint,reference_key", [
    ("uniaxial", "uniaxial"),
    ("biaxial", "biaxial"),
    ("triaxial", "triaxial"),
])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_hex_restraint_family_matches_closed_form(steel, restraint, reference_key, n):
    mesh = ctb.hex_mesh(n, (0.2, 0.2, 0.2))
    res = ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint=restraint)
    s = res.sigma.mean(dim=0)
    rel = ctb.rel_error(float(s[0]), res.exact_sigma)
    assert abs(rel) < ctb.TASK_REL_TOL, (restraint, n, rel)
    assert abs(rel) < ctb.ROUNDOFF_REL_TOL, (restraint, n, rel)
    # no shear anywhere: the field is a pure volumetric thermal strain
    assert float(s[3:].abs().max()) < 1e-6


def test_hex_free_expansion_control(steel):
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    mesh = ctb.hex_mesh(2, (0.2, 0.2, 0.2))
    res = ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint="free")
    assert float(res.sigma.abs().max()) < ctb.ROUNDOFF_REL_TOL * E * alpha * DT
    # free expansion: u_x(x) = alpha dT x
    X = mesh.nodes
    assert torch.allclose(res.displacement[:, 0], alpha * DT * X[:, 0],
                          atol=1e-12)


def test_hex_uniaxial_is_poisson_invariant(steel):
    from dataclasses import replace

    mesh = ctb.hex_mesh(2, (0.2, 0.2, 0.2))
    values = []
    for nu in (0.0, 0.2, 0.3, 0.45):
        res = ctb.solve_hex(mesh, replace(steel, nu=nu), DT, T_ref=T_REF,
                            restraint="uniaxial")
        values.append(float(res.sigma.mean(dim=0)[0]))
    drift = max(values) - min(values)
    assert abs(drift) < ctb.ROUNDOFF_REL_TOL * abs(values[0]), drift


def test_hex_uniaxial_lateral_stress_vanishes(steel):
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    mesh = ctb.hex_mesh(2, (0.2, 0.2, 0.2))
    res = ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint="uniaxial")
    s = res.sigma.mean(dim=0)
    scale = E * alpha * DT
    assert abs(float(s[1])) < ctb.ROUNDOFF_REL_TOL * scale
    assert abs(float(s[2])) < ctb.ROUNDOFF_REL_TOL * scale


def test_hex_temperature_dependent_material_uses_element_temperature():
    tdep = ctb.default_steel("EH36", T_ref=T_REF)
    mesh = ctb.hex_mesh(2, (0.2, 0.2, 0.2))
    res = ctb.solve_hex(mesh, tdep, DT, T_ref=T_REF, restraint="uniaxial")
    sxx = float(res.sigma.mean(dim=0)[0])
    assert abs(ctb.rel_error(sxx, res.exact_sigma)) < ctb.ROUNDOFF_REL_TOL
    # the reference must be the hot-temperature evaluation, not the frozen one
    frozen = ctb.restrained_uniaxial_stress(tdep.E(T_REF), tdep.alpha(T_REF), DT)
    assert abs(sxx - frozen) > 1e-3 * abs(frozen)


# ---------------------------------------------------------------------------
# restraint bookkeeping
# ---------------------------------------------------------------------------
def test_restraint_dofs_remove_all_rigid_body_modes(steel):
    """The constrained stiffness block must be positive definite, no zero mode."""
    from tensorfem.polar_thermal_stress import (assemble_elasticity, solve_dirichlet,
                                                structured_hex_mesh)

    mesh = structured_hex_mesh(2, 2, 2, 0.2, 0.2, 0.2)
    K = assemble_elasticity(mesh, torch.full((mesh.n_elements,), steel.E(T_REF),
                                             dtype=torch.float64), steel.nu)
    for restraint in ("uniaxial", "biaxial", "triaxial", "free"):
        fixed = ctb.restraint_dofs(mesh, restraint)
        mask = torch.ones(3 * mesh.n_nodes, dtype=torch.bool)
        mask[fixed] = False
        free = torch.arange(3 * mesh.n_nodes)[mask]
        Kff = K[free][:, free]
        eig = torch.linalg.eigvalsh(Kff)
        assert float(eig.min()) > 0.0, (restraint, float(eig.min()))
    # the restraint family is ordered: free < uniaxial <= biaxial < triaxial
    counts = [ctb.restraint_dofs(mesh, r).numel()
              for r in ("free", "uniaxial", "biaxial", "triaxial")]
    assert counts == sorted(counts) and counts[0] < counts[-1], counts


def test_evidence_artifacts_agree_when_present():
    """If the benchmark artefacts exist, their headline numbers must match this test."""
    import json
    from pathlib import Path

    ev = Path(__file__).resolve().parents[1] / "benchmarks" / \
        "constrained_thermal_bar_evidence.json"
    if not ev.exists():
        pytest.skip("evidence JSON not generated yet")
    data = json.loads(ev.read_text())
    steel = ctb.constant_steel("EH36", T_ref=T_REF)
    ref = ctb.restrained_uniaxial_stress(steel.E(T_REF), steel.alpha(T_REF), DT)
    assert data["reference"]["value_Pa"] == pytest.approx(ref, rel=1e-15)
    assert data["verdict"]["task_gate_3pct"] is True
    assert math.isclose(data["gates"]["task_rel_tol"], ctb.TASK_REL_TOL)