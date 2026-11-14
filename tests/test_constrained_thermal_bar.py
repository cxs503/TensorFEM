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
    # round-2 gates must be present and green in the published evidence
    for key in ("graded_model_gate", "graded_order_gate",
                "graded_discrimination_gate", "gate_strength_gate"):
        assert data["verdict"][key] is True, key
    assert data["gates"]["graded_model_roundoff_max"] < ctb.ROUNDOFF_REL_TOL
    assert data["gates"]["graded_task_gate_fail_levels_3d"], \
        "the graded rung must contain at least one cell that fails the 3% gate"
    assert math.isclose(data["gates"]["task_rel_tol"], ctb.TASK_REL_TOL)


def test_gate_strength_probe_catches_the_node_sampling_mutation():
    """Mutation B must pass every uniform gate and be caught by the graded rung."""
    import importlib.util
    import sys
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "benchmarks" / \
        "constrained_thermal_bar.py"
    if not bench.exists():  # pragma: no cover
        pytest.skip("benchmark driver not present")
    spec = importlib.util.spec_from_file_location("ctb_bench_probe", bench)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ctb_bench_probe"] = mod
    try:
        spec.loader.exec_module(mod)
        probe = mod.gate_strength_probe(ctb.constant_steel("EH36", T_ref=T_REF))
    finally:
        sys.modules.pop("ctb_bench_probe", None)
    assert probe["A_caught_on_uniform"] is True
    assert probe["B_invisible_on_uniform"] is True, probe
    assert probe["B_caught_on_graded"] is True, probe
    assert probe["gate"] is True



# ---------------------------------------------------------------------------
# graded dT: the rung on which the 3% criterion discriminates
# ---------------------------------------------------------------------------
GRADED_LEVELS = (2, 4, 8, 16)


def test_graded_bias_identity(steel):
    """``graded_field_stress`` must reproduce the analytic trapezoid bias 1/(2 n^2)."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    for n in GRADED_LEVELS:
        refs = ctb.graded_field_stress(E, alpha, DT, n, 1.0)
        assert refs["predicted_bias_rel"] == pytest.approx(ctb.graded_bias(n), rel=1e-15)
        assert refs["bias_rel"] == pytest.approx(ctb.graded_bias(n), rel=1e-14)
        assert refs["continuum_Pa"] == pytest.approx(-E * alpha * DT / 3.0, rel=1e-15)
    # the bias is a temperature-interpolation effect: second order, so 1/(2n^2)
    assert ctb.graded_bias(4) == pytest.approx(1.0 / 32.0, rel=1e-15)


def test_graded_bar_and_hex_match_element_average_model(steel):
    """Both paths must reproduce ``-E alpha <dT>_trap`` to round-off.

    This is the gate that catches an assembly / thermal-load / restraint bug on a
    *graded* field; the uniform rung cannot distinguish those from round-off.
    """
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    for n in GRADED_LEVELS:
        bar = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=1.0,
                            temperature="quadratic")
        model = ctb.graded_field_stress(E, alpha, DT, n, 1.0)["element_average_Pa"]
        assert abs(ctb.rel_error(float(bar.sigma.mean()), model)) < 1e-12
    for nx in (2, 4, 8):
        hexr = ctb.solve_hex(ctb.graded_mesh(nx, 1.0, 0.10), steel, DT, T_ref=T_REF,
                             restraint="uniaxial", temp_profile="graded")
        # the model is rebuilt from the nodal geometry, not from the solver's own
        # element temperatures, so a wrong element sampling cannot agree with it
        model = ctb.graded_field_stress(E, alpha, DT, nx, 1.0)["element_average_Pa"]
        assert abs(ctb.rel_error(float(hexr.sigma.mean(dim=0)[0]), model)) < 1e-12
        assert hexr.model_sigma == pytest.approx(model, rel=1e-14)
        # axial stress is uniform over the graded block, and equals the 1-D answer
        bar = ctb.solve_bar(nx, E, alpha, DT, T_ref=T_REF, length=1.0,
                            temperature="quadratic")
        assert abs(ctb.rel_error(float(hexr.sigma.mean(dim=0)[0]),
                                 float(bar.sigma.mean()))) < 1e-12


def test_graded_task_gate_discriminates_on_the_predeclared_mesh_window(steel):
    """The 3% criterion must fail on coarse graded meshes and pass on fine ones.

    Predeclared from the analytic bias ``1/(2 n**2)`` (see the module docstring):
    3% is exceeded exactly for ``n <= GRADED_COARSE_MAX`` and met for
    ``n >= GRADED_FINE_MIN``.  This is what gives the acceptance tolerance its
    meaning -- on the uniform rung the discrete answer is exact, so no tolerance
    can bite.
    """
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    for n in GRADED_LEVELS:
        res = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=1.0,
                            temperature="quadratic")
        err = abs(ctb.rel_error(float(res.sigma.mean()),
                                ctb.graded_field_stress(E, alpha, DT, n, 1.0)["continuum_Pa"]))
        if n <= ctb.GRADED_COARSE_MAX:
            assert err > ctb.TASK_REL_TOL, (n, err)
        elif n >= ctb.GRADED_FINE_MIN:
            assert err < ctb.TASK_REL_TOL, (n, err)


def test_graded_error_is_second_order(steel):
    """The graded stress error must fall at order 2 on both paths."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    errors, hs = [], []
    for n in (4, 8, 16):
        res = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=1.0,
                            temperature="quadratic")
        ref = ctb.graded_field_stress(E, alpha, DT, n, 1.0)["continuum_Pa"]
        errors.append(abs(ctb.rel_error(float(res.sigma.mean()), ref)))
        hs.append(1.0 / n)
    assert abs(ctb.observed_order(errors, hs) - 2.0) < ctb.ORDER_TOL

    errors3, hs3 = [], []
    for nx in (2, 4, 8):
        res = ctb.solve_hex(ctb.graded_mesh(nx, 1.0, 0.10), steel, DT, T_ref=T_REF,
                            restraint="uniaxial", temp_profile="graded")
        sxx = float(res.sigma.mean(dim=0)[0])
        errors3.append(abs(ctb.rel_error(sxx, res.exact_sigma)))
        hs3.append(1.0 / nx)
    assert abs(ctb.observed_order(errors3, hs3) - 2.0) < ctb.ORDER_TOL


def test_graded_profile_requires_uniaxial_restraint(steel):
    """A graded field has no closed-form lateral stress, so it is not claimed."""
    mesh = ctb.graded_mesh(2, 1.0, 0.10)
    with pytest.raises(ValueError):
        ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint="biaxial",
                      temp_profile="graded")
    with pytest.raises(ValueError):
        ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, temp_profile="linear-ish")
