"""CASE-CANTILEVER-TIP distributed-load ladder -- gates and fail-closed checks.

Layers gated here:

* the oracles themselves (symbolic unit-load residual, symbolic clamped-clamped
  Timoshenko element solution, ``phi``-independence of the fixed-end forces);
* the load path that the published tip-load rungs never touch
  (``FrameModel.distributed_load_y`` and ``timoshenko_fixed_end_forces``) --
  nodal exactness against the oracle valid for each model;
* the falsifiable pair at ``L/h = 5`` where the two beam theories separate by
  more than the 3 % case-pool tolerance: the shear-rigid frame *must* be
  rejected there and the shear-flexible beam must pass;
* the generated artifact: every gate in ``gates_verified.json`` passes and its
  value matches ``result.json``.
"""
import json
import math
from pathlib import Path

import pytest

from tensorfem.advanced_beam import (
    exact_distributed_tip_deflection,
    exact_partial_tip_deflection,
    exact_tip_deflection,
    solve_timoshenko_cantilever,
    timoshenko_fixed_end_forces,
    timoshenko_stiffness,
)
from tensorfem.frame2d import FrameModel, solve_frame_static

E = 200.0e9
NU = 0.3
B = 0.01
H = 0.01
KAPPA = 5.0 / 6.0
G = E / (2.0 * (1.0 + NU))
A = B * H
I = B * H**3 / 12.0
Q = 5.0
TOL = 0.03
ANCHOR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "fem_cantilever_distributed"


def frame(length, n, *, q=None, partial_fraction=None, tip_moment=0.0, loaded_elements=None):
    import torch

    x = torch.linspace(0.0, length, n + 1, dtype=torch.float64)
    nodes = torch.stack([x, torch.zeros_like(x)], dim=1)
    elements = torch.stack([torch.arange(n), torch.arange(1, n + 1)], dim=1)
    forces = torch.zeros(3 * (n + 1), dtype=torch.float64)
    if tip_moment:
        forces[3 * n + 2] = tip_moment
    qs = None
    if q is not None:
        mids = 0.5 * (x[:-1] + x[1:])
        if loaded_elements is not None:
            mask = torch.zeros(n, dtype=torch.bool)
            for idx in loaded_elements:
                mask[idx] = True
        else:
            frac = 1.0 if partial_fraction is None else partial_fraction
            mask = mids <= frac * length + 1e-12
        qs = torch.where(mask, torch.tensor(float(q), dtype=torch.float64),
                         torch.tensor(0.0, dtype=torch.float64))
    model = FrameModel(
        nodes=nodes, elements=elements,
        young_modulus=torch.tensor([E], dtype=torch.float64),
        area=torch.tensor([A], dtype=torch.float64),
        inertia=torch.tensor([I], dtype=torch.float64),
        forces=forces, fixed_dofs=torch.tensor([0, 1, 2]),
        distributed_load_y=qs,
    )
    result = solve_frame_static(model)
    return {
        "tip_deflection": float(result.displacement[3 * n + 1]),
        "tip_rotation": float(result.displacement[3 * n + 2]),
        "reaction": [float(v) for v in result.reaction[:3]],
        "root_end_forces": [float(v) for v in result.element_end_forces_local[0]],
    }


# --------------------------------------------------------------------------- #
# Oracles
# --------------------------------------------------------------------------- #
def test_udl_oracle_has_shear_term_and_rotation_does_not():
    length = 10.0 * H
    bend = Q * length**4 / (8 * E * I)
    shear = Q * length**2 / (2 * KAPPA * G * A)
    assert exact_distributed_tip_deflection(length, Q, E, I, G, A, KAPPA) == pytest.approx(
        bend + shear, rel=1e-14)
    assert shear / bend == pytest.approx((E / (3 * KAPPA * G)) * (H / length) ** 2, rel=1e-12)


def test_partial_oracle_reduces_to_udl_at_full_span():
    length = 10.0 * H
    full = Q * length**4 / (8 * E * I)
    assert exact_partial_tip_deflection(length, length, Q, E, I, G, A, KAPPA) == pytest.approx(
        full + Q * length**2 / (2 * KAPPA * G * A), rel=1e-14)


def test_symbolic_oracles_are_self_consistent():
    """Castigliano residuals of all closed forms must vanish symbolically."""
    import sympy as sp

    x, L, EI, kGA, q, mm, a = sp.symbols("x L EI kGA q M a", positive=True)
    M_udl = q * (L - x) ** 2 / 2
    udl_delta = (sp.integrate(M_udl * (L - x) / EI, (x, 0, L))
                 + sp.integrate(q * (L - x) / kGA, (x, 0, L))
                 - (q * L**4 / (8 * EI) + q * L**2 / (2 * kGA)))
    udl_theta = (sp.integrate(M_udl / EI, (x, 0, L)) - q * L**3 / (6 * EI))
    M_par = q * (a - x) ** 2 / 2
    par_delta = (sp.integrate(M_par * (L - x) / EI, (x, 0, a))
                 + sp.integrate(q * (a - x) / kGA, (x, 0, a))
                 - (q * a**3 * (4 * L - a) / (24 * EI) + q * a**2 / (2 * kGA)))
    mom_theta = sp.integrate(mm / EI, (x, 0, L)) - mm * L / EI
    for residual in (udl_delta, udl_theta, par_delta, mom_theta):
        assert sp.simplify(residual) == 0


def test_fixed_end_forces_carry_no_shear_flexibility():
    """Clamped-clamped Timoshenko element: end forces are the Euler-Bernoulli ones."""
    import sympy as sp

    x, L, EI, kGA, q = sp.symbols("x L EI kGA q", positive=True)
    c1, aa, bb, dd = sp.symbols("c1 A B D")
    V = c1 - q * x
    theta = (sp.integrate(sp.integrate(V, x), x) + aa * x + bb) / EI
    w = sp.integrate(theta + V / kGA, x) + dd
    sol = sp.solve([sp.Eq(theta.subs(x, 0), 0), sp.Eq(w.subs(x, 0), 0),
                    sp.Eq(theta.subs(x, L), 0), sp.Eq(w.subs(x, L), 0)],
                   [aa, bb, c1, dd], dict=True)[0]
    shear_i = sp.simplify(V.subs(sol).subs(x, 0))
    moment_i = sp.simplify(EI * sp.diff(theta, x).subs(sol).subs(x, 0))
    assert sp.simplify(shear_i - q * L / 2) == 0
    assert sp.simplify(moment_i + q * L**2 / 12) == 0
    # the shear flexibility phi = 12 EI/(kappa G A L^2) must not appear
    assert not moment_i.has(EI)


def test_fixed_end_force_vector_matches_the_eb_vector():
    length = 0.5
    vec = timoshenko_fixed_end_forces(Q, length).tolist()
    assert vec == pytest.approx([Q * length / 2, Q * length**2 / 12,
                                 Q * length / 2, -Q * length**2 / 12], rel=1e-15)


def test_fixed_end_force_vector_is_fail_closed():
    """A flipped fixed-end moment must break nodal exactness -- otherwise the gate is blind."""
    import torch

    length = 10.0 * H
    exact = exact_distributed_tip_deflection(length, Q, E, I, G, A, KAPPA)
    ndof = 2 * 5
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    ke = timoshenko_stiffness(E, I, G, A, KAPPA, length / 4)
    for e in range(4):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += ke
    for sign in (+1, -1):
        fe = timoshenko_fixed_end_forces(Q, length / 4) * torch.tensor(
            [1.0, sign, 1.0, sign], dtype=torch.float64)
        f = torch.zeros(ndof, dtype=torch.float64)
        for e in range(4):
            f[2 * e:2 * e + 4] += fe
        free = torch.arange(2, ndof)
        u = torch.zeros_like(f)
        u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
        err = abs(float(u[-2]) - exact) / exact
        if sign > 0:
            assert err < 1e-10, "consistent load vector must be nodally exact"
        else:
            assert err > TOL, "flipped fixed-end moment must be rejected"


# --------------------------------------------------------------------------- #
# Timoshenko beam under a distributed load (new load path)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("slenderness", [100.0, 5.0])
def test_timoshenko_distributed_load_is_nodally_exact(slenderness):
    length = slenderness * H
    exact = exact_distributed_tip_deflection(length, Q, E, I, G, A, KAPPA)
    for n in (1, 2, 4, 8):
        u = solve_timoshenko_cantilever(length, n, E, I, G, A, 0.0, KAPPA,
                                        distributed_load=Q)
        assert abs(float(u.displacement[-2]) - exact) / exact < 1e-10


def test_timoshenko_distributed_load_rotation_is_shear_free():
    """At L/h = 5 the shear share of the deflection is 4.2 %, the rotation has none."""
    length = 5.0 * H
    u = solve_timoshenko_cantilever(length, 4, E, I, G, A, 0.0, KAPPA, distributed_load=Q)
    rotation = Q * length**3 / (6 * E * I)
    assert abs(float(u.displacement[-1]) - rotation) / rotation < 1e-10
    bend = Q * length**4 / (8 * E * I)
    assert Q * length**2 / (2 * KAPPA * G * A) / bend > TOL   # the shear term is above tolerance
    assert float(u.displacement[-2]) == pytest.approx(bend + Q * length**2 / (2 * KAPPA * G * A),
                                                      rel=1e-10)


def test_timoshenko_partial_load_is_nodally_exact():
    length = 10.0 * H
    a = 0.6 * length
    exact = exact_partial_tip_deflection(length, a, Q, E, I, G, A, KAPPA)
    per_element = [Q, Q, Q, 0.0, 0.0]
    u = solve_timoshenko_cantilever(length, 5, E, I, G, A, 0.0, KAPPA,
                                    distributed_load=per_element)
    assert abs(float(u.displacement[-2]) - exact) / exact < 1e-10


def test_tip_load_path_is_unchanged():
    """Backwards compatibility of the pre-existing tip-load signature."""
    length = 1.0
    u = solve_timoshenko_cantilever(length, 4, E, I, G, A, -Q, KAPPA)
    exact = exact_tip_deflection(length, E, I, G, A, Q, KAPPA)
    assert abs(-float(u.displacement[-2]) - exact) / exact < 1e-10
    # shear-rigid limit: kappa -> large must approach the Euler-Bernoulli answer
    u_rigid = solve_timoshenko_cantilever(length, 4, E, I, G, A, -Q, 1e9)
    assert abs(-float(u_rigid.displacement[-2]) - Q * length**3 / (3 * E * I)) / (
        Q * length**3 / (3 * E * I)) < 1e-8


def test_timoshenko_distributed_load_rejects_bad_input():
    with pytest.raises(ValueError):
        solve_timoshenko_cantilever(1.0, 4, E, I, G, A, 0.0, KAPPA, distributed_load=[1.0, 2.0])


# --------------------------------------------------------------------------- #
# Euler-Bernoulli frame under a distributed load (untested path before)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_frame_uniform_load_is_nodally_exact(n):
    length = 1.0
    exact = Q * length**4 / (8 * E * I)
    out = frame(length, n, q=Q)
    assert abs(out["tip_deflection"] - exact) / exact < 1e-10
    rotation = Q * length**3 / (6 * E * I)
    assert abs(out["tip_rotation"] - rotation) / rotation < 1e-10


def test_frame_reactions_and_end_forces_recover_the_exact_root_forces():
    length = 5.0 * H
    out = frame(length, 8, q=Q)
    assert abs(out["reaction"][1] + Q * length) / (Q * length) < 1e-10     # root shear
    assert abs(out["reaction"][2] + Q * length**2 / 2) / (Q * length**2 / 2) < 1e-10
    # end force recovery subtracts the fixed-end forces (equivalent_local)
    assert abs(abs(out["root_end_forces"][2]) - Q * length**2 / 2) / (Q * length**2 / 2) < 1e-10


def test_frame_partial_load_needs_the_mesh_to_break_at_the_boundary():
    length = 10.0 * H
    a = 0.6 * length
    exact = Q * a**3 * (4 * length - a) / (24 * E * I)
    conforming = frame(length, 5, q=Q, partial_fraction=0.6)
    assert abs(conforming["tip_deflection"] - exact) / exact < 1e-10
    nonconforming = frame(length, 4, q=Q, loaded_elements=[2])
    assert abs(nonconforming["tip_deflection"] - exact) / exact > 1e-2   # measured cost


def test_frame_tip_moment_is_exact():
    length = 10.0 * H
    out = frame(length, 4, tip_moment=5.0)
    assert abs(out["tip_rotation"] - 5.0 * length / (E * I)) / (5.0 * length / (E * I)) < 1e-10
    assert abs(out["tip_deflection"] - 5.0 * length**2 / (2 * E * I)) / (
        5.0 * length**2 / (2 * E * I)) < 1e-10


# --------------------------------------------------------------------------- #
# Artifact consistency
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (ANCHOR / "gates_verified.json").exists(),
                    reason="anchor artifacts not generated in this checkout")
def test_anchor_gates_all_pass_and_match_the_evidence():
    manifest = json.loads((ANCHOR / "gates_verified.json").read_text())
    result = json.loads((ANCHOR / "result.json").read_text())
    assert manifest["schema"] == "tensorfem.verified-gates.v1"
    assert manifest["anchor"] == "fem_cantilever_distributed"
    assert manifest["mission"] == "CASE-CANTILEVER-TIP"
    assert manifest["passed"] is True, manifest["failed_gates"]
    assert manifest["failed_gates"] == []
    assert not manifest["undefined_terms"]
    by_id = {g["id"]: g for g in result["gates"]}
    for gate in manifest["gates"]:
        assert gate["passed"], gate["id"]
        assert math.isfinite(gate["value"])
        assert by_id[gate["id"]]["value"] == pytest.approx(gate["value"], rel=1e-12, abs=0.0)
        assert gate["operator"] in ("<=", ">=")
        if gate["operator"] == "<=":
            assert gate["value"] <= gate["threshold"], gate["id"]
        else:
            assert gate["value"] >= gate["threshold"], gate["id"]


@pytest.mark.skipif(not (ANCHOR / "result.json").exists(),
                    reason="anchor artifacts not generated in this checkout")
def test_evidence_is_reproducible():
    """Recompute the driver's headline numbers and compare against the artifact."""
    result = json.loads((ANCHOR / "result.json").read_text())
    for row in result["ladder"]:
        length = row["length"]
        assert row["delta_eb"] == pytest.approx(Q * length**4 / (8 * E * I), rel=1e-13)
        assert row["oracle_gap"] == pytest.approx(
            Q * length**2 / (2 * KAPPA * G * A) / (Q * length**4 / (8 * E * I)), rel=1e-12)
        out = frame(length, 1, q=Q)
        assert out["tip_deflection"] == pytest.approx(row["frame"]["1"]["tip_deflection"], rel=1e-12)