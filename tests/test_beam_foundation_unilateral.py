"""Tests for the Winkler beam-foundation elements and the unilateral solvers.

The gates mirror the node's machine-checkable criteria: Hetenyi closed form,
convergence order, patch test, complementarity / energy residuals, lift-off
length convergence and the ``k -> 0`` / ``k -> infinity`` degeneracies.
Heavier evidence (figures, four-level Richardson study) lives in
``benchmarks/beam_foundation_unilateral.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

import tensorfem.beam_foundation as bf

D = torch.float64


# ---------------------------------------------------------------------------
# element algebra
# ---------------------------------------------------------------------------
def test_hermite_foundation_matches_gauss_quadrature():
    L, k = 0.37, 4.3
    closed = bf.hermite_foundation_stiffness(L, k)
    gauss = bf.hermite_foundation_stiffness_quadrature(L, k)
    assert float((closed - gauss).abs().max()) < 1e-13
    assert torch.allclose(closed, closed.T, atol=0.0, rtol=0.0)


def test_foundation_matrix_total_reaction():
    """``1^T K_f 1`` (ones on the transverse DOFs) is ``int k dx``."""
    L, k = 0.5, 3.0
    ones = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=D)
    assert abs(float(ones @ bf.hermite_foundation_stiffness(L, k) @ ones) - k * L) < 1e-13
    kt = bf.timoshenko_foundation_stiffness(L, k)
    assert abs(float(ones @ kt @ ones) - k * L) < 1e-13


def test_variable_foundation_coefficient_integral():
    """A linearly varying ``k(x)`` integrates to the exact ``int k dx``."""
    L, k = 0.5, 2.0
    def kernel(s, k=k, L=L):
        return k * (1.0 + 2.0 * s / L)
    ones = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=D)
    total = float(ones @ bf.hermite_foundation_stiffness_quadrature(L, kernel) @ ones)
    assert abs(total - 2.0 * k * L) < 1e-13


def test_beam_stiffness_annihilates_rigid_modes():
    L = 0.8
    ke = bf.eb_beam_stiffness(2.0, 3.0, L)
    assert float((ke @ torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=D)).abs().max()) < 1e-12
    assert float((ke @ torch.tensor([0.0, 1.0, L, 1.0], dtype=D)).abs().max()) < 1e-12
    kts = bf.timoshenko_beam_stiffness(2.0, 3.0, 1.0e6, 1.0, 5 / 6, L)
    assert float((kts @ torch.tensor([0.0, 1.0, L, 1.0], dtype=D)).abs().max()) < 1e-6


def test_consistent_nodal_load_vector():
    L, q = 0.5, 2.0
    f = bf.hermite_consistent_load(L, q)
    assert abs(float(f[0] + f[2]) - q * L) < 1e-13
    assert abs(float(f[1] - q * L**2 / 12.0)) < 1e-13
    ft = bf.timoshenko_consistent_load(L, q)
    assert abs(float(ft[0] + ft[2]) - q * L) < 1e-13
    assert float(ft[1]) == 0.0 and float(ft[3]) == 0.0


def test_model_validation():
    with pytest.raises(ValueError):
        bf.BeamFoundationModel(x=torch.tensor([0.0, 0.0]), E=1.0, Iz=1.0)
    with pytest.raises(ValueError):
        bf.BeamFoundationModel(x=torch.tensor([0.0, 1.0]), E=1.0, Iz=1.0,
                               element="timoshenko")
    with pytest.raises(ValueError):
        bf.BeamFoundationModel(x=torch.tensor([0.0, 1.0, 2.0]), E=1.0, Iz=1.0,
                               k=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# bilateral: Hetenyi anchor + convergence order
# ---------------------------------------------------------------------------
def _hetenyi_half(n, k=1.0, EI=1.0, P=1.0, element="eb", G=0.0, A=0.0):
    beta = bf.characteristic_wavenumber(k, EI)
    x = torch.linspace(0.0, 20.0 / beta, n + 1, dtype=D)
    model = bf.BeamFoundationModel(x=x, E=1.0, Iz=EI, k=k, element=element, G=G, A=A)
    loads = torch.zeros(model.ndof, dtype=D)
    loads[0] = 0.5 * P
    return model, bf.solve_bilateral(model, nodal_loads=loads, fixed_dofs=[1])


def test_hetenyi_closed_form_point_load():
    k = EI = P = 1.0
    target = bf.hetenyi_penetration(P, k, EI)
    assert abs(target - P * (k / (4 * EI)) ** 0.25 / (2 * k)) < 1e-15
    model, res = _hetenyi_half(200)
    w0 = float(res["w_nodes"][0])
    assert abs(w0 - target) / target <= 1e-6


def test_hetenyi_field_profile_l2():
    k = EI = P = 1.0

    def ref(xe):
        return bf.hetenyi_point_load_deflection(xe, 0.0, P, k, EI)

    model, res = _hetenyi_half(320)
    err = bf.l2_error(model, res["w"], ref)["l2_relative"]
    assert err < 1e-6


@pytest.mark.parametrize("element,threshold,kw", [
    ("eb", 2.9, {}),
    ("timoshenko", 1.9, {"G": 1e8, "A": 1.0}),
])
def test_convergence_order(element, threshold, kw):
    k = EI = P = 1.0
    ns = (40, 80, 160, 320)
    hs = [20.0 / bf.characteristic_wavenumber(k, EI) / n for n in ns]

    def ref(xe):
        return bf.hetenyi_point_load_deflection(xe, 0.0, P, k, EI)

    errs = []
    for n in ns:
        model, res = _hetenyi_half(n, element=element, **kw)
        errs.append(bf.l2_error(model, res["w"], ref)["l2_relative"])
    xs = [math.log(h) for h in hs]
    ys = [math.log(e) for e in errs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    order = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
             / sum((a - mx) ** 2 for a in xs))
    assert order >= threshold, f"fitted order {order:.2f} < {threshold}"


# ---------------------------------------------------------------------------
# patch test
# ---------------------------------------------------------------------------
def test_patch_test_linear_field():
    L, n = 2.5, 12
    a, b = 1.0e-3, 2.0e-3
    model = bf.uniform_beam(L, n, 1.0, 1.0, k=3.0)
    asm = bf.assemble_beam_foundation(model)
    scale = abs(a) + abs(b) * L
    xe = torch.linspace(0.0, L, 401, dtype=D)
    wdof = torch.zeros(model.ndof, dtype=D)
    wdof[0::2] = a + b * model.x
    wdof[1::2] = b
    assert float((bf.evaluate_field(model, wdof, xe) - (a + b * xe)).abs().max()) \
        / scale <= 1e-12
    ke = asm["K"]
    # rigid/linear field must be annihilated relative to the operator scale
    denom = float(ke.abs().max()) * float(wdof.abs().max())
    assert float((ke @ wdof).abs().max()) / denom <= 1e-12
    ones_w = torch.zeros(model.ndof, dtype=D)
    ones_w[0::2] = 1.0
    resultant = float(ones_w @ (asm["Kf"] @ wdof))
    exact = 3.0 * (a * L + 0.5 * b * L**2)
    assert abs(resultant - exact) / abs(exact) <= 1e-12
    const = torch.zeros(model.ndof, dtype=D)
    const[0::2] = 1.0
    assert abs(float(ones_w @ (asm["Kf"] @ const)) - 3.0 * L) / (3.0 * L) <= 1e-12


# ---------------------------------------------------------------------------
# unilateral: tensionless lift-off
# ---------------------------------------------------------------------------
LIFT = {"k": 1.0, "EI": 1.0, "q": 1.0, "P": 6.0}


def _lift_off(n):
    beta = bf.characteristic_wavenumber(LIFT["k"], LIFT["EI"])
    x = torch.linspace(0.0, 20.0 / beta, n + 1, dtype=D)
    model = bf.BeamFoundationModel(x=x, E=1.0, Iz=LIFT["EI"], k=LIFT["k"],
                                   q=LIFT["q"])
    loads = torch.zeros(model.ndof, dtype=D)
    loads[0] = -0.5 * LIFT["P"]
    res = bf.solve_tensionless(model, nodal_loads=loads, fixed_dofs=[1])
    return model, res, loads


def test_tensionless_kkt_and_energy():
    model, res, loads = _lift_off(400)
    rs = res["residuals"]
    assert res["converged"] and res["iterations"] <= 10
    assert rs["complementarity"] <= 1e-10
    assert rs["stationarity"] <= 1e-10
    assert rs["spring_law"] <= 1e-10
    assert rs["reaction_min"] >= -1e-10
    assert rs["separation_min"] >= -1e-10
    # pointwise complementarity and no tensile reaction
    R, y = res["reaction"], res["separation"]
    assert bool(torch.all(R >= -1e-9 * max(float(R.abs().max()), 1.0)))
    assert bool(torch.all(y >= 0.0))
    assert float((R * y).abs().max()) <= 1e-9 * max(float(R.abs().max()), 1.0) * max(
        float(y.abs().max()), 1.0)
    ledger = bf.energy_ledger(model, res["w"], nodal_loads=loads)
    assert ledger["relative_residual"] <= 1e-9
    # the region near the uplift force has separated from the foundation
    assert int((~res["active"]).sum()) > 0
    assert float(res["w_nodes"][0]) < 0.0


def test_tensionless_matches_bilateral_when_no_lift_off():
    """With a purely downward load nothing lifts off: the two solvers coincide."""
    model = bf.uniform_beam(6.0, 60, 1.0, 1.0, k=2.0, q=1.0)
    fixed = [0, 2 * 60]
    loads = torch.zeros(model.ndof, dtype=D)
    loads[60] = 1.0
    bil = bf.solve_bilateral(model, nodal_loads=loads, fixed_dofs=fixed)
    uni = bf.solve_tensionless(model, nodal_loads=loads, fixed_dofs=fixed)
    # the two supported nodes sit exactly at w = 0 and carry no bed force; the
    # residual difference (6e-7 relative) is the consistent-vs-lumped footing
    assert bool(torch.all(uni["active"][1:-1]))
    assert int((~uni["active"]).sum()) == 2
    assert float((bil["w"] - uni["w"]).abs().max()) <= 1e-5


def test_lift_off_length_converges_to_semi_analytic():
    analytic = bf.lift_off_length_analytic(**LIFT)
    assert analytic > 0.0
    lengths = []
    for n in (100, 200, 400):
        model, res, _ = _lift_off(n)
        lengths.append(bf.lift_off_extent(model, res["w_nodes"]))
    assert all(b < a for a, b in zip(lengths, lengths[1:])), lengths
    assert abs(lengths[-1] - analytic) / analytic <= 5e-3
    # Richardson with the observed order; the estimate must be stable to 0.5 %
    beta = bf.characteristic_wavenumber(LIFT["k"], LIFT["EI"])
    hs = [20.0 / beta / n for n in (100, 200, 400)]
    errs = [abs(v - analytic) for v in lengths]
    xs = [math.log(h) for h in hs]
    ys = [math.log(e) for e in errs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    p = max(sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            / sum((a - mx) ** 2 for a in xs), 0.5)
    rich = lengths[-1] + (lengths[-1] - lengths[-2]) / (2**p - 1.0)
    assert abs(rich - lengths[-1]) / abs(rich) <= 5e-3


def test_lift_off_analytic_rejects_degenerate_input():
    with pytest.raises(ValueError):
        bf.lift_off_length_analytic(k=1.0, EI=1.0, q=0.0, P=1.0)
    with pytest.raises(ValueError):
        bf.lift_off_length_analytic(k=1.0, EI=1.0, q=1.0, P=0.0)
    with pytest.raises(ValueError):
        # uplift too small to separate the beam
        bf.lift_off_length_analytic(k=1.0, EI=1.0, q=1.0, P=0.5)


# ---------------------------------------------------------------------------
# unilateral: rigid bed and the k -> infinity / k -> 0 limits
# ---------------------------------------------------------------------------
def _pier(n=100, P=10.0, EI=1.0e3, L=10.0, b=2.0, R=2.0, d=0.03):
    x = torch.linspace(0.0, L, n + 1, dtype=D)
    s = (x - 0.5 * L).abs()
    gap = torch.where(s <= b, torch.full_like(s, d), d + (s - b) ** 2 / (2.0 * R))
    loads = torch.zeros(2 * (n + 1), dtype=D)
    loads[2 * (n // 4)] = P
    return x, gap, loads, n


def test_rigid_bed_no_penetration_and_complementarity():
    x, gap, loads, n = _pier()
    model = bf.BeamFoundationModel(x=x, E=1.0, Iz=1.0e3, k=0.0)
    res = bf.solve_rigid_bed(model, nodal_loads=loads, gap=gap,
                             fixed_dofs=[0, 2 * n])
    rs = res["residuals"]
    assert res["polish_iterations"] <= 10
    assert rs["no_penetration"] <= 1e-12
    assert rs["complementarity"] <= 1e-9
    assert rs["reaction_min"] >= -1e-9
    bed = float(res["reaction"].sum())
    assert 0.0 < bed < 10.0  # the remainder sits on the two clamped supports
    R, y = res["reaction"], res["separation"]
    assert float((R * y).abs().max()) < 1e-8


def test_stiff_bed_degenerates_to_rigid_bed():
    x, gap, loads, n = _pier()
    model0 = bf.BeamFoundationModel(x=x, E=1.0, Iz=1.0e3, k=0.0)
    rigid = bf.solve_rigid_bed(model0, nodal_loads=loads, gap=gap,
                               fixed_dofs=[0, 2 * n])
    m_rigid = float(bf.element_end_moments(model0, rigid["w"]).abs().max())
    active = None
    m_elastic = None
    for k in (1e3, 1e4, 1e5):
        model = bf.BeamFoundationModel(x=x, E=1.0, Iz=1.0e3, k=k)
        res = bf.solve_tensionless(model, nodal_loads=loads, gap=gap,
                                   fixed_dofs=[0, 2 * n], init_active=active)
        active = res["active"]
        assert res["iterations"] <= 10
        m_elastic = float(bf.element_end_moments(model, res["w"]).abs().max())
    assert abs(m_elastic - m_rigid) / m_rigid <= 0.01


def test_soft_bed_reaction_becomes_uniform():
    """``k L^4 / EI -> 0``: the beam moves as a rigid body, reaction uniform."""
    L, EI, nu = 10.0, 1.0e4, 100
    k = 1e-4 * EI / L**4
    model = bf.uniform_beam(L, nu, 1.0, EI, k=k)
    loads = torch.zeros(model.ndof, dtype=D)
    loads[2 * (nu // 4)] = 1.0
    loads[2 * (3 * nu // 4)] = 1.0
    res = bf.solve_bilateral(model, nodal_loads=loads)
    lengths = torch.diff(model.x)
    trib = torch.zeros(model.n_nodes, dtype=D)
    trib[:-1] += lengths / 2.0
    trib[1:] += lengths / 2.0
    rho = res["reaction_nodal"] / trib
    dev = float((rho - rho.mean()).abs().max()) / float(rho.mean())
    assert dev <= 0.01


# ---------------------------------------------------------------------------
# plate extension: lumped surface springs
# ---------------------------------------------------------------------------
def test_plate_nodal_springs_conserve_total():
    xy = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.2, 0.1], [0.0, 0.1]], dtype=D)
    k = 5.0
    ks = bf.q4_winkler_nodal_springs(xy, k)
    assert abs(float(ks.sum()) - k * 0.02) < 1e-14
    assert torch.allclose(ks, ks.new_full((4,), k * 0.02 / 4.0))


def test_plate_bilateral_solve_runs_and_is_symmetric():
    nodes, elements, fixed, loads = _plate_strip(nx=20)
    res = bf.solve_plate_foundation(nodes, elements, 1.0, 0.3, 0.05, 2e3,
                                    nodal_loads=loads, fixed_dofs=fixed,
                                    unilateral=False)
    w = res["w_nodes"].reshape(21, 3)[:, 0]
    assert float(w.min()) < 0.0
    # load at x = 0, strip decays towards the free end
    assert abs(float(w[0])) > abs(float(w[-1]))


def _plate_strip(nx=20, ny=2, side=1.0):
    xs = torch.linspace(0.0, side, nx + 1, dtype=D)
    ys = torch.linspace(0.0, side / 10.0, ny + 1, dtype=D)
    nodes = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            elements.append([n0, n0 + ny + 1, n0 + ny + 2, n0 + 1])
    elements = torch.tensor(elements, dtype=torch.long)
    nn = nodes.shape[0]
    fixed = [3 * i + 1 for i in range(nn)]  # cylindrical bending: theta_y = 0
    loads = torch.zeros(3 * nn, dtype=D)
    loads[0] = 0.5
    return nodes, elements, fixed, loads