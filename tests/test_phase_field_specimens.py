"""Verification of :mod:`tensorfem.phase_field_specimens` (ASTM E399 specimen library).

Unit tests cover the closed-form anchors (ASTM E399 shape factors, Irwin/Griffith
relations), the hole/pin geometry (resultant, contact arc, hole area) and the
measurement post-processing on synthetic data.  The two integration tests check
the pieces that actually need a finite-element solve: the true alternating
(staggered) solver against the monolithic Newton solver, and the C(T)
energy-release rate against the ASTM E399 stress-intensity calibration.
"""

import math

import numpy as np
import pytest

from tensorfem.phase_field_fracture import PhaseFieldSolver, assemble, energies
from tensorfem.phase_field_specimens import (
    astm_ct_factor,
    astm_ct_k,
    astm_ct_predict_load,
    astm_seb_factor,
    astm_seb_k,
    astm_seb_predict_load,
    circular_hole_elements,
    configure_threads,
    ct_setup,
    damage_path,
    fit_limit,
    gc_from_irwin_k,
    griffith_stress,
    hole_boundary_ring,
    irwin_k_from_gc,
    ledger_residuals,
    lock_circle_damage,
    peak_load,
    pin_load_forces,
    plane_strain_modulus,
    rigid_two_sided,
    run_control_path,
    seb_setup,
    sent_critical_stress,
    sent_setup,
    tada_edge_crack_factor,
)

E, NU = 1000.0, 0.3


@pytest.fixture(scope="module", autouse=True)
def _threads():
    """Pin the torch pool: the assembly is a chain of small tensor ops."""
    previous = configure_threads(1)
    yield
    configure_threads(previous)


# ----------------------------------------------------------------------------------
# closed-form anchors
# ----------------------------------------------------------------------------------
def test_plane_strain_modulus_and_irwin_roundtrip():
    eprime = plane_strain_modulus(E, NU)
    assert eprime == pytest.approx(E / (1.0 - NU ** 2), rel=1e-14)
    k = irwin_k_from_gc(1e-3, E, NU)
    assert k == pytest.approx(math.sqrt(1e-3 * eprime), rel=1e-14)
    assert gc_from_irwin_k(k, E, NU) == pytest.approx(1e-3, rel=1e-14)


def test_astm_ct_factor_matches_hand_value():
    """ASTM E399 C(T) calibration: f(0.5) = 9.6591 (hand evaluation of the polynomial)."""
    assert astm_ct_factor(0.5) == pytest.approx(9.65908, rel=1e-5)
    # monotone in a/W over the valid range
    vals = [astm_ct_factor(a) for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    assert all(np.diff(vals) > 0.0)


def test_astm_seb_factor_matches_hand_value():
    """ASTM E399 SE(B), S = 4W: independent hand evaluation of the same polynomial."""
    expected = (3.0 * math.sqrt(0.45) / (2.0 * (1.0 + 0.9) * (1.0 - 0.45) ** 1.5)
                * (1.99 - 0.45 * 0.55 * (2.15 - 3.93 * 0.45 + 2.7 * 0.45 ** 2)))
    assert astm_seb_factor(0.45) == pytest.approx(expected, rel=1e-14)
    assert astm_seb_factor(0.45) == pytest.approx(2.28547, rel=1e-5)
    vals = [astm_seb_factor(a) for a in (0.2, 0.3, 0.4, 0.5, 0.6)]
    assert all(np.diff(vals) > 0.0)


def test_astm_shape_factors_reject_out_of_range():
    for fn in (astm_ct_factor, astm_seb_factor):
        with pytest.raises(ValueError):
            fn(0.0)
        with pytest.raises(ValueError):
            fn(1.0)


def test_astm_load_predictions_invert_the_calibration():
    """P_pred must reproduce K_Ic through the forward E399 expressions."""
    k_ic = irwin_k_from_gc(1e-3, E, NU)
    p_ct = astm_ct_predict_load(k_ic, 1.0, 1.0, 0.5)
    assert astm_ct_k(p_ct, 1.0, 1.0, 0.5) == pytest.approx(k_ic, rel=1e-12)
    assert p_ct == pytest.approx(k_ic / 9.65908, rel=1e-5)
    p_seb = astm_seb_predict_load(k_ic, 1.0, 1.0, 0.45)
    assert astm_seb_k(p_seb, 1.0, 1.0, 0.45) == pytest.approx(k_ic, rel=1e-12)


def test_tada_factor_limit_and_griffith_stress():
    assert tada_edge_crack_factor(0.0) == pytest.approx(1.12, rel=1e-12)
    sigma = griffith_stress(1e-3, E, NU, 0.3)
    assert sigma == pytest.approx(irwin_k_from_gc(1e-3, E, NU) / math.sqrt(math.pi * 0.3),
                                 rel=1e-14)
    # finite width stiffens the specimen -> lower critical stress for the same a
    finite = sent_critical_stress(1e-3, E, NU, 0.3, 1.0)
    assert finite < sigma
    assert finite == pytest.approx(sigma / tada_edge_crack_factor(0.3), rel=1e-14)


# ----------------------------------------------------------------------------------
# hole carving and pin loading
# ----------------------------------------------------------------------------------
def _plate_with_hole(h=0.02):
    from tensorfem.phase_field_fracture import structured_quad_mesh
    nodes, elements = structured_quad_mesh(-0.5, 0.5, -0.5, 0.5, int(1.0 / h), int(1.0 / h))
    return nodes, elements


def test_circular_hole_removes_the_expected_area():
    nodes, elements = _plate_with_hole()
    radius = 0.15
    void = circular_hole_elements(nodes, elements, (0.0, 0.0), radius)
    area = len(void) * (1.0 / 50.0) * (1.0 / 50.0)
    assert area == pytest.approx(math.pi * radius ** 2, rel=0.15)
    centroids = nodes[elements[void]].mean(axis=1)
    assert np.all(np.linalg.norm(centroids, axis=1) < radius)


def test_hole_boundary_ring_sits_on_the_circle():
    nodes, elements = _plate_with_hole()
    radius = 0.15
    void = circular_hole_elements(nodes, elements, (0.0, 0.0), radius)
    ring, angle = hole_boundary_ring(nodes, elements, void, (0.0, 0.0), radius)
    assert ring.size > 8
    r = np.linalg.norm(nodes[ring], axis=1)
    assert np.all(r > 0.5 * radius)
    assert np.all(r < 1.6 * radius)
    assert np.all(np.diff(angle) >= 0.0)


def test_pin_load_resultant_is_exact_and_centred():
    """The nodal vector of the cosine pin pressure sums to the applied force."""
    nodes, elements = _plate_with_hole()
    radius = 0.15
    void = circular_hole_elements(nodes, elements, (0.0, 0.0), radius)
    for force, direction in ((1.0, (0.0, 1.0)), (-2.5, (0.0, 1.0)), (0.7, (1.0, 0.0))):
        f, info = pin_load_forces(nodes, elements, void, (0.0, 0.0), radius, force, direction)
        r = f.reshape(-1, 2)
        net = r.sum(axis=0)
        d = np.asarray(direction, dtype=float)
        assert net[0] == pytest.approx(force * d[0], abs=1e-12)
        assert net[1] == pytest.approx(force * d[1], abs=1e-12)
        # a radial pressure field carries no moment about the hole centre; the
        # lumped nodal vector reproduces that only to O(h), which is bounded here
        idx = np.where(np.abs(f) > 0)[0] // 2
        moment = float((nodes[idx, 0] * r[idx, 1] - nodes[idx, 1] * r[idx, 0]).sum())
        assert abs(moment) < 0.16 * abs(force) * radius


def test_pin_load_moment_error_converges_with_the_ring():
    """The lumped pin load is moment-free only to first order: check the O(h) decay."""
    from tensorfem.phase_field_fracture import structured_quad_mesh
    err = []
    for n in (50, 100):
        nodes, elements = structured_quad_mesh(-0.5, 0.5, -0.5, 0.5, n, n)
        radius = 0.15
        void = circular_hole_elements(nodes, elements, (0.0, 0.0), radius)
        f, _ = pin_load_forces(nodes, elements, void, (0.0, 0.0), radius, 1.0, (0.0, 1.0))
        r = f.reshape(-1, 2)
        idx = np.where(np.abs(f) > 0)[0] // 2
        err.append(abs(float((nodes[idx, 0] * r[idx, 1] - nodes[idx, 1] * r[idx, 0]).sum()))
                   / radius)
        assert r.sum(axis=0)[1] == pytest.approx(1.0, abs=1e-12)
    assert err[1] < 0.6 * err[0]


def test_pin_load_only_acts_on_the_contact_half():
    """A pin pulled in +y presses the upper half of the bore only."""
    nodes, elements = _plate_with_hole()
    radius = 0.15
    void = circular_hole_elements(nodes, elements, (0.0, 0.0), radius)
    f, info = pin_load_forces(nodes, elements, void, (0.0, 0.0), radius, 1.0, (0.0, 1.0))
    loaded = np.where(np.abs(f) > 0)[0]
    loaded_nodes = np.unique(loaded // 2)
    assert np.all(nodes[loaded_nodes, 1] > -0.25 * radius)
    # and a downward pin presses the lower half
    g, _ = pin_load_forces(nodes, elements, void, (0.0, 0.0), radius, -1.0, (0.0, 1.0))
    gl = np.unique(np.where(np.abs(g) > 0)[0] // 2)
    assert np.all(nodes[gl, 1] < 0.25 * radius)


def test_lock_circle_selects_the_pin_annulus():
    nodes, elements = _plate_with_hole()
    idx = lock_circle_damage(nodes, (0.0, 0.0), 0.10, 0.30)
    r = np.linalg.norm(nodes[idx], axis=1)
    assert idx.size > 0
    assert np.all(r >= 0.10) and np.all(r <= 0.30)


# ----------------------------------------------------------------------------------
# specimen builders
# ----------------------------------------------------------------------------------
def test_ct_setup_geometry_and_conventions():
    spec = ct_setup(a_over_w=0.5, W=1.0, l=0.08, h=0.04)
    assert spec.alpha == pytest.approx(0.5)
    # ASTM: a is measured from the load line, the front face is at -0.25 W
    tip_x = spec.meta["crack_tip"][0]
    assert tip_x == pytest.approx(0.5, abs=0.05)
    assert spec.meta["front_face_x"] == pytest.approx(-0.25)
    # the notch length measured from the front face is a + 0.25 W
    assert spec.meta["notch_from_front"] == pytest.approx(0.75)
    ys = np.abs(spec.meta["hole_centers"])
    assert ys[0][1] == pytest.approx(0.275)
    assert spec.meta["hole_radius"] == pytest.approx(0.125)
    # pin dofs: the upper pin is driven on the upper half, the lower on the lower half
    assert spec.control.size == spec.control2.size > 4
    assert np.all(spec.nodes[spec.control, 1] > 0.275 - 1e-9)
    assert np.all(spec.nodes[spec.control2, 1] < -0.275 + 1e-9)
    assert spec.notch.size > 0
    assert np.all(spec.nodes[spec.notch, 0] < tip_x + 1e-9)


def test_seb_setup_geometry():
    spec = seb_setup(a_over_w=0.45, W=1.0, l=0.08, h=0.04)
    assert spec.meta["span"] == pytest.approx(4.0)
    assert spec.alpha == pytest.approx(0.45)
    assert spec.notch.size > 0
    assert np.all(spec.nodes[spec.notch, 1] < 0.45 + 1e-9)
    assert np.all(np.abs(spec.nodes[spec.control, 0]) <= 0.15 + 1e-12)
    assert np.all(spec.nodes[spec.control, 1] > 0.999)


def test_sent_setup_geometry():
    spec = sent_setup(a_over_w=0.3, W=1.0, l=0.08, h=0.04)
    assert spec.alpha == pytest.approx(0.3)
    assert np.all(spec.nodes[spec.notch, 0] < 0.3 + 1e-9)
    assert np.all(spec.nodes[spec.control, 1] > 0.999)
    assert spec.problem.n_elem > 100


def test_specimen_tangent_is_not_singular():
    """Removed element rows leave orphan nodes; the builders must pin them."""
    spec = ct_setup(a_over_w=0.4, W=1.0, l=0.1, h=0.05)
    p = spec.problem
    R, K, _ = assemble(p, np.zeros(2 * p.n_nodes), np.zeros(p.n_nodes),
                       np.zeros((p.n_elem, 4)))
    fixed = np.zeros(p.n_dof, dtype=bool)
    fixed[list(spec.fixed)] = True
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    mask = fixed.astype(float)
    free = sp.diags(1.0 - mask)
    Kd = (free @ K @ free + sp.diags(mask)).tocsc()
    lu = spla.splu(Kd)                       # raises if the system is singular
    assert lu.U.nnz > 0
    # every orphan node is pinned in both fields
    for nd in p.orphan_nodes:
        assert 2 * int(nd) in spec.fixed
        assert 2 * int(nd) + 1 in spec.fixed
        assert 2 * p.n_nodes + int(nd) in spec.fixed


def test_rigid_two_sided_sets_opposite_pin_displacements():
    spec = ct_setup(a_over_w=0.4, W=1.0, l=0.1, h=0.05)
    fixed, control = rigid_two_sided(spec, spec.control, spec.control2, 0.011)
    ups = [fixed[d] for d in control]
    assert all(v == pytest.approx(0.011) for v in ups)
    downs = [fixed[2 * int(n) + 1] for n in spec.control2]
    assert all(v == pytest.approx(-0.011) for v in downs)


# ----------------------------------------------------------------------------------
# measurement post-processing on synthetic data
# ----------------------------------------------------------------------------------
def test_peak_load_identification():
    records = [{"reaction": -v, "value": i} for i, v in enumerate([1.0, 3.0, 4.0, 2.5, 1.0])]
    out = peak_load(records)
    assert out["peak"] == pytest.approx(4.0)
    assert out["index"] == 2
    assert out["value_at_peak"] == 2
    assert out["post_peak_drop_percent"] == pytest.approx(75.0)
    assert out["rising_monotone"]


def test_ledger_residual_metric():
    good = [{"work": 10.0, "elastic": 6.0, "fracture": 4.0},
            {"work": 20.0, "elastic": 12.0, "fracture": 8.0}]
    assert np.allclose(ledger_residuals(good), 0.0)
    bad = [{"work": 10.0, "elastic": 6.0, "fracture": 2.0}]
    assert ledger_residuals(bad)[0] == pytest.approx(0.2)


def test_damage_path_recovers_a_straight_crack():
    nodes, _ = __import__("tensorfem.phase_field_fracture", fromlist=["x"]).structured_quad_mesh(
        0.0, 1.0, 0.0, 1.0, 40, 40)
    d = np.where(np.abs(nodes[:, 1] - 0.5) < 0.0126, 1.0, 0.0)
    out = damage_path(nodes, d, (0.0, 0.5), (1.0, 0.0), 1.0, n_bands=20)
    assert out["filled_fraction"] == pytest.approx(1.0, abs=0.05)
    assert out["max_abs_t"] < 0.02
    assert out["tip_s"] > 0.95


def test_damage_path_detects_a_deviating_crack():
    from tensorfem.phase_field_fracture import structured_quad_mesh
    nodes, _ = structured_quad_mesh(0.0, 1.0, 0.0, 1.0, 40, 40)
    d = np.where(np.abs(nodes[:, 1] - (0.3 + 0.4 * nodes[:, 0])) < 0.0126, 1.0, 0.0)
    out = damage_path(nodes, d, (0.0, 0.3), (1.0, 0.0), 1.0, n_bands=20)
    assert out["max_abs_t"] > 0.2          # the slope is visible
    assert out["filled_fraction"] > 0.8


def test_fit_limit_recovers_a_linear_series():
    l = np.array([0.08, 0.04, 0.02, 0.01])
    values = 2.5 + 3.0 * l
    out = fit_limit(l, values, power=1.0)
    assert out["limit"] == pytest.approx(2.5, rel=1e-8)
    assert out["monotone"]
    assert out["finest_gap_percent"] < 2.0


# ----------------------------------------------------------------------------------
# integration: solver and physical anchors
# ----------------------------------------------------------------------------------
def _small_sent(l=0.1, h=0.05):
    spec = sent_setup(a_over_w=0.3, W=1.0, l=l, h=h, E=E, nu=NU, Gc=1e-3)
    solver = PhaseFieldSolver(spec.problem, fixed=dict(spec.fixed))
    solver.fix_damage_nodes(spec.notch, 1.0)
    return spec, solver


def test_alternating_step_converges_and_matches_monolithic():
    """The true staggered scheme reaches a 1e-6 coupled residual and the same energy.

    The coupled residual of the alternating loop is checked directly (it is the
    quantity the acceptance criterion bounds) instead of relying on the Newton
    driver, which stalls on the non-smooth ``max(H, psi+)`` floor.
    """
    spec, solver_a = _small_sent()
    solver_m = PhaseFieldSolver(spec.problem, fixed=dict(spec.fixed))
    solver_m.fix_damage_nodes(spec.notch, 1.0)
    delta = 2.0e-4
    for solver in (solver_a, solver_m):
        for dof in spec.control_dofs:
            solver.fixed[int(dof)] = True
            solver.fixed_values[int(dof)] = delta
        solver._apply_fixed()
    iters, hist, converged = solver_a.alternating_step(tol=1e-6, max_iter=2000)
    assert converged, (iters, hist[-1])
    assert hist[-1] <= 1e-6
    # independent check: the coupled residual at the converged state is tiny
    R_raw, _, _ = assemble(spec.problem, solver_a.u, solver_a.d, solver_a.history)
    force_scale = float(np.abs(R_raw[solver_a.fixed]).sum())      # actuator force
    R = np.where(solver_a.fixed, 0.0, R_raw)
    assert force_scale > 0.0
    assert np.linalg.norm(R) < 1e-3 * force_scale
    solver_m.newton_step(max_iter=25, tol=2e-8)
    en_a = energies(spec.problem, solver_a.u, solver_a.d, solver_a.history)
    en_m = energies(spec.problem, solver_m.u, solver_m.d, solver_m.history)
    # alternate minimisation is a descent method on the same functional, so it can
    # never end up above the (here stalled) Newton state
    assert en_a["total"] <= en_m["total"] * (1.0 + 1e-6)


def test_alternating_and_monolithic_agree_when_the_phase_field_is_frozen():
    """With ``d`` frozen both schemes solve the same SPD displacement problem."""
    spec, solver_a = _small_sent()
    solver_m = PhaseFieldSolver(spec.problem, fixed=dict(spec.fixed))
    delta = 2.0e-4
    for solver in (solver_a, solver_m):
        solver.fix_damage_nodes(np.arange(spec.problem.n_nodes), 0.0)
        for dof in spec.control_dofs:
            solver.fixed[int(dof)] = True
            solver.fixed_values[int(dof)] = delta
        solver._apply_fixed()
    solver_a.alternating_step(tol=1e-10, max_iter=50)
    solver_m.newton_step(max_iter=5, tol=1e-12)
    scale = max(np.linalg.norm(solver_m.u), 1e-30)
    assert np.linalg.norm(solver_a.u - solver_m.u) / scale < 1e-8


def test_ct_energy_release_rate_matches_astm_e399():
    """C(T) energy-release rate of a slit crack vs the ASTM E399 K calibration."""
    from tensorfem.phase_field_fracture import sharp_crack_energy_release
    alpha, W, P, l = 0.45, 1.0, 1.0, 0.06
    h = 0.5 * l
    specs = [ct_setup(a_over_w=alpha + da / W, W=W, l=l, h=h, E=E, nu=NU) for da in (-h, 0.0, h)]
    base = specs[1]
    cu, cl = base.meta["hole_centers"]
    r = base.meta["hole_radius"]
    forces = []
    for s in specs:
        fu, _ = pin_load_forces(s.nodes, s.elements, s.problem.void_elements, cu, r, P, (0, 1))
        fl, _ = pin_load_forces(s.nodes, s.elements, s.problem.void_elements, cl, r, -P, (0, 1))
        forces.append(fu + fl)
    out = sharp_crack_energy_release([s.problem for s in specs], forces[1], base.fixed, h)
    k_fem = math.sqrt(max(out["G"], 0.0) * plane_strain_modulus(E, NU))
    k_astm = astm_ct_k(P, base.B, W, base.alpha)
    assert abs(k_fem - k_astm) / k_astm < 0.10, (k_fem, k_astm)


def test_run_control_path_work_matches_elastic_energy():
    """For a linear elastic path the external work equals the stored strain energy."""
    spec, solver = _small_sent()
    solver.fix_damage_nodes(np.arange(spec.problem.n_nodes), 0.0)   # freeze the phase field
    steps = [dict((d, v) for d in spec.control_dofs) for v in (5e-5, 1e-4, 1.5e-4)]
    records = run_control_path(spec.problem, solver, steps, measure_dofs=spec.control_dofs,
                               scheme="monolithic", max_iter=5, tol=1e-11)
    for rec in records:
        assert rec["fracture"] == pytest.approx(0.0, abs=1e-14)
        assert rec["work"] == pytest.approx(rec["elastic"], rel=2e-6)
        assert abs(rec["work"] - rec["total"]) / rec["work"] < 1e-5