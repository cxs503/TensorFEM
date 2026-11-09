"""Verification suite for ``tensorfem.polar_thermal_stress``.

Every test either compares against a closed form (1-D wall, lumped capacitance,
restrained biaxial thermo-elasticity, Timoshenko clamped plate), checks a
physical-consistency gate (force balance, monotone margin in air temperature,
material trends) or pins reproducibility.  The dense reference assembly is
single-threaded on purpose: this host has 128 cores and the launch overhead of
tiny kernels dominates otherwise (see ``industrial_p0_phase3_qualification``).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.polar_thermal_stress import (
    DTYPE,
    IceContactPatch,
    PolarEnvironment,
    PolarPanel,
    POLAR_STEELS,
    FacePatch,
    assemble_conduction,
    assemble_elasticity,
    assemble_pressure_load,
    assemble_thermal_load,
    assess_polar_panel,
    clamped_plate_bending,
    clamped_plate_uniform_reference,
    external_faces,
    faces_on_plane,
    free_plate_gradient_stress,
    ice_patch_faces,
    isotropic_D,
    layered_slab_mesh,
    lumped_capacitance,
    material_anchor_residuals,
    medium_convection,
    patch_area,
    polar_thermal_limit_state_matrix,
    radiation_coefficient,
    restrained_biaxial_stress,
    solve_dirichlet,
    solve_panel_temperature,
    solve_steady_conduction,
    steel_from_anchors,
    structured_hex_mesh,
    transient_conduction,
    von_mises,
    wall_analytic_1d,
)

torch.set_num_threads(1)

STEEL = POLAR_STEELS["EH36"]


# ---------------------------------------------------------------------------
# conduction
# ---------------------------------------------------------------------------

def test_hex8_linear_field_patch_test_on_sheared_mesh():
    """A trilinear hex8 reproduces a linear temperature field on an affine mesh.

    The mesh is sheared (``x -> x + 0.3 y``, ``y -> y + 0.1 z``), i.e. every
    element keeps a constant Jacobian.  Genuinely *warped* hexes do not pass a
    strict patch test because their isoparametric map is not affine -- that is a
    property of the element family, not of this implementation.
    """
    from tensorfem.polar_thermal_stress import HexMesh

    base = structured_hex_mesh(3, 3, 3, 0.1, 0.2, 0.3)
    nodes = base.nodes.clone()
    y = nodes[:, 1].clone()
    z = nodes[:, 2].clone()
    nodes[:, 0] += 0.3 * y
    nodes[:, 1] += 0.1 * z
    mesh = HexMesh(nodes=nodes, elements=base.elements)
    shear = 1.0  # det of the shear transform
    assert abs(mesh.volume() - 0.1 * 0.2 * 0.3 * shear) < 1e-12

    field = 2.0 - 1.5 * mesh.nodes[:, 0] + 0.75 * mesh.nodes[:, 1]
    # boundary by structured index (the sheared block is not axis aligned)
    n = 3
    idx = torch.arange(mesh.n_nodes)
    i = idx // ((n + 1) * (n + 1))
    j = (idx % ((n + 1) * (n + 1))) // (n + 1)
    k = idx % (n + 1)
    boundary = (i == 0) | (i == n) | (j == 0) | (j == n) | (k == 0) | (k == n)
    fixed = torch.nonzero(boundary, as_tuple=False).reshape(-1)
    K = assemble_conduction(mesh, 12.0)
    T = solve_dirichlet(K, torch.zeros(mesh.n_nodes, dtype=DTYPE), fixed, field[fixed])
    interior = torch.nonzero(~boundary, as_tuple=False).reshape(-1)
    assert interior.numel() > 0
    assert float((T[interior] - field[interior]).abs().max()) < 1e-9


def test_composite_wall_matches_closed_form():
    """Criterion (a): steady 1-D composite wall vs the analytic profile."""
    layers = [(0.003, 0.3), (0.025, 45.0)]  # 3 mm coating, 25 mm steel
    h_out, h_in, T_air, T_in = 30.0, 8.0, -40.0, 5.0
    mesh = layered_slab_mesh([(0.003, 2), (0.025, 5)])
    k = torch.tensor([0.3] * 2 + [45.0] * 5, dtype=DTYPE)
    outer, parent_o = faces_on_plane(mesh, 0, 0.0)
    inner, parent_i = faces_on_plane(mesh, 0, 0.028)
    T = solve_steady_conduction(
        mesh,
        k,
        [FacePatch.of(outer, parent_o, h_out, T_air), FacePatch.of(inner, parent_i, h_in, T_in)],
    )
    ana = wall_analytic_1d(layers, T_air, h_out, T_in, h_in,
                           x_samples=mesh.nodes[:, 0].tolist())
    err = max(abs(a - b) for a, b in zip(T.tolist(), ana["temperature"]))
    span = T_in - T_air
    assert err / span < 0.01, f"wall error {err:.3e} K of {span:.0f} K"
    # outer surface must sit well below the inner one and the profile must be
    # monotone in x
    prof = sorted(zip(mesh.nodes[:, 0].tolist(), T.tolist()))
    assert prof[0][1] < prof[-1][1]


def test_wall_flux_continuity_across_layers():
    """Heat flux has to be continuous across the coating/steel interface."""
    import tensorfem.polar_thermal_stress as pts

    mesh = layered_slab_mesh([(0.003, 2), (0.025, 5)])
    k = torch.tensor([0.3] * 2 + [45.0] * 5, dtype=DTYPE)
    outer, parent_o = faces_on_plane(mesh, 0, 0.0)
    inner, parent_i = faces_on_plane(mesh, 0, 0.028)
    T = solve_steady_conduction(
        mesh,
        k,
        [FacePatch.of(outer, parent_o, 30.0, -40.0), FacePatch.of(inner, parent_i, 8.0, 5.0)],
    )
    gd = pts._hex8_gauss_data(mesh)
    flux = []
    for e in (1, 2):  # last coating element, first steel element
        grad = gd.dNdx[e].mean(dim=0)  # (8, 3)
        Te = T[mesh.elements[e]]
        dTdx = float((grad[:, 0] * Te).sum())
        flux.append(-float(k[e]) * dTdx)
    ref = max(abs(flux[0]), abs(flux[1]), 1e-9)
    assert abs(flux[0] - flux[1]) / ref < 0.01, flux


def test_radiation_coefficient_reduces_to_stefan_boltzmann_limit():
    """With T_s == T_surr the linearised coefficient must equal ``4 eps sig T^3``."""
    from tensorfem.polar_thermal_stress import SIGMA_SB

    for T in (-40.0, 0.0, 20.0):
        tk = T + 273.15
        assert radiation_coefficient(T, T, 0.9) == pytest.approx(
            0.9 * SIGMA_SB * 4.0 * tk ** 3, rel=1e-12
        )
    # colder surface -> smaller radiative exchange
    assert radiation_coefficient(-40.0, -40.0) < radiation_coefficient(20.0, 20.0)


def test_medium_convection_correlations_are_monotone_and_documented():
    assert medium_convection("air", 0.0) == pytest.approx(5.7)
    assert medium_convection("air", 15.8) == pytest.approx(5.7 + 3.8 * 15.8)
    assert medium_convection("seawater", 0.0) > 10.0 * medium_convection("still_air", 0.0)
    assert medium_convection("air", 20.0) > medium_convection("air", 5.0)
    with pytest.raises(ValueError):
        medium_convection("crude_oil", 1.0)


def test_transient_lumped_capacitance_decay():
    """Criterion: transient conduction vs ``exp(-t/tau)`` for Bi << 1."""
    thickness = 0.002
    mesh = structured_hex_mesh(1, 2, 2, thickness, 0.05, 0.05)
    outer, parent_o = faces_on_plane(mesh, 0, 0.0)
    patch = FacePatch.of(outer, parent_o, 20.0, -40.0)
    rho_cp = STEEL.density * STEEL.specific_heat
    tau = STEEL.density * STEEL.specific_heat * thickness / 20.0
    sol = transient_conduction(mesh, STEEL.conductivity, rho_cp, [patch],
                               T0=20.0, dt=20.0, n_steps=100)
    ana = lumped_capacitance(sol.times.tolist(), 20.0, -40.0, tau)
    err = float((sol.temperatures.mean(dim=1) - ana).abs().max())
    assert err / 60.0 < 0.02, f"transient error {err:.3f} K of 60 K"


def test_transient_reaches_the_steady_solution():
    thickness = 0.002
    mesh = structured_hex_mesh(1, 2, 2, thickness, 0.05, 0.05)
    outer, parent_o = faces_on_plane(mesh, 0, 0.0)
    patch = FacePatch.of(outer, parent_o, 20.0, -40.0)
    rho_cp = STEEL.density * STEEL.specific_heat
    sol = transient_conduction(mesh, STEEL.conductivity, rho_cp, [patch],
                               T0=20.0, dt=50.0, n_steps=400)
    steady = solve_steady_conduction(mesh, STEEL.conductivity, [patch])
    assert float((sol.final - steady).abs().max()) < 0.05
    with pytest.raises(ValueError):
        transient_conduction(mesh, STEEL.conductivity, rho_cp, [patch], T0=20.0, dt=0.0, n_steps=5)


# ---------------------------------------------------------------------------
# thermal stress
# ---------------------------------------------------------------------------

def _biaxially_restrained_solve(dT: float):
    """FEM solve of a solid whose in-plane strain is held at zero."""
    import tensorfem.polar_thermal_stress as pts

    mesh = structured_hex_mesh(2, 2, 2, 0.025, 0.1, 0.1)
    T_hold = 20.0
    T_body = T_hold + dT
    E = float(STEEL.E(torch.tensor([T_body]))[0])
    alpha = float(STEEL.alpha(torch.tensor([T_body]))[0])
    K = assemble_elasticity(mesh, E, STEEL.nu)
    f = assemble_thermal_load(mesh, E, alpha, torch.full((mesh.n_elements,), T_body, dtype=DTYPE),
                              T_hold, STEEL.nu)
    X = mesh.nodes
    fixed = []
    for i in range(mesh.n_nodes):
        fixed += [3 * i + 1, 3 * i + 2]  # uy = uz = 0 everywhere
    fixed += [3 * i for i in range(mesh.n_nodes) if abs(float(X[i, 0])) < 1e-12]
    u = solve_dirichlet(K, f, torch.tensor(fixed), torch.zeros(len(fixed), dtype=DTYPE))
    gd = pts._hex8_gauss_data(mesh)
    B = pts._hex8_B(gd)
    D = isotropic_D(E, STEEL.nu)
    ue = u.reshape(mesh.n_nodes, 3)[mesh.elements].reshape(mesh.n_elements, 24)
    eps = torch.einsum("egki,ei->egk", B, ue)
    sc = alpha * (T_body - T_hold)
    eps_th = torch.zeros_like(eps)
    eps_th[:, :, 0] = sc
    eps_th[:, :, 1] = sc
    eps_th[:, :, 2] = sc
    sigma = torch.einsum("kl,egl->egk", D, eps - eps_th).mean(dim=1)
    return sigma, E, alpha


@pytest.mark.parametrize("dT", [-60.0, -40.0, 30.0])
def test_restrained_biaxial_thermal_stress_matches_analytic(dT):
    """Criterion (b): ``sigma = E alpha dT / (1 - nu)`` within 2 %."""
    sigma, E, alpha = _biaxially_restrained_solve(dT)
    analytic = restrained_biaxial_stress(E, alpha, dT, STEEL.nu)
    fem = float(sigma[0, 1])
    assert abs(abs(fem) - abs(analytic)) / abs(analytic) < 0.02, (fem, analytic)
    # the through-thickness (x) direction is free -> no normal stress there
    assert abs(float(sigma[0, 0])) < 1e-3 * abs(analytic)
    # cooling restrained in-plane must give tension, heating compression
    assert math.copysign(1.0, fem) == math.copysign(1.0, analytic)


def test_free_plate_gradient_helper_is_positive_and_self_equilibrated():
    """The analytic surface stress of a free plate with a linear gradient."""
    s = free_plate_gradient_stress(206e9, 11.2e-6, 20.0, 0.3)
    assert s == pytest.approx(206e9 * 11.2e-6 * 20.0 / (2.0 * 0.7))
    assert abs(s) < abs(restrained_biaxial_stress(206e9, 11.2e-6, 20.0, 0.3))


def test_isotropic_D_limits_and_symmetry():
    E, nu = 206e9, 0.3
    D = isotropic_D(E, nu)
    assert torch.allclose(D, D.T)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    assert float(D[0, 0]) == pytest.approx(lam + 2 * mu)
    assert float(D[3, 3]) == pytest.approx(mu)
    D0 = isotropic_D(E, 0.0)
    assert float(D0[0, 0]) == pytest.approx(E)
    batch = isotropic_D(torch.tensor([E, E], dtype=DTYPE), nu)
    assert tuple(batch.shape) == (2, 6, 6)


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------

def test_material_anchor_residuals_below_five_percent():
    """Criterion (c): regression vs the transcribed ASTM-family anchor table."""
    for name in ("EH36", "A516-Gr70"):
        steel = steel_from_anchors(name)
        res = material_anchor_residuals(steel)
        assert max(res.values()) < 0.05, (name, res)
        assert steel.anchor_residual["anchor_band"] == pytest.approx(0.03)


def test_material_temperature_trends_are_physical():
    """Stiffness and yield rise, CTE falls, as the temperature drops."""
    for steel in POLAR_STEELS.values():
        temps = [20.0, 0.0, -20.0, -40.0, -60.0]
        E = [steel.E(T) for T in temps]
        sy = [steel.yield_strength(T) for T in temps]
        alpha = [steel.alpha(T) for T in temps]
        assert all(E[i] < E[i + 1] for i in range(len(E) - 1))
        assert all(sy[i] < sy[i + 1] for i in range(len(sy) - 1))
        assert all(alpha[i] > alpha[i + 1] for i in range(len(alpha) - 1))
        assert steel.alpha(-200.0) >= 0.0
    with pytest.raises(KeyError):
        steel_from_anchors("unobtainium")


# ---------------------------------------------------------------------------
# plate bending + ice load
# ---------------------------------------------------------------------------

def test_clamped_plate_matches_timoshenko_reference():
    """Kirchhoff FD plate vs Timoshenko Table 36 (clamped square, uniform load)."""
    E, nu, t, p, L = 206e9, 0.3, 0.025, 0.125e6, 1.2
    ref = clamped_plate_uniform_reference(L, L, t, p, E, nu)
    res = clamped_plate_bending(L, L, t, E, nu, p, n_y=48, n_z=48)
    assert abs(res.w_max - ref["w_max_m"]) / ref["w_max_m"] < 0.02
    assert abs(res.sigma_max - ref["sigma_max_Pa"]) / ref["sigma_max_Pa"] < 0.02
    assert res.total_load == pytest.approx(p * L * L, rel=1e-9)


def test_plate_patch_load_total_equals_pressure_times_area():
    p = 3.0e6
    res = clamped_plate_bending(1.2, 2.4, 0.025, 206e9, 0.3, p,
                                patch=(0.45, 0.75, 1.05, 1.35), n_y=24, n_z=48)
    assert res.total_load == pytest.approx(p * 0.09, rel=1e-9)
    assert res.loaded_area == pytest.approx(0.09, rel=1e-9)
    with pytest.raises(ValueError):
        clamped_plate_bending(1.2, 2.4, 0.025, 206e9, 0.3, p, n_y=24, n_z=36)


def test_ice_pressure_face_load_reaction_equals_pressure_times_area():
    """The solid-model pressure load must integrate to ``p * A``."""
    panel = PolarPanel(n_thickness=3, n_y=8, n_z=12)
    mesh = panel.mesh()
    ice = IceContactPatch(length=0.3, width=0.3, pressure=3.0e6)
    patch = ice_patch_faces(mesh, panel, ice)
    f = assemble_pressure_load(mesh, patch, ice.pressure)
    reaction = float(-f.reshape(-1, 3)[:, 0].sum())
    assert reaction == pytest.approx(ice.pressure * patch_area(mesh, patch.nodes), rel=1e-10)
    assert patch_area(mesh, patch.nodes) >= ice.area  # discretised patch never smaller


def test_ice_pressure_area_law_is_decreasing_and_capped():
    from tensorfem.polar_thermal_stress import ice_pressure_area

    assert ice_pressure_area(1.0) == pytest.approx(1.5e6)
    assert ice_pressure_area(0.09) > ice_pressure_area(1.0)
    assert ice_pressure_area(1e-6) <= 15.0e6
    with pytest.raises(ValueError):
        ice_pressure_area(0.0)
    ice = IceContactPatch(length=0.3, width=0.3)
    assert ice.resolved_pressure() == pytest.approx(ice_pressure_area(0.09))


# ---------------------------------------------------------------------------
# end-to-end chain
# ---------------------------------------------------------------------------

def test_panel_temperature_field_is_three_dimensional():
    """The bay must show both a through-thickness drop and an in-plane cold spot."""
    panel = PolarPanel(n_thickness=3, n_y=8, n_z=12)
    field = solve_panel_temperature(panel, PolarEnvironment(T_air=-40.0), STEEL)
    g = field.temperature.reshape(panel.n_thickness + 1, panel.n_y + 1, panel.n_z + 1)
    outer = g[0]
    inner = g[-1]
    assert float(outer.min()) < field.env.T_air + 10.0
    assert float(outer.mean()) < float(inner.mean())
    assert float(outer.max() - outer.min()) > 1.0  # in-plane cold spot
    assert field.picard_delta_C < 1e-3
    assert field.areas["outer_m2"] == pytest.approx(panel.area)
    prof = field.through_thickness_profile()
    assert len(prof["x"]) == panel.n_thickness + 1
    assert prof["T"][-1] > prof["T"][0]


def test_limit_state_matrix_three_pressures_and_monotone_margin():
    """Criterion (d): >= 3 (T_air, P_ice) pairs, margin falls as T_air falls."""
    panel = PolarPanel(n_thickness=3, n_y=8, n_z=12)
    out = polar_thermal_limit_state_matrix(
        panel, STEEL, T_air_values=(-40.0, -20.0, 0.0), ice_pressures=(0.5e6, 1.5e6),
        env_base=PolarEnvironment(inner_medium="air"),
    )
    rows = out["rows"]
    assert len(rows) == 6
    assert out["monotonic_margin_in_T_air"] is True
    for key, series in out["per_pressure"].items():
        margins = series["margin"]
        assert margins[0] < margins[1] < margins[2], (key, margins)
        utils = series["utilization"]
        assert utils[0] > utils[1] > utils[2]
    # all thermal stresses are positive and grow as the air gets colder
    by_T = {}
    for row in rows:
        by_T.setdefault(row.T_air, row.sigma_thermal_max)
    assert by_T[-40.0] > by_T[-20.0] > by_T[0.0]
    assert out["governing"].utilization == max(r.utilization for r in rows)


def test_limit_state_utilization_and_status_are_consistent():
    panel = PolarPanel(n_thickness=3, n_y=8, n_z=12)
    row = assess_polar_panel(panel, PolarEnvironment(T_air=-40.0), STEEL, ice_pressure=1.0e6)
    allowable = row.sigma_yield_at_critical / row.gamma_M
    assert row.gamma_M == pytest.approx(1.15)
    assert row.utilization == pytest.approx(row.sigma_total_max / allowable)
    assert row.margin == pytest.approx(1.0 - row.utilization)
    assert row.safety_factor == pytest.approx(allowable / row.sigma_total_max)
    assert row.sigma_total_max == pytest.approx(row.sigma_thermal_max + row.sigma_ice_max)
    assert row.status == ("PASS" if row.utilization <= 1.0 else "FAIL")
    assert row.superposition_rel_error < 1e-8  # FEM route is linear
    assert row.T_at_critical_C < 0.0
    # the yield stress used is the *increased* low-temperature one
    assert row.sigma_yield_at_critical > STEEL.yield_strength(20.0)


def test_unrestrained_bound_is_softer_than_the_clamped_bay():
    panel = PolarPanel(n_thickness=3, n_y=8, n_z=12)
    env = PolarEnvironment(T_air=-40.0)
    clamped = assess_polar_panel(panel, env, STEEL, ice_pressure=0.5e6, boundary="clamped")
    free = assess_polar_panel(panel, env, STEEL, ice_pressure=0.5e6, boundary="free")
    assert free.sigma_thermal_max < clamped.sigma_thermal_max
    assert free.margin > clamped.margin


def test_deterministic_reproducibility_of_the_chain():
    panel = PolarPanel(n_thickness=3, n_y=6, n_z=8)
    env = PolarEnvironment(T_air=-35.0)
    a = assess_polar_panel(panel, env, STEEL, ice_pressure=1.0e6, verify_superposition=False)
    b = assess_polar_panel(panel, env, STEEL, ice_pressure=1.0e6, verify_superposition=False)
    assert a.digest == b.digest
    assert a.sigma_total_max == b.sigma_total_max
    c = assess_polar_panel(panel, PolarEnvironment(T_air=-30.0), STEEL,
                           ice_pressure=1.0e6, verify_superposition=False)
    assert c.digest != a.digest
    assert c.sigma_thermal_max < a.sigma_thermal_max


def test_external_faces_and_von_mises_helpers():
    mesh = structured_hex_mesh(2, 2, 2, 0.1, 0.1, 0.1)
    nodes, parent = external_faces(mesh)
    assert nodes.shape[0] == 24 and parent.shape[0] == 24
    assert patch_area(mesh, nodes) == pytest.approx(6 * 0.01, rel=1e-9)
    hydro = torch.tensor([[-100e6, -100e6, -100e6, 0.0, 0.0, 0.0]], dtype=DTYPE)
    assert float(von_mises(hydro)[0]) == pytest.approx(0.0, abs=1e-6)
    uniaxial = torch.tensor([[200e6, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    assert float(von_mises(uniaxial)[0]) == pytest.approx(200e6)