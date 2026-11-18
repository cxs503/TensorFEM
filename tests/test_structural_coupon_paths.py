import numpy as np
from scipy.integrate import trapezoid
import pytest

from tensorfem.cohesive_bending import BilinearCohesiveLaw, build_three_point_bend
from tensorfem.ice_bend_initiation import first_crack
from tensorfem.ice_indentation import spherical_bearing_load, spherical_bearing_work
from tensorfem.parallel.structure import solve_panel


def test_first_crack_location_emerges_from_an_intact_mesh():
    law = BilinearCohesiveLaw.from_material(2.0e8, 1.0e6, 1.5)
    model = build_three_point_bend(
        span=0.18,
        depth=0.04,
        thickness=0.08,
        young_modulus=2.0e8,
        poisson_ratio=0.3,
        law=law,
        nx=16,
        ny=8,
    )
    history = first_crack(model, dt=0.001)
    assert history["y_m"] == pytest.approx(0.0)
    assert history["x_error_span"] <= 1 / 16
    assert history["energy_balance_rel"] < 1e-7
    assert history["max_surface_tension_Pa"] >= law.peak_traction
    assert 0 < history["sample_deflection_m"] < history["punch_depth_m"]


def test_bearing_work_is_the_integral_of_force():
    r, hardness = 0.0016, 22e6
    h = np.linspace(0.0, 0.0003, 2001)
    f = [spherical_bearing_load(x, r, hardness) for x in h]
    assert trapezoid(f, h) == pytest.approx(spherical_bearing_work(h[-1], r, hardness), rel=1e-7)
    with pytest.raises(ValueError):
        spherical_bearing_load(2 * r, r, hardness)


def test_unequal_parallel_strips_recover_dense_and_sparse_panel_fields():
    dense = solve_panel(9, 3, "dense")
    for method in ("sparse", "parallel"):
        result = solve_panel(9, 3, method, workers=2)
        for key in ("stress", "displacement", "reaction", "force"):
            scale = np.max(np.abs(dense[key]))
            assert np.max(np.abs(result[key] - dense[key])) < 1e-8 * scale
        assert result["energy"] == pytest.approx(result["work"], rel=1e-8)
        assert len(set(result["worker_pids"])) == (2 if method == "parallel" else 1)


def test_stroke_loading_recovers_elastic_energy_before_damage_and_postpeak_drop():
    from tensorfem.ice_bend_stroke import stroke_control

    law = BilinearCohesiveLaw.from_material(2e8, 1e6, 1.5)
    model = build_three_point_bend(
        span=0.18,
        depth=0.04,
        thickness=0.08,
        young_modulus=2e8,
        poisson_ratio=0.3,
        law=law,
        nx=12,
        ny=8,
    )
    elastic = stroke_control(model, max_stroke=1e-5, increments=4)
    assert elastic["fracture_dissipation_J"] == pytest.approx(0.0, abs=1e-12)
    assert elastic["stored_energy_J"] == pytest.approx(elastic["external_work_J"], rel=1e-6)
    broken = stroke_control(model, max_stroke=0.0016, increments=240)
    assert broken["converged"] and broken["stop_reason"] == "residual_load_below_2pct"
    assert broken["force_N"][-1] < 0.02 * broken["peak_N"]
    assert broken["irreversibility_violation_J"] < 1e-12
    assert 0 < broken["fracture_dissipation_J"] <= 1.5 * model.depth * model.thickness


def test_transient_break_converts_released_energy_to_fragment_motion():
    from tensorfem.ice_bend_transient import rupture

    law = BilinearCohesiveLaw.from_material(2e8, 1e6, 1.5)
    model = build_three_point_bend(
        span=0.18,
        depth=0.04,
        thickness=0.08,
        young_modulus=2e8,
        poisson_ratio=0.3,
        law=law,
        nx=12,
        ny=8,
    )
    result = rupture(model, step_fraction=0.65)
    assert result["postpeak_stop"] == "sustained_postpeak_unloading"
    assert result["energy_balance_rel"] < 1e-3
    assert result["failed_area_fraction"] > 0.7
    assert result["kinetic_energy_J"] > result["initial_kinetic_J"]
    with pytest.raises(ValueError, match="stability bound"):
        rupture(model, fixed_dt=1.0)
