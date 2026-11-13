"""Tests for the compressible baffled-panel radiation impedance.

Each test is an anchor, not a smoke run: the assertion is always against an
exact closed form, an independent code path, or a convergence rate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.acoustic_radiation import rayleigh_radiated_power
from tensorfem.hydro_added_mass import (
    panel_added_mass_ratio,
    piston_radiation_mass,
)
from tensorfem.plate_radiation_impedance import (
    disc_piston_impedance_panels,
    panel_modal_impedance,
    panel_wet_dynamics,
    piston_normalized_impedance,
    piston_radiation_impedance,
    self_patch_integral,
    struve_h1_series,
)

RHO_W = 1025.0
C_W = 1500.0
RHO_S = 7850.0

PANEL = dict(lx=3.0, ly=1.5, thickness=0.012, rho_steel=RHO_S,
             rho_water=RHO_W, sound_speed=C_W)


def test_struve_series_matches_scipy():
    """A1: own Struve series vs scipy (independent implementations)."""
    scipy_special = pytest.importorskip("scipy.special")
    for x in (0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
        own = struve_h1_series(x)
        ref = float(scipy_special.struve(1, x))
        assert abs(own - ref) <= 1e-12 * abs(ref), (x, own, ref)


def test_piston_low_frequency_asymptotes():
    """A2: R1 -> (ka)^2/2 and X1 -> 8ka/(3pi) as ka -> 0."""
    for ka in (0.02, 0.05, 0.1):
        r1, x1 = piston_normalized_impedance(ka)
        assert abs(x1 - 8.0 * ka / (3.0 * math.pi)) / (8.0 * ka / (3.0 * math.pi)) < 5e-3
        assert abs(r1 - 0.5 * ka * ka) / (0.5 * ka * ka) < 1e-2


def test_piston_low_frequency_added_mass_matches_rayleigh():
    """A2': X/omega at ka -> 0 is (8/3) rho a^3 (cross-module anchor)."""
    radius = 1.0
    res = piston_radiation_impedance(1e-3, radius, RHO_W, C_W)
    exact = piston_radiation_mass(radius, RHO_W)
    assert res["added_mass"] == pytest.approx(exact, rel=1e-4)
    assert exact == pytest.approx(8.0 / 3.0 * RHO_W * radius ** 3)


def test_piston_high_frequency_limits():
    """A3: R1 -> 1 and X1 -> 0 for large ka."""
    for ka in (10.0, 20.0, 40.0):
        r1, x1 = piston_normalized_impedance(ka)
        assert abs(r1 - 1.0) < 0.02, (ka, r1)
        assert abs(x1) < 0.06, (ka, x1)


def test_disc_panel_operator_matches_exact_piston():
    """A4: the *panel* operator on a disc reproduces the exact piston."""
    for ka in (0.5, 1.0):
        got = disc_piston_impedance_panels(
            radius=1.0, ka=ka, rho=RHO_W, sound_speed=C_W, n_cells=60
        )
        r1, x1 = piston_normalized_impedance(ka)
        assert abs(got["R1_panels"] - r1) / r1 < 2e-2, (ka, got, r1)
        assert abs(got["X1_panels"] - x1) / x1 < 3e-2, (ka, got, x1)
        # staircase disc area converges to pi a^2
        assert abs(got["area_ratio"] - 1.0) < 1e-2


def test_radiation_damping_equals_twice_radiated_power():
    """A5: R = 2 P_rad, with P_rad from the independent sine-kernel quadrature."""
    for freq in (20.0, 200.0):
        imp = panel_modal_impedance(frequency=freq, m=1, n=1, nx=40, ny=20, **PANEL)
        power = rayleigh_radiated_power(
            length=PANEL["lx"], width=PANEL["ly"], frequency=freq,
            rho=RHO_W, c=C_W, m=1, n=1, velocity_amplitude=1.0, n_points=32,
        )
        assert imp["radiation_damping"] > 0.0
        assert imp["radiation_damping"] == pytest.approx(2.0 * power, rel=1e-2)


def test_inphase_reduction_converges_first_order():
    """A6: mu(k->0) vs hydro_added_mass, residual is O(h) (not zero)."""
    errors = []
    for nx, ny in ((20, 10), (40, 20), (80, 40)):
        ref = panel_added_mass_ratio(
            PANEL["lx"], PANEL["ly"], PANEL["thickness"], RHO_S, RHO_W,
            nx=nx, ny=ny, m=1, n=1,
        )
        imp = panel_modal_impedance(frequency=1.0, m=1, n=1, nx=nx, ny=ny, **PANEL)
        errors.append(abs(imp["added_mass_ratio"] - ref["mu"]) / ref["mu"])
    assert errors[-1] < 1.5e-2, errors
    # halving h must roughly halve the difference (first-order self-term gap)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(1.7 <= r <= 2.3 for r in ratios), (errors, ratios)


def test_self_patch_integral_limits():
    """Diagonal cell integral: exact real part at k=0, imag -> -k h^2."""
    h = 0.05
    val = self_patch_integral(h, 0.0)
    assert val.imag == 0.0
    assert val.real == pytest.approx(2.0 * h * math.log(1.0 + math.sqrt(2.0)))
    k = 0.4
    val = self_patch_integral(h, k)
    for target, got in ((-k * h * h, val.imag),):
        assert abs(got - target) / abs(target) < 5e-2
    assert val.real < 2.0 * h * math.log(1.0 + math.sqrt(2.0))


def test_added_mass_decreases_with_frequency():
    """The reactance falls from the in-phase value: mu(200 Hz) < mu(1 Hz)."""
    low = panel_modal_impedance(frequency=1.0, m=1, n=1, nx=40, ny=20, **PANEL)
    high = panel_modal_impedance(frequency=200.0, m=1, n=1, nx=40, ny=20, **PANEL)
    assert high["added_mass_ratio"] < low["added_mass_ratio"]
    assert high["loss_factor"] > low["loss_factor"] > 0.0


def test_wet_dynamics_physical_and_converged():
    """A8: wet < dry frequency, small in-phase error, converged fixed point."""
    res = panel_wet_dynamics(
        lx=PANEL["lx"], ly=PANEL["ly"], thickness=PANEL["thickness"],
        rho_steel=RHO_S, rho_water=RHO_W, sound_speed=C_W,
        youngs_modulus=2.1e11, poisson=0.3, m=1, n=1, nx=20, ny=10,
    )
    assert 0.0 < res["f_wet_compressible"] < res["f_dry"]
    assert res["iteration_residual"] < 1e-9
    # compressibility shifts the wet frequency by well under 0.5% here
    assert abs(res["inphase_model_error"]) < 5e-3
    assert res["loss_factor"] > 0.0
    assert res["mu_at_wet"] < res["mu_inphase_same_operator"]


def test_input_validation():
    with pytest.raises(ValueError):
        panel_modal_impedance(frequency=-1.0, m=1, n=1, nx=4, ny=2, **PANEL)
    with pytest.raises(ValueError):
        piston_normalized_impedance(-0.5)
    with pytest.raises(ValueError):
        self_patch_integral(0.0, 0.0)
    with pytest.raises(ValueError):
        # non-square panels have no analytic square-cell self term
        panel_modal_impedance(frequency=10.0, m=1, n=1, nx=7, ny=5, **PANEL)
    with pytest.raises(ValueError):
        panel_wet_dynamics(
            lx=3.0, ly=1.5, thickness=0.012, rho_steel=RHO_S,
            rho_water=RHO_W, sound_speed=C_W, youngs_modulus=2.1e11,
            poisson=0.7, m=1, n=1, nx=10, ny=5,
        )


def test_committed_validation_artifact_is_green():
    """The committed benchmark artifact must claim only gated, passing anchors."""
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/plate_radiation_impedance/validation_report.json"
    )
    if not path.exists():  # pragma: no cover - artifact ships with the change
        pytest.skip("validation artifact not present")
    report = json.loads(path.read_text())
    assert report["gated_anchors_passed"] is True
    assert report["capability"] == "fem.plate.radiation_impedance_complex"
    for name, anchor in report["anchors"].items():
        assert anchor.get("pass", True) is True, name
        assert anchor["criterion"], name