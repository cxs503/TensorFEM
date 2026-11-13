"""CASE FEM-HERTZ-CONTACT — field-level verification of the Hertz surface deflection.

The anchor certificate lives in ``benchmarks/verified/fem_hertz_contact_field``;
these tests pin the three properties the certificate rests on:

1. the oracle really is the exact half-space solution (it reproduces, from the
   Boussinesq kernel alone, the Hertz relations and the flat contact interior),
2. the field metric is not vacuous (it rejects a 10 % contact-radius error),
3. the FE surface deflection reproduces the exact profile on a coarse mesh.

The heavy multi-level certificate is not re-run here; a single reduced case
(h/a = 0.075, depth 48a) is solved so the suite stays fast.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

RUN = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "fem_hertz_contact_field"
    / "run.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("hertz_field_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def state(mod):
    return mod.hertz_state(mod.load_for_radius_ratio(mod.RATIO))


# ---------------------------------------------------------------------------
# 1. the oracle is exact: three closed forms it is not derived from
# ---------------------------------------------------------------------------


def test_oracle_reproduces_hertz_indentation(mod, state):
    """w(0) = pi p0 a / (2 E*) must equal the Hertz indentation a^2/R."""
    w0 = mod.deflection_exact(
        0.0,
        contact_radius=state["contact_radius"],
        peak_pressure=state["peak_pressure"],
        e_star=state["e_star"],
    )
    assert w0 == pytest.approx(state["indentation"], rel=1e-10)
    # ... and that value carries the Hertz relation p0 = 2 E* a / (pi R)
    assert state["peak_pressure"] == pytest.approx(
        2.0 * state["e_star"] * state["contact_radius"] / (math.pi * mod.R), rel=1e-12
    )


def test_oracle_is_flat_inside_the_contact(mod, state):
    """Under a paraboloid indenter the exact solution is w = delta - r^2/(2R)."""
    a = state["contact_radius"]
    for rel in (0.2, 0.5, 0.9):
        r = rel * a
        w = mod.deflection_exact(
            r,
            contact_radius=a,
            peak_pressure=state["peak_pressure"],
            e_star=state["e_star"],
        )
        assert w == pytest.approx(state["indentation"] - r * r / (2.0 * mod.R), rel=1e-8)


def test_oracle_reaches_the_boussinesq_point_load_asymptote(mod, state):
    """Far field of the same pressure field is a point load: w -> P/(pi E* r)."""
    a = state["contact_radius"]
    r = 3.0 * a
    w = mod.deflection_exact(
        r, contact_radius=a, peak_pressure=state["peak_pressure"], e_star=state["e_star"]
    )
    point = state["load"] / (math.pi * state["e_star"] * r)
    assert w == pytest.approx(point, rel=0.03)  # O((a/r)^2) correction


def test_oracle_decays_monotonically_outside_contact(mod, state):
    a = state["contact_radius"]
    radii = np.linspace(1.0, 3.0, 21) * a
    w = np.array(
        [
            mod.deflection_exact(
                float(r),
                contact_radius=a,
                peak_pressure=state["peak_pressure"],
                e_star=state["e_star"],
            )
            for r in radii
        ]
    )
    assert np.all(np.diff(w) < 0.0)
    # the analytic half-space solution does not radiate a contact ring artefact
    assert w[0] < state["indentation"]


# ---------------------------------------------------------------------------
# 2. the metric discriminates: a 10 % radius error must be rejected
# ---------------------------------------------------------------------------


def test_shape_metric_rejects_a_ten_percent_radius_error(mod, state):
    case = {
        "radii": np.linspace(0.0, 3.0, 61) * state["contact_radius"],
        "deflection": mod.oracle_profile(
            np.linspace(0.0, 3.0, 61) * state["contact_radius"], state
        ),
    }
    right = mod.profile_metrics(case, state)
    wrong = mod.discrimination_control(case, state, 1.10)
    # the metric is a residual, so the correct state matches itself to round-off
    assert right["max_residual_over_delta"] < 1e-12
    # a 10 % contact radius error is amplified far beyond the 1 % tolerance
    assert wrong["max_residual_over_delta"] > 0.05
    assert wrong["max_residual_over_local"] > 0.10


def test_profile_metrics_ignore_a_rigid_offset(mod, state):
    """The graded quantity is the profile shape, not the absolute deflection."""
    a = state["contact_radius"]
    radii = np.linspace(0.0, 3.0, 61) * a
    exact = mod.oracle_profile(radii, state)
    shifted = dict(state)
    case = {"radii": radii, "deflection": exact - 0.004 * state["indentation"]}
    metrics = mod.profile_metrics(case, shifted)
    assert metrics["max_residual_over_delta"] < 1e-12
    assert metrics["offset_over_delta"] == pytest.approx(-0.004, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. end-to-end: the FE surface deflection on a coarse mesh
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coarse_case(mod):
    return mod.run_case(h_rel=0.075, depth_factor=48.0)


def test_fe_surface_deflection_matches_the_exact_profile(mod, state, coarse_case):
    metrics = mod.profile_metrics(coarse_case, state)
    # generous bound for the coarse CI mesh; the certificate uses 1 % at h/a=0.015
    assert metrics["max_residual_over_delta"] < 0.03
    assert metrics["max_residual_over_local"] < 0.10
    # the residual is a *residual*: it must be far below the deflection itself
    assert coarse_case["deflection"][0] == pytest.approx(state["indentation"], rel=0.05)


def test_fe_contact_constraint_is_satisfied(coarse_case):
    assert coarse_case["penetration_over_delta"] < 1e-4
    assert coarse_case["contact_constraint_residual_rel"] < 1e-3


def test_fe_reproduces_the_certified_indentation_error(mod):
    """The same model used by the certified anchor must give the same number."""
    certified = mod.certified_indentation_error(0.05, 0.030)
    if certified is None:  # pragma: no cover - committed artifact is the default
        pytest.skip("certified anchor artifact not present in the work tree")
    assert certified == pytest.approx(-0.995682141721177, abs=1e-9)