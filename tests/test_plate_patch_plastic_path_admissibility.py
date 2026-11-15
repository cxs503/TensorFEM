"""Admissibility gates on the elastoplastic plate path (anchor GAP-PP-3).

The anchored acceptance item *plastic limit load vs Ilyushin within 10 %* is
measured by :func:`tensorfem.plate_patch_plastic_limit.limit_load_from_path`.
These tests pin the two fail-open holes found while mining the queue artifacts of
the three-mesh ladder (job ``cj3424``):

1. ``artifacts/patch_plate_plastic_ladder_nx12_free.json`` contains a final load
   step that is **not an equilibrium state** of the layer model: the centre
   deflection goes ``0.0696 m -> 133.49 m`` (a 20 mm plate), the equivalent
   plastic strain reaches ``35.4`` and the section-moment invariant reaches
   ``|M|/m_p = 1.077`` although the 4-layer capacity ratio is exactly ``1.0``.
   The two-line rule still returned ``1.825 MPa`` (= +20.5 % above the Ilyushin
   bound) with a plausible 6.8 % criteria spread.  Removing that one point makes
   the same rule return a *negative* limit load, so the number was manufactured
   entirely by the inadmissible point.
2. A path that merely *stiffens* (``k1 > k0``, which is what the layered shell
   does while the moment field redistributes) satisfies ``k1 > k0`` and therefore
   triggers the two-line rule, but the intersection of two stiffening lines lies
   to the left of the origin.

Both are now withheld: :func:`path_admissibility` declares the path and
:func:`limit_load_from_path` refuses to report a limit load, and
:func:`run_patch_plastic_path` refuses to *commit* such a step in the first
place (so the history never contains it).
"""
import math

import pytest

from tensorfem.plate_patch_plastic_limit import (
    PatchPath,
    PatchPathPoint,
    limit_load_from_path,
    low_temperature_flow_stress,
    path_admissibility,
    run_patch_plastic_path,
    section_capacity_ratio_of,
    section_moment_capacity,
)

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
GRADE, TEMP_C = "DH36", -40.0
PATCH = (0.5 * A, 0.5 * B)

#: measured (p [Pa], w [m], max alpha, |M|/m_p) of the nx = 12 ramp, verbatim
#: from ``artifacts/patch_plate_plastic_ladder_nx12_free.json`` (job cj3424,
#: mesh nx = ny = 12, 4 layers).  Reproducing the defect from the archived
#: numbers is the point: the gate must reject exactly what the artifact holds.
NX12_TAIL = [
    (1564620.564250666, 0.028919128638832777, 0.0031585576669364147, 0.9362322743914009),
    (1651543.928931259, 0.03470126696791508, 0.004427905858213083, 0.9775887435026549),
    (1738467.293611851, 0.043871709777229706, 0.00722865807627221, 1.0000145846047817),
    (1825390.658292444, 0.06961885989222877, 0.014471383297962027, 1.0000302566841943),
    (1912314.022973037, 133.48563691946416, 35.395458187521896, 1.0768519772439316),
]


def _nx12_path() -> PatchPath:
    return PatchPath(model=None, info={}, loading=[
        PatchPathPoint(p, w, alpha, 0.0, ratio)
        for p, w, alpha, ratio in NX12_TAIL])


def test_archived_runaway_point_is_declared_inadmissible():
    """The step that produced the +20.5 % 'limit load' is not an equilibrium state."""
    report = path_admissibility(_nx12_path(), capacity_ratio=1.0)
    assert report["admissible"] is False
    joined = " ".join(report["violations"])
    assert "section capacity exceeded" in joined
    assert report["max_moment_ratio"] == pytest.approx(1.0768519772439316, rel=1e-9)


def test_inadmissible_path_withholds_the_limit_load():
    """Regression of the measured fail-open: the number must become ``None``."""
    estimate = limit_load_from_path(_nx12_path(), criteria=(0.02, 0.05, 0.10))
    assert estimate["limit_load"] is None
    assert estimate["measurements"] == {}
    assert estimate["spread"] is None
    assert "inadmissible" in estimate["reason"]


def test_the_runaway_step_changed_every_reported_number():
    """Evidence of the corruption: the same rule on the admissible prefix.

    ``1.825 MPa`` with a 6.8 % spread (three stiffness criteria) was reported on
    the polluted path; on the admissible prefix the two-line rule gives
    ``1.696 MPa`` and the 0.02 stiffness criterion no longer fires, because the
    ratios had been computed with the runaway slope ``1.5e-3`` instead of
    ``2.96e-7``.  The reported number therefore moved by ``+7.6 %`` -- entirely
    because of a step that is not an equilibrium state.
    """
    healthy = PatchPath(model=None, info={}, loading=[
        PatchPathPoint(p, w, alpha, 0.0, ratio)
        for p, w, alpha, ratio in NX12_TAIL[:4]])
    report = path_admissibility(healthy, capacity_ratio=1.0)
    assert report["admissible"] is True, report["violations"]
    estimate = limit_load_from_path(healthy, criteria=(0.02, 0.05, 0.10))
    assert estimate["limit_load"] is not None
    assert 0.0 < estimate["limit_load"] <= healthy.loading[-1].load_factor
    assert estimate["limit_load"] == pytest.approx(1.696e6, rel=0.01)
    assert "stiffness_criterion_0.02" not in estimate["measurements"]


def test_stiffening_path_does_not_produce_a_negative_limit_load():
    """``k1 > k0`` alone is not a collapse: the intersection must be in range."""
    points = [
        PatchPathPoint(0.5e6, 0.005, 0.0, 0.0, 0.40),
        PatchPathPoint(1.0e6, 0.030, 0.0, 0.0, 0.80),
        PatchPathPoint(2.0e6, 0.045, 0.0, 0.0, 1.00),
    ]
    # a coarse ramp that jumps across yield and then stiffens (the nx = 8
    # record looks like this): k0 = 1e-8, k1 = 1.5e-8, and the middle point lies
    # above the through-origin line of slope k1, so the two-line rule evaluates
    # and lands at p_c = -3 MPa
    path = PatchPath(model=None, info={}, loading=points)
    assert path_admissibility(path, capacity_ratio=1.0)["admissible"] is True
    estimate = limit_load_from_path(path, criteria=(0.02,))
    rejected = estimate.get("rejected_measurements") or {}
    assert "intersection_of_tangents" in rejected
    assert rejected["intersection_of_tangents"] < 0.0
    assert estimate["limit_load"] is None or estimate["limit_load"] > 0.0


def test_healthy_plateau_path_is_still_measured():
    """A genuine collapse branch must keep its number (no over-gating)."""
    points = []
    for factor in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        deflection = (0.004 * factor if factor <= 1.5
                      else 0.006 + 0.020 * (factor - 1.5))
        points.append(PatchPathPoint(factor * 1e6, deflection, 0.004, 0.0, 0.95))
    path = PatchPath(model=None, info={}, loading=points)
    assert path_admissibility(path, capacity_ratio=1.0)["admissible"] is True
    estimate = limit_load_from_path(path, criteria=(0.05, 0.1))
    assert estimate["limit_load"] is not None
    assert 0.0 < estimate["limit_load"] <= 3.0e6


def test_section_capacity_ratio_of_a_real_model():
    """The gate's capacity is the model's own layered value, not a constant."""
    flow = low_temperature_flow_stress(GRADE, TEMP_C)
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=flow.yield_stress,
        hardening=1e6, patch=PATCH, load_factors=[0.2e6, 0.4e6],
        nx=4, ny=4, layers=3, unload_steps=0)
    assert section_capacity_ratio_of(path) == pytest.approx(
        section_moment_capacity(T, flow.yield_stress, 3)["ratio"])
    assert section_capacity_ratio_of(path) == pytest.approx(8.0 / 9.0)


def test_integrator_refuses_to_commit_a_runaway_step():
    """The small-rotation guard truncates *before* the point enters the history."""
    flow = low_temperature_flow_stress(GRADE, TEMP_C)
    p_y = 0.87e6
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=flow.yield_stress,
        hardening=1e6, patch=PATCH,
        load_factors=[0.4 * p_y, 0.8 * p_y, 1.2 * p_y, 1.6 * p_y],
        nx=8, ny=8, layers=4, unload_steps=0, max_deflection_factor=1.05)
    assert path.ramp_truncated is True
    assert path.truncation_reason
    assert path.ramp_completed is False
    first = path.loading[0]
    k0 = first.centre_deflection / first.load_factor
    last = path.loading[-1]
    # every committed point respects the guard it was checked against
    assert last.centre_deflection <= 1.05 * last.load_factor * k0 * (1.0 + 1e-9)
    report = path_admissibility(path)
    assert report["admissible"] is True