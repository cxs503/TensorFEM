"""Contact-box closure tests for GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.

The anchor's patch load is solved on a finite analysis box.  A relief-driven
contact patch closes (``p -> 0``) at a finite radius; a box smaller than that
radius makes ``contact_area``/``mean``/``peak_pressure`` describe the truncation
instead of the ice--structure contact.  These tests pin the diagnostic
(:func:`boundary_pressure_ratio`, :attr:`ContactPatch.contact_closed`), the
auto-sizing of a derived box, and the magnitude of the bias that the previously
shipped 0.10 m box produced at the production 40 kN force.

Closed-form reference (section A of
``tools/verify_gap_pp1_contact_box_closure.py``): paraboloid indenter on a
linear penalty (Winkler) foundation,
``delta0 = sqrt(2 amp F/(pi k R^2))``, ``p0 = k delta0``,
``r_c = sqrt(delta0 R^2/amp)``, ``mean = p0/2``.
"""

from __future__ import annotations

import math

import pytest

from tensorfem.ice_panel_patch_contact import (
    CLOSURE_TOL,
    coupled_patch_solve,
    czm_contact_patch,
    default_configurations,
    default_law,
)

PANEL = (0.8, 0.7)
CENTRE = (0.4, 0.35)
FORCE = 4.0e4          # production force of nonuniformity_scan
CEILING = 2.0e6
R, AMP = 1.0, 0.02
K = default_law().stiffness


def _analytic(force: float = FORCE) -> dict:
    d0 = math.sqrt(2.0 * AMP * force / (math.pi * K * R**2))
    rc = math.sqrt(d0 * R**2 / AMP)
    return {"delta0": d0, "p0": K * d0, "rc": rc, "area": math.pi * rc * rc}


def _solve(**kw):
    edge = {"kind": "spherical", "radius": R, "amplitude": AMP, **kw.pop("edge", {})}
    return czm_contact_patch(
        law=default_law(),
        ceiling=CEILING,
        edge=edge,
        total_force=FORCE,
        panel=PANEL,
        centre=CENTRE,
        **kw,
    )


def test_solver_recovers_the_closed_form_at_the_operating_force():
    """The analytic check that shipped was run at 7.07 kN; the production force
    is 40 kN.  At 40 kN with a box that contains the contact the solver must
    still hit the closed form -- that is the regime the scan actually uses."""
    ref = _analytic()
    patch = _solve(grid=61)  # box derived and grown automatically
    assert patch.contact_closed is True
    assert patch.boundary_pressure_ratio <= CLOSURE_TOL
    assert patch.peak_pressure == pytest.approx(ref["p0"], rel=1e-2)
    assert patch.approach == pytest.approx(ref["delta0"], rel=1e-2)
    assert patch.contact_area == pytest.approx(ref["area"], rel=1e-2)
    mean = patch.total_force / patch.contact_area
    assert patch.peak_pressure / mean == pytest.approx(2.0, rel=2e-2)


def test_derived_box_is_grown_until_the_contact_closes():
    """No ``box_half`` and no ``contact_box``: the returned box must be larger
    than the old heuristic 2*amp + 0.05 = 0.09 m and the patch must be closed."""
    patch = _solve(grid=61)
    x0, x1, _, _ = patch.config["contact_box"]
    half = 0.5 * (x1 - x0)
    assert half > 2.0 * AMP + 0.05
    assert half > _analytic()["rc"]
    assert patch.contact_closed is True


def test_truncating_box_is_flagged_and_raises_when_required():
    """The shipped 0.10 m box cuts the 40 kN contact: the diagnostic must say so."""
    patch = _solve(grid=97, edge={"box_half": 0.10})
    assert patch.contact_closed is False
    assert patch.boundary_pressure_ratio > 0.4
    with pytest.raises(ValueError, match="truncated"):
        _solve(grid=41, edge={"box_half": 0.10}, require_closed=True)


def test_truncation_inflates_the_peak_pressure_and_shrinks_the_contact_area():
    """Quantify the bias: +7 % peak, -29 % contact area at 40 kN."""
    truncated = _solve(grid=97, edge={"box_half": 0.10})
    closed = _solve(grid=97, edge={"box_half": 0.20})
    assert closed.contact_closed is True
    assert truncated.peak_pressure / closed.peak_pressure > 1.05
    assert truncated.contact_area / closed.contact_area < 0.80
    # the analytic value belongs to the closed box, not to the truncated one
    ref = _analytic()
    assert closed.peak_pressure == pytest.approx(ref["p0"], rel=1e-2)
    assert truncated.peak_pressure / ref["p0"] > 1.05


def test_flat_edge_has_no_closure_to_diagnose():
    """A flat edge fills whatever box it is given; ``contact_closed`` is None."""
    patch = czm_contact_patch(
        law=default_law(),
        ceiling=CEILING,
        edge={"kind": "flat", "amplitude": 0.0, "box_half": 0.10},
        total_force=FORCE,
        panel=PANEL,
        centre=CENTRE,
        grid=41,
    )
    assert patch.contact_closed is None
    assert patch.boundary_pressure_ratio == 0.0


def test_coupled_patch_grows_its_box_and_reports_closure():
    """Two-way coupling widens the contact (the panel deflects away from the
    ice), so a box sized on the one-way solve is not enough: the coupled solve
    has to grow it on the coupled fixed point."""
    edge = {"kind": "spherical", "radius": R, "amplitude": AMP}
    patch, _, coup = coupled_patch_solve(
        nx=12, ny=10, panel=PANEL, law=default_law(), ceiling=CEILING,
        edge=edge, total_force=FORCE, centre=CENTRE, grid=33,
        tol=1e-4, max_coupling=40,
    )
    assert coup["contact_closed"] is True
    assert coup["boundary_pressure_ratio"] <= CLOSURE_TOL
    assert coup["converged"] is True

    # the historically shipped box truncates the coupled patch, and the
    # truncation is larger than in the one-way case
    _, _, coup_trunc = coupled_patch_solve(
        nx=12, ny=10, panel=PANEL, law=default_law(), ceiling=CEILING,
        edge={**edge, "box_half": 0.10}, total_force=FORCE, centre=CENTRE,
        grid=33, tol=1e-4, max_coupling=40,
    )
    assert coup_trunc["contact_closed"] is False
    assert coup_trunc["boundary_pressure_ratio"] > 0.4


def test_shipped_configurations_close_and_converge():
    """Acceptance gate for :func:`default_configurations`: no configuration may
    return a truncated contact patch, and every coupling loop must converge."""
    for cfg in default_configurations():
        edge = dict(cfg["edge"])
        centre = (
            (0.5 * PANEL[0] + cfg["offset"][0], 0.5 * PANEL[1] + cfg["offset"][1])
            if cfg.get("offset") else None
        )
        if cfg["coupled"]:
            patch, _, coup = coupled_patch_solve(
                nx=12, ny=10, panel=PANEL, law=default_law(), ceiling=CEILING,
                edge=edge, total_force=FORCE, centre=centre, grid=41,
                tol=1e-4, max_coupling=40,
            )
            closed, bnd, conv = coup["contact_closed"], coup["boundary_pressure_ratio"], coup["converged"]
        else:
            patch = czm_contact_patch(
                law=default_law(), ceiling=CEILING, edge=edge, total_force=FORCE,
                panel=PANEL, centre=centre, grid=41,
            )
            closed, bnd, conv = patch.contact_closed, patch.boundary_pressure_ratio, True
        assert closed is not False, (cfg["name"], bnd, patch.config.get("contact_box"))
        assert bnd <= CLOSURE_TOL, (cfg["name"], bnd)
        assert conv, cfg["name"]