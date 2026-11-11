"""Convergence / consistency tests for the curved-interface penalty mortar coupling.

Backs the convergence statement in
``docs/mortar-curved-interface-convergence.md`` (anchor GAP-FEM-MORTAR).
The planar patch tests already in the suite prove exactness on flats; they do
not say how fast the transfer error decays on a *curved* interface, nor how the
error splits into geometry (faceting) and quadrature parts.  Three independent
references are used here:

  * exact circle + radial normal gap  -> adaptive quadrature (`oracle_radial`)
  * exact integral of the faceted model -> 8-point Gauss (`oracle_facet`)
  * closed-form paraboloid limit -> cross-check of the first

Kept small (a few thousand facet projections) so the suite stays fast; the
production study lives in
``benchmarks/mortar_curved_interface_convergence.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import mortar_curved_interface_convergence as conv  # noqa: E402

from tensorfem.mortar_contact3d import integrate_mortar_contact  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_oracle_is_self_consistent():
    chk = conv.oracle_selfcheck()
    # the paraboloid closed form is the O(delta/R) limit of the exact circle
    assert chk["rel_gap_paraboloid"] < 0.02, chk
    # F = dE/d(delta) must hold for the quadrature oracle itself
    assert chk["rel_gap_energy_identity"] < 1e-6, chk
    # normal gap vs classical vertical gap differ at O((a/R)^2), a real effect
    assert 1e-3 < chk["rel_normal_vs_vertical"] < 2e-2, chk


def test_planar_patch_is_exact_at_every_refinement():
    for r in conv.series_flat_patch():
        assert r["rel_err_F"] < 1e-12, r
        assert r["rel_err_E"] < 1e-12, r


def test_faceting_error_is_second_order():
    rows = conv.series_faceting(levels=(16, 32, 64, 128))
    order = conv.observed_order([r["h_master"] for r in rows],
                                [r["faceting_err_F"] for r in rows])
    assert order > 1.8, (order, rows)
    assert rows[-1]["faceting_err_F"] < rows[0]["faceting_err_F"]


def test_assembly_matches_the_faceted_model():
    """FE must reproduce the exact integral of its own (faceted) description."""
    d = conv.equilibrium(0.0, n_slave=40, n_master=32)
    fac = conv.oracle_facet(0.0, 32)
    assert conv._rel(d["F"], fac["F"]) < 2e-3, (d["F"], fac["F"])
    assert conv._rel(d["E"], fac["E"]) < 2e-3, (d["E"], fac["E"])


def test_production_mesh_is_below_three_percent_and_balanced():
    rad = conv.oracle_radial(0.0)
    d = conv.equilibrium(0.0, n_slave=40, n_master=32)
    assert abs(d["F"] - rad["F"]) / rad["F"] < 0.03, (d["F"], rad["F"])
    assert abs(d["E"] - rad["E"]) / rad["E"] < 0.03, (d["E"], rad["E"])
    assert d["balance_roundoff"] / abs(d["F"]) < 1e-12, d["balance_roundoff"]
    assert abs(d["pen_max"] - rad["pen_max"]) / rad["pen_max"] < 0.01


def test_tilted_contact_matches_analytical_moment():
    """Asymmetric contact: the mortar moment must match the exact integral."""
    rad = conv.oracle_radial(conv.TILT)
    assert abs(rad["M"]) > 1.0
    d = conv.equilibrium(conv.TILT, n_slave=40, n_master=32)
    assert abs(d["M"] - rad["M"]) / abs(rad["M"]) < 0.03, (d["M"], rad["M"])


def test_penetration_profile_tracks_exact_circle():
    prof = conv.penetration_profile(20, 32)
    rad = conv.oracle_radial(0.0)
    assert prof
    assert max(r["pen"] for r in prof) > 0.0
    assert abs(max(r["pen"] for r in prof) - rad["pen_max"]) / rad["pen_max"] < 0.01


def test_invalid_master_geometry_still_fails_closed():
    sv, sf = conv.slave_plate(10)
    mv, mf = conv.master_cylinder(8)
    mv[5] = mv[4]  # collapse a master facet
    with pytest.raises(ValueError):
        integrate_mortar_contact(sv, sf, mv, mf, normal_penalty=conv.EPS_N)