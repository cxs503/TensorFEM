"""Cylindrical shell4: rigid-translation defect and the Scordelis-Lo residual.

These tests pin the attribution published in
``docs/CASE-FEM-SCORDELIS-LO-ROOF-reaction-residual-attribution.md``:

* the metric-consistent cylindrical shell4 annihilates the *axial* rigid
  translation exactly;
* it does **not** annihilate the vertical one, and the spurious strain follows
  the closed form ``eps,gamma = pitch^2 (cos,sin)(theta_c) / (12 R^3)``;
* consequently the Scordelis-Lo reaction residual is exactly the rigid-mode
  projection ``(K 1_z)^T u / (-1^T f)`` and contracts as ``h^2``;
* the flat-facet family on the same problem is exact, so this is a curved-element
  discretisation limit, not a load or reaction-extraction defect.

Kept deliberately coarse (4x4, 8x8) so the file runs in seconds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

STUDY_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "fem_shell_scordelis_lo"
if str(STUDY_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(STUDY_DIR))

import reaction_residual_study as study  # noqa: E402


def test_library_exposes_assembled_stiffness():
    res = study.scordelis_lo_cylindrical(4, 4)
    assert res.stiffness is not None
    assert res.stiffness.shape == (res.displacement.numel(),) * 2
    # the recovered load K u - r is the assembled vertical surface load: no other
    # component is loaded, and the resultant is the applied pressure * area.
    f = res.stiffness @ res.displacement - res.reaction
    others = torch.cat([f[k::6] for k in (0, 1, 3, 4, 5)])
    assert float(others.abs().max()) < 1e-6
    applied = 90.0 * study.L0 * study.R0 * 2.0 * study.HALF
    assert float(f[2::6].sum()) == pytest.approx(-applied, rel=1e-12)


def test_axial_rigid_translation_is_exact():
    facet = study.facet_energy(study.PITCH_REF, study.R0)
    scale = study.E * study.T * study.PITCH_REF
    assert facet["norm_K_1x"] / scale < 1e-12
    assert abs(facet["energy_1x"]) / (scale * study.PITCH_REF) < 1e-12


@pytest.mark.parametrize("pitch", [study.PITCH_REF, 2.0 * study.PITCH_REF])
def test_vertical_translation_spurious_strain_matches_closed_form(pitch):
    facet = study.facet_energy(pitch, study.R0)
    assert not facet["round_off_limited"]
    assert facet["spurious_strain_rms"] > 0.0            # K 1_z != 0
    assert facet["strain_over_predicted"] == pytest.approx(1.0, rel=0.05)


def test_spurious_energy_scales_with_pitch_fourth_power():
    coarse = study.facet_energy(study.PITCH_REF, study.R0)["spurious_energy_density"]
    fine = study.facet_energy(0.5 * study.PITCH_REF, study.R0)["spurious_energy_density"]
    # halving the pitch must divide the energy by ~16 (strain ~ pitch^2)
    assert coarse / fine == pytest.approx(16.0, rel=0.05)


def test_spurious_energy_scales_with_radius_to_the_minus_six():
    r25 = study.facet_energy(study.CURV_PITCH, 25.0)["spurious_energy_density"]
    r50 = study.facet_energy(study.CURV_PITCH, 50.0)["spurious_energy_density"]
    # doubling the radius must divide the energy by ~64 (strain ~ 1/R^3)
    assert r25 / r50 == pytest.approx(64.0, rel=0.05)


def test_reaction_residual_is_the_rigidmode_projection():
    coarse = study.level(4)
    fine = study.level(8)
    for lvl in (coarse, fine):
        assert lvl["residual_rel"] > 0.0                 # rails are still carrying load
        assert lvl["identity_difference"] < 1e-10
        assert lvl["residual_rel"] == pytest.approx(lvl["rigidmode_identity"], abs=1e-10)
    # h^2 contraction: doubling the mesh quarters the residual
    assert coarse["residual_rel"] / fine["residual_rel"] == pytest.approx(4.0, rel=0.10)


def test_flat_facet_control_is_exact():
    control = study.flat_facet_control(4)
    assert abs(control["residual_rel"]) < 1e-9


def test_mesh_strain_matches_integrated_closed_form():
    for nx in (4, 8):
        lvl = study.level(nx)
        assert lvl["predicted_strain_ratio"] == pytest.approx(1.0, rel=0.05)