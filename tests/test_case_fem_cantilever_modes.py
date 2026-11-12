"""Regression gate for CASE-FEM-CANTILEVER-MODES.

The case is the calculation-pool anchor ``CASE-FEM-CANTILEVER-MODES``: the first
three natural frequencies of a clamped-free beam against Euler--Bernoulli beam
theory (Blevins 1979, Table 8-1).  These tests re-derive the reference from the
governing transcendental equation and re-run the two cheap model levels (1-D
Euler--Bernoulli beam FE and 3-D HEX20 solid at a coarse mesh), so a regression
in either the beam eigenvalue solver or the solid modal path is caught without
paying for the full study.  The heavy convergence/sensitivity sweep lives in
``benchmarks/case_fem_cantilever_modes.py``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "case_fem_cantilever_modes",
    ROOT / "benchmarks" / "case_fem_cantilever_modes.py",
)
_case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_case)

GEOMETRY = _case.GEOMETRY
MATERIAL = _case.MATERIAL
REFERENCE_MESH = _case.REFERENCE_MESH
TOLERANCE = _case.TOLERANCE
attach_reference = _case.attach_reference
eb_level = _case.eb_level
hex20_level = _case.hex20_level
reference_table = _case.reference_table
root_residual = _case.root_residual
sections = _case.sections
structure_modes = _case.structure_modes
study_mode_shapes = _case.study_mode_shapes


@pytest.fixture(scope="module")
def reference():
    return reference_table()


def test_reference_roots_satisfy_characteristic_equation(reference):
    for beta_l in reference["roots"]["beta_l"]:
        assert abs(root_residual(beta_l)) < 1e-10


def test_reference_roots_match_blevins_table_8_1(reference):
    """Independent cross-check against the documented table values."""
    for computed, documented in zip(reference["roots"]["beta_l"],
                                    reference["roots"]["documented_blevins_table_8_1"]):
        assert abs(computed - documented) / documented < 1e-8


def test_reference_frequency_ratios(reference):
    weak = reference["weak"]
    assert weak["ratios"][1] == pytest.approx(6.2669, rel=1e-4)
    assert weak["ratios"][2] == pytest.approx(17.5472, rel=1e-4)
    # strong axis scales by sqrt(I_strong / I_weak)
    sec = sections()
    scale = math.sqrt(sec["strong"]["second_moment"] / sec["weak"]["second_moment"])
    assert reference["strong"]["frequencies_hz"][0] == pytest.approx(
        weak["frequencies_hz"][0] * scale, rel=1e-12)


def test_beam_theory_formula_closed_form(reference):
    """f = (beta L)^2 / (2 pi L^2) sqrt(EI/(rho A)) evaluated independently."""
    sec = sections()
    beta_l = reference["roots"]["beta_l"][0]
    ref = (beta_l ** 2 / (2.0 * math.pi * sec["length"] ** 2)
           * math.sqrt(MATERIAL["young"] * sec["weak"]["second_moment"]
                       / (MATERIAL["density"] * sec["area"])))
    assert reference["weak"]["frequencies_hz"][0] == pytest.approx(ref, rel=1e-14)


def test_euler_bernoulli_beam_fe_within_tolerance(reference):
    level = eb_level(REFERENCE_MESH["eb"])
    families = {fam: level[fam]["frequencies_hz"] for fam in ("weak", "strong")}
    rows = attach_reference(families, reference)
    assert [r["structure_mode"] for r in rows] == [1, 2, 3]
    for row in rows:
        assert row["rel_error"] < TOLERANCE, row


def test_euler_bernoulli_frequency_ordering(reference):
    """f1 = weak 1, f2 = strong 1, f3 = weak 2 for b/h = 2."""
    level = eb_level(REFERENCE_MESH["eb"])
    families = {fam: level[fam]["frequencies_hz"] for fam in ("weak", "strong")}
    labels = [row["family"] for row in structure_modes(families)]
    assert labels == ["weak", "strong", "weak"]
    rows = attach_reference(families, reference)
    assert [r["family"] for r in rows] == [reference["weak"]["label"],
                                           reference["strong"]["label"],
                                           reference["weak"]["label"]]


def test_beam_convergence_order_is_positive(reference):
    """Refined meshes must be at least as accurate as the coarse one."""
    coarse = eb_level(8)
    fine = eb_level(32)
    for fam in ("weak", "strong"):
        err_c = [abs(f - r) / r for f, r in
                 zip(coarse[fam]["frequencies_hz"], reference[fam]["frequencies_hz"])]
        err_f = [abs(f - r) / r for f, r in
                 zip(fine[fam]["frequencies_hz"], reference[fam]["frequencies_hz"])]
        for c, f in zip(err_c, err_f):
            assert f <= c + 1e-15


def test_mode_shapes_match_analytic(reference):
    shapes = study_mode_shapes()
    for fam in ("weak", "strong"):
        for row in shapes[fam]["rows"]:
            assert row["shape_error_l2"] < 1e-3, (fam, row)


def test_hex20_solid_modal_within_tolerance(reference):
    """Coarse 3-D solid check (nx=8): exercises the solid modal path end to end."""
    level = hex20_level(8)
    rows = attach_reference(level["families"], reference)
    assert len(rows) == 3
    for row in rows:
        assert row["rel_error"] < TOLERANCE, row


def test_hex20_mode_classification(reference):
    """First three modes must be bending (no axial/torsion leaking into rank 1-3)."""
    level = hex20_level(8)
    labels = {reference["weak"]["label"], reference["strong"]["label"]}
    for mode in level["modes"][:3]:
        assert mode["shape"] in labels, mode
    assert level["modes"][0]["shape"] == reference["weak"]["label"]


def test_geometry_declared_slenderness():
    sec = sections()
    assert GEOMETRY["length"] / GEOMETRY["height"] == pytest.approx(75.0)
    assert GEOMETRY["width"] / GEOMETRY["height"] == pytest.approx(2.0)
    assert sec["area"] == pytest.approx(GEOMETRY["width"] * GEOMETRY["height"])
