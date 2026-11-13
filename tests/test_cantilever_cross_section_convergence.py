"""CASE-CANTILEVER-TIP -- guards for the cross-section convergence bundle.

The numbers live in ``benchmarks/verified/fem_cantilever_shear/cross_section_convergence.json``.
The published slenderness ladder (§5 of ``docs/CASE-CANTILEVER-TIP-shear-validity.md``) had to
report "the refinement trend ... is reported instead of a converged value" because the dense
HEX20 solve capped the model near 5e3 dof.  This module keeps three things honest:

1. the new sparse path is *the same discretisation* as the dense one -- it is certifed against
   the dense solve and against the tip deflections already published in ``result.json``;
2. the extrapolations quoted in the document are recomputed here from the stored sweep, so the
   prose cannot drift away from the data;
3. the scientific verdict is recorded as an honest FAIL (the metric is not converged and is not
   rung-independent) and the tests fail loudly if anyone silently upgrades it to a pass.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "verified" / "fem_cantilever_shear"
EVIDENCE = BENCH / "cross_section_convergence.json"
PUBLISHED = BENCH / "result.json"
DOC = REPO / "docs" / "CASE-CANTILEVER-TIP-cross-section-convergence.md"
FIG = REPO / "docs" / "assets" / "cantilever_shear" / "cross_section_convergence.png"

for extra in (REPO / "src", BENCH):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture(scope="module")
def evidence() -> dict:
    if not EVIDENCE.exists():
        pytest.skip("cross_section_convergence.json not generated")
    return json.loads(EVIDENCE.read_text())


@pytest.fixture(scope="module")
def module():
    import cross_section_convergence as cx

    return cx


def _level(row: dict, n: int) -> dict:
    return next(lvl for lvl in row["levels"] if lvl["cross_subdivisions"] == n)


# --------------------------------------------------------------------------- #
# 1. the sparse path is the same discretisation, not a different model
# --------------------------------------------------------------------------- #
def test_sparse_matches_dense_on_published_mesh(module):
    """SuperLU vs LAPACK on one published mesh: identical to round-off."""
    dense = module.cantilever(4.0, (30, 2, 2), sparse=False)
    sparse = module.cantilever(4.0, (30, 2, 2), sparse=True)
    rel = float((dense["displacement"] - sparse["displacement"]).norm()
                / dense["displacement"].norm())
    assert rel < 1e-9, rel
    assert abs(sparse["tip_deflection_m"] - dense["tip_deflection_m"]) / dense["tip_deflection_m"] < 1e-9


def test_sparse_reproduces_published_tip_deflections(module, evidence):
    """The sparse solve must land on the numbers already published in result.json."""
    published = json.loads(PUBLISHED.read_text())
    by_rung = {row["slenderness_L_over_h"]: row["solid_hex20_levels"] for row in published["ladder"]}
    for record in evidence["sparse_crosscheck"]:
        assert record["published_relative_difference"] < 1e-9, record
    # one independent re-solve, so the check is not only a comparison of two JSON files
    out = module.cantilever(4.0, (40, 2, 2), sparse=True)
    reference = next(
        lvl["tip_deflection_m"] for lvl in by_rung[4.0]
        if lvl["mesh"] == {"axial": 40, "cross_y": 2, "cross_z": 2}
    )
    assert abs(out["tip_deflection_m"] - reference) / reference < 1e-9


def test_sparse_path_removes_the_dof_cap(evidence):
    """The published ladder could not exceed ~5e3 dof; the new path must."""
    heaviest = max(lvl["dofs"] for row in evidence["ladder"] for lvl in row["levels"])
    assert heaviest >= 50_000, heaviest
    assert evidence["sparse_crosscheck"], "cross-check block is empty"


# --------------------------------------------------------------------------- #
# 2. the quoted extrapolation is the one the data supports
# --------------------------------------------------------------------------- #
def test_richardson_limit_is_recomputed_from_the_sweep(evidence):
    """Re-derive the L/h = 4 limit from the stored levels; it must match the record."""
    row = next(r for r in evidence["ladder"] if r["slenderness_L_over_h"] == 4.0)
    fine, coarse = row["levels"][-1], row["levels"][-2]
    r = fine["relative_element_size"] / coarse["relative_element_size"]
    r = 1.0 / r
    limit = fine["shear_recovery"] + (fine["shear_recovery"] - coarse["shear_recovery"]) / (r**2 - 1.0)
    assert math.isclose(limit, row["richardson"]["extrapolated_limit"], rel_tol=1e-9)


def test_observed_order_supports_the_quadratic_extrapolation(evidence):
    """O(h^2) is defensible at the thickest rung; reported where it is not."""
    row = next(r for r in evidence["ladder"] if r["slenderness_L_over_h"] == 4.0)
    assert row["observed_order"] is not None
    assert 1.7 <= row["observed_order"] <= 2.3, row["observed_order"]


def test_converged_residual_vs_timoshenko_is_the_documented_one(evidence):
    """The document quotes a converged ~-1.7 % at L/h = 4; recompute it from the limit."""
    row = next(r for r in evidence["ladder"] if r["slenderness_L_over_h"] == 4.0)
    rec = row["richardson"]["extrapolated_limit"]
    gap = row["shear_fraction_of_eb"]
    err_vs_timoshenko = ((1.0 + rec * gap) - (1.0 + gap)) / (1.0 + gap)
    assert -0.020 < err_vs_timoshenko < -0.014, err_vs_timoshenko
    # ... and it stays inside the case-pool tolerance, which is the anchor's own verdict.
    assert abs(err_vs_timoshenko) < 0.03


# --------------------------------------------------------------------------- #
# 3. honest FAIL: the debt is not closed, and must stay recorded as open
# --------------------------------------------------------------------------- #
def test_infrastructure_gates_pass(evidence):
    assert evidence["infrastructure_passed"] is True
    for gate in evidence["gates"]["infrastructure"]:
        assert gate["pass"] is True, gate


def test_scientific_claims_are_recorded_as_failed(evidence):
    """If a future run converges, this test is the place to re-grade -- not a silent edit."""
    claim = {g["name"]: g for g in evidence["gates"]["claim"]}
    assert evidence["claim_supported"] is False
    assert claim["fine_increment_below_1pp"]["pass"] is False
    assert claim["timoshenko_recovery_inside_extrapolation_band"]["pass"] is False
    assert claim["recovery_rung_independent"]["pass"] is False
    assert evidence["timoshenko_supported"] is False


def test_shear_recovery_is_not_rung_independent(evidence):
    """A kappa measurement is a section property; the metric varies by >1.5 across rungs."""
    finest = [row["finest_recovery"] for row in evidence["ladder"]]
    assert max(finest) - min(finest) > 0.5
    assert max(finest) > 0.0 and min(finest) < 0.0, "sign change across rungs is the finding"


def test_published_slender_residual_shrinks_under_section_refinement(evidence):
    """The falsified attribution: the -0.40 % 'clamp' residual is largely discretisation."""
    published = {r["slenderness_L_over_h"]: r["error_vs_eb"] for r in evidence["published_residuals"]}
    refined = {row["slenderness_L_over_h"]: row["finest_error_vs_eb"] for row in evidence["ladder"]}
    # L/h = 20: published -0.38 % on a (32,2,2) mesh, refined at least a factor 2 smaller in
    # magnitude, i.e. more than half of the published residual was mesh error.
    assert abs(refined[20.0]) < 0.5 * abs(published[20.0]), (refined[20.0], published[20.0])
    # L/h = 10: the published residual (-0.146 %) even changes sign once the section refines.
    assert refined[10.0] > 0.0 > published[10.0]


# --------------------------------------------------------------------------- #
# 4. published artefacts
# --------------------------------------------------------------------------- #
def test_document_and_figure_exist():
    assert DOC.exists(), f"missing {DOC}"
    assert FIG.exists(), f"missing {FIG}"
    text = DOC.read_text()
    for section in ("①", "②", "③", "④", "⑤", "⑥"):
        assert section in text, section
    # the honest verdicts must be written down, not just machine-recorded
    assert "honest FAIL" in text or "FAIL" in text
    assert "未闭合" in text or "not closed" in text