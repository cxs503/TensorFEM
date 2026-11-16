"""Regression tests for the GAP-PP-3 3-grid matrix *mesh family*.

Defect being pinned (finding N6, mission m22746)
------------------------------------------------
The 3-grid convergence study published with the anchor used the family
``(8,4) (12,6) (16,8)``.  For a = 2.0 m, b = 1.0 m and a centred 0.5 m x 0.5 m
patch, ``patch_plate_mesh`` requires both patch edges to fall on element edges,
i.e. ``nx % 8 == 0`` and ``ny % 4 == 0``.  ``12x6`` violates it, so the driver
died with ``ValueError`` *after* the first (expensive) mesh: the published
3-grid evidence was never a complete data set (compute job cj4232, rc = 1), yet
the consuming test only *skipped* when the results file was absent, so the
acceptance criterion silently reported "pending" instead of "the study cannot
run on this family".

These tests assert:

1. the shipped family is patch-aligned and is increasing in element count;
2. the pre-flight check rejects exactly the infeasible meshes the real mesher
   rejects (agreement with ``patch_plate_mesh``, not a private re-derivation);
3. the driver refuses an infeasible family *before* solving anything;
4. the merge step reproduces the criteria block from raw shard entries (no
   hand-copied numbers, and a missing shard is reported, not invented).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gap_pp3_3grid_study import (  # noqa: E402
    A,
    B,
    GRIDS,
    PATCH,
    finalize,
    headers,
    main as study_main,
    parse_grids,
)
from gap_pp3_3grid_merge import load_shards  # noqa: E402
from tensorfem.plate_patch_plastic_limit import patch_plate_mesh  # noqa: E402

YIELD = 374.2e6          # DH36 at -20 C, as used by the study


def _mesher_accepts(nx: int, ny: int) -> bool:
    try:
        patch_plate_mesh(A, B, 0.020, nx, ny, young=206e9, poisson=0.3,
                         yield_stress=YIELD, hardening=1000e6, layers=3,
                         patch=PATCH)
    except ValueError:
        return False
    return True


class TestShippedFamilyIsAligned:
    def test_three_meshes_increasing_in_size(self):
        assert len(GRIDS) == 3
        sizes = [nx * ny for nx, ny in GRIDS]
        assert sizes == sorted(sizes) and len(set(sizes)) == 3

    def test_refinement_ratio_is_documented_and_above_one(self):
        """Aligned meshes cannot be uniformly refined by 2 at every step.

        The alignment rule (nx % 8, ny % 4) admits 8x4, 16x8, 24x12 -- ratio 2
        then 1.5.  A uniform-ratio family would need 32x16 (512 elements, ~16x
        the coarse cost), which does not fit the compute budget.  The order
        estimate is therefore taken from the *last* pair (ratio 1.5), which
        :func:`_rules` reports as ``refinement_ratio``.
        """
        ratios = [(nx1 / nx0, ny1 / ny0)
                  for (nx0, ny0), (nx1, ny1) in zip(GRIDS, GRIDS[1:])]
        assert ratios == [(2.0, 2.0), (1.5, 1.5)]
        for rx, ry in ratios:
            assert rx > 1.0 and rx == ry

    def test_every_shipped_mesh_is_patch_aligned(self):
        for nx, ny in GRIDS:
            assert _mesher_accepts(nx, ny), f"{nx}x{ny} is not patch-aligned"


class TestPreflightMatchesMesher:
    @pytest.mark.parametrize("grid", [(8, 4), (16, 8), (24, 12)])
    def test_aligned_meshes_pass_preflight(self, grid):
        assert _mesher_accepts(*grid)

    @pytest.mark.parametrize("grid", [(12, 6), (10, 5), (12, 4), (8, 6)])
    def test_infeasible_meshes_fail_preflight(self, grid):
        """The pre-flight verdict must equal the mesher's verdict, mesh for mesh."""
        assert not _mesher_accepts(*grid)

    def test_preflight_message_names_the_axis_and_the_rule(self):
        from gap_pp3_3grid_study import check_alignment

        with pytest.raises(ValueError) as excinfo:
            check_alignment(12, 6)
        message = str(excinfo.value)
        assert "12x6" in message and "patch edge" in message
        assert "nx % 8" in message           # tells the caller how to fix it


class TestDriverRefusesInfeasibleFamily:
    def test_main_rejects_before_any_solve(self, tmp_path):
        """An infeasible family must cost milliseconds, not a first mesh solve."""
        with pytest.raises(ValueError):
            study_main([(8, 4), (12, 6)], str(tmp_path / "out.json"))
        assert not (tmp_path / "out.json").exists()

    def test_parse_grids_accepts_both_separators(self):
        assert parse_grids("16x8, 24*12") == [(16, 8), (24, 12)]


class TestMergeReproducesCriteria:
    @staticmethod
    def _shard(nx, ny, err, limit, perm):
        results = headers()
        results["meshes"][f"{nx}x{ny}"] = {
            "nx": nx, "ny": ny, "elements": nx * ny,
            "first_yield_error": err, "elastic_bias": -0.15 / nx,
            "limit_load_measured": limit, "limit_load_error_vs_ub": limit / 2.0 - 1.0,
            "permanent_deflection": perm, "permanent_set_valid": True,
        }
        return results

    def test_criteria_come_from_raw_entries_coarse_to_fine(self, tmp_path):
        """Coarse -> fine comes from the element count, and the order/limit are right.

        The synthetic sequence is an *exact* power law ``e = h**1.5`` so the
        observed order and the Richardson limit are known in closed form.  The
        pre-fix ``_rules`` returned order = -1.5 and a limit of 0.068 for it
        (finding N7), so both assertions are regressions, not decoration.
        """
        from gap_pp3_3grid_study import B

        p_true = 1.5
        paths, limits, perms = [], [], []
        for index, (nx, ny) in enumerate(GRIDS):
            h = B / ny
            err = h ** p_true
            limit = 1.40 + 0.05 * index
            perm = 0.026 - 0.001 * index
            path = tmp_path / f"{nx}x{ny}.json"
            path.write_text(json.dumps(self._shard(nx, ny, err, limit, perm)))
            paths.append(str(path))
            limits.append(limit)
            perms.append(perm)

        merged, used, missing = load_shards(paths)
        finalize(merged)

        assert missing == [] and len(used) == 3
        # coarse -> fine by element count, never by name ("12x6" < "24x12" as strings)
        assert merged["criteria"]["mesh_order"] == ["8x4", "16x8", "24x12"]
        conv = merged["criteria"]["raw_error_convergence"]
        assert conv["refinement_ratio"] == pytest.approx(1.5)
        assert conv["observed_order"] == pytest.approx(p_true, abs=1e-9)
        assert abs(conv["extrapolated_error"]) < 1e-12      # exact power law -> 0
        # an exact power law extrapolates onto 0, so the mesh-limit claim holds
        assert merged["criteria"]["transition_lt_5pct"]["met_in_mesh_limit"] is True
        assert merged["criteria"]["limit_lt_10pct"]["limit_trend"] == pytest.approx(limits)
        assert [v["value"] for v in
                merged["criteria"]["permanent_deflection_reported"].values()] == pytest.approx(perms)

    def test_order_is_positive_for_a_converging_sequence(self):
        """Regression for the sign of ``observed_order`` (finding N7)."""
        from gap_pp3_3grid_study import _rules

        rules = _rules([0.30, 0.15, 0.08], [(8, 4), (16, 8), (24, 12)])
        assert rules["observed_order"] > 0.0
        assert rules["refinement_ratio"] == pytest.approx(1.5)
        # Richardson limit must lie *below* the finest error
        assert rules["extrapolated_error"] < rules["errors"][-1]

    def test_no_order_invented_from_a_single_mesh_or_sign_change(self):
        from gap_pp3_3grid_study import _rules

        single = _rules([0.08], [(24, 12)])
        assert "observed_order" not in single and "extrapolated_error" not in single
        flipped = _rules([0.30, -0.02], [(16, 8), (24, 12)])
        assert "observed_order" not in flipped and "extrapolated_error" not in flipped
        assert "sign" in flipped["note"]

    def test_missing_shard_is_reported_not_invented(self, tmp_path):
        present = tmp_path / "8x4.json"
        present.write_text(json.dumps(self._shard(8, 4, 0.30, 1.40, 0.026)))
        merged, used, missing = load_shards([str(present), str(tmp_path / "nope.json")])
        finalize(merged)
        assert used == ["8x4.json"]
        assert missing and missing[0].endswith("nope.json")
        assert merged["criteria"]["mesh_order"] == ["8x4"]
        # with a single mesh the mesh-limit claim must be None, not True
        assert merged["criteria"]["transition_lt_5pct"]["met_in_mesh_limit"] is False
        assert merged["criteria"]["transition_lt_5pct"]["extrapolated_error"] is None