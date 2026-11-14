"""Gates on ``benchmarks/cohesive3d_verification_tables.{json,md}``.

The tables are the widest evidence artefact of the 3-D cohesive element: 29 rows
across two kinematics, each an oracle-vs-measurement statement.  Until this
module existed, nothing tied them to the code that produced them:

* the committed ``provenance`` block carried a *host* path
  (``/root/tp-agent/work/m16164/...``) of somebody else's worktree plus
  ``git_dirty: true``, i.e. it described a checkout that cannot be inspected;
* no source hash was recorded, so the numbers could outlive the module they
  describe.  They had: the committed ``specimen_compliance_a40mm`` row read
  ``C_fem = 4.715234e-05 m/N`` while the current code gives ``4.713796e-05``
  (3e-4 relative) and no test noticed;
* nothing hashed the rows, so a hand-edited ``value`` was indistinguishable from
  a regenerated one, and the ``.md`` render could drift from the ``.json``.

These tests make the pair a *gate*:

* **schema** -- every row carries ``check``/``oracle``/``measure``/``value``/
  ``tolerance``/``status``; ``passed``/``failed``/``known_limits`` really are
  counts of the rows;
* **tamper gate** -- ``checks_digest`` must hash the stored
  ``parameters + checks``, so editing one number breaks the digest without
  re-running the 40 s table;
* **freshness** -- every declared SHA-256 in ``source_files`` must still match
  the checkout, i.e. the table cannot outlive ``cohesive3d.py``;
* **no host leakage** -- no absolute path anywhere in the artefact;
* **markdown consistency** -- the rendered table is the JSON rows, in order,
  with the same statuses;
* **reproducibility** -- the cheap 23 rows are recomputed from this checkout and
  compared leaf by leaf with the committed numbers within
  ``atol + rtol * max(|a|, |b|)`` (``rtol = 1e-8``, ``atol = 1e-10``, declared in
  ``check_artifact`` before any result was seen).  The six multi-second rows are
  the ``--check`` CLI mode's job
  (``python benchmarks/cohesive3d_verification_tables.py --check``), so this
  module stays inside the default suite's budget.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "cohesive3d_verification_tables.py"
ARTIFACT = REPO / "benchmarks" / "cohesive3d_verification_tables.json"
MARKDOWN = REPO / "benchmarks" / "cohesive3d_verification_tables.md"

ROW_FIELDS = ("check", "oracle", "measure", "value", "tolerance", "status")
STATUSES = {"PASS", "KNOWN_LIMIT"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: The two rows that publish a measured limitation instead of a claim.
KNOWN_LIMIT_ROWS = {
    "element_rigid_rotation_small_strain",
    "small_strain_dissipation_at_rotation",
}

#: Rows only the full (slow) run produces; ``build_tables(quick=True)`` drops them.
SLOW_ROWS = {
    "corotational_tangent_vs_finite_difference",
    "corotational_dissipation_rotation_invariance",
    "corotational_dissipation_vs_fracture_energy",
    "small_strain_dissipation_at_rotation",
    "nonproportional_path_dissipation",
    "corotational_mesh_objectivity",
}


@pytest.fixture(scope="module")
def bench():
    """The table generator, imported (it sets the global torch default to float64).

    The default dtype is restored at teardown so this gate cannot change the
    numerics of any other test module in the same session.
    """
    prior = torch.get_default_dtype()
    spec = importlib.util.spec_from_file_location("cohesive3d_verification_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        torch.set_default_dtype(prior)


@pytest.fixture(scope="module")
def committed():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def quick_rows(bench):
    """The 23 cheap rows, recomputed once from this checkout."""
    prior = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return bench.build_tables(quick=True)
    finally:
        torch.set_default_dtype(prior)


# --------------------------------------------------------------------------- #
# Schema, counts, published statuses
# --------------------------------------------------------------------------- #


def test_artifact_schema(committed):
    for key in (
        "module",
        "parameters",
        "provenance",
        "checks",
        "passed",
        "failed",
        "known_limits",
        "checks_digest",
    ):
        assert key in committed, f"missing top-level key {key}"
    assert committed["module"] == "tensorfem.cohesive3d"
    rows = committed["checks"]
    assert len(rows) == 29, f"published table should have 29 rows, has {len(rows)}"
    for row in rows:
        assert tuple(row) == ROW_FIELDS, f"{row.get('check')}: fields {tuple(row)}"
        assert isinstance(row["value"], float)
        assert isinstance(row["tolerance"], float)


def test_counts_are_the_conjunction_of_the_rows(committed):
    rows = committed["checks"]
    assert committed["passed"] == sum(row["status"] == "PASS" for row in rows)
    assert committed["failed"] == sum(row["status"] == "FAIL" for row in rows)
    assert committed["known_limits"] == sum(row["status"] == "KNOWN_LIMIT" for row in rows)
    assert (committed["passed"], committed["failed"], committed["known_limits"]) == (27, 0, 2)


def test_statuses_are_published_and_no_row_fails(committed):
    rows = committed["checks"]
    assert {row["status"] for row in rows} <= STATUSES
    assert committed["failed"] == 0


def test_row_names_are_unique(committed):
    names = [row["check"] for row in committed["checks"]]
    assert len(names) == len(set(names))


def test_known_limit_rows_are_measurements_not_claims(committed):
    by_name = {row["check"]: row for row in committed["checks"]}
    assert set(by_name) >= KNOWN_LIMIT_ROWS
    for name in KNOWN_LIMIT_ROWS:
        row = by_name[name]
        assert row["status"] == "KNOWN_LIMIT", name
        assert abs(row["value"]) > 0.0, f"{name} must publish its measured error"
    # the small-strain rotation row is the documented failure, not a small residue
    assert by_name["element_rigid_rotation_small_strain"]["value"] > 1e-2


def test_every_row_has_an_oracle_and_a_measurement(committed):
    for row in committed["checks"]:
        assert row["oracle"].strip(), row["check"]
        assert row["measure"].strip(), row["check"]


def test_parameters_match_the_generator_constants(bench, committed):
    params = committed["parameters"]
    assert params["normal_stiffness_Pa_per_m"] == bench.K_N
    assert params["shear_stiffness_Pa_per_m"] == bench.K_S
    assert params["normal_strength_Pa"] == bench.T_N0
    assert params["shear_strength_Pa"] == bench.T_S0
    assert params["mode_i_energy_J_per_m2"] == bench.G_IC
    assert params["mode_ii_energy_J_per_m2"] == bench.G_IIC
    assert params["bk_eta"] == bench.ETA


# --------------------------------------------------------------------------- #
# Tamper gate
# --------------------------------------------------------------------------- #


def test_checks_digest_hashes_the_stored_rows(bench, committed):
    assert HEX64.match(committed["checks_digest"])
    assert committed["checks_digest"] == bench.checks_digest(
        committed["parameters"], committed["checks"]
    )


def test_checks_digest_breaks_on_a_hand_edited_number(bench, committed):
    import copy

    tampered = copy.deepcopy(committed)
    tampered["checks"][0]["value"] = tampered["checks"][0]["value"] + 1e-6
    assert (
        bench.checks_digest(tampered["parameters"], tampered["checks"])
        != (committed["checks_digest"])
    )


# --------------------------------------------------------------------------- #
# Freshness (the defect that motivated this module)
# --------------------------------------------------------------------------- #


def test_source_hashes_are_recorded_for_every_dependency(bench, committed):
    declared = committed["provenance"]["source_files"]
    assert set(declared) == set(bench.SOURCE_FILES), sorted(declared)
    for rel, digest in declared.items():
        assert HEX64.match(digest), f"{rel}: {digest!r} is not a SHA-256"
    # the module the table is about, and the generator itself, must both be covered
    assert "src/tensorfem/cohesive3d.py" in declared
    assert "benchmarks/cohesive3d_verification_tables.py" in declared


def test_source_hashes_match_the_checkout(bench, committed):
    """The table may not outlive the code it describes."""
    stale = bench.stale_sources(committed)
    assert stale == [], f"stale source hashes (regenerate the table): {stale}"


def test_artifact_records_this_checkout_and_no_host_path(committed):
    prov = committed["provenance"]
    assert prov["imported_from_this_checkout"] is True
    assert prov["tensorfem_module_file"] in {"src/tensorfem/__init__.py", "tensorfem/__init__.py"}
    blob = json.dumps(committed)
    assert "/root/" not in blob and "/home/" not in blob
    assert str(REPO) not in blob
    for rel in prov["source_files"]:
        assert not Path(rel).is_absolute(), rel


# --------------------------------------------------------------------------- #
# Markdown render is the JSON rows
# --------------------------------------------------------------------------- #


def test_markdown_matches_the_json(bench, committed):
    expected = [
        (row["check"], "KNOWN_LIMIT" if row["status"] == "KNOWN_LIMIT" else row["status"])
        for row in committed["checks"]
    ]
    assert bench.markdown_rows(MARKDOWN.read_text()) == expected


def test_markdown_is_the_render_of_the_committed_rows(bench, committed):
    assert MARKDOWN.read_text().strip() == bench.render_markdown(committed["checks"]).strip()


# --------------------------------------------------------------------------- #
# Reproducibility (cheap half) and the --check machinery
# --------------------------------------------------------------------------- #


def test_quick_rows_exclude_exactly_the_slow_groups(bench, committed, quick_rows):
    recomputed = {row["check"] for row in quick_rows}
    published = {row["check"] for row in committed["checks"]}
    assert recomputed == published - SLOW_ROWS
    assert set(bench.SLOW_ROWS) == SLOW_ROWS


def test_quick_recompute_reproduces_the_committed_numbers(bench, committed, quick_rows):
    comparison = bench.compare_rows(committed["checks"], quick_rows)
    assert comparison["n_diffs"] == 0, comparison["diffs"]


def test_check_mode_accepts_the_committed_artifact(bench, quick_rows):
    report = bench.check_artifact(quick=True, fresh=quick_rows)
    assert report["ok"] is True, report
    assert report["exit_code"] == 0
    assert report["digest_ok"] is True
    assert report["markdown_ok"] is True
    assert report["stale"] == []
    assert report["rows_compared"] == len(quick_rows)


def test_check_mode_flags_a_hand_edited_number(bench, committed, quick_rows, tmp_path):
    import copy

    tampered = copy.deepcopy(committed)
    for row in tampered["checks"]:
        if row["check"] == "law_energy_mode_I":
            row["value"] = 1e-3
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    report = bench.check_artifact(path, quick=True, fresh=quick_rows)
    assert report["ok"] is False
    assert report["exit_code"] == 3  # digest mismatch is a staleness-class failure
    assert report["digest_ok"] is False
    flagged = {diff["check"] for diff in report["comparison"]["diffs"]}
    assert "law_energy_mode_I" in flagged


def test_check_mode_flags_a_stale_source_hash(bench, committed, quick_rows, tmp_path):
    import copy

    tampered = copy.deepcopy(committed)
    tampered["provenance"]["source_files"]["src/tensorfem/cohesive3d.py"] = "0" * 64
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(tampered))
    report = bench.check_artifact(path, quick=True, fresh=quick_rows)
    assert report["ok"] is False
    assert report["exit_code"] == 3
    assert [row["file"] for row in report["stale"]] == ["src/tensorfem/cohesive3d.py"]


def test_check_mode_flags_a_drifted_value_even_with_a_valid_digest(
    bench, committed, quick_rows, tmp_path
):
    """A regeneration that silently moves a number is caught by the numbers, not the digest."""
    import copy

    tampered = copy.deepcopy(committed)
    for row in tampered["checks"]:
        if row["check"] == "element_patch_small_strain":
            row["value"] = 1.0  # outside every bracket this row can produce
    tampered["checks_digest"] = bench.checks_digest(tampered["parameters"], tampered["checks"])
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(tampered))
    report = bench.check_artifact(path, quick=True, fresh=quick_rows)
    assert report["digest_ok"] is True
    assert report["ok"] is False
    assert report["exit_code"] == 1
    flagged = [(diff["check"], diff["field"]) for diff in report["comparison"]["diffs"]]
    assert ("element_patch_small_strain", "value") in flagged


def test_check_mode_reports_a_missing_artifact(bench):
    report = bench.check_artifact("/nonexistent/cohesive3d_tables.json", quick=True)
    assert report["ok"] is False
    assert report["exit_code"] == 2
    assert "missing" in report["reason"]
