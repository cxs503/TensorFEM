"""Release-to-release compatibility gate for the public API manifest.

The point-in-time snapshot gate (``tests/test_public_api_contract.py``) can only
see the tree it is standing in: manifest and code are always edited together, so
a deleted stable symbol looks perfectly "consistent".  These tests pin the other
axis -- what a *transition* between two reviewed snapshots is allowed to change
for a given version bump -- using synthetic manifests (fast, no torch import).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import public_api_compat as compat  # noqa: E402


def manifest(version, *entries):
    """Build a manifest dict from (name, stability, module, object) tuples."""
    payload = []
    for row in entries:
        name, stability, module = row[0], row[1], row[2]
        obj = row[3] if len(row) > 3 else name
        payload.append(
            {
                "introduced": "0.1.0",
                "module": module,
                "name": name,
                "object": obj,
                "stability": stability,
            }
        )
    return {
        "package_version": version,
        "policy": {"stable": "semver compatibility"},
        "entries": payload,
    }


BASE = manifest(
    "1.0.0",
    ("StableFn", "stable", "tensorfem.core"),
    ("BetaFn", "beta", "tensorfem.core"),
    ("ExpFn", "experimental", "tensorfem.kernels"),
    ("InternalFn", "internal_candidate", "tensorfem.kernels"),
)


def verdict(head, *, mode="pr", base=None, zero=False):
    return compat.classify(base or BASE, head, mode=mode, zero_major_minor_is_breaking=zero)


# --------------------------------------------------------------------------- #
# additive changes never break a user, whatever the version does
# --------------------------------------------------------------------------- #
def test_pure_addition_is_not_breaking_and_does_not_require_a_pr_bump():
    head = manifest(
        "1.0.0",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "beta", "tensorfem.core"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
        ("NewFn", "experimental", "tensorfem.kernels"),
    )
    result = verdict(head)
    assert result["counts"] == {"added": 1}
    assert result["breaking"] == []
    assert result["required_bump"] == "none"
    assert result["compliant"] is True
    assert result["accumulated_minor_delta"] == 1


def test_release_mode_charges_additions_to_the_declared_bump():
    """Shipping new surface under an unchanged version number is not a release."""
    head = manifest(
        "1.0.0",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "beta", "tensorfem.core"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
        ("NewFn", "experimental", "tensorfem.kernels"),
    )
    assert verdict(head)["compliant"] is True  # PR gate: additive PR is fine
    released = verdict(head, mode="release")
    assert released["compliant"] is False  # release gate: needs 1.1.0
    assert released["required_bump"] == "minor"
    moved = dict(head, package_version="1.1.0")
    assert verdict(moved, mode="release")["compliant"] is True


# --------------------------------------------------------------------------- #
# withdrawals are charged to the stability class of the withdrawn name
# --------------------------------------------------------------------------- #
BASE_ROWS = [
    ("StableFn", "stable", "tensorfem.core"),
    ("BetaFn", "beta", "tensorfem.core"),
    ("ExpFn", "experimental", "tensorfem.kernels"),
    ("InternalFn", "internal_candidate", "tensorfem.kernels"),
]


@pytest.mark.parametrize(
    "removed,needs",
    [("StableFn", "major"), ("BetaFn", "major"), ("ExpFn", "minor"), ("InternalFn", "minor")],
)
def test_removal_requires_a_bump_won_by_the_stability_class(removed, needs):
    head = manifest("1.0.0", *[row for row in BASE_ROWS if row[0] != removed])
    result = verdict(head)
    assert result["required_bump"] == needs
    assert result["compliant"] is False
    assert [v["name"] for v in result["violations"]] == [removed]
    assert result["violations"][0]["needs"] == needs


def test_removal_is_accepted_once_the_declared_bump_pays_for_it():
    head = manifest(
        "2.0.0",
        ("BetaFn", "beta", "tensorfem.core"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
    )
    result = verdict(head)
    assert result["required_bump"] == "major"
    assert result["actual_bump"] == "major"
    assert result["compliant"] is True
    assert result["violations"] == []


def test_zero_major_convention_is_opt_in():
    """0.x projects may bill breaking changes to the minor position (SemVer §4)."""
    base = manifest(
        "0.45.0", ("StableFn", "stable", "tensorfem.core"), ("BetaFn", "beta", "tensorfem.core")
    )
    head = manifest("0.46.0", ("BetaFn", "beta", "tensorfem.core"))
    strict = verdict(head, base=base)
    assert strict["required_bump"] == "major"
    assert strict["compliant"] is False
    assert strict["violations"][0]["name"] == "StableFn"
    lenient = verdict(head, base=base, zero=True)
    assert lenient["required_bump"] == "minor"
    assert lenient["compliant"] is True


# --------------------------------------------------------------------------- #
# in-place mutations of the guarantee
# --------------------------------------------------------------------------- #
def test_stability_demotion_of_a_stable_name_needs_a_major_bump():
    head = manifest(
        "1.0.1",
        ("StableFn", "beta", "tensorfem.core"),
        ("BetaFn", "beta", "tensorfem.core"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
    )
    result = verdict(head)
    assert result["breaking"] == [
        {"kind": "stability_changed", "name": "StableFn", "from": "stable", "to": "beta"}
    ]
    assert result["required_bump"] == "major"
    assert result["compliant"] is False


def test_stability_demotion_of_beta_is_the_documented_deprecation_path():
    head = manifest(
        "1.1.0",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "experimental", "tensorfem.core"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
    )
    result = verdict(head)
    assert result["required_bump"] == "minor"
    assert result["compliant"] is True


def test_promotion_only_strengthens_the_guarantee_so_a_patch_is_enough():
    head = manifest(
        "1.0.1",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "stable", "tensorfem.core"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
    )
    result = verdict(head)
    assert result["breaking"] == []
    assert result["required_bump"] == "none"  # PR gate: nothing withdrawn
    assert result["compliant"] is True
    released = verdict(head, mode="release")  # release gate bills the promotion
    assert released["required_bump"] == "patch"
    assert released["compliant"] is True


def test_module_move_and_object_rename_break_imports():
    moved = manifest(
        "1.0.1",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "beta", "tensorfem.legacy"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
    )
    renamed = manifest(
        "1.0.1",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "beta", "tensorfem.core", "BetaFunction"),
        ("ExpFn", "experimental", "tensorfem.kernels"),
        ("InternalFn", "internal_candidate", "tensorfem.kernels"),
    )
    for head in (moved, renamed):
        result = verdict(head)
        assert [change["name"] for change in result["breaking"]] == ["BetaFn"]
        assert result["required_bump"] == "major"
        assert result["compliant"] is False


def test_reordering_alone_is_not_a_break():
    base = manifest(
        "1.0.0",
        ("StableFn", "stable", "tensorfem.core"),
        ("BetaFn", "beta", "tensorfem.core"),
    )
    head = manifest(
        "1.0.0",
        ("BetaFn", "beta", "tensorfem.core"),
        ("StableFn", "stable", "tensorfem.core"),
    )
    result = verdict(head, base=base)
    assert result["counts"] == {"reordered": 1}
    assert result["breaking"] == []
    assert result["required_bump"] == "none"  # nothing user-facing changes
    assert result["compliant"] is True
    released = verdict(head, base=base, mode="release")  # publishing the new order needs a patch
    assert released["required_bump"] == "patch"
    assert released["compliant"] is False
    patched = manifest(
        "1.0.1",
        ("BetaFn", "beta", "tensorfem.core"),
        ("StableFn", "stable", "tensorfem.core"),
    )
    assert verdict(patched, base=base, mode="release")["compliant"] is True


# --------------------------------------------------------------------------- #
# input hygiene + CLI
# --------------------------------------------------------------------------- #
def test_duplicate_names_are_rejected_rather_than_silently_collapsed():
    broken = manifest(
        "1.0.0",
        ("StableFn", "stable", "tensorfem.core"),
        ("StableFn", "beta", "tensorfem.core"),
    )
    with pytest.raises(compat.ManifestError, match="duplicate name"):
        compat.classify(BASE, broken)


def test_unknown_stability_class_is_rejected():
    broken = manifest("1.0.0", ("StableFn", "deprecated", "tensorfem.core"))
    with pytest.raises(compat.ManifestError, match="invalid stability"):
        compat.classify(BASE, broken)


def test_cli_exit_codes_and_git_spec(tmp_path):
    """The gate must be usable from CI shell and must read baselines out of git."""
    head_path = tmp_path / "head.json"
    head_path.write_text(json.dumps(manifest("1.0.1", ("BetaFn", "beta", "tensorfem.core"))))
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(manifest("2.0.0", ("BetaFn", "beta", "tensorfem.core"))))
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(BASE))
    fail = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "public_api_compat.py"),
            "--base",
            str(base_path),
            "--head",
            str(head_path),
        ],
        capture_output=True,
        text=True,
    )
    assert fail.returncode == 1, fail.stdout + fail.stderr
    assert "VIOLATION: removed StableFn needs major" in fail.stdout
    good = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "public_api_compat.py"),
            "--base",
            str(base_path),
            "--head",
            str(ok),
        ],
        capture_output=True,
        text=True,
    )
    assert good.returncode == 0, good.stdout + good.stderr
    # git spec form used by the recorded history artifact
    if (ROOT / ".git").exists():
        git_verdict = compat.classify(
            compat.load_manifest("git:HEAD:api/public-api-v1.json"),
            compat.load_manifest(str(ROOT / "api/public-api-v1.json")),
        )
        assert git_verdict["changes"] == []
        assert git_verdict["compliant"] is True


# --------------------------------------------------------------------------- #
# recorded evidence stays true
# --------------------------------------------------------------------------- #
ARTIFACT = ROOT / "artifacts/m18005_public_api_compat_baseline.json"


def test_recorded_evidence_matches_what_the_gate_says_today():
    """The committed artifact must not drift away from the gate it documents."""
    payload = json.loads(ARTIFACT.read_text())
    contrast = payload["detection_contrast"]
    assert contrast["injected_faults"] == len(contrast["rows"]) == 2
    # baseline: the point-in-time snapshot audit is blind to code-consistent breaks
    assert contrast["snapshot_gate_detections"] == 0
    assert contrast["compat_gate_detections"] == contrast["injected_faults"]
    for row in contrast["rows"]:
        assert row["import_ok"] is True  # the fault tree really imported
        assert row["snapshot_gate_detected"] is False
        assert row["compat_gate_detected"] is True
        assert "VIOLATION" in row["compat_gate_stdout"]

    history = payload["history"]["transitions"]
    assert len(history) >= 5
    # the shipped API has only ever grown: every historical transition is additive
    for row in history:
        assert set(row["counts"]) <= {"added"}, row
        assert row["breaking"] == []
        assert row["pr_mode"]["compliant"] is True

    # and re-deriving the newest transition reproduces the recorded verdict
    newest = history[-1]
    base = compat.load_manifest(f"git:{newest['base_commit']}:api/public-api-v1.json")
    head = compat.load_manifest(f"git:{newest['head_commit']}:api/public-api-v1.json")
    recomputed = compat.classify(base, head, mode="pr")
    assert recomputed["counts"] == newest["counts"]
    assert recomputed["required_bump"] == newest["pr_mode"]["required_bump"]
    assert recomputed["compliant"] == newest["pr_mode"]["compliant"]
