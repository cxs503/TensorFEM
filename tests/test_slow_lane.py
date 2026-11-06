"""Guard the test-lane contract: one body set, a ``slow`` registry, two lanes.

TensorFEM's suite mixes sub-second element/benchmark/contract checks with
minute-scale qualification studies (mesh-convergence sweeps, long nonlinear and
contact paths, industrial sparse factorisations).  The lane split -- see
``tests/conftest.py`` and ``docs/test-lanes.md`` -- lets a developer iterate on
the fast population without ever deleting a test or loosening an assertion:

* full lane: ``pytest``          -- everything; what CI and the release
                                    evidence run certify;
* fast lane: ``pytest --fast``   -- identical to ``pytest -m "not slow"``; the
                                    registered slow cases are deselected.

Four properties can rot silently, and each is pinned here:

1. the marker is registered in ``pyproject.toml`` and the default lane stays
   the full lane (``--strict-markers`` makes a typo an error, and CI must never
   run the reduced lane);
2. ``--fast`` really deselects, and selects exactly what ``-m "not slow"``
   selects -- the alias is only honest if the two agree;
3. the lanes change *selection* only: an injected defect in an unmarked test
   still reddens the fast lane, and an injected defect in a registered test
   still reddens the full lane (the registry is not a place to hide a failure);
4. the registry ``tests/slow_lane_manifest.txt`` stays in step with the tree
   in both directions: every entry names a live test function, and every entry
   is needed (the opt-in heavyweight studies are all present).

The lane mechanics are exercised hermetically in throwaway projects built from
*this* repository's conftest and registry format, so the guards cost
milliseconds and do not depend on this tree's own test durations.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
CONFTEST = TESTS_DIR / "conftest.py"
MANIFEST = TESTS_DIR / "slow_lane_manifest.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "slow_lane_audit.py"

THIS_TEST = "tests/test_slow_lane.py::test_registry_matches_a_fresh_collection"

# "1842/1967 tests collected (125 deselected)" / "1967 tests collected".
_COLLECTED = re.compile(r"(?:(\d+)/)?(\d+) tests? collected(?: \((\d+) deselected\))?")

# The four opt-in qualification studies: minute-scale by construction, so they
# are registered even though a default run skips them and measures 0 s.
OPT_IN_HEAVY = {
    "tests/test_elastic_halfspace.py::test_refinement_ladder_converges_towards_boussinesq",
    "tests/test_finite_strain_curved_friction_contact3d.py::test_two_level_curved_finite_strain_friction_qualification",
    "tests/test_frictional_surface_path.py::test_curved_frictional_three_mesh_slow_qualification",
    "tests/test_surface_surface_contact3d.py::test_curved_nonmatching_complete_newton_qualification",
}

UNMARKED_GREEN = "def test_unmarked_and_green():\n    assert 2 + 2 == 4\n"
UNMARKED_RED = (
    "def test_unmarked_and_red():\n"
    "    assert 2 + 2 == 5, 'injected defect in an unmarked test'\n"
)
REGISTERED_GREEN = (
    "import pytest\n"
    "\n"
    "\n"
    "@pytest.mark.slow\n"
    "def test_registered_and_green():\n"
    "    assert 1 + 1 == 2\n"
)
# The registration mechanism *is* the registry file, so the hermetic case
# exercises it the way the real tree does: a plain test function plus a
# registry entry, no source decorator.
REGISTERED_BY_FILE = ("def test_registered_by_registry():\n    assert 1 + 1 == 2\n")
REGISTERED_RED = (
    "def test_registered_and_red():\n"
    "    assert 1 + 1 == 3, 'injected defect in a registered slow test'\n"
)


def _lane_project(tmp_path, files, registry):
    """Throwaway project wired to *this* repository's lane conftest + registry."""
    root = tmp_path / "lane_project"
    root.mkdir(parents=True, exist_ok=True)
    (root / "conftest.py").write_text(CONFTEST.read_text())
    (root / "slow_lane_manifest.txt").write_text("# test registry\n" + "\n".join(registry) + "\n")
    for name, body in files.items():
        (root / name).write_text(body)
    return root


def _run_pytest(root, *args):
    """Run pytest hermetically in ``root``; never inherit the caller's addopts."""
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


def _summary(result):
    """Last non-empty output line -- e.g. '1 passed, 1 deselected in 0.02s'."""
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _counts(result):
    """'1 passed, 1 deselected' -> {'passed': 1, 'deselected': 1}."""
    outcomes = {"passed", "failed", "errors", "error", "deselected", "skipped", "xfailed", "xpassed"}
    counts = {}
    for token in _summary(result).split(" in ")[0].split(","):
        parts = token.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1] in outcomes:
            counts[parts[1]] = int(parts[0])
    return counts


# --- 1. the marker and the default lane are declared in pyproject -----------


def test_pyproject_registers_the_slow_marker():
    config = tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]
    markers = config["markers"]
    names = [marker.split(":", 1)[0].strip() for marker in markers]
    assert "slow" in names, f"'slow' must be registered in tool.pytest.ini_options.markers, got {markers!r}"
    # An undocumented marker is how a registry rots into "some decorator
    # somebody added once": every registration carries its description.
    for marker in markers:
        assert ":" in marker and marker.split(":", 1)[1].strip(), f"marker without description: {marker!r}"


def test_default_lane_is_the_full_lane():
    config = tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]
    addopts = config.get("addopts", "")
    assert "--strict-markers" in addopts, "unregistered markers must be an error, not an inert decorator"
    # The fast lane is opt-in: a bare `pytest` -- what CI and the release
    # evidence run execute -- must never deselect tests through configuration.
    assert "--fast" not in addopts, "the fast lane must not be the configured default"
    assert "-m" not in addopts.split(), f"addopts must not pre-select by marker: {addopts!r}"


def test_ci_runs_the_full_lane():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    pytest_lines = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("run:") and "pytest" in line
    ]
    assert pytest_lines, "the CI workflow must run pytest"
    for line in pytest_lines:
        assert "--fast" not in line, f"CI must certify the full lane, found: {line!r}"
        assert "-m " not in line, f"CI must certify the full lane, found: {line!r}"


# --- 2. the lane mechanics, exercised hermetically --------------------------


def test_fast_lane_deselects_registered_tests_and_keeps_the_rest(tmp_path):
    root = _lane_project(
        tmp_path,
        {"test_lane.py": UNMARKED_GREEN + "\n" + REGISTERED_BY_FILE},
        ["test_lane.py::test_registered_by_registry"],
    )
    fast = _run_pytest(root, "--fast")
    assert fast.returncode == 0, fast.stdout + fast.stderr
    assert _counts(fast) == {"passed": 1, "deselected": 1}, _summary(fast)

    full = _run_pytest(root)
    assert full.returncode == 0, full.stdout + full.stderr
    assert _counts(full) == {"passed": 2}, _summary(full)


def test_registry_marks_the_item_for_pytest_mark_expressions(tmp_path):
    """The marker is real: `-m slow` selects the registered case alone."""
    root = _lane_project(
        tmp_path,
        {"test_lane.py": UNMARKED_GREEN + "\n" + REGISTERED_BY_FILE},
        ["test_lane.py::test_registered_by_registry"],
    )
    selected = _run_pytest(root, "-m", "slow")
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert _counts(selected) == {"passed": 1, "deselected": 1}, _summary(selected)


def test_fast_lane_is_exactly_the_not_slow_expression(tmp_path):
    files = {"test_lane.py": UNMARKED_GREEN + "\n" + REGISTERED_BY_FILE}
    registry = ["test_lane.py::test_registered_by_registry"]
    fast = _run_pytest(_lane_project(tmp_path / "a", files, registry), "--fast")
    expression = _run_pytest(_lane_project(tmp_path / "b", files, registry), "-m", "not slow")
    assert fast.returncode == expression.returncode == 0
    assert _counts(fast) == _counts(expression), (_summary(fast), _summary(expression))


def test_fast_lane_still_reddens_on_a_defect_in_an_unmarked_test(tmp_path):
    """Negative control: the lane changes selection, never the verdict.

    A defect injected into an *unmarked* test must turn the fast lane red --
    otherwise the fast lane has quietly become "the tests that cannot fail".
    """
    root = _lane_project(
        tmp_path,
        {"test_lane.py": UNMARKED_RED + "\n" + REGISTERED_BY_FILE},
        ["test_lane.py::test_registered_by_registry"],
    )
    result = _run_pytest(root, "--fast")
    assert result.returncode != 0, "fast lane must fail on an injected defect:\n" + result.stdout
    assert "injected defect in an unmarked test" in result.stdout
    assert _counts(result).get("failed") == 1, _summary(result)


def test_full_lane_still_reddens_on_a_defect_in_a_registered_test(tmp_path):
    """The registry must not become a hiding place: the full lane still fails."""
    root = _lane_project(
        tmp_path,
        {"test_lane.py": UNMARKED_GREEN + "\n" + REGISTERED_RED},
        ["test_lane.py::test_registered_and_red"],
    )
    full = _run_pytest(root)
    assert full.returncode != 0, "full lane must fail on an injected defect:\n" + full.stdout
    assert "injected defect in a registered slow test" in full.stdout

    # ... and the fast lane is green on the same tree, because it does not run it.
    fast = _run_pytest(root, "--fast")
    assert fast.returncode == 0, fast.stdout
    assert _counts(fast) == {"passed": 1, "deselected": 1}, _summary(fast)


def test_fast_lane_skips_nothing_it_deselects(tmp_path):
    """Deselection, not skipping/xfail: no reporting bookkeeping is introduced."""
    root = _lane_project(
        tmp_path,
        {"test_lane.py": UNMARKED_GREEN + "\n" + REGISTERED_GREEN},
        ["test_lane.py::test_registered_and_green"],
    )
    result = _run_pytest(root, "--fast", "-rsxX")
    assert result.returncode == 0, result.stdout
    summary = _summary(result)
    assert "skipped" not in summary and "xfailed" not in summary, summary


# --- 3. the registry cannot drift out of step with the tree ----------------


def read_manifest(path=MANIFEST):
    entries = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def test_registry_entries_are_canonical_and_sorted():
    entries = read_manifest()
    assert entries, "the slow registry is empty; the fast lane would equal the full lane"
    assert entries == sorted(entries), "keep the registry sorted so reviews diff cleanly"
    assert len(entries) == len(set(entries)), "duplicate registry entries"
    parametrised = [entry for entry in entries if "[" in entry]
    assert not parametrised, (
        "registry entries must be base node ids (no parametrisation suffix); "
        f"such an entry silently registers nothing: {parametrised}"
    )


def test_every_registry_entry_names_a_live_test_function():
    """A renamed/deleted test must fail here instead of silently unregistering."""
    dead = []
    for entry in read_manifest():
        relative, _, function = entry.partition("::")
        path = REPO_ROOT / relative
        if not relative.startswith("tests/") or path.suffix != ".py" or not path.is_file():
            dead.append(f"{entry} (no such module)")
            continue
        if "::" in function:  # tests/test_x.py::TestClass::test_y
            continue
        source = path.read_text()
        if f"def {function}(" not in source:
            dead.append(f"{entry} (function not defined in {relative})")
        elif not function.startswith("test"):
            dead.append(f"{entry} (not a test function)")
    assert not dead, f"registry entries that no longer name a live test: {dead}"


def test_registry_covers_the_opt_in_heavyweight_studies():
    registered = set(read_manifest())
    missing = sorted(OPT_IN_HEAVY - registered)
    assert not missing, (
        "these opt-in (TENSORFEM_RUN_SLOW_*) qualification studies are minute-scale "
        f"whenever they run and must stay registered: {missing}"
    )


def test_registry_covers_the_heavy_tail():
    entries = read_manifest()
    assert len(entries) >= 100, (
        f"only {len(entries)} slow entries; the measured heavy tail is ~115 cases, "
        "so the fast lane would silently start paying for qualification studies"
    )
    families = {"qualification", "convergence", "scaling", "contact", "plasticity", "arc"}
    covered = {family for family in families if any(family in entry for entry in entries)}
    assert len(covered) >= 5, f"registry only spans {sorted(covered)}; the heavy families drifted"


# --- 4. the maintenance tool that keeps the registry honest ----------------


def test_audit_tool_reads_a_durations_log(tmp_path):
    log = tmp_path / "durations.log"
    log.write_text(
        "slowest durations\n"
        "12.50s call     tests/test_qualification.py::test_heavy\n"
        " 0.25s setup    tests/test_qualification.py::test_heavy\n"
        " 0.01s call     tests/test_light.py::test_light\n"
    )
    proc = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), str(log), "--threshold", "2", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["over_threshold"] == {"tests/test_qualification.py::test_heavy": 12.75}
    assert report["unregistered_over_threshold"] == ["tests/test_qualification.py::test_heavy"]


def test_audit_tool_reads_a_junit_report(tmp_path):
    """CI stores JUnit; the audit must accept it as well as a durations log."""
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_qualification" name="test_heavy" time="7.5"/>'
        '<testcase classname="tests.test_light" name="test_light" time="0.02"/>'
        "</testsuite></testsuites>"
    )
    proc = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--junitxml", str(junit), "--threshold", "2", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert list(report["over_threshold"]) == ["tests/test_qualification.py::test_heavy"]


def test_audit_tool_check_mode_flags_a_registration_gap(tmp_path):
    """Negative control for the drift gate used to maintain the registry."""
    log = tmp_path / "durations.log"
    log.write_text("9.00s call     tests/test_new_heavy.py::test_sweep\n")
    gap = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), str(log), "--threshold", "2", "--check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert gap.returncode == 1, "an unregistered heavyweight test must fail --check"
    assert "NOT REGISTERED" in gap.stdout or "without a 'slow' marker" in gap.stderr

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("tests/test_new_heavy.py::test_sweep\n")
    clean = subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(log),
            "--threshold",
            "2",
            "--check",
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert clean.returncode == 0, clean.stdout + clean.stderr


# --- 5. the collection cross-check (full lane only: it collects the suite) --


@pytest.mark.slow
def test_registry_matches_a_fresh_collection():
    """Integration cross-check on the real tree: the registry and pytest agree.

    The hermetic guards above prove the mechanics; this one asks pytest about
    *this* tree: the fast lane and ``-m slow`` must partition the suite
    (fast items + slow items == everything collected), and both must agree on
    the slow count.  It collects the whole suite twice, so it is registered in
    the very registry it verifies.
    """
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)

    def collect(*args):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *args,
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                str(TESTS_DIR),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
        matches = [_COLLECTED.search(line) for line in proc.stdout.splitlines()]
        matches = [match for match in matches if match is not None]
        assert matches, f"pytest printed no collection summary:\n{proc.stdout[-2000:]}"
        selected, total, deselected = matches[-1].groups()
        return int(selected or total), int(total), int(deselected or 0)

    assert THIS_TEST in read_manifest(), "this collection cross-check must register itself"
    full_collected, full_total, full_deselected = collect()
    fast_collected, fast_total, fast_deselected = collect("--fast")
    slow_collected, slow_total, slow_deselected = collect("-m", "slow")

    assert full_deselected == 0 and full_collected == full_total, (full_collected, full_total)
    assert fast_total == slow_total == full_total, (fast_total, slow_total, full_total)
    assert fast_collected + fast_deselected == full_total
    assert slow_collected + slow_deselected == full_total
    assert fast_deselected == slow_collected, (
        f"--fast deselected {fast_deselected} but -m slow selected {slow_collected}"
    )
    assert fast_collected == slow_deselected
    assert fast_deselected >= len(read_manifest()), (
        f"only {fast_deselected} items deselected for {len(read_manifest())} registry entries"
    )
