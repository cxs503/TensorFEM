"""Guard the change-scoped lint gate wired into the CI ``lint`` job.

The tree carries a large pre-existing ruff backlog, so CI lints the *change*
(the Python files a PR/push touches inside src/, tests/, scripts/, examples/
and benchmarks/) instead of the whole tree.  ``scripts/lint_scope.sh`` computes
that set; if it broke, the gate would either silently lint nothing (green but
meaningless) or fall back to the whole tree (red on every commit).  These tests
pin the two behaviours the workflow relies on, hermetically: each one builds a
throwaway repository, so the result does not depend on the checkout that runs
the suite.
"""

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "lint_scope.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

LINTED_TREES = {"src", "tests", "scripts", "examples", "benchmarks"}


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=lint-gate", "-c", "user.email=lint@example.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _scope(repo, base="", event=""):
    """Run the scope script inside ``repo``; return the emitted path list."""
    args = ["bash", str(SCRIPT)]
    if base:
        args.append(base)
    env = {"PATH": "/usr/bin:/bin", "GITHUB_EVENT_NAME": event}
    out = subprocess.run(args, cwd=repo, capture_output=True, text=True, check=True, env=env).stdout
    return out.split()


def _seed_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("A = 1\n")
    (repo / "src" / "notes.txt").write_text("not python\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "b.py").write_text("B = 2\n")
    _git(repo, "init", "-q")
    return repo, _commit(repo, "seed")


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "CI calls scripts/lint_scope.sh via bash"


def test_changed_files_scope_is_diff_limited_to_linted_trees(tmp_path):
    repo, base = _seed_repo(tmp_path)
    (repo / "src" / "a.py").write_text("A = 10\n")  # modified -> linted
    (repo / "src" / "c.py").write_text("C = 3\n")  # added -> linted
    (repo / "docs" / "b.py").write_text("B = 20\n")  # outside the linted trees
    _commit(repo, "touch three trees")

    assert _scope(repo, base=base) == ["src/a.py", "src/c.py"]


def test_unchanged_tree_yields_empty_scope(tmp_path):
    # "Nothing changed" must print nothing rather than everything: the CI steps
    # are skipped on an empty scope instead of ruff failing on a missing file.
    repo, base = _seed_repo(tmp_path)
    assert _scope(repo, base=base) == []


def test_pull_request_event_uses_base_branch_and_falls_back_to_full_tree(tmp_path):
    repo, base = _seed_repo(tmp_path)
    (repo / "src" / "c.py").write_text("C = 3\n")
    _commit(repo, "add c")

    # No ``origin/<base>`` ref in this throwaway repo and no origin/main merge
    # base either: the script must widen to the whole tree (manual audit mode)
    # rather than emit a partial, misleading list.
    files = _scope(repo, event="pull_request")
    assert files == ["src/a.py", "src/c.py"]
    assert all(path.endswith(".py") for path in files)
    assert all(path.split("/")[0] in LINTED_TREES for path in files)


def test_workflow_runs_the_scope_script_and_guards_empty_scope():
    text = WORKFLOW.read_text()
    assert "bash scripts/lint_scope.sh" in text
    assert "steps.scope.outputs.count != '0'" in text
    assert "ruff check" in text
    assert "ruff format --check" in text
    # The scope diff needs real history, not a depth-1 clone.
    assert "fetch-depth: 0" in text


# --- rule-set determinism ---------------------------------------------------
# The change-scope gate above answers *which files* to lint.  The second
# failure mode is the opposite one: the definition of *what counts as clean*
# floating with the tool.  ruff promotes newly stable rules into its default
# ``select`` at minor releases, so an unpinned gate changes meaning without a
# line of code changing -- main went red at 696666b8 exactly that way (see
# docs/lint-gate-determinism.md).  These tests keep the rule set and the ruff
# version qualified in the repository.

PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject():
    return tomllib.loads(PYPROJECT.read_text())


def _ruff_requirement():
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    return next(req for req in dev if re.match(r"\s*ruff\b", req))


def test_lint_rule_set_is_pinned_in_the_repository():
    lint = _pyproject()["tool"]["ruff"].get("lint", {})
    select = lint.get("select")
    assert select, "tool.ruff.lint.select must pin the gate's rule set explicitly"
    assert "ALL" not in select
    # The pyflakes + error groups are the contract the tree is qualified
    # against; dropping one would silently weaken the gate.
    assert {"E4", "E7", "E9", "F"} <= set(select)


def test_workflow_lints_through_the_pinned_configuration():
    text = WORKFLOW.read_text()
    # ``--isolated`` would ignore pyproject.toml entirely (both line-length and
    # the pinned rule set) and an inline ``--select`` would bypass the pin.
    assert "--isolated" not in text
    assert "--select" not in text


def test_ruff_version_is_qualified_within_one_minor_release():
    req = _ruff_requirement()
    floor = re.search(r">=\s*(\d+)\.(\d+)", req)
    ceiling = re.search(r"<\s*(\d+)\.(\d+)", req)
    assert floor and ceiling, f"{req!r} must carry both a floor and a ceiling"
    floor_ver = tuple(int(part) for part in floor.groups())
    ceiling_ver = tuple(int(part) for part in ceiling.groups())
    assert floor_ver < ceiling_ver, f"{req!r} is an empty range"
    # One minor line: the runner and the development nodes must execute the
    # same rule implementations and the same formatter.
    assert ceiling_ver[0] == floor_ver[0] and ceiling_ver[1] - floor_ver[1] == 1, (
        f"{req!r} spans more than one minor release"
    )


# --- gate content: the format check has to actually bite ---------------------
# The tests above pin *which* files the gate sees and *what counts as clean*.
# The third failure mode is the gate not being run on the change at all: commit
# f5be5cf1 added tests/test_smoke_m4622.py without a newline at end of file,
# ``ruff check`` passed on it, ``ruff format --check`` failed, and nothing in
# the pytest job noticed -- main carried a red ``lint (advisory)`` job for a
# one-byte defect (mission m4663).  These tests execute both CI commands the
# way the workflow does, so the formatting half of the gate is tested instead
# of assumed, and ``scripts/lint_gate.sh`` gives the same verdict locally.

GATE_SCRIPT = REPO_ROOT / "scripts" / "lint_gate.sh"
RUFF = shutil.which("ruff")
requires_ruff = pytest.mark.skipif(RUFF is None, reason="ruff is not installed (dev extra)")


def _run_gate(*paths):
    """Run the two CI gate commands over ``paths``; return (check, format)."""
    return tuple(
        subprocess.run(
            [RUFF, *command, "--config", str(PYPROJECT), *(str(path) for path in paths)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        for command in (["check"], ["format", "--check"])
    )


def _gate(repo, base):
    return subprocess.run(
        ["bash", str(GATE_SCRIPT), base],
        cwd=repo,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )


def test_gate_script_exists_and_is_executable():
    assert GATE_SCRIPT.is_file()
    assert GATE_SCRIPT.stat().st_mode & 0o111, "run scripts/lint_gate.sh, not python"


@requires_ruff
def test_format_check_rejects_a_missing_trailing_newline(tmp_path):
    # The f5be5cf1 defect, reproduced hermetically: a formatting-only slip that
    # ``ruff check`` accepts and ``ruff format --check`` rejects.
    broken = tmp_path / "test_missing_newline.py"
    broken.write_text("def test_smoke() -> None:\n    assert 1 + 1 == 2")  # no final newline
    check, fmt = _run_gate(broken)
    assert check.returncode == 0, "the defect is formatting-only; ruff check must accept it"
    assert fmt.returncode != 0, "ruff format --check must reject a file with no trailing newline"
    assert broken.name in fmt.stdout


@requires_ruff
def test_format_check_accepts_the_same_file_with_a_trailing_newline(tmp_path):
    ok = tmp_path / "test_with_newline.py"
    ok.write_text("def test_smoke() -> None:\n    assert 1 + 1 == 2\n")
    check, fmt = _run_gate(ok)
    assert check.returncode == 0, check.stdout + check.stderr
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr


@requires_ruff
@pytest.mark.parametrize("name", ["tests/test_smoke_m4622.py", "smoke_marker_m4622.py"])
def test_smoke_marker_artifacts_are_gate_clean(name):
    # Regression lock for f5be5cf1: the test module and the root marker it
    # imports must both stay clean under both gate commands.
    path = REPO_ROOT / name
    assert path.is_file(), f"{name} is required by tests/test_smoke_m4622.py"
    check, fmt = _run_gate(path)
    assert check.returncode == 0, check.stdout + check.stderr
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr


@requires_ruff
def test_gate_script_is_green_on_a_clean_change_and_red_on_a_formatting_slip(tmp_path):
    # End-to-end over the script itself: clean change -> 0, the f5be5cf1 defect
    # -> non-zero with the offending path named.
    repo, base = _seed_repo(tmp_path)
    (repo / "src" / "c.py").write_text("C = 3\n")
    _commit(repo, "add a clean file")
    green = _gate(repo, base)
    assert green.returncode == 0, green.stdout + green.stderr

    (repo / "src" / "d.py").write_text("D = 4")  # no trailing newline
    _commit(repo, "add a file without a trailing newline")
    red = _gate(repo, base)
    assert red.returncode != 0, red.stdout + red.stderr
    assert "src/d.py" in red.stdout
