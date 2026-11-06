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

import re
import subprocess
import tomllib
from pathlib import Path

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
