"""Guard pins for the two-tier lint gate in .github/workflows/ci.yml.

Context (PR #125 cure): the lint job used to run ``ruff check`` /
``ruff format --check`` over ``src tests scripts examples``.  The tree carries
a large pre-existing backlog (~4.3k ``ruff check`` findings, mostly
semicolon-packed E701/E702 statements, plus ~620 files that predate
``ruff format``), so that whole-tree gate was red for every commit — including
branches whose own diff is lint-clean.  PR #125 was reported as
``lint (advisory)=failure`` for exactly that reason.

The gate is now split in two tiers:

* changed-files tier — the job's verdict: ruff runs only on the Python files
  the PR / push touches, so new code must be clean;
* repo tier — informational only: whole-tree counters are printed for trending
  and can never fail the job.

These tests pin that structure so a later edit cannot silently restore the
all-red whole-tree gate (or silently drop the changed-files enforcement).
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"


@pytest.fixture(scope="module")
def lint_job():
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = doc["jobs"]
    assert "lint" in jobs, "lint job disappeared from ci.yml"
    return jobs["lint"]


def _steps(job):
    return [s for s in job["steps"] if isinstance(s, dict) and "run" in s]


def test_lint_job_keeps_advisory_semantics(lint_job):
    # Owner design: advisory (continue-on-error) until the backlog is retired.
    assert lint_job.get("continue-on-error") is True


def test_changed_files_tier_is_enforced(lint_job):
    runs = {s.get("name"): s["run"] for s in _steps(lint_job)}
    collect = runs["Collect changed Python files"]
    # The base commit must be resolvable, so the checkout needs full history.
    checkout = [
        s for s in lint_job["steps"] if isinstance(s, dict) and "checkout" in str(s.get("uses"))
    ]
    assert checkout and checkout[0]["with"]["fetch-depth"] == 0

    assert "changed-py.txt" in collect
    assert "git diff --name-only" in collect
    # Scope filter: only the four linted trees, only .py files.
    assert "grep -E '^(src|tests|scripts|examples)/'" in collect

    for name in ("Lint changed files (ruff)", "Format check changed files (ruff format)"):
        cmd = runs[name]
        assert "changed-py.txt" in cmd, f"{name} must consume the changed-file list"
        assert "xargs -r" in cmd, f"{name} must be a no-op on an empty changed set"


def test_repo_tier_is_informational_only(lint_job):
    runs = {s.get("name"): s["run"] for s in _steps(lint_job)}
    repo = runs["Repo-wide lint backlog (informational, never fails the job)"]
    # Every whole-tree ruff call must be followed by a shell guard, otherwise a
    # pre-existing finding would flip the job red again.
    for line in [ln for ln in repo.splitlines() if "ruff" in ln]:
        assert "|| true" in line, f"unguarded whole-tree call: {line}"


def test_no_whole_tree_ruff_step_left_blocking(lint_job):
    for step in _steps(lint_job):
        run = step["run"]
        if "src tests scripts examples" in run:
            assert "|| true" in run, f"whole-tree ruff call without guard: {step.get('name')}"
