#!/usr/bin/env bash
# Reproduce the CI lint gate locally, on exactly the scope CI lints.
#
# The ``lint`` job in .github/workflows/ci.yml runs two commands over the Python
# files a change touches inside src/, tests/, scripts/, examples/ and
# benchmarks/:
#
#   ruff check <files>
#   ruff format --check <files>
#
# Only the first one gets exercised by hand ("run ruff check before pushing"),
# so a purely formatting defect can reach main unnoticed: commit f5be5cf1 added
# tests/test_smoke_m4622.py without a trailing newline, ``ruff check`` passed,
# ``ruff format --check`` did not, and main carried a red ``lint (advisory)``
# job for a one-byte defect (m4663, see docs/lint-gate-determinism.md §6).
# This script runs *both* commands over the *same* scope as CI so the gate can
# be reproduced -- and fixed -- before the push, and returns the CI status.
#
# Usage:
#   scripts/lint_gate.sh [BASE_REV]
#
#   BASE_REV   git revision to diff against.  Default (no argument, no GitHub
#              environment): the merge base with origin/main, i.e. "everything
#              my branch adds on top of main".  When the merge base cannot be
#              resolved the scope script widens to the whole tree, which is its
#              documented audit mode (expected red against the legacy backlog).
#              Requires the ruff version CI uses -- pinned in the ``dev`` extra
#              of pyproject.toml (ruff>=0.15,<0.16).
#
# Exit status: 0 when every scoped file passes both commands (or when the scope
# is empty), 1 otherwise -- the same signal the CI job produces.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${1:-}"

if [ -z "$base" ] && [ -z "${GITHUB_EVENT_NAME:-}" ]; then
    # Local default: everything this branch adds on top of main.  Keep it
    # non-fatal: an unresolved ref leaves ``base`` empty and lint_scope.sh
    # falls back to the full-tree audit mode on purpose.
    base="$(git merge-base HEAD origin/main 2>/dev/null || true)"
fi

files="$(bash "$here/lint_scope.sh" "$base" | sed '/^$/d')"
if [ -z "$files" ]; then
    echo "lint_gate: nothing to lint (empty change scope); ruff steps skipped."
    exit 0
fi

printf '%s\n' "$files"
status=0
if ! printf '%s\n' "$files" | xargs -r -d '\n' ruff check; then
    status=1
fi
if ! printf '%s\n' "$files" | xargs -r -d '\n' ruff format --check; then
    status=1
fi
echo "lint_gate: $(printf '%s\n' "$files" | wc -l | tr -d ' ') file(s) scoped, exit status ${status}."
exit "$status"