#!/usr/bin/env bash
# Reproduce the CI lint gate locally, on exactly the scope CI lints.
#
# The ``lint`` job in .github/workflows/ci.yml runs two commands on the Python
# files a change touches inside src/, tests/, scripts/, examples/ and
# benchmarks/:
#
#   ruff check <files>
#   ruff format --check <files>
#
# Only the first one used to be exercised by hand before pushing, so a purely
# formatting defect could reach main unnoticed: commit f5be5cf1 added
# tests/test_smoke_m4622.py without a newline at end of file, ``ruff check``
# passed and ``ruff format --check`` did not, and main carried a red
# ``lint (advisory)`` job for a one-byte defect (m4663).  This script runs both
# commands over the same scope so the gate can be reproduced -- and fixed --
# before the push.
#
# Usage:
#   scripts/lint_gate.sh [BASE_REV]
#
#   BASE_REV   git revision to diff against (defaults to the merge base with
#              origin/main, i.e. "everything my branch adds on top of main").
#              Requires the same ``ruff`` version as CI (see the ``dev`` extra
#              in pyproject.toml).
#
# Exit status: 0 when the scoped files pass both commands (or when the scope is
# empty), 1 otherwise -- the same signal the CI job produces.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${1:-}"

if [ -z "$base" ] && [ -z "${GITHUB_EVENT_NAME:-}" ]; then
    # Local default: everything this branch adds on top of main.  Without a
    # resolvable merge base the scope script widens to the whole tree, which is
    # the documented audit mode (expected red against the legacy backlog).
    if merge_base="$(git merge-base HEAD origin/main 2>/dev/null)"; then
        base="$merge_base"
    fi
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