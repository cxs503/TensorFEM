#!/usr/bin/env bash
# Print the Python files the lint gate must cover for the current change.
#
# The tree carries a large pre-existing ruff backlog (thousands of E701/E702
# statements spread over hundreds of legacy files).  Running the gate over the
# whole tree therefore reports red on every commit, including commits that add
# perfectly clean code, which makes the signal useless.  CI instead lints the
# *change*: the Python files this PR/push touches inside src/, tests/,
# scripts/ and examples/.  Legacy debt stays tracked separately and can be
# paid down file-by-file without blocking unrelated work.
#
# Usage:
#   scripts/lint_scope.sh [BASE_REV]
#
#   BASE_REV      git revision to diff against.  Omitted -> derived from the
#                 GitHub environment (pull_request: origin/$GITHUB_BASE_REF,
#                 push: $GITHUB_BEFORE).  When no base can be determined
#                 (workflow_dispatch, unreachable base commit) the script
#                 falls back to the full tree so a manual run still audits
#                 everything.
#
# Output: one repository-relative path per line, sorted, no duplicates.  An
# empty output means "nothing to lint" and the CI steps are skipped.
set -euo pipefail

scope_dirs=(src tests scripts examples benchmarks)
zero_sha="0000000000000000000000000000000000000000"

resolve_commit() {
    # Echo the commit SHA for $1 when it resolves locally, nothing otherwise.
    git rev-parse --verify --quiet "$1^{commit}" 2>/dev/null || true
}

base="${1:-}"

if [ -z "$base" ]; then
    case "${GITHUB_EVENT_NAME:-}" in
        pull_request | pull_request_target)
            base="origin/${GITHUB_BASE_REF:-main}"
            ;;
        push | merge_group)
            before="${GITHUB_BEFORE:-}"
            if [ -n "$before" ] && [ "$before" != "$zero_sha" ]; then
                base="$before"
            fi
            ;;
    esac
fi

base_commit=""
if [ -n "$base" ]; then
    base_commit="$(resolve_commit "$base")"
    if [ -z "$base_commit" ]; then
        # A shallow/unfetched base must not silently shrink the scope to the
        # single commit diff; fall back to the merge base with origin/main,
        # then to the full tree.
        echo "lint_scope: base '$base' unresolvable, trying merge-base with origin/main" >&2
        base_commit="$(git merge-base HEAD origin/main 2>/dev/null || true)"
    fi
fi

if [ -n "$base_commit" ]; then
    echo "lint_scope: diffing ${base_commit:0:12}...HEAD" >&2
    git diff --name-only --diff-filter=ACMR "${base_commit}...HEAD" -- "${scope_dirs[@]}" \
        | grep -E '\.py$' || true
else
    echo "lint_scope: no base revision, auditing the whole tree" >&2
    git ls-files -- "${scope_dirs[@]}" | grep -E '\.py$' || true
fi | sed '/^$/d' | sort -u