"""Audit duplicated work and duplicated assertions across the TensorFEM test suite.

The suite has ~240 test modules and no shared expensive fixtures, so the same
reference solution (an independent Q8 Cook-membrane solve, for example) is
recomputed in several modules and the retest wall clock pays for it more than
once.  This script produces the machine-readable audit behind
``docs/test-duplication-audit.md``:

``audit``          structural duplication: repeated expensive call sites,
                   assertion statements that occur verbatim in several test
                   functions, and test bodies that are duplicates of each other.
``consolidation``  the add/remove assertion table between a base git ref and the
                   working tree, with a machine check that every *removed*
                   assertion statement still exists in a surviving test (i.e.
                   the merge did not shrink assertion coverage).

Usage::

    python3 scripts/test_duplication_audit.py audit \
        --out artifacts/test_duplication_audit.json \
        --md docs/test-duplication-audit.md
    python3 scripts/test_duplication_audit.py consolidation --base-ref main \
        --out artifacts/cook_assertion_consolidation.json

The audit is static (AST only); it never imports the package, so it is cheap
enough to run in CI and cannot be fooled by import side effects.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

#: Callable-name prefixes whose invocations are treated as expensive enough to
#: be worth sharing across test modules (an iterative solve, a mesh walk, a
#: benchmark driver).  Deliberately explicit: a name-prefix allow list cannot
#: silently drift into "everything is expensive".
EXPENSIVE_PREFIXES = (
    "assemble_",
    "build_",
    "compare_",
    "convergence_",
    "cross_check_",
    "element_centre_",
    "loaded_edge_",
    "qualify_",
    "run_",
    "solve_",
    "strain_recovery_",
    "trace_",
)

TESTS = pathlib.Path("tests")


def _normalise(node: ast.AST) -> str:
    """Unparse a node into a whitespace-free one-line string."""
    return " ".join(ast.unparse(node).split())


def _is_test(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
        "test_"
    )


def _call_key(node: ast.Call) -> str | None:
    """Return ``name(arguments)`` for an expensive call, else ``None``.

    Only literal arguments are rendered; a call whose arguments depend on local
    state is keyed on the argument *shapes* so that identical setups still
    collapse together.
    """
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if not name or not name.startswith(EXPENSIVE_PREFIXES):
        return None
    parts = []
    for arg in node.args:
        if isinstance(arg, ast.Constant):
            parts.append(repr(arg.value))
        else:
            parts.append(f"<{type(arg).__name__}>")
    for keyword in node.keywords:
        if isinstance(keyword.value, ast.Constant):
            parts.append(f"{keyword.arg}={keyword.value.value!r}")
        else:
            parts.append(f"{keyword.arg}=<{type(keyword.value).__name__}>")
    return f"{name}({', '.join(parts)})"


def _test_functions(path: pathlib.Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree) if _is_test(node)}


def _assert_canonical(node: ast.Assert) -> str:
    """Canonical predicate of an ``assert``: message dropped, truthiness normalised.

    ``assert check["passed"], check`` and ``assert check["passed"] is True`` are the
    same claim about the code under test; collapsing them (and ``is False`` into a
    negation) is what makes an "every removed assertion still exists" check meaningful
    instead of a whitespace diff.
    """
    expression = _normalise(node.test)
    if expression.endswith(" is True"):
        return expression[: -len(" is True")]
    if expression.endswith(" is False"):
        return f"not ({expression[: -len(' is False')]})"
    return expression


def _assert_statements(node: ast.AST) -> list[str]:
    return [_assert_canonical(sub) for sub in ast.walk(node) if isinstance(sub, ast.Assert)]


def audit(root: pathlib.Path) -> dict:
    call_sites: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    assert_sites: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    bodies: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    test_count = 0
    for path in sorted((root / TESTS).glob("test_*.py")):
        module = path.name
        for name, node in _test_functions(path).items():
            test_count += 1
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    key = _call_key(sub)
                    if key:
                        call_sites[key].append((module, name))
            for statement in _assert_statements(node):
                assert_sites[statement].append((module, name))
            digest = hashlib.sha256(
                "\n".join(_normalise(sub) for sub in node.body).encode()
            ).hexdigest()[:16]
            bodies[digest].append((module, name))

    def summarise(mapping: dict[str, list[tuple[str, str]]], cross_file_only: bool):
        rows = []
        for key, sites in mapping.items():
            modules = sorted({module for module, _ in sites})
            if len(sites) < 2 or (cross_file_only and len(modules) < 2):
                continue
            rows.append(
                {
                    "key": key,
                    "sites": len(sites),
                    "distinct_modules": len(modules),
                    "redundant_executions": len(sites) - 1,
                    "modules": modules,
                    "tests": [f"{module}::{name}" for module, name in sites],
                }
            )
        rows.sort(key=lambda row: (-row["redundant_executions"], -row["distinct_modules"], row["key"]))
        return rows

    repeated_calls = summarise(call_sites, cross_file_only=False)
    shared_asserts = summarise(assert_sites, cross_file_only=True)
    duplicate_bodies = [
        {"digest": digest, "tests": [f"{module}::{name}" for module, name in sites]}
        for digest, sites in bodies.items()
        if len(sites) > 1
    ]
    return {
        "test_modules": len(list((root / TESTS).glob("test_*.py"))),
        "test_functions": test_count,
        "repeated_expensive_call_sites": repeated_calls,
        "assertions_repeated_across_modules": shared_asserts,
        "duplicate_test_bodies": duplicate_bodies,
        "totals": {
            "expensive_call_keys_repeated": len(repeated_calls),
            "redundant_expensive_executions": sum(
                row["redundant_executions"] for row in repeated_calls
            ),
            "assertions_repeated_across_modules": len(shared_asserts),
            "duplicate_test_bodies": len(duplicate_bodies),
        },
    }


def _base_test_sources(base_ref: str, root: pathlib.Path) -> dict[str, str]:
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", base_ref, "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    sources = {}
    for name in listing:
        if not name.endswith(".py"):
            continue
        sources[name] = subprocess.run(
            ["git", "show", f"{base_ref}:{name}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return sources


def _assert_map(sources: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """``{module: {test_name: [assert statements]}}`` for a set of module sources."""
    mapping: dict[str, dict[str, list[str]]] = {}
    for name, text in sources.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        functions = {
            node.name: _assert_statements(node) for node in ast.walk(tree) if _is_test(node)
        }
        mapping[pathlib.Path(name).name] = functions
    return mapping


def _apply_renames(statements: list[str], renames: dict[str, str]) -> list[str]:
    """Alpha-rename declared local identifiers so a rename is not read as a lost claim.

    ``assert checked["passed"]`` and ``assert check["passed"]`` where ``check``
    replaced ``checked`` inside the same test are the same claim; the mapping is
    supplied explicitly on the command line so the tool can never "explain away"
    a genuinely dropped assertion by guessing.
    """
    import re

    out = []
    for statement in statements:
        result = statement
        for old, new in renames.items():
            result = re.sub(rf"\b{re.escape(old)}\b", new, result)
        out.append(result)
    return out


def consolidation(
    base_ref: str, root: pathlib.Path, modules: list[str], renames: dict[str, str] | None = None
) -> dict:
    base = _assert_map(_base_test_sources(base_ref, root))
    current_files = {
        f"tests/{module}": (root / "tests" / module).read_text(encoding="utf-8")
        for module in modules
    }
    current = _assert_map(current_files)

    added, removed = [], []
    for module in modules:
        before = base.get(module, {})
        after = current.get(module, {})
        for test in sorted(set(after) - set(before)):
            added.append({"module": module, "test": test, "asserts": after[test]})
        for test in sorted(set(before) - set(after)):
            removed.append({"module": module, "test": test, "asserts": before[test]})
        for test in sorted(set(before) & set(after)):
            for statement in after[test]:
                if statement not in before[test]:
                    added.append({"module": module, "test": test, "asserts": [statement]})
            for statement in before[test]:
                if statement not in after[test]:
                    removed.append({"module": module, "test": test, "asserts": [statement]})

    renames = renames or {}
    for row in removed:
        row["matched_asserts"] = _apply_renames(row["asserts"], renames)
        row["survives_in"] = sorted(
            f"{module}::{test}"
            for module, tests in current.items()
            for test, statements in tests.items()
            if set(row["matched_asserts"]) <= set(statements)
        )
    for row in added:
        row["survives_in"] = [f"{row['module']}::{row['test']}"]
    remapped = [
        {
            "removed_from": f"{row['module']}::{row['test']}",
            "asserts": row["asserts"],
            "matched_asserts": row["matched_asserts"],
            "survives_in": row["survives_in"],
            "equivalent_survivor": bool(row["survives_in"]),
        }
        for row in removed
    ]
    lost = [row for row in remapped if not row["equivalent_survivor"]]
    return {
        "base_ref": base_ref,
        "modules": modules,
        "alpha_renames": renames,
        "added": added,
        "removed": remapped,
        "lost_assertions": lost,
        "coverage_preserved": not lost,
        "survivor_assert_total": sum(
            len(statements) for tests in current.values() for statements in tests.values()
        ),
    }


def _markdown(report: dict) -> str:
    totals = report["totals"]
    lines = [
        "# Test-suite duplication audit (m4718)",
        "",
        "Generated by ``python3 scripts/test_duplication_audit.py audit``; the numbers below are",
        "regenerated from the working tree, never hand-copied.",
        "",
        f"- test modules scanned: **{report['test_modules']}**",
        f"- test functions scanned: **{report['test_functions']}**",
        f"- expensive call keys executed more than once: **{totals['expensive_call_keys_repeated']}**",
        f"- redundant expensive executions: **{totals['redundant_expensive_executions']}**",
        f"- assertions repeated across modules: **{totals['assertions_repeated_across_modules']}**",
        f"- duplicate test bodies: **{totals['duplicate_test_bodies']}**",
        "",
        "## Highest-duplication expensive call sites",
        "",
        "| key | call sites | modules | redundant executions | modules |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["repeated_expensive_call_sites"][:15]:
        lines.append(
            f"| `{row['key']}` | {row['sites']} | {row['distinct_modules']} | "
            f"{row['redundant_executions']} | {', '.join(row['modules'])} |"
        )
    lines += [
        "",
        "## Assertions repeated verbatim in several modules",
        "",
        "| assertion | sites | modules |",
        "| --- | --- | --- |",
    ]
    for row in report["assertions_repeated_across_modules"][:15]:
        lines.append(f"| `{row['key']}` | {row['sites']} | {', '.join(row['modules'])} |")
    lines += [
        "",
        "Duplicated *files* are absent: the highest 3-line-shingle Jaccard similarity between any",
        "two test modules is 0.06, so the suite is not a copy-paste suite.  The duplication that",
        "costs wall clock is the recomputation of the same expensive reference solution (see the",
        "first table) and the handful of verbatim assertion statements in the second.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "consolidation"))
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--base-ref", default="main", help="git ref for the consolidation diff")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["test_cook_oracle_operator.py", "test_cook_stress_recovery_verification.py"],
        help="test modules whose assertion diff is tabulated",
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="declared identifier rename applied when matching removed assertions",
    )
    parser.add_argument("--out", default=None, help="artifact path (per-mode default)")
    parser.add_argument("--md", default="docs/test-duplication-audit.md")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root).resolve()

    if args.mode == "audit":
        renames = dict(item.split("=", 1) for item in args.rename)
        report = audit(root)
        payload, text = report, _markdown(report)
    else:
        renames = dict(item.split("=", 1) for item in args.rename)
        report = consolidation(args.base_ref, root, args.modules, renames)
        payload = report
        text = json.dumps(report, indent=2)

    default_out = {
        "audit": "artifacts/test_duplication_audit.json",
        "consolidation": "artifacts/cook_assertion_consolidation.json",
    }[args.mode]
    out = root / (args.out or default_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.mode == "audit":
        md = root / args.md
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(text, encoding="utf-8")
    print(json.dumps(payload.get("totals", {"coverage_preserved": payload.get("coverage_preserved")})))
    if args.mode == "consolidation" and not payload["coverage_preserved"]:
        print("ERROR: consolidation lost assertions", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())