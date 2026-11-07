#!/usr/bin/env python3
"""Deterministic device-readiness ("device-isation") debt census for TensorFEM.

The census walks every Python module below ``src/`` (``tests/`` and ``examples/``
subtrees are excluded — they are allowed to be host-only) and counts four
classes of technical debt that block a full device (NPU/GPU) port.  All four
definitions are frozen here and mirrored verbatim in
``docs/debt_census_report.md`` so the numbers are reproducible and auditable.

D1 ``numpy_only``
    Number of scanned modules that import **no** device-capable array
    framework (``torch``, ``torch_npu``, ``torch_sdaa``, ``torch_mlu``,
    ``jax``, ``tensorflow``, ``paddle``, ``mindspore``, ``cupy``).  Such a
    module cannot run on an accelerator at all; it is the hardest debt class.

D2 ``sync_points``
    Number of ``<expr>.cpu()`` plus ``<expr>.numpy()`` call sites, i.e.
    tensor->host synchronising round trips that serialise the device queue.
    ``.item()``/``.tolist()``/``.synchronize()`` are reported separately in
    ``breakdown.sync_extended`` (they are also synchronising but the task
    metric is the cpu/numpy pair).

D3 ``no_device_constructs``
    Number of tensor-constructor call sites that do not state ``device=`` and
    cannot inherit it, i.e. ``torch.tensor/as_tensor/from_numpy/zeros/ones/
    full/empty/arange/linspace/logspace/eye/rand/randn/randint/randperm/
    normal/bernoulli/scalar_tensor/frombuffer`` calls with
    (a) no ``device=`` keyword, (b) no ``**kwargs`` spread and
    (c) no device inheritance (``*_like`` constructors inherit from their
    first argument, so they are *not* debt and live in
    ``breakdown.device_inherited_like``).  Constructors that forward
    ``**opts``/``**kwargs`` are reported as ``device_unverified_spread``:
    the device may be supplied by the caller, static analysis cannot tell.

D4 ``host_loops``
    Number of Python ``for``/``async for``/``while`` statement nodes.  Any
    interpreter-level loop executes on the host; loops whose body contains an
    indexed store (``a[i] = ...``, ``a[i] += ...``) or an accumulator
    ``.append(...)`` are additionally counted in
    ``breakdown.vectorizable_loops`` because those are the per-index loops
    that must become tensor ops for a device port.  ``while`` loops are solver
    time/iteration loops and land in ``breakdown.host_loops_by_kind.while``.

Coverage
    ``ast_coverage_pct = 100 * files_parsed / files_scanned``.  Files that
    fail to parse/decode/module-analysis are listed **in full** under
    ``uncovered_files`` with the failure reason (honest-FAIL rule: coverage
    below 70 % must name every uncovered file).

Determinism: files are visited in sorted order, every emitted list is sorted,
no timestamp / absolute path / environment value enters the JSON, and floats
are rounded.  Two consecutive runs therefore produce byte-identical output
(``source_digest`` is a content hash of the scanned files, not a wall clock).

Usage::

    python tools/debt_census.py                       # write artifacts/fem_debt_census.json
    python tools/debt_census.py --root src --out /tmp/a.json
    python tools/debt_census.py --roots src --quiet    # one-line summary to stdout
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pathlib
import sys
import tokenize

SCHEMA_VERSION = "fem-debt-census/1"

# Directories whose Python sources are deliberately host-only (test/example
# code) and therefore excluded from the debt population.
EXCLUDED_DIR_NAMES = frozenset({"tests", "test", "examples", "example", "__pycache__"})

# Importable roots that make a module device-capable.
DEVICE_FRAMEWORK_ROOTS = (
    "torch",
    "torch_npu",
    "torch_sdaa",
    "torch_mlu",
    "jax",
    "tensorflow",
    "paddle",
    "mindspore",
    "cupy",
)

# torch tensor constructors (module-level names, i.e. torch.<name>).
TENSOR_FACTORY_NAMES = frozenset(
    {
        "tensor",
        "as_tensor",
        "from_numpy",
        "frombuffer",
        "zeros",
        "ones",
        "full",
        "empty",
        "arange",
        "linspace",
        "logspace",
        "eye",
        "rand",
        "randn",
        "randint",
        "randperm",
        "normal",
        "bernoulli",
        "scalar_tensor",
        "zeros_like",
        "ones_like",
        "full_like",
        "empty_like",
    }
)
# Constructors that inherit the device of their first argument.
DEVICE_INHERITING_FACTORIES = frozenset({"zeros_like", "ones_like", "full_like", "empty_like"})
# Host->device boundary constructors (always CPU by definition).
HOST_BOUNDARY_FACTORIES = frozenset({"from_numpy", "frombuffer"})

SYNC_METHODS = frozenset({"cpu", "numpy"})
SYNC_METHODS_EXTENDED = frozenset({"item", "tolist", "synchronize"})

# Builtins / methods whose result is a host iterator or host sequence; a loop
# over them runs the interpreter per iteration.
HOST_ITERATOR_CALLS = frozenset(
    {
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "list",
        "tuple",
        "set",
        "dict",
        "iter",
        "divmod",
    }
)
HOST_ITERATOR_METHODS = frozenset(
    {
        "items",
        "keys",
        "values",
        "splitlines",
        "split",
        "tolist",
        "numpy",
        "cpu",
        "reshape",
        "ravel",
        "flatten",
        "copy",
        "sort",
    }
)


def discover_files(roots, repo_root):
    """Return (scanned_paths, excluded_paths) as sorted repo-relative strings."""
    scanned, excluded = [], []
    for root in roots:
        root_path = (repo_root / root).resolve()
        if not root_path.exists():
            raise SystemExit(f"root not found: {root}")
        for path in sorted(root_path.rglob("*.py")):
            rel = path.relative_to(repo_root.resolve()).as_posix()
            if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(root_path).parts[:-1]):
                excluded.append(rel)
            else:
                scanned.append(rel)
    return sorted(scanned), sorted(excluded)


def _import_aliases(tree):
    """Map local torch-ish aliases, direct factory names, frameworks and numpy."""
    torch_aliases, direct_factories, frameworks = set(), set(), set()
    imports_numpy = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DEVICE_FRAMEWORK_ROOTS:
                    frameworks.add(root)
                    torch_aliases.add(alias.asname or root)
                if root == "numpy":
                    imports_numpy = True
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
            if root in DEVICE_FRAMEWORK_ROOTS:
                frameworks.add(root)
                for alias in node.names:
                    if alias.name in TENSOR_FACTORY_NAMES:
                        direct_factories.add(alias.asname or alias.name)
            if root == "numpy":
                imports_numpy = True
    return torch_aliases, direct_factories, frameworks, imports_numpy


def _name_chain(node):
    """Dotted name of a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _factory_of(call, torch_aliases, direct_factories):
    """Return the factory name if ``call`` is a torch tensor constructor."""
    func = call.func
    if isinstance(func, ast.Attribute):
        owner = _name_chain(func.value)
        if owner in torch_aliases and func.attr in TENSOR_FACTORY_NAMES:
            return func.attr
        return None
    if isinstance(func, ast.Name) and func.id in direct_factories:
        return func.id
    return None


def _iterable_kind(loop):
    """Classify a ``for`` loop's iterable (documentation value only)."""
    if isinstance(loop, ast.While):
        return "while"
    it = loop.iter
    if isinstance(it, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "comprehension"
    if isinstance(it, ast.Call):
        func = it.func
        if isinstance(func, ast.Name) and func.id in HOST_ITERATOR_CALLS:
            return f"builtin:{func.id}"
        if isinstance(func, ast.Attribute) and func.attr in HOST_ITERATOR_METHODS:
            return f"method:{func.attr}"
        return "call"
    if isinstance(it, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return "literal"
    if isinstance(it, ast.Subscript):
        return "host_index"
    if isinstance(it, (ast.Name, ast.Attribute)):
        return "host_sequence"
    if isinstance(it, ast.BinOp):
        return "host_sequence"
    return "other"


def _has_vectorizable_body(loop):
    """True if the loop body stores through an index or accumulates by append."""
    for node in ast.walk(loop):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
            "append",
            "extend",
            "add",
            "update",
        }:
            return True
    return False


def analyse_module(rel_path, text):
    """Count the four debt classes in one module; raises on unparsable input."""
    tree = ast.parse(text, filename=rel_path)
    torch_aliases, direct_factories, frameworks, imports_numpy = _import_aliases(tree)
    numpy_only = not frameworks and imports_numpy
    # A module with no device framework import is numpy-only (host-only) when it
    # does import numpy; pure-python helper modules without numpy are reported
    # as python_only and kept out of the D1 count.
    sync_sites, construct_sites = [], []
    counts = {
        "cpu": 0,
        "numpy": 0,
        "item": 0,
        "tolist": 0,
        "synchronize": 0,
        "device_explicit": 0,
        "device_inherited_like": 0,
        "device_unverified_spread": 0,
        "no_device_plain": 0,
        "no_device_host_boundary": 0,
        "host_loops_for": 0,
        "host_loops_while": 0,
        "vectorizable_loops": 0,
    }
    loop_kinds = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in SYNC_METHODS | SYNC_METHODS_EXTENDED:
                kind = func.attr
                counts[kind] += 1
                if kind in SYNC_METHODS:
                    sync_sites.append({"file": rel_path, "line": node.lineno, "kind": kind})
            factory = _factory_of(node, torch_aliases, direct_factories)
            if factory is not None:
                keyword_names = {kw.arg for kw in node.keywords}
                if "device" in keyword_names:
                    counts["device_explicit"] += 1
                elif None in keyword_names:
                    counts["device_unverified_spread"] += 1
                elif factory in DEVICE_INHERITING_FACTORIES:
                    counts["device_inherited_like"] += 1
                else:
                    bucket = "no_device_host_boundary" if factory in HOST_BOUNDARY_FACTORIES else "no_device_plain"
                    counts[bucket] += 1
                    construct_sites.append(
                        {
                            "file": rel_path,
                            "line": node.lineno,
                            "factory": factory,
                            "bucket": bucket,
                        }
                    )
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            kind = _iterable_kind(node)
            loop_kinds[kind] = loop_kinds.get(kind, 0) + 1
            if isinstance(node, ast.While):
                counts["host_loops_while"] += 1
            else:
                counts["host_loops_for"] += 1
            if _has_vectorizable_body(node):
                counts["vectorizable_loops"] += 1
    return {
        "path": rel_path,
        "numpy_only": numpy_only,
        "frameworks": sorted(frameworks),
        "imports_numpy": imports_numpy,
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "counts": counts,
        "loop_kinds": dict(sorted(loop_kinds.items())),
        "sync_sites": sorted(sync_sites, key=lambda s: (s["file"], s["line"], s["kind"])),
        "construct_sites": sorted(construct_sites, key=lambda s: (s["file"], s["line"], s["factory"])),
    }


def token_cross_check(path, text):
    """Independent tokenize-based count of sync attribute calls (no AST)."""
    cpu = numpy = device_kw = 0
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    for index, tok in enumerate(tokens):
        if tok.type == tokenize.NAME and tok.string in ("cpu", "numpy"):
            prev = [t for t in tokens[:index] if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
            if prev and prev[-1].string == ".":
                following = [t for t in tokens[index + 1 :] if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
                if following and following[0].string == "(":
                    if tok.string == "cpu":
                        cpu += 1
                    else:
                        numpy += 1
        if tok.type == tokenize.NAME and tok.string == "device":
            following = [t for t in tokens[index + 1 :] if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
            if following and following[0].string == "=":
                device_kw += 1
    return {"cpu": cpu, "numpy": numpy, "device_kwarg": device_kw}


def build_census(repo_root, roots=("src",)):
    repo_root = pathlib.Path(repo_root)
    scanned, excluded = discover_files(roots, repo_root)
    per_file, uncovered, tokens_total, digest = [], [], {}, hashlib.sha256()
    for rel in scanned:
        path = repo_root / rel
        raw = path.read_bytes()
        digest.update(rel.encode() + b"\0" + hashlib.sha256(raw).hexdigest().encode() + b"\0")
        text = None
        try:
            text = raw.decode("utf-8")
            record = analyse_module(rel, text)
            tokens_total[rel] = token_cross_check(rel, text)
        except (UnicodeDecodeError, SyntaxError, ValueError, tokenize.TokenError) as exc:
            uncovered.append({"file": rel, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        per_file.append(record)

    coverage = 100.0 * len(per_file) / len(scanned) if scanned else 0.0
    numpy_only = [r for r in per_file if r["numpy_only"]]
    python_only = [r for r in per_file if not r["frameworks"] and not r["imports_numpy"]]

    def total(key):
        return sum(r["counts"][key] for r in per_file)

    sync_points = total("cpu") + total("numpy")
    no_device = total("no_device_plain") + total("no_device_host_boundary")
    host_loops = total("host_loops_for") + total("host_loops_while")

    loop_kinds = {}
    for record in per_file:
        for kind, value in record["loop_kinds"].items():
            loop_kinds[kind] = loop_kinds.get(kind, 0) + value

    token_cpu = sum(v["cpu"] for v in tokens_total.values())
    token_numpy = sum(v["numpy"] for v in tokens_total.values())

    def deviation(ast_value, token_value):
        base = max(ast_value, 1)
        return round(100.0 * abs(ast_value - token_value) / base, 4)

    census = {
        "schema_version": SCHEMA_VERSION,
        "repo": "TensorFEM",
        "roots": list(roots),
        "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
        "files_scanned": len(scanned),
        "files_parsed": len(per_file),
        "files_excluded": len(excluded),
        "excluded_files": excluded,
        "ast_coverage_pct": round(coverage, 2),
        "uncovered_files": uncovered,
        "source_digest": "sha256:" + digest.hexdigest(),
        # ---- the four debt classes (task metrics) -------------------------
        "numpy_only_count": len(numpy_only),
        "sync_points": sync_points,
        "no_device_constructs": no_device,
        "host_loops": host_loops,
        # ---- audit detail --------------------------------------------------
        "breakdown": {
            "sync_points": {"cpu": total("cpu"), "numpy": total("numpy")},
            "sync_extended": {
                "item": total("item"),
                "tolist": total("tolist"),
                "synchronize": total("synchronize"),
            },
            "tensor_constructs": {
                "total_factory_calls": (
                    no_device
                    + total("device_explicit")
                    + total("device_inherited_like")
                    + total("device_unverified_spread")
                ),
                "no_device_constructs": no_device,
                "no_device_plain": total("no_device_plain"),
                "no_device_host_boundary": total("no_device_host_boundary"),
                "device_explicit": total("device_explicit"),
                "device_inherited_like": total("device_inherited_like"),
                "device_unverified_spread": total("device_unverified_spread"),
            },
            "host_loops_by_kind": dict(sorted(loop_kinds.items())),
            "vectorizable_loops": total("vectorizable_loops"),
            "loops_in_numpy_only_modules": sum(
                r["counts"]["host_loops_for"] + r["counts"]["host_loops_while"] for r in numpy_only
            ),
            "python_only_modules": {
                "count": len(python_only),
                "modules": sorted(r["path"] for r in python_only),
            },
            "device_module_count": len(per_file) - len(numpy_only) - len(python_only),
        },
        "cross_check": {
            "method": "tokenize-based recount of '.cpu()'/'.numpy()' call sites vs AST",
            "ast": {"cpu": total("cpu"), "numpy": total("numpy")},
            "tokenize": {"cpu": token_cpu, "numpy": token_numpy},
            "deviation_pct": {"cpu": deviation(total("cpu"), token_cpu), "numpy": deviation(total("numpy"), token_numpy)},
        },
        "numpy_only_modules": sorted(r["path"] for r in numpy_only),
        "sync_sites": sorted(
            (site for r in per_file for site in r["sync_sites"]), key=lambda s: (s["file"], s["line"], s["kind"])
        ),
        "no_device_sites": sorted(
            (site for r in per_file for site in r["construct_sites"]),
            key=lambda s: (s["file"], s["line"], s["factory"]),
        ),
        "per_file": [
            {
                "path": r["path"],
                "numpy_only": r["numpy_only"],
                "frameworks": r["frameworks"],
                "lines": r["lines"],
                "sync_points": r["counts"]["cpu"] + r["counts"]["numpy"],
                "no_device_constructs": r["counts"]["no_device_plain"] + r["counts"]["no_device_host_boundary"],
                "host_loops": r["counts"]["host_loops_for"] + r["counts"]["host_loops_while"],
                "vectorizable_loops": r["counts"]["vectorizable_loops"],
                "device_explicit": r["counts"]["device_explicit"],
            }
            for r in sorted(per_file, key=lambda r: r["path"])
        ],
    }
    return census


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=None, help="repository root (default: parent of tools/)")
    parser.add_argument("--roots", nargs="+", default=["src"], help="source roots to scan (default: src)")
    parser.add_argument("--out", default="artifacts/fem_debt_census.json", help="JSON output path, relative to repo root")
    parser.add_argument("--quiet", action="store_true", help="print a one-line summary instead of the JSON")
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root) if args.repo_root else pathlib.Path(__file__).resolve().parents[1]
    census = build_census(repo_root, tuple(args.roots))
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(census, indent=2, sort_keys=False, ensure_ascii=True) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    if args.quiet:
        print(
            "numpy_only={numpy_only_count} sync_points={sync_points} "
            "no_device_constructs={no_device_constructs} host_loops={host_loops} "
            "ast_coverage_pct={ast_coverage_pct} uncovered={u} -> {out}".format(
                u=len(census["uncovered_files"]), out=out_path
            , **census)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())