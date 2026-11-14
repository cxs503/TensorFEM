"""Reproducibility gate for the CASE-SPHERE-PRESSURE record (``result.json``).

What this checks, and what it deliberately does not
---------------------------------------------------
The committed ``result.json`` is evidence only if a re-run reproduces it.  This
tool re-executes the case into a throw-away directory and compares the two
records **leaf by leaf** (1419 leaves today), partitioning every difference into
one of three declared classes:

``drift``
    a deterministic leaf moved.  Any difference here is a red gate: the record
    no longer describes what the code does (stale run, hand edit, code change).
``timing``
    ``flat_facet_control/wall_clock_s`` and ``flat_facet_control/timing/*``.
    Wall clock is host-load dependent by construction, so a re-run *cannot*
    reproduce it.  It is excluded from the drift comparison but is *not*
    unchecked (see below).
``out_dir_dependent``
    ``figures[*]``: the record stores figure paths relative to the run's output
    directory, so the directory part is expected to differ.  The panel basenames
    and their order are compared instead.

Timing is not evidence, but a timing record can still be *wrong*.  Two checks
keep it honest:

* **internal consistency** -- ``timing`` must be a repeated measurement
  (``repeats >= 3``), ordered (``min_s <= median_s <= max_s``), with
  ``samples_s`` of that length and ``wall_clock_s == median_s``.  The closed PR
  #860 changed only the bare scalar ``wall_clock_s`` (13.23 -> 0.54) with no
  measurement behind it; with the block in place that edit is internally
  inconsistent, so the record cannot be typed into shape one number at a time.
* **plausibility bound** -- the re-run median must stay within
  ``TIMING_PLAUSIBILITY_FACTOR`` (8x) of the recorded median.  The bound exists
  to catch a frozen outlier or a typed-in number, *not* to grade performance:
  the same leaf was recorded once at 13.23 s and measured at 0.44-1.64 s
  elsewhere (ratio 8-30x), while legitimately observed run-to-run spread stayed
  around 3.4x.  A hand edit *inside* the factor is not detectable here -- wall
  clock cannot be authenticated, only bounded; that limitation is stated rather
  than papered over.

Usage
-----
``python3 benchmarks/verified/fem_sphere_pressure_membrane/repro_check.py``
    re-run and grade; exit 0 iff no drift and the timing record is consistent
    and plausible.
``... repro_check.py --json``
    the same verdict as machine-readable JSON on stdout.
``... repro_check.py --committed <path>``
    grade a specific record (default: this directory's ``result.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.sphere_pressure_case import (  # noqa: E402
    LSB_TOL,
    TIMING_LEAVES,
    TIMING_PLAUSIBILITY_FACTOR,
    TIMING_REPEATS,
    run_case,
)

COMMITTED = os.path.join(HERE, "result.json")


# --------------------------------------------------------------------------
# leaf helpers
# --------------------------------------------------------------------------
def flatten(obj, prefix: str = "") -> dict:
    """Flatten a JSON record into ``{"/a/b[0]/c": value}`` leaves."""
    out: dict = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}/{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{index}]"))
    else:
        out[prefix] = obj
    return out


def is_timing_leaf(path: str) -> bool:
    """Is ``path`` one of the declared wall-clock leaves?  (Accepts "/a/b" or "a/b".)"""
    normalised = path.lstrip("/")
    return any(normalised == leaf or normalised.startswith(leaf + "/") for leaf in TIMING_LEAVES)


def is_figure_leaf(path: str) -> bool:
    return path.startswith("/figures[")


def _same(left, right, tol: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right or left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left == right:
            return True
        scale = max(abs(left), abs(right))
        return scale > 0 and abs(left - right) / scale <= tol
    return left == right


# --------------------------------------------------------------------------
# the two graders
# --------------------------------------------------------------------------
def timing_consistency(record: dict) -> list[str]:
    """Internal consistency of the timing record (empty list == consistent)."""
    problems: list[str] = []
    flat = flatten(record)
    timing = record.get("flat_facet_control", {}).get("timing")
    if not isinstance(timing, dict):
        return [
            "flat_facet_control/timing: missing (timing must be a measured block, "
            "not a bare scalar)"
        ]
    repeats = timing.get("repeats")
    samples = timing.get("samples_s")
    median = timing.get("median_s")
    try:
        lo, hi = float(timing["min_s"]), float(timing["max_s"])
        med = float(median)
    except (KeyError, TypeError, ValueError):
        return [f"flat_facet_control/timing: min_s/median_s/max_s missing or not numeric: {timing}"]
    if not isinstance(repeats, int) or repeats < 3:
        problems.append(f"flat_facet_control/timing/repeats: {repeats!r} < 3 samples")
    if not isinstance(samples, list) or len(samples) != repeats:
        problems.append(
            f"flat_facet_control/timing/samples_s: {len(samples or [])} samples "
            f"for repeats={repeats!r}"
        )
    elif len(samples) >= 1:
        if min(samples) != lo or max(samples) != hi:
            problems.append(
                f"flat_facet_control/timing: min_s/max_s {lo}/{hi} do not bracket "
                f"the samples {samples}"
            )
        ordered = sorted(samples)
        if ordered[len(ordered) // 2] != med:
            problems.append(
                f"flat_facet_control/timing/median_s: {med} is not the median of {samples}"
            )
    if not lo <= med <= hi:
        problems.append(f"flat_facet_control/timing: median_s {med} outside [{lo}, {hi}]")
    spread = timing.get("spread_ratio")
    if hi > 0 and spread is not None and abs(float(spread) - round(hi / lo, 2)) > 1e-9:
        problems.append(
            f"flat_facet_control/timing/spread_ratio: {spread} != max/min ({round(hi / lo, 2)})"
        )
    if timing.get("gated") is not False:
        problems.append(
            "flat_facet_control/timing/gated: wall clock must be declared non-gating provenance"
        )
    legacy = flat.get("/flat_facet_control/wall_clock_s")
    if legacy is None:
        problems.append("/flat_facet_control/wall_clock_s: missing")
    elif abs(float(legacy) - med) > 0.005 + 1e-9:
        problems.append(
            f"/flat_facet_control/wall_clock_s: {legacy} != timing/median_s {med} "
            "(a lone edit of the scalar is not a measurement)"
        )
    return problems


def compare(
    stored: dict, fresh: dict, *, tol: float = LSB_TOL, factor: float = TIMING_PLAUSIBILITY_FACTOR
) -> dict:
    """Grade a fresh record against a stored one.  Returns a verdict dict."""
    a, b = flatten(stored), flatten(fresh)
    drift: list[dict] = []
    timing: list[dict] = []
    figures: list[str] = []
    for path in sorted(set(a) | set(b)):
        if path not in a or path not in b:
            report = {
                "leaf": path,
                "stored": a.get(path, "<missing>"),
                "fresh": b.get(path, "<missing>"),
            }
            (figures if is_figure_leaf(path) else drift).append(
                path if is_figure_leaf(path) else report
            )
            continue
        if _same(a[path], b[path], tol):
            continue
        if is_timing_leaf(path):
            timing.append({"leaf": path, "stored": a[path], "fresh": b[path]})
        elif is_figure_leaf(path):
            figures.append(path)
        else:
            drift.append({"leaf": path, "stored": a[path], "fresh": b[path]})

    # figure panels: the directory prefix is out-dir dependent, the panels are not
    stored_panels = [os.path.basename(p) for p in (stored.get("figures") or [])]
    fresh_panels = [os.path.basename(p) for p in (fresh.get("figures") or [])]
    if stored_panels != fresh_panels:
        drift.append({"leaf": "/figures", "stored": stored_panels, "fresh": fresh_panels})

    # timing: bounded plausibility, not equality
    stored_med = (stored.get("flat_facet_control", {}).get("timing") or {}).get("median_s")
    fresh_med = (fresh.get("flat_facet_control", {}).get("timing") or {}).get("median_s")
    ratio = None
    plausible = True
    if stored_med and fresh_med:
        ratio = max(float(stored_med), float(fresh_med)) / min(float(stored_med), float(fresh_med))
        plausible = ratio <= factor
    else:
        plausible = False
        ratio = None

    problems = timing_consistency(stored)
    if not plausible:
        problems.append(
            f"flat_facet_control/timing: recorded median_s {stored_med} vs re-run {fresh_med} "
            f"(ratio {ratio if ratio is None else round(ratio, 2)}x > plausibility factor "
            f"{factor:g}x) -- the recorded timing is a frozen outlier or was typed in; re-run "
            f"the case to regenerate the record"
        )

    return {
        "stored_record": "committed" if stored is not None else None,
        "leaves": {"stored": len(a), "fresh": len(b), "union": len(set(a) | set(b))},
        "drift": drift,
        "timing_leaf_moves": timing,
        "out_dir_dependent_leaf_moves": len(figures),
        "timing_ratio": None if ratio is None else round(ratio, 3),
        "timing_plausibility_factor": factor,
        "timing_problems": problems,
        "passed": not drift and not problems,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="repro_check",
        description="Re-run CASE-SPHERE-PRESSURE and grade the committed record leaf by leaf.",
    )
    parser.add_argument("--committed", default=COMMITTED, help="record to grade")
    parser.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress the case summary")
    args = parser.parse_args(argv)

    with open(args.committed) as fh:
        stored = json.load(fh)
    with tempfile.TemporaryDirectory() as tmp:
        if args.quiet:
            stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
            try:
                fresh = run_case(out_dir=tmp)
            finally:
                sys.stdout.close()
                sys.stdout = stdout
        else:
            fresh = run_case(out_dir=tmp)
    fresh["figures"] = [os.path.join(HERE, os.path.basename(p)) for p in fresh.get("figures") or []]
    verdict = compare(stored, fresh)
    verdict["stored_record"] = os.path.abspath(args.committed)

    if args.json:
        print(json.dumps(verdict, indent=1, sort_keys=True))
    else:
        print(
            f"leaves compared: {verdict['leaves']['union']} "
            f"(declared timing leaves: {list(TIMING_LEAVES)}, repeats={TIMING_REPEATS})"
        )
        print(f"deterministic drift: {len(verdict['drift'])}")
        for row in verdict["drift"][:20]:
            print(f"  DRIFT {row['leaf']}: {row['stored']!r} -> {row['fresh']!r}")
        print(f"declared timing moves (tolerated): {len(verdict['timing_leaf_moves'])}")
        for row in verdict["timing_leaf_moves"][:10]:
            print(f"  timing {row['leaf']}: {row['stored']!r} -> {row['fresh']!r}")
        print(f"out-dir dependent figure leaves: {verdict['out_dir_dependent_leaf_moves']}")
        print(
            f"timing plausibility: ratio={verdict['timing_ratio']}x "
            f"(factor {TIMING_PLAUSIBILITY_FACTOR:g}x)"
        )
        for problem in verdict["timing_problems"]:
            print(f"  TIMING {problem}")
        print("PASSED" if verdict["passed"] else "FAILED")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
