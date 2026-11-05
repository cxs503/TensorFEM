#!/usr/bin/env python3
"""Byte-budgeted figure pipeline for the evidence PNGs under ``docs/assets``.

Why this exists
---------------
Evidence figures are committed to git so the documentation renders on GitHub,
but a 400 KB matplotlib render of a smooth field carries far more resolution
than a reader needs and it inflates every diff/PR payload.  The merge gate in
this ecosystem caps the artifact payload of a pull request at 200 KB
(``max_artifact_kb``), so oversized renders block merges instead of merely
being wasteful.

Policy implemented here (two tiers):

* **committed tier** -- ``docs/assets/compact/<area>/<name>.png``: a
  palette-quantised, downsampled preview.  Per-file budget is small (default
  20 KB) and the *whole* compact tier has a total budget (default 180 KB) so
  a future gate that sums real blob sizes still passes.
* **regenerable tier** -- the full-resolution render stays a build artifact:
  it is produced by the benchmark/audit script that owns the figure and is not
  committed.  ``--apply`` reads it, writes the compact preview and removes the
  full-size blob from the tree.

Usage
-----
    # list committed PNGs that are over the per-file cap, do nothing else
    python scripts/compact_figures.py --oversized

    # (re)build the compact tier from the full-size renders
    python scripts/compact_figures.py --apply [PATH ...]

    # machine gate: per-file cap, tier total, and doc link integrity
    python scripts/compact_figures.py --check

The compaction is deterministic (fixed resize ladder, fixed palette sizes,
``optimize=True``) so re-running it on the same input reproduces the same
bytes.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

PNG = ".png"
COMPACT_DIR = Path("docs/assets/compact")
#: per-file ceiling for a committed figure (matches the ecosystem gate)
MAX_KB = 200
#: per-file budget the compactor aims for
FILE_BUDGET_KB = 20
#: budget for the whole compact tier (whole tier, real bytes)
TOTAL_BUDGET_KB = 180
#: images referenced from documentation
DOC_DIRS = ("docs", "README.md", "CHANGELOG.md")
IMG_REF = re.compile(r"!\[[^\]]*\]\(([^)]+\.(?:png|svg|jpg))\)")

_RESIZE_LADDER = (1.0, 0.85, 0.72, 0.60, 0.50, 0.42, 0.35, 0.29, 0.24)
_PALETTE_LADDER = (256, 128, 64, 32, 16)


def compact_target(src: Path) -> Path:
    """Committed preview path for a full-size render path.

    ``docs/assets/thermal_2d/field.png`` -> ``docs/assets/compact/thermal_2d/field.png``
    Paths already inside the compact tier are returned unchanged.
    """
    src = Path(src)
    if COMPACT_DIR.as_posix() in src.as_posix():
        return src
    prefix = Path("docs/assets")
    try:
        rel = src.relative_to(prefix)
    except ValueError:
        rel = Path(src.name)
    return COMPACT_DIR / rel


def compact_png(src: Path, dst: Path, budget_bytes: int) -> dict:
    """Downsample + palette-quantise ``src`` into ``dst`` under ``budget_bytes``.

    Returns a small report dict.  Raises ``RuntimeError`` when even the
    smallest ladder step cannot reach the budget (that means the figure needs a
    redesign, not more compression).
    """
    from PIL import Image

    im = Image.open(src)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        im = flat
    else:
        im = im.convert("RGB")
    src_w, src_h = im.size

    best = None
    for scale in _RESIZE_LADDER:
        w = max(240, int(round(min(src_w, 1000) * scale)))
        h = max(160, int(round(src_h * w / src_w)))
        small = im if (w, h) == (src_w, src_h) else im.resize((w, h), Image.LANCZOS)
        for colors in _PALETTE_LADDER:
            q = small.quantize(colors=colors, method=Image.MEDIANCUT,
                               dither=Image.FLOYDSTEINBERG)
            buf = io.BytesIO()
            q.save(buf, "PNG", optimize=True, compress_level=9)
            size = buf.tell()
            if best is None or size < best[0]:
                best = (size, w, h, colors, buf.getvalue())
            if size <= budget_bytes:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(buf.getvalue())
                return {
                    "src": src.as_posix(),
                    "dst": dst.as_posix(),
                    "src_bytes": src.stat().st_size,
                    "bytes": size,
                    "width": w,
                    "height": h,
                    "colors": colors,
                    "over_budget": False,
                }
    # nothing fit: still write the smallest variant so the doc link resolves,
    # and report the overrun instead of failing silently.
    size, w, h, colors, data = best
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return {
        "src": src.as_posix(),
        "dst": dst.as_posix(),
        "src_bytes": src.stat().st_size,
        "bytes": size,
        "width": w,
        "height": h,
        "colors": colors,
        "over_budget": True,
    }


def committed_pngs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "docs").rglob("*.png"))


def oversized(root: Path, cap_kb: int = MAX_KB) -> list[tuple[int, Path]]:
    out = []
    for p in committed_pngs(root):
        n = p.stat().st_size
        if n > cap_kb * 1024:
            out.append((n, p.relative_to(root)))
    return sorted(out, reverse=True)


def doc_image_refs(root: Path) -> list[tuple[Path, str]]:
    """``(markdown file, referenced path)`` for every image embed in the docs."""
    refs = []
    files: list[Path] = []
    for entry in DOC_DIRS:
        p = root / entry
        if p.is_dir():
            files += sorted(p.rglob("*.md"))
        elif p.is_file():
            files.append(p)
    for f in files:
        for m in IMG_REF.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            refs.append((f, m.group(1)))
    return refs


def missing_refs(root: Path) -> list[tuple[str, str]]:
    """Image embeds whose target is absent.

    A reference is resolved twice, because documents in this tree use both
    conventions: relative to the markdown file (GitHub rendering) and relative
    to the repository root (report prose).  Non-local references (URLs,
    ``attachment:`` uploads, ``<placeholder>`` templates) are not checkable and
    are skipped.
    """
    bad = []
    for f, ref in doc_image_refs(root):
        if "://" in ref or ref.startswith("attachment:") or "<" in ref:
            continue
        candidates = [(f.parent / ref).resolve(), (root / ref).resolve()]
        if not any(c.is_file() for c in candidates):
            bad.append((f.relative_to(root).as_posix(), ref))
    return bad


def tier_report(root: Path, file_budget_kb: int = FILE_BUDGET_KB,
                total_budget_kb: int = TOTAL_BUDGET_KB) -> dict:
    tier = sorted((root / COMPACT_DIR).rglob("*.png")) if (root / COMPACT_DIR).is_dir() else []
    rows = []
    for p in tier:
        n = p.stat().st_size
        rows.append({"path": p.relative_to(root).as_posix(), "bytes": n,
                     "over_file_budget": n > file_budget_kb * 1024})
    total = sum(r["bytes"] for r in rows)
    return {
        "files": rows,
        "total_bytes": total,
        "file_budget_kb": file_budget_kb,
        "total_budget_kb": total_budget_kb,
        "over_total": total > total_budget_kb * 1024,
        "violations": [r["path"] for r in rows if r["over_file_budget"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path,
                    help="full-size PNG renders to compact (docs/assets/... paths)")
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root")
    ap.add_argument("--apply", action="store_true", help="write compact previews")
    ap.add_argument("--remove-sources", action="store_true",
                    help="delete the full-size renders after a successful --apply")
    ap.add_argument("--oversized", action="store_true",
                    help="list committed PNGs above --max-kb and exit")
    ap.add_argument("--check", action="store_true",
                    help="validate the committed tiers and doc links")
    ap.add_argument("--strict-links", action="store_true",
                    help="with --check: a broken document image link fails the run "
                         "(the tree still carries legacy broken links, so CI only "
                         "warns; tests/test_figure_artifact_budget.py enforces the "
                         "no-new-breakage ratchet)")
    ap.add_argument("--max-kb", type=int, default=MAX_KB)
    ap.add_argument("--file-budget-kb", type=int, default=FILE_BUDGET_KB)
    ap.add_argument("--total-budget-kb", type=int, default=TOTAL_BUDGET_KB)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--manifest", type=Path, default=Path("docs/assets/figure-budget.json"),
                    help="budget manifest {src, budget_kb} used when no PATH is given")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    if args.oversized:
        rows = oversized(root, args.max_kb)
        for n, p in rows:
            print(f"{n:>8}  {p.as_posix()}")
        if not rows:
            print(f"no committed PNG above {args.max_kb} KB")
        return 0

    if args.check:
        rep = tier_report(root, args.file_budget_kb, args.total_budget_kb)
        big = oversized(root, args.max_kb)
        bad_refs = missing_refs(root)
        if args.json:
            print(json.dumps({"tier": rep,
                              "oversized": [{"bytes": n, "path": p.as_posix()} for n, p in big],
                              "missing_refs": bad_refs}, indent=2))
        else:
            print(f"compact tier: {len(rep['files'])} files, "
                  f"{rep['total_bytes'] / 1024:.1f} KB / {args.total_budget_kb} KB")
            for r in rep["files"]:
                flag = "  OVER-FILE-BUDGET" if r["over_file_budget"] else ""
                print(f"  {r['bytes']:>7}  {r['path']}{flag}")
            if big:
                print(f"oversized committed renders (>{args.max_kb} KB): "
                      f"{len(big)} file(s) totalling {sum(n for n, _ in big) / 1024:.0f} KB")
                for n, p in big:
                    print(f"  {n:>8}  {p.as_posix()}")
            for f, ref in bad_refs:
                print(f"MISSING IMAGE  {f} -> {ref}")
        ok = not rep["over_total"] and not rep["violations"]
        if args.strict_links and bad_refs:
            ok = False
        return 0 if ok else 1

    jobs: list[tuple[Path, int]] = []
    for raw in args.paths:
        jobs.append((raw, args.file_budget_kb * 1024))
    if not jobs:
        manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
        if not manifest.is_file():
            ap.error("nothing to do: pass PNG paths with --apply, or provide a manifest")
        spec = json.loads(manifest.read_text())
        for entry in spec.get("figures", []):
            jobs.append((Path(entry["src"]),
                         int(entry.get("budget_kb", args.file_budget_kb)) * 1024))

    results = []
    for raw, budget in jobs:
        src = (root / raw).resolve() if not raw.is_absolute() else raw
        if not src.is_file():
            print(f"skip (absent): {raw}", file=sys.stderr)
            continue
        dst_rel = compact_target(src.relative_to(root) if src.is_relative_to(root) else raw)
        dst = root / dst_rel
        rep = compact_png(src, dst, budget)
        results.append(rep)
        if not args.json:
            print(f"{rep['src_bytes']:>8} -> {rep['bytes']:>7} bytes  "
                  f"{rep['width']}x{rep['height']} {rep['colors']}c  {rep['dst']}"
                  + ("  OVER-BUDGET" if rep["over_budget"] else ""))

    if args.apply and args.remove_sources:
        for rep in results:
            src = root / rep["src"]
            if src.is_file() and Path(rep["src"]) != Path(rep["dst"]):
                src.unlink()
                if not args.json:
                    print(f"removed full-size render {rep['src']}")

    total = sum(r["bytes"] for r in results)
    if not args.json:
        print(f"compact tier written: {len(results)} file(s), {total / 1024:.1f} KB")
    else:
        print(json.dumps({"results": results, "total_bytes": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())