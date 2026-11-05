#!/usr/bin/env python3
"""Resolve the two PR #61 conflicts as a semantic union (no side wins).

Both conflicts are documentation prose that each side extended; the merged
tree contains both implementations, so the prose must describe both.  The
resolution keeps every sentence of both sides (only line wrapping and the
"this module is not ..." connector are rewritten) and never deletes an
assertion from either side.

Run from the worktree root while the merge is in progress.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

MARINE = """This module itself still provides no large-angle `GZ` curve; two separate
services cover that question. The exact wall-sided `GZ` oracle lives in
`tensorfem.marine_large_angle_stability` and is documented in
[the audit report](marine-large-angle-stability-audit.md). The explicit
large-angle `GZ` curves and `KN` cross-curves are provided separately by
`tensorfem.marine_large_angle_gz` (see `docs/marine-large-angle-stability.md`),
which solves the exact clipped-section geometry for prismatic barges and is
verified against the wall-sided closed form to machine precision.

Beyond those two services this module offers no large-angle `GZ`, cross-curves,
free-surface, damage stability, compartment flooding, trim, arbitrary-hull
integration, wave-load, seakeeping, mooring, or classification-rule
implementation. Passing its tests does not constitute design approval or
regulatory/classification certification."""


def resolve_marine() -> None:
    path = ROOT / "docs" / "marine-hydrostatics.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("<<<<<<< "))
    other = next(i for i, ln in enumerate(lines) if ln.startswith(">>>>>>> "))
    # The shared sentence tail ("stability, compartment flooding, ...") is
    # restated by the resolved prose, so it is replaced rather than duplicated.
    end = other + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    path.write_text(
        "\n".join(lines[:start] + MARINE.split("\n") + lines[end:]) + "\n",
        encoding="utf-8",
    )


def resolve_changelog() -> None:
    path = ROOT / "CHANGELOG.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("<<<<<<< "))
    mid = next(i for i, ln in enumerate(lines) if ln.startswith("======="))
    other = next(i for i, ln in enumerate(lines) if ln.startswith(">>>>>>> "))
    side_main = lines[start + 1 : mid]          # prestressed-modal entry (newest)
    side_branch = lines[mid + 1 : other]        # 2026-11-04 large-angle GZ section
    # newest-first file order: keep main's entry, then the branch's dated section.
    resolved = lines[:start] + side_main + [""] + side_branch + lines[other + 1 :]
    path.write_text("\n".join(resolved) + "\n", encoding="utf-8")


if __name__ == "__main__":
    resolve_marine()
    resolve_changelog()
    for rel in ("CHANGELOG.md", "docs/marine-hydrostatics.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "<<<<<<<" not in text and ">>>>>>>" not in text, rel
        print(f"{rel}: markers removed, {len(text.splitlines())} lines")