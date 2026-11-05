"""Guards for the committed evidence-figure budget (docs/figure-artifact-budget.md).

The merge gate in this ecosystem caps the artifact payload of a pull request at
200 KB, and a 400 KB render of a smooth field is unreadable *and* unmergeable.
These tests keep the discipline machine-checkable:

* the committed preview tier stays inside its per-file and total budgets, and
  is rebuilt deterministically by ``scripts/compact_figures.py``;
* every image a document embeds actually exists in the tree;
* no *new* oversized render is committed.  The set below is a ratchet: entries
  may only be removed (by compacting them), never added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compact_figures as cf  # noqa: E402

#: Renders that predate the budget policy and still exceed the per-file cap.
#: Every entry is either fixed by ``scripts/compact_figures.py`` or reported as
#: platform-level debt; the list must not grow.
LEGACY_OVERSIZED = {
    "docs/assets/benchmark-clouds/hemisphere-hole/displacement.png",
    "docs/assets/benchmark-clouds/hemisphere-hole/von_mises.png",
    "docs/assets/benchmark-clouds/sphere-pressure/displacement.png",
    "docs/assets/benchmark-clouds/sphere-pressure/von_mises.png",
    "docs/assets/suboff-ice/fields.png",
    "docs/benchmarks/exports/beam_vibration_convergence.png",
}


#: Document image links that were already broken before the budget policy
#: (their renders live in sibling pull requests that have not merged yet).
#: Ratchet: fix or drop them, never add new ones.
KNOWN_MISSING_REFS = {
    ("docs/hetanyi-free-end-reference.md", "docs/assets/hetanyi_free_end/deflection_end_conditions.png"),
    ("docs/hetanyi-free-end-reference.md", "docs/assets/hetanyi_free_end/actions_free_end.png"),
    ("docs/hetanyi-free-end-reference.md", "docs/assets/hetanyi_free_end/fe_convergence.png"),
    ("docs/lame-cylinder-plane-strain.md", "assets/lame_cylinder_profiles.png"),
    ("docs/lame-cylinder-plane-strain.md", "assets/lame_cylinder_convergence.png"),
    ("docs/lame-cylinder-plane-strain.md", "assets/lame_cylinder_field.png"),
    ("docs/plate-vibration-validation.md", "assets/plate_vibration_convergence.png"),
    ("docs/plate-vibration-validation.md", "assets/plate_vibration_mode_shapes.png"),
    ("docs/plate-vibration-validation.md", "assets/plate_vibration_winkler.png"),
}


def test_compact_tier_exists_and_is_within_budget():
    rep = cf.tier_report(ROOT)
    assert rep["files"], "compact tier is empty; run scripts/compact_figures.py --apply"
    assert not rep["violations"], f"files over the per-file budget: {rep['violations']}"
    assert not rep["over_total"], (
        f"compact tier is {rep['total_bytes'] / 1024:.1f} KB, "
        f"budget {rep['total_budget_kb']} KB"
    )
    for row in rep["files"]:
        assert (ROOT / row["path"]).stat().st_size <= cf.MAX_KB * 1024


def test_no_new_oversized_figures():
    found = {p.as_posix() for _, p in cf.oversized(ROOT)}
    new = sorted(found - LEGACY_OVERSIZED)
    assert not new, (
        "committed renders above %d KB must go through scripts/compact_figures.py: %s"
        % (cf.MAX_KB, new)
    )


def test_document_image_links_resolve():
    missing = set(cf.missing_refs(ROOT))
    new = sorted(missing - KNOWN_MISSING_REFS)
    assert not new, f"documents reference missing images: {new}"
    fixed = sorted(KNOWN_MISSING_REFS - missing)
    assert not fixed, f"links no longer broken, drop them from KNOWN_MISSING_REFS: {fixed}"


def test_budget_manifest_matches_committed_tier():
    spec = json.loads((ROOT / "docs/assets/figure-budget.json").read_text())
    entries = spec["figures"]
    assert entries, "manifest lists no figures"
    total_kb = sum(int(e.get("budget_kb", spec.get("default_budget_kb", 10))) for e in entries)
    assert total_kb <= spec["total_budget_kb"], (
        f"manifest budgets sum to {total_kb} KB > {spec['total_budget_kb']} KB"
    )
    for e in entries:
        target = cf.compact_target(Path(e["src"]))
        assert (ROOT / target).is_file(), f"missing committed preview for {e['src']}"


def test_compact_target_layout():
    assert cf.compact_target(Path("docs/assets/thermal_2d/x.png")).as_posix() == \
        "docs/assets/compact/thermal_2d/x.png"
    # already-compacted paths are stable under a second pass
    once = cf.compact_target(Path("docs/assets/thermal_2d/x.png"))
    assert cf.compact_target(once) == once


def _sample_figure(path: Path) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    ax.plot(x, np.sin(8 * x) * np.exp(-2 * x), label="u")
    ax.plot(x, np.cos(5 * x), label="v")
    ax.set_xlabel("x")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def test_compaction_respects_budget_and_fidelity(tmp_path):
    import numpy as np
    from PIL import Image

    src = tmp_path / "render.png"
    _sample_figure(src)
    assert src.stat().st_size > 8 * 1024

    dst = tmp_path / "compact.png"
    rep = cf.compact_png(src, dst, budget_bytes=8 * 1024)
    assert not rep["over_budget"], rep
    assert dst.stat().st_size <= 8 * 1024
    assert rep["width"] < Image.open(src).size[0]

    a = Image.open(src).convert("RGB")
    b = Image.open(dst).convert("RGB")
    lo = np.asarray(a.resize(b.size, Image.LANCZOS), dtype=float)
    hi = np.asarray(b, dtype=float)
    rel = float(np.sqrt(((lo - hi) ** 2).mean())) / max(float(lo.max() - lo.min()), 1.0)
    assert rel < 0.10, f"relative RMSE {rel:.3f} too high for a preview"


def test_compaction_is_deterministic(tmp_path):
    src = tmp_path / "render.png"
    _sample_figure(src)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    cf.compact_png(src, a, budget_bytes=8 * 1024)
    cf.compact_png(src, b, budget_bytes=8 * 1024)
    assert a.read_bytes() == b.read_bytes()


def test_check_mode_passes_on_clean_tree():
    assert cf.main(["--check", "--root", str(ROOT)]) == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))