"""Certificate figures must be re-renderable from committed data, not re-solved.

``fem_inverse_identification``, ``fem_sent_j_integral`` and
``fem_winkler_plate_k_sweep`` named figures that were not in the delivered tree
(``E_ARTIFACT_MISSING`` in ``tools/verified_anchor_audit.py``, and two of them
were new rows in the shrink-only offender set of
``tests/test_anchor_figure_gate_safety.py``).

The repair is ``tools/anchor_figure_rebuild.py``: every plotted number comes
from the anchor's committed ``result.json``, the PNG is written under
``docs/assets/verified/<anchor>/`` (the only path class the push gate carries),
the certificate names that path, and ``figures_manifest.json`` pins the sha256
of the render.

These tests enforce the part that a green audit alone cannot see: that the
figures *are* the data, i.e. re-rendering reproduces the committed bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED = REPO_ROOT / "benchmarks" / "verified"
FIGURES = REPO_ROOT / "docs" / "assets" / "verified"
REBUILD = REPO_ROOT / "tools" / "anchor_figure_rebuild.py"
AUDIT = REPO_ROOT / "tools" / "verified_anchor_audit.py"

ANCHORS = ("fem_inverse_identification", "fem_sent_j_integral", "fem_winkler_plate_k_sweep")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rebuild():
    return _load(REBUILD, "anchor_figure_rebuild")


@pytest.fixture(scope="module")
def manifest():
    out = {}
    for anchor in ANCHORS:
        path = FIGURES / anchor / "figures_manifest.json"
        assert path.is_file(), f"{anchor}: figures_manifest.json missing"
        out[anchor] = json.loads(path.read_text())
    return out


@pytest.mark.parametrize("anchor", ANCHORS)
def test_certificate_names_figures_that_exist_under_docs(anchor):
    payload = json.loads((VERIFIED / anchor / "gates_verified.json").read_text())
    figures = [
        value for value in (payload.get("artifacts") or {}).values()
        if isinstance(value, str) and value.endswith(".png")
    ]
    assert figures, f"{anchor}: certificate no longer names any figure"
    for name in figures:
        assert name.startswith("docs/"), (
            f"{anchor}: {name} is outside docs/, the push gate strips it"
        )
        assert (REPO_ROOT / name).is_file(), f"{anchor}: {name} does not resolve"


@pytest.mark.parametrize("anchor", ANCHORS)
def test_committed_png_matches_its_manifest_hash(anchor, manifest):
    entries = manifest[anchor]["figures"]
    assert entries, f"{anchor}: manifest lists no figures"
    for name, record in entries.items():
        blob = (FIGURES / anchor / name).read_bytes()
        assert len(blob) == record["bytes"]
        assert hashlib.sha256(blob).hexdigest() == record["sha256"], (
            f"{anchor}/{name}: committed bytes differ from the manifest hash"
        )


@pytest.mark.parametrize("anchor", ANCHORS)
def test_render_is_deterministic_and_data_driven(anchor, rebuild, manifest, tmp_path):
    """A fresh render must reproduce the committed bytes exactly."""
    fresh = rebuild.render(anchor, tmp_path / anchor)
    committed = manifest[anchor]["figures"]
    assert set(fresh) == set(committed), (
        f"{anchor}: renderer emits {sorted(fresh)}, manifest has {sorted(committed)}"
    )
    for name, record in fresh.items():
        assert record["sha256"] == committed[name]["sha256"], (
            f"{anchor}/{name}: render is not reproducible from result.json"
        )


@pytest.mark.parametrize("anchor", ("fem_sent_j_integral", "fem_winkler_plate_k_sweep"))
def test_manifest_declares_the_committed_source(anchor, manifest):
    entry = manifest[anchor]
    assert entry["source"] == f"benchmarks/verified/{anchor}/result.json"
    assert entry["renderer"] == "tools/anchor_figure_rebuild.py"


def test_repaired_anchors_carry_no_artifact_defect():
    tool = _load(AUDIT, "verified_anchor_audit")
    tracked = tool.tracked_files()
    for anchor in ANCHORS:
        record = tool.audit_anchor(VERIFIED / anchor, tracked)
        hard = list(record.hard_defects)
        assert not any(code.startswith("E_ARTIFACT") for code in hard), (
            f"{anchor}: artifact defect survived the repair: {hard}"
        )