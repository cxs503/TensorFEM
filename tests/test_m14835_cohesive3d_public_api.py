"""Export contract for the mission m14835 deliverable (3-D cohesive interface element).

Mission m14835 shipped ``tensorfem.cohesive3d`` (BK mixed-mode cohesive law plus an
8-node zero-thickness interface element) but its only code change was the eight
lines that re-exported nine symbols from the package root.  That is a contract, not
an implementation: it survives only as long as

* every name in ``tensorfem.cohesive3d.__all__`` is reachable as ``tensorfem.<name>``,
* ``from tensorfem import *`` really binds those names to the *same* objects,
* the checked API snapshot (``api/public-api-v1.json`` + its packaged twin used by
  wheels) records each of them against ``tensorfem.cohesive3d``, so a later
  ``scripts/generate_api_manifest.py`` run cannot quietly drop the mission's
  public surface.

The last point is not hypothetical: on 2026-11 the snapshot had drifted five
symbols behind ``tensorfem.__all__`` and ``tests/test_public_api_contract.py``
went red; a mission re-export with no snapshot entry would fail the same way.
These checks are metadata-only (sub-second) and deliberately do not re-run the
element's numerics -- those live in ``tests/test_cohesive3d.py``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

import tensorfem

ROOT = Path(__file__).resolve().parents[1]

COHESIVE_MODULE = "tensorfem.cohesive3d"
EXPECTED_EXPORTS = {
    "BKCohesive3DLaw",
    "Cohesive3DResponse",
    "cohesive_hex8_quadrature",
    "cohesive_shape_functions",
    "mid_surface_frame",
    "interface_area",
    "cohesive_separations",
    "cohesive_hex8_response",
    "assemble_cohesive_interface",
}
ALLOWED_STABILITY = {"stable", "beta", "experimental", "internal_candidate"}


def _module():
    return importlib.import_module(COHESIVE_MODULE)


def test_cohesive3d_module_exports_expected_nine_symbols():
    module = _module()
    assert set(module.__all__) == EXPECTED_EXPORTS


def test_root_package_reexports_every_cohesive3d_symbol():
    module = _module()
    missing = [name for name in module.__all__ if name not in tensorfem.__all__]
    assert missing == [], f"cohesive3d symbols dropped from tensorfem.__all__: {missing}"
    for name in module.__all__:
        exported = getattr(tensorfem, name)
        original = getattr(module, name)
        assert exported is original, f"tensorfem.{name} is not {COHESIVE_MODULE}.{name}"
        assert getattr(exported, "__module__", None) == COHESIVE_MODULE


def test_star_import_binds_cohesive3d_names():
    namespace: dict[str, object] = {}
    exec("from tensorfem import *", namespace)  # noqa: S102 -- contract under test
    module = _module()
    for name in EXPECTED_EXPORTS:
        assert name in namespace, f"star-import does not bind {name}"
        assert namespace[name] is getattr(module, name)


def _snapshot_entries() -> dict[str, dict]:
    entries = {}
    for relative in ("api/public-api-v1.json", "src/tensorfem/public_api_manifest.json"):
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for entry in payload["entries"]:
            previous = entries.setdefault(entry["name"], entry)
            assert previous == entry, f"snapshot copies disagree on {entry['name']}"
    return entries


def test_checked_api_snapshot_records_each_cohesive3d_symbol():
    entries = _snapshot_entries()
    module = _module()
    for name in sorted(EXPECTED_EXPORTS):
        assert name in entries, f"{name} missing from the checked API snapshot"
        entry = entries[name]
        assert entry["module"] == COHESIVE_MODULE
        assert entry["object"] == getattr(getattr(module, name), "__name__", name)
        assert entry["stability"] in ALLOWED_STABILITY


@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
def test_symbol_is_callable_or_dataclass_backed(name):
    obj = getattr(_module(), name)
    if name in {"BKCohesive3DLaw", "Cohesive3DResponse"}:
        assert hasattr(obj, "__init__")
        assert obj.__doc__, f"{name} lost its docstring"
    else:
        assert callable(obj), f"{name} is not callable"
        assert obj.__doc__, f"{name} lost its docstring"


def test_bk_law_default_exponent_is_reachable_from_root():
    """The documented default (BK eta = 1.6) must survive the re-export intact."""
    import inspect

    signature = inspect.signature(tensorfem.BKCohesive3DLaw.__init__)
    assert signature.parameters["bk_eta"].default == 1.6

EVIDENCE = ROOT / "artifacts" / "m14835_public_api_evidence.json"


def test_evidence_artifact_is_fresh_and_untampered():
    """The committed evidence must describe *this* checkout (hash-gated)."""
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for relative, digest in payload["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, f"{relative} changed after the evidence was generated"
    body = {key: value for key, value in payload.items() if key != "checks_digest"}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert payload["checks_digest"] == expected, "evidence payload was edited by hand"
    assert payload["counts"]["root_exports_missing_from_snapshot"] == []
