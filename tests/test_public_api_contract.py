import json
from pathlib import Path

import pytest
import tensorfem

from tensorfem.deprecation import TensorFEMDeprecationWarning, deprecated, warn_deprecated

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_snapshot_is_exact_and_packaged_copy_matches():
    source = json.loads((ROOT / "api/public-api-v1.json").read_text())
    packaged = json.loads((ROOT / "src/tensorfem/public_api_manifest.json").read_text())
    assert source == packaged
    assert [entry["name"] for entry in source["entries"]] == list(tensorfem.__all__)
    assert len(source["entries"]) == len({entry["name"] for entry in source["entries"]})
    assert {entry["stability"] for entry in source["entries"]} <= {
        "stable", "beta", "experimental", "internal_candidate"
    }
    assert any(entry["stability"] == "internal_candidate" for entry in source["entries"])


def test_snapshot_objects_remain_importable_from_recorded_modules():
    payload = json.loads((ROOT / "api/public-api-v1.json").read_text())
    for entry in payload["entries"]:
        value = getattr(tensorfem, entry["name"])
        assert getattr(value, "__module__", "tensorfem") == entry["module"]


def test_deprecation_warning_and_metadata():
    @deprecated(since="0.19", remove="1.1", replacement="new_api")
    def legacy(value):
        return value + 1

    with pytest.warns(TensorFEMDeprecationWarning, match="new_api"):
        assert legacy(2) == 3
    assert legacy.__tensorfem_deprecation__["remove"] == "1.1"
    with pytest.raises(ValueError):
        warn_deprecated("", since="0.19", remove="1.1")


def test_generator_preserves_reviewed_metadata_on_regeneration(tmp_path):
    """Regeneration must keep curated introduced/stability (history is squashed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_api_manifest_under_test", ROOT / "scripts" / "generate_api_manifest.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    payload = json.loads((ROOT / "api/public-api-v1.json").read_text())
    reviewed = json.loads(json.dumps(payload))
    sentinel_name = reviewed["entries"][0]["name"]
    for entry in reviewed["entries"]:
        if entry["name"] == sentinel_name:
            entry["introduced"] = "0.0.1"
            entry["stability"] = "beta"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(reviewed), encoding="utf-8")

    out = gen.build(manifest)
    by_name = {entry["name"]: entry for entry in out["entries"]}
    assert by_name[sentinel_name]["introduced"] == "0.0.1"
    assert by_name[sentinel_name]["stability"] == "beta"
    # every other reviewed entry keeps its reviewed values too
    for entry in reviewed["entries"]:
        assert by_name[entry["name"]]["introduced"] == entry["introduced"]
        assert by_name[entry["name"]]["stability"] == entry["stability"]
    # names missing from the reviewed manifest fall back to derived values
    assert by_name[sentinel_name]["module"] == payload["entries"][0]["module"]
