import json
import pytest
from tensorfem.snapthrough_shell_qualification import karatas_yuksel_ring_load_audit


def test_published_shell_candidate_is_fail_closed_not_false_qualified(tmp_path):
    audit=karatas_yuksel_ring_load_audit()
    assert not audit.qualified and len(audit.missing)==3
    with pytest.raises(RuntimeError,match="not qualified"):audit.require_qualified()
    path=tmp_path/"shell-snapthrough-audit.json";audit.write_json(path)
    saved=json.loads(path.read_text())
    assert saved["geometry"]["radius_mm"]==254.
    assert saved["qualified"] is False


def test_audit_never_contains_digitized_or_invented_limit_values():
    audit=karatas_yuksel_ring_load_audit();keys=set(audit.geometry)
    assert "critical_load" not in keys and "critical_displacement" not in keys
