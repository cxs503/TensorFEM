from pathlib import Path

def test_negative_runner_is_fail_closed_and_cannot_impersonate_qualification():
    text=Path("scripts/run_hertz_3d_fe_failure_evidence.py").read_text()
    assert "tensorfem.hertz-3d-fe-negative-evidence/1.0" in text
    assert '"passed":False' in text
    assert '"qualification_status":"blocked"' in text
    assert "tensorfem.hertz-3d-qualification/1.0" not in text
    assert "evidence_sha256" in text and "NamedTemporaryFile" in text
