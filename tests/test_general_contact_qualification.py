import hashlib,json
from pathlib import Path
import pytest
from tensorfem.general_contact_qualification import qualify_general_contact3d

def hashed(tmp_path,name,body):
    body={**body,"evidence_sha256":hashlib.sha256(json.dumps(body,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()}
    path=tmp_path/name;path.write_text(json.dumps(body));return path

def evidence(tmp_path):
    common={"passed":True,"rollback_exact":True}
    a=hashed(tmp_path,"a.json",{**common,"schema":"tensorfem.curved-surface-contact3d-qualification/1.0",
        "mesh_sequence":[{"relative_error":.02}]})
    b=hashed(tmp_path,"b.json",{**common,"schema":"tensorfem.frictional-surface-mesh-qualification/1.0",
        "mesh_sequence":[{"normal_relative_error":.018,"coulomb_relative_error":.001}]})
    c=hashed(tmp_path,"c.json",{**common,"schema":"tensorfem.curved-finite-strain-friction-qualification/1.0",
        "mesh_sequence":[{"coulomb_relative_error":.002}]})
    d=hashed(tmp_path,"d.json",{"schema":"tensorfem.hertz-3d-qualification/1.0",
        "status":"qualified","errors":{"pressure":.025},"rollback_exact":True})
    return dict(curved_frictionless=a,curved_frictional=b,
        curved_finite_strain_friction=c,hertz_3d=d)

def test_composite_emits_exact_general_scope_only_after_all_gates(tmp_path):
    report=qualify_general_contact3d(**evidence(tmp_path))
    assert report["passed"] and report["scope"]=="general_surface_to_surface"
    assert report["maximum_relative_error"]==.025 and report["rollback_exact"]
    assert len(report["evidence_sha256"])==64
    assert "production-scale segmentation/search" in report["excluded_capabilities"]

def test_missing_and_tampered_sources_fail_closed(tmp_path):
    paths=evidence(tmp_path);Path(paths["curved_frictional"]).unlink()
    with pytest.raises(ValueError,match="missing or unreadable"):
        qualify_general_contact3d(**paths)
    paths=evidence(tmp_path);p=Path(paths["hertz_3d"])
    value=json.loads(p.read_text());value["errors"]["pressure"]=.001;p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        qualify_general_contact3d(**paths)

def test_above_three_percent_never_promotes_scope(tmp_path):
    paths=evidence(tmp_path);p=Path(paths["hertz_3d"]);v=json.loads(p.read_text())
    v.pop("evidence_sha256");v["errors"]["pressure"]=.031;paths["hertz_3d"]=hashed(tmp_path,"bad.json",v)
    report=qualify_general_contact3d(**paths)
    assert not report["passed"] and report["scope"]=="blocked"

