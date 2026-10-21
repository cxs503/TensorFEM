"""Tests for fracture energy ledger."""
import pytest
import json
import tempfile
from pathlib import Path

from tensorfem.fracture_energy import FractureEnergyLedger, MassAuditLedger


def test_ledger_construction():
    """Ledger should be constructable with default values."""
    ledger = FractureEnergyLedger()
    assert ledger.total_released_energy == 0.0
    assert ledger.n_failed_faces == 0
    assert len(ledger.failed_faces) == 0


def test_ledger_add_single_face():
    """Adding a single face should update totals correctly."""
    ledger = FractureEnergyLedger()
    ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=1.0)
    assert ledger.total_released_energy == 1.0
    assert ledger.n_failed_faces == 1
    assert len(ledger.failed_faces) == 1


def test_ledger_add_multiple_faces():
    """Adding multiple faces should accumulate correctly."""
    ledger = FractureEnergyLedger()
    ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=1.0)
    ledger.add_failed_face(face_id=1, area=2.0, G=2.0, released_energy=4.0)
    assert ledger.total_released_energy == 5.0
    assert ledger.n_failed_faces == 2


def test_ledger_negative_energy_rejected():
    """Negative released energy should be rejected."""
    ledger = FractureEnergyLedger()
    with pytest.raises(ValueError):
        ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=-1.0)


def test_ledger_verify_energy_conservation():
    """Energy conservation check should pass when totals match."""
    ledger = FractureEnergyLedger()
    ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=1.0)
    ledger.add_failed_face(face_id=1, area=2.0, G=2.0, released_energy=4.0)
    assert ledger.verify_energy_conservation(5.0)
    assert not ledger.verify_energy_conservation(6.0)


def test_ledger_verify_energy_sum():
    """Energy sum check should pass when total equals sum of individuals."""
    ledger = FractureEnergyLedger()
    ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=1.0)
    ledger.add_failed_face(face_id=1, area=2.0, G=2.0, released_energy=4.0)
    assert ledger.verify_energy_sum()


def test_ledger_to_dict():
    """Ledger should serialize to dictionary correctly."""
    ledger = FractureEnergyLedger()
    ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=1.0)
    data = ledger.to_dict()
    assert data["total_released_energy"] == 1.0
    assert data["n_failed_faces"] == 1
    assert len(data["faces"]) == 1
    assert data["units"] == "SI"


def test_ledger_from_dict():
    """Ledger should deserialize from dictionary correctly."""
    data = {
        "total_released_energy": 5.0,
        "n_failed_faces": 2,
        "faces": [
            {"id": 0, "area": 1.0, "G": 1.0, "released_energy": 1.0},
            {"id": 1, "area": 2.0, "G": 2.0, "released_energy": 4.0},
        ],
        "units": "SI",
    }
    ledger = FractureEnergyLedger.from_dict(data)
    assert ledger.total_released_energy == 5.0
    assert ledger.n_failed_faces == 2


def test_ledger_json_roundtrip():
    """Ledger should survive JSON serialization roundtrip."""
    ledger = FractureEnergyLedger()
    ledger.add_failed_face(face_id=0, area=1.0, G=1.0, released_energy=1.0)
    ledger.add_failed_face(face_id=1, area=2.0, G=2.0, released_energy=4.0)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    
    try:
        ledger.to_json(path)
        ledger2 = FractureEnergyLedger.from_json(path)
        assert ledger2.total_released_energy == ledger.total_released_energy
        assert ledger2.n_failed_faces == ledger.n_failed_faces
    finally:
        Path(path).unlink()


def test_mass_audit_ledger():
    """Mass audit ledger should track energy-mass equivalence."""
    ledger = MassAuditLedger()
    ledger.add_energy_release(1.0)  # 1 J
    # E = m * c^2 => m = E / c^2
    expected_mass = 1.0 / ledger.c_squared
    assert abs(ledger.total_mass_equivalent - expected_mass) < 1e-30


def test_mass_audit_verify():
    """Mass audit verification should work."""
    ledger = MassAuditLedger()
    ledger.add_energy_release(1.0)
    expected_mass = 1.0 / ledger.c_squared
    assert ledger.verify_mass_conservation(expected_mass)
    # Use a large relative tolerance since mass values are tiny (~1e-17 kg)
    assert not ledger.verify_mass_conservation(expected_mass * 2, tol=1e-30)
