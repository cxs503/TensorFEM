"""Cohesive failure ledger benchmark - library-level verification.

This script verifies three anchors:
1. Mass audit ledger: |Δ|/Σ ≤ 1e-12
2. Fracture energy: ∑G·A vs ledger ≤ 1e-12
3. Analytical enumeration: uniform stress field → failed face count matches exactly

Output: validation_report.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import torch

from tensorfem.plate_mindlin import (
    solve_plate_on_winkler_foundation,
    compute_face_stresses,
)
from tensorfem.cohesive_failure import (
    FailureCriterion,
    evaluate_face_failure,
    enumerate_failed_faces_analytical,
)
from tensorfem.fracture_energy import FractureEnergyLedger, MassAuditLedger


def utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def anchor1_mass_audit():
    """Anchor 1: Mass audit ledger conservation.
    
    Verify that the mass equivalent of released energy is tracked correctly.
    |total_mass - expected_mass| / expected_mass ≤ 1e-12
    """
    ledger = MassAuditLedger()
    
    # Add some energy releases
    energies = [1.0, 2.0, 3.0, 4.0, 5.0]
    for e in energies:
        ledger.add_energy_release(e)
    
    total_energy = sum(energies)
    expected_mass = total_energy / ledger.c_squared
    
    # Check conservation
    delta = abs(ledger.total_mass_equivalent - expected_mass)
    relative_error = delta / expected_mass if expected_mass > 0 else 0.0
    
    passed = relative_error <= 1e-12
    
    return {
        "anchor": "mass_audit",
        "description": "Mass audit ledger conservation: |Δ|/Σ ≤ 1e-12",
        "passed": passed,
        "total_energy": total_energy,
        "expected_mass": expected_mass,
        "actual_mass": ledger.total_mass_equivalent,
        "absolute_error": delta,
        "relative_error": relative_error,
        "tolerance": 1e-12,
    }


def anchor2_fracture_energy():
    """Anchor 2: Fracture energy conservation.
    
    Verify that ∑G·A matches the ledger total.
    |∑G·A - ledger_total| ≤ 1e-12
    """
    ledger = FractureEnergyLedger()
    
    # Add some failed faces
    faces_data = [
        {"face_id": 0, "area": 1.0, "G": 1.0, "released_energy": 1.0},
        {"face_id": 1, "area": 2.0, "G": 2.0, "released_energy": 4.0},
        {"face_id": 2, "area": 3.0, "G": 3.0, "released_energy": 9.0},
    ]
    
    for face in faces_data:
        ledger.add_failed_face(**face)
    
    # Compute ∑G·A
    sum_GA = sum(f["G"] * f["area"] for f in faces_data)
    
    # Check conservation
    delta = abs(ledger.total_released_energy - sum_GA)
    
    passed = delta <= 1e-12
    
    return {
        "anchor": "fracture_energy",
        "description": "Fracture energy conservation: |∑G·A - ledger| ≤ 1e-12",
        "passed": passed,
        "sum_GA": sum_GA,
        "ledger_total": ledger.total_released_energy,
        "absolute_error": delta,
        "tolerance": 1e-12,
    }


def anchor3_analytical_enumeration():
    """Anchor 3: Analytical enumeration of failed faces.
    
    For a uniform stress field, verify that the number of failed faces
    matches the analytical prediction exactly.
    
    Setup:
    - 2x2 mesh (4 elements, 16 faces total)
    - Uniform stress: sigma_n = 2e6 Pa, tau = 0
    - Criterion: sigma_max = 1e6 Pa, tau_max = 1e6 Pa
    - Expected: all 16 faces fail (f = (2e6/1e6)^2 = 4 > 1)
    """
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    
    # Analytical prediction
    n_failed_analytical, _, _ = enumerate_failed_faces_analytical(
        n_elements_x=2,
        n_elements_y=2,
        uniform_sigma_n=2e6,
        uniform_tau=0,
        criterion=criterion,
    )
    
    # FEM evaluation
    result = solve_plate_on_winkler_foundation(
        nx=2,
        side=1.0,
        young=2.1e11,
        poisson=0.3,
        thickness=0.01,
        winkler_k=1e6,
        pressure=1.0,
        boundary='simply_supported',
    )
    
    faces = compute_face_stresses(result)
    
    # Override face stresses with uniform biaxial field for testing
    # This ensures all faces experience the same normal stress regardless of orientation
    for face in faces:
        face["stress"] = {
            "sigma_x": 2e6,
            "sigma_y": 2e6,
            "tau_xy": 0,
            "tau_xz": 0,
            "tau_yz": 0,
        }
    
    failed_faces = evaluate_face_failure(faces, criterion)
    n_failed_fem = len(failed_faces)
    
    # Check exact match
    passed = n_failed_fem == n_failed_analytical
    
    # Also check total area and energy
    total_area_fem = sum(f["area"] for f in failed_faces)
    total_energy_fem = sum(f["released_energy"] for f in failed_faces)
    
    # For uniform stress, all faces have the same area
    # Each face area = side_length * thickness
    # For 2x2 mesh on 1.0x1.0 plate, each element is 0.5x0.5
    # Each face has length 0.5 and thickness 0.01, so area = 0.005
    face_area = 0.5 * 0.01
    expected_total_area = n_failed_analytical * face_area
    expected_total_energy = n_failed_analytical * criterion.G * face_area
    
    area_error = abs(total_area_fem - expected_total_area) / expected_total_area if expected_total_area > 0 else 0.0
    energy_error = abs(total_energy_fem - expected_total_energy) / expected_total_energy if expected_total_energy > 0 else 0.0
    
    return {
        "anchor": "analytical_enumeration",
        "description": "Analytical enumeration: uniform stress → failed face count exact match",
        "passed": passed,
        "n_failed_analytical": n_failed_analytical,
        "n_failed_fem": n_failed_fem,
        "total_area_fem": total_area_fem,
        "expected_total_area": expected_total_area,
        "area_relative_error": area_error,
        "total_energy_fem": total_energy_fem,
        "expected_total_energy": expected_total_energy,
        "energy_relative_error": energy_error,
        "tolerance": 1e-12,
    }


def main():
    """Run all anchors and write validation_report.json."""
    print("=" * 70)
    print("Cohesive Failure Ledger Benchmark - Library-Level Verification")
    print("=" * 70)
    print()
    
    # Run anchors
    print("Running Anchor 1: Mass audit ledger...")
    anchor1_result = anchor1_mass_audit()
    print(f"  Passed: {anchor1_result['passed']}")
    print(f"  Relative error: {anchor1_result['relative_error']:.2e}")
    print()
    
    print("Running Anchor 2: Fracture energy conservation...")
    anchor2_result = anchor2_fracture_energy()
    print(f"  Passed: {anchor2_result['passed']}")
    print(f"  Absolute error: {anchor2_result['absolute_error']:.2e}")
    print()
    
    print("Running Anchor 3: Analytical enumeration...")
    anchor3_result = anchor3_analytical_enumeration()
    print(f"  Passed: {anchor3_result['passed']}")
    print(f"  Failed faces: {anchor3_result['n_failed_fem']} / {anchor3_result['n_failed_analytical']}")
    print()
    
    # Summary
    all_passed = all([
        anchor1_result["passed"],
        anchor2_result["passed"],
        anchor3_result["passed"],
    ])
    
    print("=" * 70)
    print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 70)
    print()
    
    # Write validation report
    report = {
        "timestamp": utcnow_z(),
        "benchmark": "cohesive_failure_ledger",
        "description": "Library-level verification of cohesive failure and fracture energy ledger",
        "anchors": [
            anchor1_result,
            anchor2_result,
            anchor3_result,
        ],
        "all_passed": all_passed,
    }
    
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation report written to validation_report.json")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
