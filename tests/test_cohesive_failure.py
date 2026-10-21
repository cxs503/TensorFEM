"""Tests for cohesive failure criterion."""
import pytest

from tensorfem.cohesive_failure import (
    FailureCriterion,
    evaluate_face_failure,
    enumerate_failed_faces_analytical,
)


def test_failure_criterion_construction():
    """Failure criterion should be constructable with positive parameters."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    assert criterion.sigma_max == 1e6
    assert criterion.tau_max == 1e6
    assert criterion.G == 1.0


def test_failure_criterion_invalid_parameters():
    """Failure criterion should reject non-positive parameters."""
    with pytest.raises(ValueError):
        FailureCriterion(sigma_max=0, tau_max=1e6, G=1.0)
    with pytest.raises(ValueError):
        FailureCriterion(sigma_max=1e6, tau_max=-1, G=1.0)
    with pytest.raises(ValueError):
        FailureCriterion(sigma_max=1e6, tau_max=1e6, G=0)


def test_failure_index_pure_tension():
    """Pure tension should give correct failure index."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    # At sigma_max, f should be 1
    assert abs(criterion.failure_index(1e6, 0) - 1.0) < 1e-12
    # Below sigma_max, f < 1
    assert criterion.failure_index(0.5e6, 0) < 1.0
    # Above sigma_max, f > 1
    assert criterion.failure_index(2e6, 0) > 1.0


def test_failure_index_pure_shear():
    """Pure shear should give correct failure index."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    # At tau_max, f should be 1
    assert abs(criterion.failure_index(0, 1e6) - 1.0) < 1e-12
    # Below tau_max, f < 1
    assert criterion.failure_index(0, 0.5e6) < 1.0
    # Above tau_max, f > 1
    assert criterion.failure_index(0, 2e6) > 1.0


def test_failure_index_combined():
    """Combined tension and shear should give correct failure index."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    # Equal tension and shear: f = 0.5^2 + 0.5^2 = 0.5
    f = criterion.failure_index(0.5e6, 0.5e6)
    assert abs(f - 0.5) < 1e-12


def test_evaluate_face_failure_none_failed():
    """No faces should fail when stress is below threshold."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    faces = [
        {
            "id": 0,
            "area": 1.0,
            "normal": [1.0, 0.0, 0.0],
            "centroid": [0.0, 0.0, 0.0],
            "stress": {"sigma_x": 0.5e6, "sigma_y": 0, "tau_xy": 0, "tau_xz": 0, "tau_yz": 0},
        }
    ]
    failed = evaluate_face_failure(faces, criterion)
    assert len(failed) == 0


def test_evaluate_face_failure_all_failed():
    """All faces should fail when stress is above threshold."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    faces = [
        {
            "id": 0,
            "area": 1.0,
            "normal": [1.0, 0.0, 0.0],
            "centroid": [0.0, 0.0, 0.0],
            "stress": {"sigma_x": 2e6, "sigma_y": 0, "tau_xy": 0, "tau_xz": 0, "tau_yz": 0},
        },
        {
            "id": 1,
            "area": 2.0,
            "normal": [0.0, 1.0, 0.0],
            "centroid": [1.0, 0.0, 0.0],
            "stress": {"sigma_x": 0, "sigma_y": 2e6, "tau_xy": 0, "tau_xz": 0, "tau_yz": 0},
        }
    ]
    failed = evaluate_face_failure(faces, criterion)
    assert len(failed) == 2
    # Check released energy
    assert abs(failed[0]["released_energy"] - 1.0 * 1.0) < 1e-12
    assert abs(failed[1]["released_energy"] - 1.0 * 2.0) < 1e-12


def test_evaluate_face_failure_mixed():
    """Some faces should fail when stress is mixed."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    faces = [
        {
            "id": 0,
            "area": 1.0,
            "normal": [1.0, 0.0, 0.0],
            "centroid": [0.0, 0.0, 0.0],
            "stress": {"sigma_x": 2e6, "sigma_y": 0, "tau_xy": 0, "tau_xz": 0, "tau_yz": 0},
        },
        {
            "id": 1,
            "area": 2.0,
            "normal": [0.0, 1.0, 0.0],
            "centroid": [1.0, 0.0, 0.0],
            "stress": {"sigma_x": 0, "sigma_y": 0.5e6, "tau_xy": 0, "tau_xz": 0, "tau_yz": 0},
        }
    ]
    failed = evaluate_face_failure(faces, criterion)
    assert len(failed) == 1
    assert failed[0]["id"] == 0


def test_enumerate_failed_faces_analytical_all_fail():
    """All faces should fail under high uniform stress."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    n_failed, _, _ = enumerate_failed_faces_analytical(
        n_elements_x=2,
        n_elements_y=2,
        uniform_sigma_n=2e6,
        uniform_tau=0,
        criterion=criterion,
    )
    # 2x2 elements, 4 faces each = 16 faces total
    assert n_failed == 16


def test_enumerate_failed_faces_analytical_none_fail():
    """No faces should fail under low uniform stress."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    n_failed, total_area, total_energy = enumerate_failed_faces_analytical(
        n_elements_x=2,
        n_elements_y=2,
        uniform_sigma_n=0.5e6,
        uniform_tau=0,
        criterion=criterion,
    )
    assert n_failed == 0
    assert total_area == 0.0
    assert total_energy == 0.0


def test_enumerate_failed_faces_analytical_combined():
    """Combined tension and shear should trigger failure."""
    criterion = FailureCriterion(sigma_max=1e6, tau_max=1e6, G=1.0)
    # sigma_n = 0.7e6, tau = 0.7e6
    # f = 0.7^2 + 0.7^2 = 0.98 < 1, so no failure
    n_failed, _, _ = enumerate_failed_faces_analytical(
        n_elements_x=2,
        n_elements_y=2,
        uniform_sigma_n=0.7e6,
        uniform_tau=0.7e6,
        criterion=criterion,
    )
    assert n_failed == 0
    
    # sigma_n = 0.8e6, tau = 0.8e6
    # f = 0.8^2 + 0.8^2 = 1.28 > 1, so all fail
    n_failed, _, _ = enumerate_failed_faces_analytical(
        n_elements_x=2,
        n_elements_y=2,
        uniform_sigma_n=0.8e6,
        uniform_tau=0.8e6,
        criterion=criterion,
    )
    assert n_failed == 16
