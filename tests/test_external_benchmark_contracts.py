import json
from dataclasses import replace
import math

import pytest

from tensorfem.external_benchmark_contracts import (
    external_p0_contracts,
    hertz_sphere_halfspace_contract,
    large_rotation_pure_bending_contract,
    shallow_spherical_shell_ring_load_contract,
    write_external_contracts,
)


def test_external_contracts_are_traceable_but_fail_closed():
    contracts = external_p0_contracts()
    assert {case.category for case in contracts} == {"nonlinear_shell", "contact_3d"}
    assert len({case.source.doi for case in contracts}) == 3
    blocked = [case for case in contracts if case.status == "blocked"]
    assert len(blocked) == 2
    for case in blocked:
        assert case.status == "blocked"
        assert case.missing_evidence
        assert case.source.url == "https://doi.org/" + case.source.doi
        with pytest.raises(RuntimeError, match="is blocked"):
            case.require_qualified()


def test_complete_large_rotation_shell_contract_is_qualified():
    case = large_rotation_pure_bending_contract().require_qualified()
    assert case.status == "qualified"
    assert case.missing_evidence == ()
    assert case.parameters["mesh_elements"] == (1, 2, 4)
    assert case.reference_quantities["tip_z"]["value"] == pytest.approx(20.0 / math.pi)
    assert "tests/test_large_rotation_shell_benchmark.py" in case.evidence_files


def test_shell_candidate_does_not_invent_curve_values():
    case = shallow_spherical_shell_ring_load_contract()
    assert case.reference_quantities == {}
    assert "tabulated primary-source" in case.missing_evidence[0]
    assert case.parameters["sphere_radius"] == {"value": 254.0, "unit": "mm"}


def test_hertz_oracle_does_not_promote_a_formula_to_fe_qualification():
    case = hertz_sphere_halfspace_contract()
    values = {key: item["value"] for key, item in case.parameters.items()}
    effective = 1.0 / (
        (1.0 - values["sphere_poisson_ratio"] ** 2)
        / values["sphere_young_modulus"]
        + (1.0 - values["halfspace_poisson_ratio"] ** 2)
        / values["halfspace_young_modulus"]
    )
    radius = (
        3.0
        * values["normal_force"]
        * values["sphere_radius"]
        / (4.0 * effective)
    ) ** (1.0 / 3.0)
    pressure = 3.0 * values["normal_force"] / (2.0 * math.pi * radius**2)
    assert case.reference_quantities["effective_modulus"]["value"] == pytest.approx(effective)
    assert case.reference_quantities["contact_radius"]["value"] == pytest.approx(radius)
    assert case.reference_quantities["indentation"]["value"] == pytest.approx(
        radius**2 / values["sphere_radius"]
    )
    assert case.reference_quantities["maximum_pressure"]["value"] == pytest.approx(pressure)
    assert case.reference_quantities["maximum_pressure"]["unit"] == "Pa"
    assert case.evidence_files == ()
    assert "global deformable-to-deformable" in case.missing_evidence[0]


def test_contract_validation_rejects_false_promotion_and_incomplete_metadata():
    blocked = hertz_sphere_halfspace_contract()
    with pytest.raises(ValueError, match="qualified contract cannot"):
        replace(blocked, status="qualified").validate()
    with pytest.raises(ValueError, match="DOI"):
        replace(blocked, source=replace(blocked.source, doi="")).validate()
    with pytest.raises(ValueError, match="tolerance"):
        replace(blocked, tolerance=0.031).validate()
    with pytest.raises(ValueError, match="blocked contract"):
        replace(blocked, missing_evidence=()).validate()


def test_machine_readable_archive_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_external_contracts(first)
    write_external_contracts(second)
    assert first.read_bytes() == second.read_bytes()
    saved = json.loads(first.read_text(encoding="utf-8"))
    assert saved["schema"] == "tensorfem.external-benchmark-contract/1.0"
    assert [row["status"] for row in saved["contracts"]] == [
        "qualified",
        "blocked",
        "blocked",
    ]
