"""Tests for the KO ontology v1 compiler/validator (FEM method axis pilot).

The suite is deliberately adversarial: besides "does the real index validate", it
pins that the validator *rejects* the failure modes the ontology exists to catch
(undeclared relation, dangling target, wrong form pair, fan-out on a one_to_one
verb, silent grade inflation), and that the compiler is deterministic and merges
multi-document cases into one object instead of duplicating anchors.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "tk_ontology" / "tk_ontology.py"
spec = importlib.util.spec_from_file_location("tk_ontology", MODULE_PATH)
onto = importlib.util.module_from_spec(spec)
sys.modules["tk_ontology"] = onto
spec.loader.exec_module(onto)


@pytest.fixture(scope="module")
def index():
    return onto.build_index(ROOT)


@pytest.fixture(scope="module")
def vocab():
    return onto.relation_vocabulary(ROOT)


def test_schema_document_is_a_valid_json_schema():
    import jsonschema

    schema = onto.schema(ROOT)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["properties"]["form"]["enum"] == list(onto.FORMS)


def test_relation_vocabulary_is_closed(vocab):
    verbs = [row["verb"] for row in vocab["verbs"]]
    inverses = [row["inverse"] for row in vocab["verbs"]]
    assert len(set(verbs)) == len(verbs), "duplicate forward verb"
    assert len(set(inverses)) == len(inverses), "duplicate inverse verb"
    assert set(verbs).isdisjoint(set(inverses)), "a word is used as both forward and inverse verb"
    for row in vocab["verbs"]:
        assert row["from"] in onto.FORMS and row["to"] in onto.FORMS
        assert row["cardinality"] in ("one_to_one", "one_to_many", "many_to_many")
        assert row["meaning"].strip()


def test_repository_index_validates(index):
    problems = onto.validate_index(index, ROOT)
    assert problems == [], "\n".join(problems[:20])


def test_every_form_is_populated(index):
    counts = {f: 0 for f in onto.FORMS}
    for ko in index["kos"]:
        counts[ko["form"]] += 1
    for form, n in counts.items():
        assert n > 0, f"no {form} object in the index: the six-form ontology is incomplete in practice"


def test_compiler_is_deterministic(index):
    again = onto.build_index(ROOT, commit=index["commit"])
    assert json.dumps(again, sort_keys=True) == json.dumps(index, sort_keys=True)


def test_cases_are_merged_not_duplicated(index):
    ids = [k["id"] for k in index["kos"]]
    assert len(ids) == len(set(ids))
    merged = [k for k in index["kos"]
              if k["form"] == "CASE" and k["provenance"].get("supporting_paths")]
    assert merged, "expected at least one case described by several documents"
    for ko in merged:
        for path in ko["provenance"]["supporting_paths"]:
            assert (ROOT / path).is_file()


def test_case_coverage_numbers_are_consistent(index):
    m = onto.metrics(index)
    cov = m["case_coverage"]
    cases = [k for k in index["kos"] if k["form"] == "CASE"]
    assert cov["cases"] == len(cases)
    assert 0 <= cov["with_real_evidence"] <= cov["cases"]
    real = 0
    by_id = {k["id"]: k for k in index["kos"]}
    for case in cases:
        if any(r["verb"] == "has_evidence" and by_id[r["target"]]["fields"]["mechanical_grade"] == "REAL"
               for r in case["relations"]):
            real += 1
    assert real == cov["with_real_evidence"]


# --------------------------------------------------------------------------
# negative tests: the validator must reject what the ontology forbids
# --------------------------------------------------------------------------
def _mutated(index, ko_id):
    idx = copy.deepcopy(index)
    return idx, next(k for k in idx["kos"] if k["id"] == ko_id)


def test_validator_rejects_undeclared_relation_verb(index):
    idx, case = _mutated(index, next(k["id"] for k in index["kos"] if k["form"] == "CASE"))
    case["relations"].append({"verb": "has_vibes", "target": case["id"]})
    problems = onto.validate_index(idx, ROOT)
    assert any("undeclared relation verb" in p for p in problems)


def test_validator_rejects_dangling_target(index):
    idx, case = _mutated(index, next(k["id"] for k in index["kos"] if k["form"] == "CASE"))
    case["relations"].append({"verb": "has_criterion", "target": "CRITERION-DOES-NOT-EXIST"})
    problems = onto.validate_index(idx, ROOT)
    assert any("dangling relation target" in p for p in problems)


def test_validator_rejects_wrong_object_form(index):
    idx, case = _mutated(index, next(k["id"] for k in index["kos"] if k["form"] == "CASE"))
    other_case = next(k["id"] for k in index["kos"] if k["form"] == "CASE" and k["id"] != case["id"])
    case["relations"].append({"verb": "has_criterion", "target": other_case})
    problems = onto.validate_index(idx, ROOT)
    assert any("expects object form CRITERION" in p for p in problems)


def test_validator_rejects_one_to_one_fan_out(index):
    # has_criterion is one_to_one: a case judged by two thresholds is an ontology
    # error, not a richer description (which threshold applies becomes unknowable).
    idx = copy.deepcopy(index)
    case = next(k for k in idx["kos"]
                if k["form"] == "CASE" and any(r["verb"] == "has_criterion" for r in k["relations"]))
    taken = {r["target"] for r in case["relations"] if r["verb"] == "has_criterion"}
    spare = next(k["id"] for k in idx["kos"] if k["form"] == "CRITERION" and k["id"] not in taken)
    case["relations"].append({"verb": "has_criterion", "target": spare})
    idx["edges"].append({"subject": case["id"], "verb": "has_criterion", "object": spare,
                         "subject_form": "CASE", "object_form": "CRITERION", "derived_inverse": False})
    problems = onto.validate_index(idx, ROOT)
    assert any("one_to_one" in p for p in problems)


def test_validator_rejects_id_prefix_mismatch(index):
    idx, case = _mutated(index, next(k["id"] for k in index["kos"] if k["form"] == "CASE"))
    case["id"] = "SOMETHING-" + case["id"]
    for e in idx["edges"]:
        for side in ("subject", "object"):
            if e[side] == case["id"].replace("SOMETHING-", ""):
                e[side] = case["id"]
    problems = onto.validate_index(idx, ROOT)
    assert any("does not match form" in p for p in problems)


def test_validator_rejects_inflated_evidence_grade(index):
    idx, ev = _mutated(index, next(k["id"] for k in index["kos"]
                                   if k["form"] == "EVIDENCE" and k["fields"]["mechanical_grade"] == "REAL"))
    ev["fields"]["sha256"] = None
    problems = onto.validate_index(idx, ROOT)
    assert any("REAL evidence without content digest" in p for p in problems)


def test_schema_rejects_unknown_grade_and_missing_field(index):
    import jsonschema

    schema = onto.schema(ROOT)
    validator = jsonschema.Draft202012Validator(schema)
    bad = copy.deepcopy(next(k for k in index["kos"] if k["form"] == "EVIDENCE"))
    bad["fields"]["mechanical_grade"] = "PROBABLY_REAL"
    assert list(validator.iter_errors(bad))
    incomplete = copy.deepcopy(next(k for k in index["kos"] if k["form"] == "CASE"))
    del incomplete["fields"]["verdict"]
    assert list(validator.iter_errors(incomplete))


# --------------------------------------------------------------------------
# mechanical grading rules (the audit surface)
# --------------------------------------------------------------------------
def _write(tmp_path: Path, rel: str, payload) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_grading_missing_artifact_is_withheld(tmp_path):
    grade, basis, digest, _ = onto.mechanical_grade(tmp_path, "benchmarks/nope/evidence.json")
    assert (grade, basis, digest) == ("WITHHELD", "missing_artifact", None)


def test_grading_numeric_payload_is_real(tmp_path):
    _write(tmp_path, "artifacts/case/result.json", {"error": 0.0123, "convergence": [1.0, 2.0]})
    grade, basis, digest, n_num = onto.mechanical_grade(tmp_path, "artifacts/case/result.json")
    assert grade == "REAL" and basis == "numeric_payload_present" and digest and n_num == 3


def test_grading_placeholder_values_is_stub(tmp_path):
    _write(tmp_path, "artifacts/case/result.json", {"error": "TODO", "note": "placeholder"})
    grade, basis, _, _ = onto.mechanical_grade(tmp_path, "artifacts/case/result.json")
    assert (grade, basis) == ("STUB", "placeholder_values_in_artifact")


def test_grading_keyword_in_meta_text_is_not_stub(tmp_path):
    """Regression: a real artefact that merely *mentions* 'placeholder' in a meta field
    (e.g. "placeholder_markers": []) must not be demoted. The earlier keyword scan did
    exactly that to three genuine evidence files."""
    _write(tmp_path, "artifacts/case/result.json",
           {"placeholder_markers": [], "non_placeholder": True, "error": 0.02})
    grade, _, _, _ = onto.mechanical_grade(tmp_path, "artifacts/case/result.json")
    assert grade == "REAL"


def test_grading_unparsable_json_is_withheld(tmp_path):
    _write(tmp_path, "artifacts/case/broken.json", "{not json")
    grade, basis, _, _ = onto.mechanical_grade(tmp_path, "artifacts/case/broken.json")
    assert (grade, basis) == ("WITHHELD", "unparsable_json")


def test_grading_resolves_bare_basename(tmp_path):
    _write(tmp_path, "benchmarks/case_a/evidence.json", {"error": 0.01})
    grade, basis, _, _ = onto.mechanical_grade(tmp_path, "evidence.json")
    assert grade == "REAL" and basis.startswith("resolved_by_name:")


def test_grading_ambiguous_basename_stays_withheld(tmp_path):
    _write(tmp_path, "benchmarks/a/evidence.json", {"error": 0.01})
    _write(tmp_path, "benchmarks/b/evidence.json", {"error": 0.02})
    grade, basis, _, _ = onto.mechanical_grade(tmp_path, "evidence.json")
    assert grade == "WITHHELD" and basis.startswith("ambiguous_relative_reference:")


def test_method_seed_paths_exist():
    seed = onto.load_json(ROOT / "tools/tk_ontology/seeds/fem_methods_v1.json")
    for method in seed["methods"]:
        for impl in method["implementation_paths"]:
            assert (ROOT / impl).is_file(), f"{method['id']} points at missing {impl}"


def test_gap_report_lists_known_classes(index):
    report = onto.gap_report(index, onto.metrics(index))
    for heading in ("## 1. Cases whose cited artefacts", "## 2. Declared grade vs mechanical",
                    "## 3. Cases with no parsed acceptance criterion", "## 4. Orphan objects"):
        assert heading in report