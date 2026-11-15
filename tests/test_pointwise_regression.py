"""Tests for the pointwise reference-regression harness and its store.

The tests assert *invariants*, not just execution:

* a stored reference must reproduce, and a perturbed value must be caught and
  named (the whole point of the harness);
* a structural change (probe set, coordinates, quantity coverage, mesh) must
  raise instead of being reported as a tolerance failure;
* a component that is zero by symmetry must remain inside the regression
  surface through the declared physical scale, not be withheld;
* the analytic gates of the two closed-form cases must pass with the measured
  margins reported below.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from tensorfem import pointwise_cases as cases
from tensorfem.pointwise_regression import (
    SCHEMA_VERSION,
    ProbeSample,
    Snapshot,
    Tolerance,
    build_snapshot,
    compare,
    dumps,
    load,
    loads,
    main,
    mesh_signature,
    save,
    verify_directory,
)

STORE = "benchmarks/pointwise_reference"


# ---------------------------------------------------------------------------
# unit-level: tolerances, construction, serialisation
# ---------------------------------------------------------------------------
def test_tolerance_rejects_ill_posed_settings():
    for kwargs in ({"rtol": -0.1}, {"rtol": 1.0}, {"atol": -1.0},
                   {"atol": None, "atol_scale": 0.0}):
        with pytest.raises(ValueError):
            Tolerance(**kwargs)


def test_tolerance_resolves_atol_from_scale():
    assert Tolerance(rtol=1e-9, atol=2.5).resolve_atol(1.0e6) == 2.5
    assert Tolerance(rtol=1e-9, atol_scale=1e-12).resolve_atol(1.0e6) == pytest.approx(1e-6)


def _probe(probe_id, values, coords=(0.0, 0.0), kind="node"):
    return ProbeSample(probe_id=probe_id, kind=kind, coordinates=tuple(coords),
                       values=values)


def test_build_snapshot_is_canonical_and_order_independent():
    a = build_snapshot("c", ("u",), [_probe("p2", {"u": 2.0}), _probe("p1", {"u": 1.0})])
    b = build_snapshot("c", ("u",), [_probe("p1", {"u": 1.0}), _probe("p2", {"u": 2.0})])
    assert [p.probe_id for p in a.probes] == ["p1", "p2"]
    assert dumps(a) == dumps(b)


def test_build_snapshot_rejects_duplicate_probe_ids():
    with pytest.raises(ValueError, match="duplicate probe ids"):
        build_snapshot("c", ("u",), [_probe("p", {"u": 1.0}), _probe("p", {"u": 2.0})])


def test_build_snapshot_rejects_undeclared_quantity():
    with pytest.raises(ValueError, match="undeclared quantities"):
        build_snapshot("c", ("u",), [_probe("p", {"u": 1.0, "v": 2.0})])


def test_build_snapshot_rejects_declared_but_never_carried_quantity():
    with pytest.raises(ValueError, match="not carried by any probe"):
        build_snapshot("c", ("u", "v"), [_probe("p", {"u": 1.0})])


def test_build_snapshot_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="not finite"):
        build_snapshot("c", ("u",), [_probe("p", {"u": math.nan})])


def test_build_snapshot_accepts_per_probe_quantity_coverage():
    snap = build_snapshot(
        "c", ("u", "s"),
        [_probe("n0", {"u": 1.0}), _probe("e0", {"s": 5.0}, kind="element_centre")],
    )
    assert snap.values("u") == [1.0]
    assert snap.values("s") == [5.0]
    assert snap.n_values == 2


def test_snapshot_json_roundtrip(tmp_path):
    snap = build_snapshot("c", ("u",), [_probe("p1", {"u": 1.25})],
                          {"mesh_signature": "deadbeef"})
    path = save(snap, tmp_path / "c.json")
    assert load(path) == snap
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_loads_rejects_unknown_schema_version():
    payload = json.loads(dumps(build_snapshot("c", ("u",), [_probe("p", {"u": 1.0})])))
    payload["schema_version"] = "0.9"
    with pytest.raises(ValueError, match="schema_version"):
        loads(json.dumps(payload))


def test_loads_rejects_hand_edited_probe_order():
    payload = json.loads(dumps(build_snapshot(
        "c", ("u",), [_probe("p1", {"u": 1.0}), _probe("p2", {"u": 2.0})])))
    payload["probes"].reverse()
    with pytest.raises(ValueError, match="canonical order"):
        loads(json.dumps(payload))


def test_mesh_signature_pins_the_discretisation():
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                         dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    digest = mesh_signature(nodes, elements)
    assert digest == mesh_signature(nodes.clone(), elements.clone())
    moved = nodes.clone()
    moved[2, 0] += 1e-12
    assert mesh_signature(moved, elements) != digest
    assert mesh_signature(nodes, torch.tensor([[0, 2, 1, 3]])) != digest


# ---------------------------------------------------------------------------
# comparison semantics
# ---------------------------------------------------------------------------
def _reference():
    return build_snapshot(
        "case", ("u", "s"),
        [_probe("n0", {"u": 1.0}, coords=(0.0, 0.5)),
         _probe("e0", {"s": 2.0}, coords=(0.5, 0.5), kind="element_centre")],
        {"mesh_signature": "sig", "quantity_scales": {"u": 0.0, "s": 0.0}},
    )


def _with_values(snapshot, **changes):
    probes = []
    for probe in snapshot.probes:
        values = dict(probe.values)
        for quantity, value in changes.items():
            if quantity in values:
                values[quantity] = value
        probes.append(ProbeSample(probe.probe_id, probe.kind, probe.coordinates, values))
    return build_snapshot(snapshot.case_id, snapshot.quantities, probes, snapshot.meta)


def test_identical_snapshots_pass():
    report = compare(_reference(), _reference(), mode="regression")
    assert report["state"] == "PASS"
    assert report["counts"]["FAIL"] == 0
    assert report["max_relative_error"] == 0.0


def test_injected_drift_is_caught_and_named():
    current = _with_values(_reference(), s=2.0 * (1.0 + 1e-6))
    report = compare(_reference(), current, mode="regression",
                     tolerance=Tolerance(rtol=1e-9))
    assert report["state"] == "FAIL"
    assert report["failed_labels"] == ["case:e0:s"]      # localised, not global
    assert report["counts"]["FAIL"] == 1
    assert report["counts"]["PASS"] == 1
    assert report["by_quantity"]["s"]["max_relative_error"] == pytest.approx(1e-6, rel=1e-3)
    assert report["by_quantity"]["u"]["verdicts"] == "PASS"


def test_drift_below_tolerance_passes():
    current = _with_values(_reference(), s=2.0 * (1.0 + 1e-12))
    report = compare(_reference(), current, mode="regression",
                     tolerance=Tolerance(rtol=1e-9))
    assert report["state"] == "PASS"


def test_nonfinite_value_fails():
    # `build_snapshot` rejects non-finite input, so reach the defensive branch in
    # `compare` the only way it can occur: a snapshot mutated after construction.
    reference = _reference()
    probes = tuple(ProbeSample(p.probe_id, p.kind, p.coordinates,
                               dict(p.values, **({"u": math.inf} if "u" in p.values else {})))
                   for p in reference.probes)
    current = Snapshot(reference.case_id, reference.quantities, probes, reference.meta)
    report = compare(reference, current, mode="regression")
    assert report["state"] == "FAIL"
    assert report["failed_labels"] == ["case:n0:u"]
    failed = [row for row in report["rows"] if row["label"] == "case:n0:u"]
    assert len(failed) == 1
    assert failed[0]["reason"] == "computed value is not finite"


def test_zero_reference_uses_declared_physical_scale_not_withheld():
    reference = build_snapshot(
        "zero", ("s",), [_probe("e0", {"s": 0.0}, kind="element_centre")],
        {"quantity_scales": {"s": 1.0e6}},
    )
    current = build_snapshot(
        "zero", ("s",), [_probe("e0", {"s": 1.0e-9}, kind="element_centre")],
        {"quantity_scales": {"s": 1.0e6}},
    )
    report = compare(reference, current, mode="regression",
                     tolerance=Tolerance(rtol=1e-9, atol_scale=1e-12))
    assert report["state"] == "PASS"                     # round-off, not withheld
    assert report["by_quantity"]["s"]["scale_source"] == "meta.quantity_scales"

    # ...and a spurious 1 % of the load scale is emphatically not a pass.
    spurious = build_snapshot(
        "zero", ("s",), [_probe("e0", {"s": 1.0e4}, kind="element_centre")],
        {"quantity_scales": {"s": 1.0e6}},
    )
    report = compare(reference, spurious, mode="regression",
                     tolerance=Tolerance(rtol=1e-9, atol_scale=1e-12))
    assert report["state"] == "FAIL"


def test_zero_reference_without_scale_is_withheld_not_passed():
    reference = build_snapshot("zero", ("s",), [_probe("e0", {"s": 0.0}, kind="element_centre")])
    current = build_snapshot("zero", ("s",), [_probe("e0", {"s": 0.0}, kind="element_centre")])
    # A zero field with no declared physical scale resolves atol = 0, so the
    # comparison is genuinely undefined and must not be reported as a pass.
    report = compare(reference, current, mode="regression",
                     tolerance=Tolerance(rtol=1e-9, atol_scale=1e-12))
    assert report["state"] == "PASS_WITH_WITHHELD"
    assert report["counts"]["WITHHELD"] == 1
    assert report["counts"]["PASS"] == 0


@pytest.mark.parametrize("mutation,message", [
    (lambda s: build_snapshot(s.case_id, s.quantities,
                              list(s.probes) + [_probe("n9", {"u": 1.0})],
                              s.meta), "probe set changed"),
    (lambda s: build_snapshot("other", s.quantities, s.probes, s.meta), "case mismatch"),
    (lambda s: build_snapshot(s.case_id, s.quantities + ("extra",),
                              list(s.probes) + [_probe("e9", {"extra": 1.0},
                                                       kind="element_centre")],
                              s.meta), "quantity set changed"),
    (lambda s: build_snapshot(s.case_id, s.quantities,
                              [_probe(p.probe_id, dict(p.values),
                                      (p.coordinates[0] + 0.1, p.coordinates[1]), p.kind)
                               for p in s.probes], s.meta), "coordinates moved"),
    (lambda s: build_snapshot(s.case_id, s.quantities, s.probes,
                              dict(s.meta, mesh_signature="other")), "discretisation changed"),
])
def test_structural_mismatch_raises_instead_of_failing(mutation, message):
    reference = _reference()
    with pytest.raises(ValueError, match=message):
        compare(reference, mutation(reference))


def test_regression_mode_default_tolerance_is_tight():
    report = compare(_reference(), _reference(), mode="regression")
    assert report["tolerance"]["rtol"] == 1e-9
    report = compare(_reference(), _reference(), mode="reference")
    assert report["tolerance"]["rtol"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# the shipped store
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def store_ledger():
    return verify_directory(STORE, {
        case_id: (lambda builder=builder: builder()[0])
        for case_id, builder in cases.CASE_BUILDERS.items()
    })


def test_shipped_store_reproduces(store_ledger):
    assert store_ledger["state"] == "PASS"
    assert store_ledger["n_cases"] == 3
    assert store_ledger["n_values"] == 421
    for case in store_ledger["cases"]:
        assert case["state"] == "PASS", case["case_id"]
        assert case["counts"]["FAIL"] == 0
        assert case["counts"]["WITHHELD"] == 0
        # deterministic float64 path: a re-run is bit-identical
        assert case["max_relative_error"] == 0.0


def test_store_files_are_present_and_well_formed():
    import pathlib

    root = pathlib.Path(STORE)
    names = {p.name for p in root.glob("*.json")}
    assert {"q4-uniaxial-patch.json", "thermal-harmonic-square.json",
            "cook-membrane-q4.json"} <= names
    for name in sorted(names - {"ledger.json", "evidence.json"}):
        snapshot = load(root / name)
        assert snapshot.schema_version == SCHEMA_VERSION
        assert snapshot.meta["mesh_signature"]
        assert snapshot.meta["case_key"] == snapshot.case_id


def test_cli_check_passes_and_writes_ledger(tmp_path):
    assert main(["check", "--store", STORE, "--ledger", str(tmp_path / "ledger.json")]) == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["state"] == "PASS"
    assert ledger["n_values"] == 421


# ---------------------------------------------------------------------------
# analytic gates (the "XVAL" half of the anchor)
# ---------------------------------------------------------------------------
def test_uniaxial_patch_is_pointwise_exact():
    computed, analytic = cases.uniaxial_patch()
    report = compare(analytic, computed, mode="reference", tolerance=Tolerance(rtol=0.03))
    assert report["state"] == "PASS"
    # A patch test must reproduce a linear field to round-off, not to 3 %.
    assert report["max_relative_error"] < 1e-12
    for quantity in ("ux", "uy", "sigma_xx"):
        assert report["by_quantity"][quantity]["max_relative_error"] < 1e-12
    # zero-by-symmetry components stay inside the surface via the declared scale
    for quantity in ("sigma_yy", "tau_xy"):
        entry = report["by_quantity"][quantity]
        assert entry["scale_source"] == "meta.quantity_scales"
        assert entry["max_abs_error"] < 1e-6          # 1e-12 of the 1 MPa scale


def test_thermal_harmonic_analytic_gate_is_met():
    computed, analytic = cases.thermal_harmonic_square()
    report = compare(analytic, computed, mode="reference", tolerance=Tolerance(rtol=0.03))
    assert report["state"] == "PASS"
    measured = {q: report["by_quantity"][q]["max_relative_error"]
                for q in ("T", "q_x", "q_y")}
    # Fixed 16x16 mesh: the discretisation error, not round-off, sets this.
    assert measured["T"] < 0.03
    assert measured["q_x"] < 0.03
    assert measured["q_y"] < 0.03
    assert measured["T"] > 0.0                       # a real error, not a stub


def test_cook_membrane_declares_absence_of_analytic_reference():
    computed, analytic = cases.cook_membrane_q4()
    assert analytic is None
    assert computed.meta["has_analytic_reference"] is False
    assert "no pointwise closed-form" in computed.meta["analytic_reference_reason"]
    # a case without an analytic reference still supports regression mode
    report = compare(computed, computed, mode="regression")
    assert report["state"] == "PASS"


def test_harness_catches_a_regression_injected_into_a_real_case():
    """Perturb one stored Cook value and require the harness to name it."""
    stored = load(f"{STORE}/cook-membrane-q4.json")
    target = stored.probes[3]
    probes = []
    for probe in stored.probes:
        values = dict(probe.values)
        if probe.probe_id == target.probe_id:
            quantity = sorted(values)[0]
            values[quantity] = values[quantity] * (1.0 + 1e-6)
            changed = f"{stored.case_id}:{probe.probe_id}:{quantity}"
        probes.append(ProbeSample(probe.probe_id, probe.kind, probe.coordinates, values))
    tampered = build_snapshot(stored.case_id, stored.quantities, probes, stored.meta)
    report = compare(stored, tampered, mode="regression", tolerance=Tolerance(rtol=1e-9))
    assert report["state"] == "FAIL"
    assert report["failed_labels"] == [changed]
    assert report["counts"]["PASS"] == stored.n_values - 1
    assert report["counts"]["FAIL"] == 1