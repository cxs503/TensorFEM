"""Tests for the TensorFEM capability registry (TensorLBM pilot pattern).

Verifies the four acceptance criteria of the capability-registry pilot:

* ``dump`` — ``import`` + ``--static`` AST dual-mode dump;
* ``triad`` — 成熟/开发中/未开发 (mature/developing/undeveloped) classification;
* ``callable`` — endpoint contract fields (signature, params, returns);
* ``decorator`` — ``@capability`` self-registration.
"""
import json
import pytest

from tensorfem.capability_registry import (
    Capability,
    CapabilityContract,
    CapabilityRegistry,
    TRIAD_ORDER,
    assert_two_mode_agreement,
    capability,
    classify_triad,
    dump_capabilities,
    registry,
)


@pytest.fixture(scope="module")
def catalog():
    """Import the catalog so every @capability is registered."""
    import tensorfem.capability_catalog  # noqa: F401
    return tensorfem.capability_catalog


def test_decorator_self_registers(catalog):
    """@capability registers the decorated function in the process-wide registry."""
    assert len(registry) >= 13
    cap = registry.get("capability.linear_static")
    assert cap is not None
    assert cap.title == "Linear Static Solver"
    assert cap.domain == "structural / statics"
    assert cap.maturity == "mature"
    # the decorated wrapper is the callable endpoint
    assert cap.contract.callable_name == "solve_linear_static"


def test_duplicate_id_rejected():
    """A true duplicate capability id is an error, not a silent overwrite."""
    r = CapabilityRegistry()
    r.register(
        Capability(
            id="x.dup",
            title="T",
            domain="d",
            maturity="mature",
            contract=CapabilityContract("_a", "_a()"),
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        r.register(
            Capability(
                id="x.dup",
                title="T2",
                domain="d",
                maturity="mature",
                contract=CapabilityContract("_b", "_b()"),
            )
        )


def test_invalid_maturity_rejected():
    """maturity must be one of the triad; anything else is an error on apply."""
    dec = capability("x.bad", title="T", domain="d", maturity="not-a-triad")

    def _f():
        return 1

    with pytest.raises(ValueError, match="maturity"):
        dec(_f)


def test_dump_import_mode(catalog):
    """import-mode dump carries id/title/domain/maturity/contract/evidence."""
    d = dump_capabilities("import")
    assert d["mode"] == "import"
    assert d["count"] == len(registry)
    ids = {c["id"] for c in d["capabilities"]}
    assert "capability.linear_static" in ids
    assert "capability.shell4" in ids
    for c in d["capabilities"]:
        assert c["title"] and c["domain"] and c["maturity"] in TRIAD_ORDER
        assert c["contract"]["callable_name"]
        assert c["contract"]["signature"]
        assert "evidence" in c


def test_dump_ast_mode(catalog):
    """ast-mode dump parses source without importing and finds the same ids."""
    d = dump_capabilities("ast")
    assert d["mode"] == "ast"
    assert d["count"] == len(registry)
    ids = {c["id"] for c in d["capabilities"]}
    assert "capability.linear_static" in ids
    assert "capability.shell4" in ids
    # ast mode records the source file it parsed
    assert all(c["source"] for c in d["capabilities"])


def test_two_mode_agreement(catalog):
    """import and ast modes agree on the *set* of capability ids (fail-closed)."""
    assert_two_mode_agreement()
    imp = {c["id"] for c in dump_capabilities("import")["capabilities"]}
    ast = {c["id"] for c in dump_capabilities("ast")["capabilities"]}
    assert imp == ast


def test_triad_classification():
    """mature/developing/undeveloped from machine-visible evidence."""
    assert classify_triad(has_benchmark=True, has_test=True) == "mature"
    assert classify_triad(has_benchmark=True, in_public_api=True) == "mature"
    assert classify_triad(has_test=True) == "developing"
    assert classify_triad(in_public_api=True) == "developing"
    assert classify_triad() == "undeveloped"


def test_cli_capabilities_carries_registry(monkeypatch, capsys):
    """`tensorfem capabilities` now emits the registry (triad/contract/dump)."""
    monkeypatch.setattr("sys.argv", ["tensorfem", "capabilities"])
    from tensorfem.cli import main

    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "0.45.0"
    reg = payload["registry"]
    assert reg["count"] == len(registry)
    assert set(reg["triad"]) == {"mature", "developing", "undeveloped"}
    assert reg["triad"]["mature"]  # at least one mature capability
    assert reg["dump_modes"]["import"] == reg["dump_modes"]["ast"] == len(registry)
    assert reg["dump_modes"]["agreement"] is True
    # contract fields present for every capability
    assert all(
        {"callable", "signature", "params", "returns"} <= set(c)
        for c in reg["contracts"].values()
    )
    # backward-compat: the hand-maintained stable/experimental lists still present
    assert payload["stable"] and payload["experimental"]
