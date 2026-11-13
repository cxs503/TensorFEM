"""Test the TensorFEM external-tool integration verification (KPI-TOOLS-5).

The suite must report >= 5 REAL tool categories, each verified against an
analytic/closed-form reference with a hard error bound.  A missing tool must be
reported as MISSING (never silently skipped), and the verdict must be PASS only
when every probed category is REAL.
"""

import json
from pathlib import Path

import pytest

from tensorfem.external_tools_verification import (
    TOOL_CATEGORIES,
    run_suite,
)


def test_suite_reports_at_least_five_real_categories(tmp_path: Path) -> None:
    """The full probe must surface >= 5 REAL tool categories (KPI-TOOLS-5)."""
    ledger = run_suite(tmp_path)
    assert ledger["real_classes"] >= 5, (
        f"expected >=5 REAL tool categories, got {ledger['real_classes']}: "
        f"{[c['tool_id'] for c in ledger['classes'] if c['status'] != 'REAL']}"
    )
    assert ledger["ok"] is True
    assert ledger["verdict"] == "PASS"


def test_every_category_has_an_analytic_reference() -> None:
    """Each probed category must carry a closed-form/analytic reference."""
    for cat in TOOL_CATEGORIES:
        assert cat["ref_source"], f"{cat['tool_id']} has no ref_source"
        assert "ref" in cat, f"{cat['tool_id']} has no ref"


def test_autodiff_matches_closed_form() -> None:
    """torch reverse-mode gradient of Σx³ must match 3x² to 1e-10."""
    from tensorfem.external_tools_verification import _probe_autodiff

    r = _probe_autodiff()
    assert r["abs_err"] <= 1e-10


def test_quadrature_matches_pi() -> None:
    """scipy quad of ∫0^1 4/(1+t²)dt must match π to 1e-10."""
    from tensorfem.external_tools_verification import _probe_quadrature

    r = _probe_quadrature()
    assert r["abs_err"] <= 1e-10


def test_symbolic_fundamental_theorem() -> None:
    """sympy integrate(diff(f)) - f must simplify to exactly 0."""
    from tensorfem.external_tools_verification import _probe_symbolic

    r = _probe_symbolic()
    assert r["abs_err"] == 0.0


def test_mpmath_zeta2_vs_pi2_over_6() -> None:
    """mpmath zeta(2) must match π²/6 to 1e-30."""
    from tensorfem.external_tools_verification import _probe_mpmath

    r = _probe_mpmath()
    assert r["abs_err"] <= 1e-30


def test_matplotlib_renders_nonblank_field(tmp_path: Path) -> None:
    """matplotlib must render a non-degenerate field to a non-blank PNG."""
    from tensorfem.external_tools_verification import _probe_matplotlib

    r = _probe_matplotlib(tmp_path)
    png = tmp_path / "external_tools_field.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert r["value"] > 0.0  # field span non-zero -> not a blank canvas


def test_ledger_is_json_serializable(tmp_path: Path) -> None:
    """The ledger must be JSON-serializable (for the platform report)."""
    ledger = run_suite(tmp_path)
    json.dumps(ledger)  # must not raise
