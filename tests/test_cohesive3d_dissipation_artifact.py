"""Gate for the committed 3-D cohesive dissipation figure/number package.

The artifact ``artifacts/cohesive3d_dissipation/`` is the single source of every
number quoted in the dissipated-energy report (figures 1-3 and ``numbers.json``).
These tests make that package auditable instead of decorative:

* every committed scalar is re-derived from the live ``tensorfem.cohesive3d``
  module and must agree to ``1e-6`` relative (the numbers cannot rot);
* the three PNGs exist and are real figures (non-trivial size, valid PNG header);
* the ``--check`` gate is *exercised*: a tampered copy of ``numbers.json`` must
  be rejected, so a green gate means something (negative control).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "cohesive3d_dissipation"
FIGURES = (
    "fig1_dissipation_envelope.png",
    "fig2_nonproportional_path.png",
    "fig3_element_energy_closure.png",
)


@functools.lru_cache(maxsize=1)
def _load_module():
    """One module instance per session: its number cache (the 2001-step path
    integrals, ~30 s) is then computed once and shared by every test."""
    path = ARTIFACT / "make_figures.py"
    spec = importlib.util.spec_from_file_location("cohesive3d_dissipation_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def package():
    return _load_module()


def test_numbers_json_matches_live_module(package):
    """Committed numbers == numbers re-derived now, to 1e-6 relative."""
    committed = package.flatten_numbers(json.loads((ARTIFACT / "numbers.json").read_text()))
    live = package.collect_flat()

    flat = committed
    assert set(flat) == set(live), "artifact keys drifted from the generator"
    # The per-ray entries (list values) must be in the gate: they are the
    # path-integral residuals the report quotes.
    assert "envelope_rays[0].rel_residual" in flat
    for key, expected in flat.items():
        got = live[key]
        denominator = max(abs(expected), abs(got), 1e-300)
        assert abs(got - expected) / denominator <= 1e-6, f"{key}: {got} != {expected}"


def test_check_mode_passes():
    assert _load_module().check() == 0


def test_check_mode_rejects_tampered_numbers(tmp_path):
    """Negative control: without it the gate could pass on a broken artifact."""
    numbers = json.loads((ARTIFACT / "numbers.json").read_text())
    numbers["envelope_rays"][0]["closed_form_at_failure"] *= 1.01
    tampered = tmp_path / "numbers.json"
    tampered.write_text(json.dumps(numbers))
    assert _load_module().check(numbers_path=tampered, figures_dir=ARTIFACT) == 1


def test_check_mode_rejects_missing_figure(tmp_path):
    numbers = tmp_path / "numbers.json"
    shutil.copy(ARTIFACT / "numbers.json", numbers)
    figures = tmp_path / "figures"
    figures.mkdir()
    assert _load_module().check(numbers_path=numbers, figures_dir=figures) == 1


@pytest.mark.parametrize("figure", FIGURES)
def test_figure_present_and_valid_png(figure):
    path = ARTIFACT / figure
    assert path.is_file(), f"{figure} missing from the artifact package"
    payload = path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n"), f"{figure} is not a PNG"
    assert len(payload) > 20_000, f"{figure} is suspiciously small ({len(payload)} bytes)"


def test_committed_numbers_meet_the_reported_accuracy(package):
    """The physical claims behind the figures, re-derived (not read from JSON)."""
    numbers = package.collect_flat()

    # Closed form == numerical path integral along four fixed mode mixes.
    worst = max(numbers[f"envelope_rays[{i}].rel_residual"] for i in range(4))
    assert worst < 1e-4, f"path-integral residual {worst:.2e} exceeds 1e-4"

    # W_d(a_f) == G_c(direction) for every ray (this is what BK means).
    for i in range(4):
        gc = numbers[f"envelope_rays[{i}].gc"]
        closed = numbers[f"envelope_rays[{i}].closed_form_at_failure"]
        assert abs(closed - gc) / gc < 1e-12

    # Element closure W_d = G_Ic * A and the non-proportional path bound.
    assert numbers["element.rel_error"] < 1e-12
    assert numbers["nonproportional.work_path"] <= numbers["nonproportional.g_IIc"] * (1 + 1e-3)
    assert numbers["nonproportional.rel_residual"] < 1e-3
