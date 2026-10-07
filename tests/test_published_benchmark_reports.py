"""Regression gates for the published, self-contained benchmark evidence."""

from pathlib import Path
import re

import pytest

from scripts.run_stiffened_panel_fe_benchmark import solve_case


ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "cantilever_beam_full.md": 2,
    "hull_girder_longitudinal_bending_full.md": 4,
    "stiffened_panel_fe_full.md": 2,
}


@pytest.mark.parametrize(("name", "minimum_images"), REPORTS.items())
def test_published_report_contains_results_and_embedded_images(name, minimum_images):
    report = ROOT / "docs" / "benchmarks" / "tutorials" / name
    text = report.read_text(encoding="utf-8")

    assert "真实计算结果" in text
    assert "相对误差" in text
    assert not re.search(r"待填|待计算|generated in JSON|gate-derived|TODO", text)

    images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
    assert len(images) >= minimum_images
    assert all((report.parent / image).resolve().is_file() for image in images)


def test_stiffened_panel_patch_test_preserves_total_load_and_constant_strain():
    case = solve_case(4, 2)

    assert case["applied_load_x"] == pytest.approx(1.0e6, rel=1e-14)
    assert case["reaction_left_x"] == pytest.approx(-1.0e6, rel=1e-12)
    assert case["reaction_left_y"] == pytest.approx(0.0, abs=1e-6)
    assert case["relative_error"] < 1e-12
    assert case["status"] == "qualified"

    sigma_x = [sample["sigma_x"] for sample in case["field"]["element_stress"]]
    assert min(sigma_x) == pytest.approx(62.5e6, rel=1e-12)
    assert max(sigma_x) == pytest.approx(62.5e6, rel=1e-12)
