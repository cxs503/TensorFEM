import json
from pathlib import Path

import pytest

from tensorfem.marine_structures import (
    solve_hull_girder_uniform_load, stiffened_panel_sine_benchmark,
)


ROOT = Path(__file__).parents[1]


def test_hull_girder_static_wave_bending_below_three_percent():
    p = json.loads((ROOT / "examples/marine/hull_girder.json").read_text())["parameters"]
    result = solve_hull_girder_uniform_load(**p)
    assert result.maximum_relative_error < 0.03
    expected_reaction = result.distributed_load*p["length"]/2
    assert abs(float(result.frame.reaction[1])/expected_reaction - 1) < 1e-10


def test_stiffened_orthotropic_panel_below_three_percent():
    p = json.loads((ROOT / "examples/marine/stiffened_panel.json").read_text())["parameters"]
    result = stiffened_panel_sine_benchmark(**p)
    assert result.relative_error < 0.03
    assert result.bending_rigidity_x > result.bending_rigidity_y


@pytest.mark.parametrize("bad", [0, 3, -2])
def test_hull_girder_rejects_invalid_mesh(bad):
    with pytest.raises(ValueError):
        solve_hull_girder_uniform_load(length=100, young=2e11, area=1, inertia=2,
                                       still_water_load=1, wave_load=1, elements=bad)


def test_panel_rejects_nonphysical_material():
    with pytest.raises(ValueError):
        stiffened_panel_sine_benchmark(length=4, width=2, plate_thickness=.01,
            plate_young=2e11, plate_poisson=.6, stiffener_spacing=.5,
            stiffener_area=.001, stiffener_young=2e11, stiffener_eccentricity=.1,
            pressure_amplitude=1e4)
