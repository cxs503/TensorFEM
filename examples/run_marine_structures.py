"""Run the two introductory marine-structure verification cases."""
import json
from pathlib import Path

from tensorfem.marine_structures import solve_hull_girder_uniform_load, stiffened_panel_sine_benchmark


root = Path(__file__).with_name("marine")
for filename, solver in (("hull_girder.json", solve_hull_girder_uniform_load),
                         ("stiffened_panel.json", stiffened_panel_sine_benchmark)):
    case = json.loads((root / filename).read_text())
    result = solver(**case["parameters"])
    error = result.maximum_relative_error if filename.startswith("hull") else result.relative_error
    print(f"{case['name']}: relative error={100*error:.6f}%")
