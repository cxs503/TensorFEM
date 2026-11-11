"""One-shot patch: exact Saint-Venant end traction + rectangular wall-peak recovery."""
from __future__ import annotations

import pathlib

path = pathlib.Path("src/tensorfem/torsion_rectangular.py")
text = path.read_text()

old_all = '''    "rectangular_torsion_bar",
    "richardson",'''
new_all = '''    "rectangular_torsion_bar",
    "richardson",
    "surface_peak_shear",'''
assert old_all in text
text = text.replace(old_all, new_all)

old_classes = '''    "RectangularBarResult",'''
new_classes = '''    "RectangularBarResult",
    "SurfacePeak",'''
assert old_classes in text
text = text.replace(old_classes, new_classes)

old_fields = '''    centroid_shear_l2_error: float
    peak_centroid_fe: float
    peak_series_sampled: float
'''
new_fields = '''    centroid_shear_l2_error: float
    peak_centroid_fe: float
    peak_series_sampled: float
    end_traction: str = "linear"
    end_traction_residual: float = 0.0
'''
assert old_fields in text
text = text.replace(old_fields, new_fields)

start = "def rectangular_torsion_bar(*, a: float = 2.0"
end = "# ------------------------------------------------------- independent PDE reference"
i0 = text.index(start)
i1 = text.index(end)
assert i0 < i1, "anchors out of order"
block = pathlib.Path("scratch/new_block.txt").read_text()
block = block.replace("long_target = (b - hz / 2.0, a / 2.0)",
                      "long_target = (b - b / solve.n_short / 2.0, a / 2.0)")
block = block.replace("short_target = (b / 2.0, a - hz / 2.0)",
                      "short_target = (b / 2.0, a - a / solve.n_long / 2.0)")
block = block.replace("mid_slab = (centroids[:, 2] - z_mid).abs() < 1e-9",
                      "mid_slab = (centroids[:, 2] - z_mid).abs() < hz / 2.0 + 1e-9")
text = text[:i0] + block + "\n\n" + text[i1:]
path.write_text(text)
print("patched", path, len(text.splitlines()), "lines")