#!/usr/bin/env python
"""Print the per-mesh table of CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md from the committed bundle."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
b = json.load(open(os.path.join(ROOT, "docs", "assets", "gap_pp3_3grid_results.json")))
for n in sorted(b["meshes"], key=lambda k: b["meshes"][k]["elements"]):
    m = b["meshes"][n]
    print(n, m["elements"], m["dofs"], "%.2f%%" % (100 * m["elastic_bias"]),
          "%.4fMPa" % (m["first_yield_factor"] / 1e6),
          "%.2f%%" % (100 * m["first_yield_error"]),
          m.get("limit_load_measured"), m.get("permanent_deflection"), m["seconds"])
