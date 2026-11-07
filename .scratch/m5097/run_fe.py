import json, sys, time
import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
from tensorfem.stiffened_panel_unit_fe import build_panel_unit, trace_panel_unit_path, trace_summary

cfg = json.loads(sys.argv[1])
panel = StiffenedPanel(**cfg["panel"])
case = build_panel_unit(panel, nx=cfg.get("nx",4), ny=cfg.get("ny",2), layers=cfg.get("layers",3),
                        pressure_ratio=cfg.get("pressure_ratio",0.0),
                        column_imperfection_ratio=cfg.get("column_imperfection",1e-3),
                        plate_imperfection_ratio=cfg.get("plate_imperfection",0.1))
sy = panel.yield_stress
ds = cfg.get("ds") or 0.02*panel.plate_thickness
ls = ds/(cfg.get("load_fraction", 0.05)*sy)
t0=time.time()
trace = trace_panel_unit_path(case, steps=cfg.get("steps",24), step_size=ds, load_scale=ls,
                              tolerance=cfg.get("tolerance",1e-3), max_iterations=cfg.get("max_iterations",12))
out = trace_summary(case, trace)
out.update({"elapsed": time.time()-t0, "config": cfg, "panel_summary": panel.summary(),
            "free_dofs": int(len(case.free_dofs())), "elements": case.element_counts})
print(json.dumps(out, indent=1))
