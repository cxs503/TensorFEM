#!/usr/bin/env python
"""Solve, render and validate the four additional cases without cached passes."""
import json
from pathlib import Path
from tensorfem.additional_benchmark_fields import build_additional_reports
from tensorfem.hemisphere_benchmark_fields import hemisphere_field_report
from tensorfem.sphere_pressure_benchmark import sphere_pressure_report
from build_additional_benchmark_reports import build
from validate_additional_benchmark_reports import validate

root=Path(__file__).resolve().parents[1]
reports=build_additional_reports()
reports['sphere-pressure']=sphere_pressure_report()
reports['hemisphere-hole']=hemisphere_field_report()
for name,r in reports.items():
    p=root/'docs/assets/benchmark-clouds'/name;p.mkdir(exist_ok=True)
    pending=p/'results.json.tmp'
    with pending.open('w') as handle:
        json.dump(r,handle,allow_nan=False,**({'separators':(',',':')} if name=='sphere-pressure' else {'indent':1}))
        handle.write('\n')
    pending.replace(p/'results.json')
build()
validate(reproduce=False)  # This invocation has just rerun every solver.
