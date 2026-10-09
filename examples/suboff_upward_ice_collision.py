#!/usr/bin/env python
"""Run a configurable initial SUBOFF upward impact against cohesive ice."""
import argparse
import json
from pathlib import Path

from tensorfem.suboff_ice_collision import IceImpactConfig, save_result, simulate_impact


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,default=root/'docs/assets/suboff-ice/geometry-16.json')
    p.add_argument('--output',type=Path,default=root/'results/suboff-ice/results.json')
    p.add_argument('--config',type=Path,help='JSON object with IceImpactConfig overrides (SI units)')
    p.add_argument('--speed',type=float)
    p.add_argument('--duration',type=float)
    p.add_argument('--time-safety',type=float)
    a=p.parse_args()
    overrides=json.loads(a.config.read_text()) if a.config else {}
    for name,value in (('upward_speed',a.speed),('duration',a.duration),('time_safety',a.time_safety)):
        if value is not None:
            overrides[name]=value
    c=IceImpactConfig(**overrides)
    r=simulate_impact(json.loads(a.geometry.read_text()),c)
    save_result(a.output,r,a.geometry)
    print(json.dumps({'status':r['status'],'physical_accuracy_qualified':False,
                      'checks':r['checks'],'metrics':r['metrics'],'solver':r['solver'],
                      'output':str(a.output)},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
