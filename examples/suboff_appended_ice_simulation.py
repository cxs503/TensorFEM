#!/usr/bin/env python
"""Run appended SUBOFF against configurable ice (SI units; initial impact only)."""
import argparse
import hashlib
import json
from pathlib import Path
from tensorfem.ice_sheet import IceSimulationConfig
from tensorfem.suboff_ice_simulation import simulate
from tensorfem.suboff_ice_collision import save_result

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('suboff_geometry.py','suboff_appendages.py','ice_sheet.py','suboff_ice_simulation.py','suboff_ice_collision.py','cohesive.py','spherical_shell.py','plate.py')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,default=ROOT/'docs/assets/suboff-ice-v2/appended-16.json')
    p.add_argument('--config',type=Path,help='JSON IceSimulationConfig overrides')
    p.add_argument('--output',type=Path,default=ROOT/'results/suboff-ice-v2/result.json')
    p.add_argument('--duration',type=float,default=None)
    p.add_argument('--speed',type=float,default=None)
    a=p.parse_args();overrides=json.loads(a.config.read_text()) if a.config else {}
    overrides.setdefault('duration',.0015)
    if a.duration is not None:overrides['duration']=a.duration
    if a.speed is not None:overrides['upward_speed']=a.speed
    c=IceSimulationConfig(**overrides)
    r=simulate(json.loads(a.geometry.read_text()),c)
    r['source_sha256']={n:hashlib.sha256((ROOT/'src/tensorfem'/n).read_bytes()).hexdigest() for n in SOURCES}
    r['geometry_file']=str(a.geometry.resolve().relative_to(ROOT))
    save_result(a.output,r,a.geometry)
    print(json.dumps({'status':r['status'],'checks':r['checks'],'metrics':r['metrics'],'output':str(a.output)},indent=2))

if __name__=='__main__':main()
