#!/usr/bin/env python3
"""Render displacement and von-Mises FE clouds from run_cantilever_fe_benchmark JSON."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); d=json.loads(a.input.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True); c=d['cases'][-1]
 for key,label,name in [('nodes','uy','cantilever_fe_displacement'),('element_stress','von_mises','cantilever_fe_von_mises')]:
  rows=c['field'][key]; x=[r['x'] for r in rows]; y=[r['y'] for r in rows]; z=[r[label] for r in rows]; fig,ax=plt.subplots(figsize=(8,3)); im=ax.scatter(x,y,c=z,s=12,cmap='turbo'); fig.colorbar(im,ax=ax,label=label); ax.set_title(name+' (refined FE mesh)'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); fig.tight_layout(); fig.savefig(a.output_dir/(name+'.png'),dpi=160); plt.close(fig)
if __name__=='__main__': main()
