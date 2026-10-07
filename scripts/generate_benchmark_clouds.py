#!/usr/bin/env python3
"""Generate reproducible analytical field clouds for the public benchmarks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); meta=[]
    # Cantilever bending stress and transverse displacement fields.
    x=np.linspace(0,1,81); y=np.linspace(-.2,.2,41); X,Y=np.meshgrid(x,y)
    U=100*X**2*(3-X)/(6*210e9*1e-6); S=-100*(1-X)*Y/1e-6
    for name,Z,label in [('cantilever_displacement',np.broadcast_to(U,X.shape),'u_y [m]'),('cantilever_bending_stress',S,'sigma_xx [Pa]')]:
        fig,ax=plt.subplots(figsize=(7,3)); im=ax.pcolormesh(X,Y,Z,shading='auto',cmap='turbo'); fig.colorbar(im,ax=ax,label=label); ax.set(xlabel='x [m]',ylabel='y [m]',title=name); fig.tight_layout(); path=out/(name+'.png'); fig.savefig(path,dpi=160); plt.close(fig); meta.append({'name':name,'path':str(path),'field':label,'min':float(Z.min()),'max':float(Z.max()),'source':'closed_form'})
    # Hertz pressure cloud over circular contact patch.
    r=np.linspace(0,1,121); X,Y=np.meshgrid(np.linspace(-1,1,161),np.linspace(-1,1,161)); R=np.sqrt(X*X+Y*Y); P=np.where(R<=1,1.5*np.sqrt(1-R*R),0.)
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.pcolormesh(X,Y,P,shading='auto',cmap='turbo'); fig.colorbar(im,ax=ax,label='normalized contact pressure'); ax.set_aspect('equal'); ax.set(title='Hertz contact pressure cloud'); fig.tight_layout(); path=out/'hertz_contact_pressure.png'; fig.savefig(path,dpi=160); plt.close(fig); meta.append({'name':'hertz_contact_pressure','path':str(path),'field':'normalized pressure','min':0.,'max':1.5,'source':'Hertz closed form'})
    (out/'cloud-metadata.json').write_text(json.dumps({'schema':'tensorfem.benchmark-clouds/1.0','fields':meta},indent=2)+'\n')
if __name__=='__main__': main()
