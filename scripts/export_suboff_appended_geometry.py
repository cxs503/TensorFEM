#!/usr/bin/env python
"""Export welded equivalent SUBOFF sail/fins using numerical TensorLBM CAD queries."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from tensorfem.suboff_appendages import appended_shell_mesh


def export(source,output,*,length=4.356,axial=16,circumferential=16,sail_chord=6,fin_chord=6,layers=3):
    source=Path(source).resolve()
    spec=importlib.util.spec_from_file_location('_suboff_appended_cad_export',source)
    cad=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cad;spec.loader.exec_module(cad)
    scale=length/cad._SUBOFF_L_FT;radius=length*cad.SuboffConfig().r_over_l
    xs=np.linspace(cad._SAIL_X1_START,cad._SAIL_X3_END,sail_chord+1)
    a=cad._sail_half_thickness_np(xs)
    sail=[[float(x*scale),float((cad._SAIL_YTMP+w/2)*scale),float(w*scale)] for x,w in zip(xs,a)]
    # Fin root is the actual hull/fin median-plane intersection. The upstream
    # triangle exporter includes span inside the hull, which is excluded here.
    fins=[]
    for s in np.linspace(0,1,fin_chord+1):
        lo=0.;hi=cad._FIN_R_OUTER
        for _ in range(60):
            r=(lo+hi)/2
            x=cad._FIN_H+(s-1)*(cad._FIN_SWEEP_K*r+cad._FIN_SWEEP_C)
            actual=radius*float(cad.suboff_radius_profile(x*scale/length))/scale
            if r<actual:lo=r
            else:hi=r
        rr=(lo+hi)/2
        xr=cad._FIN_H+(s-1)*(cad._FIN_SWEEP_K*rr+cad._FIN_SWEEP_C)
        xt=cad._FIN_H+(s-1)*(cad._FIN_SWEEP_K*cad._FIN_R_OUTER+cad._FIN_SWEEP_C)
        fins.append([float(xr*scale),float(xt*scale),float(cad._FIN_R_OUTER*scale)])
    mesh=appended_shell_mesh(cad.suboff_radius_profile,length,radius,axial,circumferential,
                              sail,fins,sail_layers=layers,fin_layers=layers)
    commit=subprocess.check_output(['git','-C',str(source.parents[2]),'rev-parse','HEAD'],text=True).strip()
    mesh.update(schema='tensorfem.suboff-appended-shell/1',length_m=length,radius_m=radius,
                source={'project':'TensorLBM','commit':commit,'relative_path':'src/tensorlbm/suboff_cad.py',
                        'file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'variant':'full',
                        'representation':'equivalent sail and four fin midsurface sheets welded to closed hull',
                        'queries':['suboff_radius_profile','_sail_half_thickness_np','sail/fin CAD dimensions']})
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(mesh,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'output':str(p),'audit':mesh['audit'],'components':[p['name'] for p in mesh['components']]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='/data/TensorLBM/src/tensorlbm/suboff_cad.py')
    p.add_argument('--output',default='docs/assets/suboff-ice-v2/appended-16.json')
    p.add_argument('--length',type=float,default=4.356)
    p.add_argument('--axial',type=int,default=16)
    p.add_argument('--circumferential',type=int,default=16)
    p.add_argument('--sail-chord',type=int,default=6)
    p.add_argument('--fin-chord',type=int,default=6)
    p.add_argument('--layers',type=int,default=3)
    a=p.parse_args();export(a.source,a.output,length=a.length,axial=a.axial,circumferential=a.circumferential,
                           sail_chord=a.sail_chord,fin_chord=a.fin_chord,layers=a.layers)
