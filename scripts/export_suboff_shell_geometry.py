#!/usr/bin/env python
"""Export numerical SUBOFF geometry from the existing TensorLBM CAD module."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from tensorfem.suboff_geometry import closed_revolved_mesh


def export(source, destination, axial, circumferential, length):
    source = Path(source).resolve()
    spec = importlib.util.spec_from_file_location('_tensorlbm_suboff_geometry_export', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    radius = length*module.SuboffConfig().r_over_l
    x,e,audit = closed_revolved_mesh(module.suboff_radius_profile,length,radius,
                                    axial,circumferential)
    git_root = source.parents[2]
    commit = subprocess.check_output(['git','-C',str(git_root),'rev-parse','HEAD'],text=True).strip()
    status = subprocess.check_output(['git','-C',str(git_root),'status','--porcelain','--',str(source)],text=True).strip()
    radial_error = abs((x[:,1:]**2).sum(1).sqrt().numpy() -
                       radius*module.suboff_radius_profile(x[:,0].numpy()/length)).max()
    record = {'schema':'tensorfem.suboff-outer-geometry/1',
              'source':{'project':'TensorLBM','relative_path':'src/tensorlbm/suboff_cad.py',
                        'commit':commit,'file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'source_file_modified':bool(status),
                        'function':'suboff_radius_profile','variant':'bare_hull',
                        'reference':'Groves, Huang, Chang (1989), DTRC/SHD-1298-01'},
              'length_m':length,'radius_m':radius,'axial':axial,
              'circumferential':circumferential,'nodes':x.tolist(),'elements':e.tolist(),
              'audit':audit,'maximum_source_profile_error_m':float(radial_error)}
    if radial_error > 1e-7*length:
        raise ValueError('CAD reproduction error exceeds geometry gate')
    destination = Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'output':str(destination),'audit':audit,
                      'source_profile_error_m':float(radial_error)},ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='/data/TensorLBM/src/tensorlbm/suboff_cad.py')
    p.add_argument('--output',default='docs/assets/suboff-ice/geometry-24.json')
    p.add_argument('--axial',type=int,default=24)
    p.add_argument('--circumferential',type=int,default=24)
    p.add_argument('--length',type=float,default=4.356)
    a=p.parse_args();export(a.source,a.output,a.axial,a.circumferential,a.length)
