"""Hemisphere response evidence and centre shell stresses recovered from DOFs."""
import torch
from .spherical_shell import hemisphere_with_hole
from .hemisphere_postprocess import build_hemisphere_post, _plain
from .plate import _shape


def hemisphere_fields(n, drilling_factor=1e-6):
    result = hemisphere_with_hole(n, drilling_factor=drilling_factor)
    post = build_hemisphere_post(result)
    record = _plain(post); record['mesh'] = n; record['drilling_factor'] = drilling_factor
    q = result.solution.reshape(-1,6)
    stress = []
    for el in result.elements:
        xyz = result.nodes[el]
        normal = torch.linalg.cross(xyz[2]-xyz[0],xyz[3]-xyz[1],dim=0)
        e3 = normal/torch.linalg.vector_norm(normal)
        raw = xyz[1]-xyz[0]; e1 = raw-torch.dot(raw,e3)*e3
        e1 /= torch.linalg.vector_norm(e1); e2 = torch.linalg.cross(e3,e1,dim=0)
        basis = torch.stack((e1,e2,e3))
        xy = (xyz-xyz.mean(0))@basis[:2].T
        _,dn = _shape(xy.new_tensor(0.),xy.new_tensor(0.))
        grad = dn@torch.linalg.inv(xy.T@dn)
        trans = q[el,:3]@basis.T; rot = q[el,3:]@basis.T
        dt = trans.T@grad; dr = rot.T@grad
        strain = torch.stack((dt[0,0],dt[1,1],dt[0,1]+dt[1,0]))
        curvature = torch.stack((dr[1,0],-dr[0,1],dr[1,1]-dr[0,0]))
        C = xy.new_tensor([[1,.3,0],[.3,1,0],[0,0,.35]])*(6.825e7/(1-.3**2))
        stress.append(torch.stack((C@(strain+.02*curvature),C@(strain-.02*curvature))))
    stress = torch.stack(stress)
    vm = torch.sqrt(stress[:,:,0]**2-stress[:,:,0]*stress[:,:,1]+stress[:,:,1]**2+3*stress[:,:,2]**2)
    record['stress_top_bottom'] = stress.tolist(); record['von_mises_top_bottom'] = vm.tolist()
    return record


def hemisphere_field_report():
    cases = [hemisphere_fields(n) for n in (8,12,16,20,24,28,32,40)]
    sensitivity = [(cases[-1] if f==1e-6 else hemisphere_fields(40,f))['validation']|{'drilling_factor':f} for f in (1e-7,1e-6,1e-5)]
    last_change = abs(cases[-1]['probe']['value']/cases[-2]['probe']['value']-1)
    passed = cases[-1]['validation']['passed'] and last_change<.03 and all(s['passed'] for s in sensitivity)
    return {'physical_case_id':'hemisphere-hole','cases':cases,'sensitivity':sensitivity,
            'coarse_sensitivity':[(next(c for c in cases if c['mesh']==24) if f==1e-6 else hemisphere_fields(24,f))['validation']|{'drilling_factor':f} for f in (1e-7,1e-6,1e-5)],
            'last_response_change':last_change,'stress_accuracy_qualified':False,
            'scope':'loaded-equator radial displacement only; recovered shell stresses have no independent oracle',
            'status':'response-qualified' if passed else 'blocked'}
