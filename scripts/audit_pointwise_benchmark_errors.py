#!/usr/bin/env python
"""Independent NumPy audit of pointwise errors, beyond published L2 gates.

The additional criterion is diagnostic and does not replace the previously
stated norms. Zero reference components require absolute-error treatment.
"""
import json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def audit():
    sources={}
    def load(folder,filename='results.json'):
        path=ROOT/'docs/assets/benchmark-clouds'/folder/filename
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())['cases'][-1]
    c=load('sphere-pressure');x=np.array(c['nodes']);cells=np.array(c['elements'])
    p=x[cells].mean(1);r=np.linalg.norm(p,axis=1);n=p/r[:,None]
    A=1e6*8**3/(10**3-8**3);B=A*10**3
    sr=A-B/r**3;st=A+B/(2*r**3)
    raw=np.array(c['element_stress']);fe=np.zeros((len(cells),3,3))
    for j in range(3):fe[:,j,j]=raw[:,j]
    for j,(a,b) in enumerate(((0,1),(1,2),(0,2)),3):fe[:,a,b]=fe[:,b,a]=raw[:,j]
    ref=st[:,None,None]*np.eye(3)+(sr-st)[:,None,None]*n[:,:,None]*n[:,None,:]
    radial=np.einsum('ei,eij,ej->e',n,fe,n)
    rel=abs((radial-sr)/sr);worst=int(np.argmax(rel))
    sphere={'criterion':'all HEX8 centre radial relative errors < 3%',
            'max_radial_relative_error':float(rel.max()),
            'max_stress_tensor_relative_error':float(np.max(np.linalg.norm(fe-ref,axis=(1,2))/np.linalg.norm(ref,axis=(1,2)))),
            'max_radial_absolute_error_over_pressure':float(np.max(abs(radial-sr))/1e6),
            'worst_element_index':worst,'worst_radius_m':float(r[worst]),
            'worst_reference_radial_stress_pa':float(sr[worst]),'worst_fe_radial_stress_pa':float(radial[worst]),
            'pointwise_radial_passed':bool(rel.max()<.03)}
    c=load('hull-girder','hull-girder-fe.json');f=c['field'];x=np.array(f['x'])
    ref=2.3e6*x*(120-x)/2;rel=abs((np.array(f['moment'])-ref)/ref)
    worst=int(rel.argmax())
    hull={'criterion':'all declared span samples: moment and nonzero fibre stress relative errors < 3%',
          'max_moment_and_sigma_x_relative_error':float(rel.max()),'worst_x_m':float(x[worst]),
          'pointwise_passed':bool(rel.max()<.03)}
    c=load('cantilever-fe');raw=c['field']['element_stress']
    a=np.array([[s[k] for k in ('x','y','sigma_x','tau_xy')] for s in raw]);a=a[(a[:,0]>=.2)&(a[:,0]<=.8)]
    I=.012*.1**3/12;sx=100*(1-a[:,0])*a[:,1]/I;sh=-100/(2*I)*(.1**2/4-a[:,1]**2)
    ex=float(np.max(abs((a[:,2]-sx)/sx)));eq=float(np.max(abs((a[:,3]-sh)/sh)))
    beam={'scope':'declared interior 0.2L <= x <= 0.8L; full domain and clamp peaks unqualified',
          'max_sigma_x_relative_error':ex,'max_tau_xy_relative_error':eq,
          'declared_interior_components_passed':max(ex,eq)<.03}
    c=load('mindlin-navier');x=np.array(c['nodes']);u=np.array(c['displacement']);p=np.array(c['centers'])
    k=np.pi;D=1e7*.01**3/(12*(1-.3**2));Ds=5/6*1e7/(2*1.3)*.01
    Wb=1/(D*(2*k*k)**2);Ws=1/(Ds*2*k*k)
    ew=(Wb+Ws)*np.sin(k*x[:,0])*np.sin(k*x[:,1]);nonzero=abs(ew)>1e-12
    werror=float(np.max(abs((u[nonzero,0]-ew[nonzero])/ew[nonzero])))
    diagonal=k*k*Wb*np.sin(k*p[:,0])*np.sin(k*p[:,1])
    curvature=np.stack((diagonal,diagonal,-2*k*k*Wb*np.cos(k*p[:,0])*np.cos(k*p[:,1])),1)
    C=D*np.array([[1,.3,0],[.3,1,0],[0,0,.35]]);exact=6*curvature@C.T/.01**2
    stress_error=float(np.max(np.linalg.norm(np.array(c['top_stress'])-exact,axis=1)/np.linalg.norm(exact,axis=1)))
    plate={'scope':'nonzero nodal w and element-centre top bending stress vector within Mindlin model',
           'max_w_relative_error':werror,'max_top_stress_vector_relative_error':stress_error,
           'checked_pointwise_metrics_passed':max(werror,stress_error)<.03}
    return {'schema':'tensorfem.pointwise-audit/1.0','relative_error_limit':.03,
            'source_sha256':sources,'sphere_pressure':sphere,'hull_girder':hull,
            'cantilever':beam,'mindlin_plate':plate,'all_cases_pointwise_qualified':False,
            'note':'Published L2 passes do not certify every component at every point. Cook and hemisphere have no independent stress oracle; the membrane case is only a patch test.'}

if __name__=='__main__':
    result=audit()
    out=ROOT/'docs/benchmarks/tutorials/pointwise-audit.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
