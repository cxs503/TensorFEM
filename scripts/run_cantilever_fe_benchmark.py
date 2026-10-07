"""Run real Q4 FE cantilever meshes and write fields, reactions and errors."""
import argparse,json
from pathlib import Path
import torch
from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum

def solve(nx,ny):
    L,H,T,E,P=1.,.1,.012,210e9,-100.; nodes,elements=rectangular_q4_mesh(L,H,nx,ny)
    forces=torch.zeros(2*len(nodes),dtype=torch.float64); right=torch.where(torch.isclose(nodes[:,0],torch.tensor(L,dtype=nodes.dtype)))[0]
    forces[2*right+1]=P/ny; forces[2*right[[0,-1]]+1]*=.5
    left=torch.where(nodes[:,0]==0)[0]; fixed=torch.stack((2*left,2*left+1),1).reshape(-1)
    result=solve_continuum(ContinuumModel(nodes,elements,torch.tensor(E),torch.tensor(.3),torch.tensor(T),forces,fixed))
    tip=float(result.displacement[2*right[len(right)//2]+1]); ref=P*L**3/(3*E*(T*H**3/12)); err=abs(tip/ref-1)
    centers=nodes[elements].mean(1); vm=torch.sqrt(result.stress[:,0]**2-result.stress[:,0]*result.stress[:,1]+result.stress[:,1]**2+3*result.stress[:,2]**2)
    return {"mesh":{"nx":nx,"ny":ny,"nodes":len(nodes),"elements":len(elements)},"tip_displacement":tip,"reference_displacement":ref,"relative_error":err,"status":"qualified" if err<.03 else "blocked","reaction_left_y":float(result.reaction[2*left+1].sum()),"field":{"nodes":[{"x":float(p[0]),"y":float(p[1]),"uy":float(result.displacement[2*i+1])} for i,p in enumerate(nodes)],"element_stress":[{"x":float(p[0]),"y":float(p[1]),"von_mises":float(v)} for p,v in zip(centers,vm)]}}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); cases=[solve(20,4),solve(40,8)]
    report={"schema":"tensorfem.cantilever-fe-benchmark/1.0","reference":"Euler-Bernoulli tip displacement","cases":cases,"acceptance":{"relative_error_lt":.03}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"output":str(a.output),"errors":[c['relative_error'] for c in cases],"statuses":[c['status'] for c in cases]}))
if __name__=='__main__': main()
