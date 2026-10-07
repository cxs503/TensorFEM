"""Three traceable end-to-end engineering case packages."""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
import torch
from .result_db import ResultDB,write_result_db
from .modeldb import ModelDB,ElementBlock
from .result_io import write_vtk

SCHEMA="tensorfem.engineering-case.v1"


def _pressure_vessel(p):
    from .hertz_axisymmetric import assemble_axisymmetric_stiffness
    ri,ro,L,E,nu,pressure,n=(float(p[k]) for k in ("inner_radius","outer_radius","length","young","poisson","pressure","radial_elements"));n=int(n)
    radii=torch.linspace(ri,ro,n+1,dtype=torch.float64);z=torch.tensor([0.,L],dtype=torch.float64)
    nodes=torch.stack(torch.meshgrid(radii,z,indexing="ij"),-1).reshape(-1,2)
    elements=torch.tensor([[2*i,2*i+2,2*i+3,2*i+1] for i in range(n)],dtype=torch.long)
    K=assemble_axisymmetric_stiffness(nodes,elements,E,nu);f=torch.zeros(2*len(nodes),dtype=torch.float64)
    f[0]=f[2]=-pressure*2*math.pi*ri*L/2
    fixed=torch.arange(1,2*len(nodes),2);mask=torch.ones(2*len(nodes),dtype=torch.bool);mask[fixed]=False
    free=torch.nonzero(mask).flatten();u=torch.zeros_like(f);u[free]=torch.linalg.solve(K[free[:,None],free],f[free])
    A=pressure*ri**2/(ro**2-ri**2);B=pressure*ri**2*ro**2/(ro**2-ri**2)
    reference=(1+nu)/E*((1-2*nu)*A*ri+B/ri);computed=-float(u[0])
    return nodes,elements,u.reshape(-1,2),computed,reference,"Lame plane-strain thick-cylinder inner displacement"


def _thin_panel(p):
    from .shell4 import shell4_stiffness
    L,W,E,nu,t,force=float(p["length"]),float(p["width"]),float(p["young"]),float(p["poisson"]),float(p["thickness"]),float(p["force"])
    nx,ny=int(p["nx"]),int(p["ny"]);xs=torch.linspace(0,L,nx+1);ys=torch.linspace(0,W,ny+1)
    xx,yy=torch.meshgrid(xs,ys,indexing="ij");nodes=torch.stack((xx,yy,torch.zeros_like(xx)),-1).reshape(-1,3).double()
    elements=torch.tensor([[i*(ny+1)+j,(i+1)*(ny+1)+j,(i+1)*(ny+1)+j+1,i*(ny+1)+j+1] for i in range(nx) for j in range(ny)])
    nd=6*len(nodes);K=torch.zeros((nd,nd),dtype=torch.float64);f=torch.zeros(nd,dtype=torch.float64)
    for el in elements:
        ids=torch.stack(tuple(6*el+k for k in range(6)),1).reshape(-1);K[ids[:,None],ids]+=shell4_stiffness(nodes[el],E,nu,t)
    right=torch.arange(nx*(ny+1),(nx+1)*(ny+1));f[6*right]=force/ny;f[6*right[[0,-1]]]*=.5
    fixed=[]
    for i in range(len(nodes)):fixed += [6*i+2,6*i+3,6*i+4]
    left=torch.arange(ny+1);fixed += (6*left).tolist();fixed += [1,5]
    fixed=torch.tensor(sorted(set(fixed)));mask=torch.ones(nd,dtype=torch.bool);mask[fixed]=False;free=torch.nonzero(mask).flatten()
    u=torch.zeros(nd,dtype=torch.float64);u[free]=torch.linalg.solve(K[free[:,None],free],f[free])
    computed=float(u.reshape(-1,6)[right,0].mean());reference=force*L/(E*t*W)
    return nodes,elements,u.reshape(-1,6)[:,:3],computed,reference,"uniform membrane panel u=FL/(EtW)"


def _contact_connector(p):
    from .contact import solve_rigid_plane_contact
    k1,k2,kc,F,g=(float(p[k]) for k in ("ground_stiffness_1","ground_stiffness_2","coupling_stiffness","compressive_force","clearance"))
    K=torch.tensor([[k1+kc,-kc],[-kc,k2+kc]],dtype=torch.float64);f=torch.tensor([-F,0.],dtype=torch.float64)
    C=torch.tensor([[1.,0.]],dtype=torch.float64);offset=torch.tensor([g],dtype=torch.float64)
    r=solve_rigid_plane_contact(K,f,C,offset,method="active_set")
    effective=k1+kc-kc*kc/(k2+kc);reference=F-effective*g;computed=float(r.contact_force[0])
    nodes=torch.tensor([[0.],[1.]],dtype=torch.float64);elements=torch.tensor([[0,1]])
    return nodes,elements,r.displacement[:,None],computed,reference,"two-spring connector Schur-complement contact reaction"


SOLVERS={"pressurized_thick_cylinder":_pressure_vessel,"thin_membrane_panel":_thin_panel,
         "contact_connector":_contact_connector}


def load_case(path):
    data=json.loads(Path(path).read_text())
    if data.get("schema")!=SCHEMA or set(data)!={"schema","name","kind","parameters","units"}:raise ValueError("invalid engineering case input")
    if data["kind"] not in SOLVERS:raise ValueError("unsupported engineering case")
    return data


def _digest(case):return hashlib.sha256(json.dumps(case,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20]


def run_case(case_or_path,root,*,replay=True,restart=True,checkpoint_only=False):
    case=load_case(case_or_path) if isinstance(case_or_path,(str,Path)) else case_or_path
    if case.get("schema")!=SCHEMA or case.get("kind") not in SOLVERS:raise ValueError("invalid engineering case")
    jid=_digest(case);d=Path(root)/jid;d.mkdir(parents=True,exist_ok=True)
    manifest=d/"input.json";job=d/"job.json";checkpoint=d/"checkpoint.json";report=d/"validation.json"
    canonical=json.dumps(case,sort_keys=True,indent=2)+"\n"
    if manifest.exists() and manifest.read_text()!=canonical:raise RuntimeError("case manifest collision")
    manifest.write_text(canonical);start=time.perf_counter()
    if replay and report.exists() and job.exists():
        status=json.loads(job.read_text());raw=report.read_bytes()
        if status.get("status")=="completed":
            if hashlib.sha256(raw).hexdigest()!=status.get("report_sha256"):raise RuntimeError("validation report checksum mismatch")
            out=json.loads(raw);out["execution"]["replayed"]=True;return out
    restarted=False
    if restart and checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved.get("job_id")!=jid:raise RuntimeError("checkpoint job mismatch")
        checksum=saved.pop("checkpoint_sha256",None)
        if checksum!=hashlib.sha256(json.dumps(saved,sort_keys=True,separators=(",",":")).encode()).hexdigest():
            raise RuntimeError("checkpoint checksum mismatch")
        nodes=torch.tensor(saved["nodes"],dtype=torch.float64);elements=torch.tensor(saved["elements"],dtype=torch.long)
        field=torch.tensor(saved["field"],dtype=torch.float64);computed=saved["computed"];reference=saved["reference"];source=saved["source"];restarted=True
    else:
        nodes,elements,field,computed,reference,source=SOLVERS[case["kind"]](case["parameters"])
        saved={"schema":"tensorfem.engineering-checkpoint.v1","job_id":jid,"nodes":nodes.tolist(),"elements":elements.tolist(),
               "field":field.tolist(),"computed":computed,"reference":reference,"source":source}
        saved["checkpoint_sha256"]=hashlib.sha256(json.dumps(saved,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        checkpoint.write_text(json.dumps(saved,sort_keys=True,indent=2)+"\n")
    if checkpoint_only:
        job.write_text(json.dumps({"status":"checkpointed","job_id":jid},indent=2));return {"job_id":jid,"checkpointed":True}
    error=abs(computed/reference-1);passed=error<.03
    db=ResultDB(torch.arange(len(nodes)),torch.arange(len(elements)),elements,
        node_fields={"coordinates":nodes,"primary_result":field},histories={"computed":torch.tensor([computed]),
        "reference":torch.tensor([reference]),"relative_error":torch.tensor([error])},
        model_metadata={"case":case["kind"],"units":case["units"]},job_metadata={"job_id":jid,"passed":passed})
    write_result_db(d/"results",db)
    element_type="T2D2" if case["kind"]=="contact_connector" else "CPS4"
    model=ModelDB(nodes={i+1:tuple(x) for i,x in enumerate(nodes.tolist())},
        elements=[ElementBlock(element_type,list(range(1,len(elements)+1)),
            [[int(i)+1 for i in conn] for conn in elements.tolist()],"CASE")])
    vtk_field=field[:,0] if field.ndim==2 and field.shape[1]==1 else field
    write_vtk(d/"results.vtk",model,point_data={"primary_result":vtk_field})
    out={"schema":"tensorfem.engineering-report.v1","job_id":jid,"case":case["kind"],"source":source,
         "computed":computed,"reference":reference,"relative_error":error,"tolerance":.03,"passed":passed,
         "mesh":{"nodes":len(nodes),"elements":len(elements)},"execution":{"replayed":False,"restarted":restarted,"seconds":time.perf_counter()-start}}
    raw=(json.dumps(out,sort_keys=True,indent=2)+"\n").encode();report.write_bytes(raw)
    (d/"validation.md").write_text(f"# {case['name']}\n\n- Computed: {computed:.9g}\n- Reference: {reference:.9g}\n- Relative error: {100*error:.4f}%\n- Source: {source}\n- Result: **{'PASS' if passed else 'FAIL'}**\n")
    job.write_text(json.dumps({"status":"completed","job_id":jid,"report_sha256":hashlib.sha256(raw).hexdigest()},indent=2))
    return out
