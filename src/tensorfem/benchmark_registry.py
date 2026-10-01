"""Traceable executable benchmark registry."""
from dataclasses import asdict, dataclass
import math
import torch

@dataclass(frozen=True)
class BenchmarkEvidence:
    id:str; capability:str; quantity:str; unit:str; source:str
    computed:float; reference:float; error:float; tolerance:float; passed:bool
    def to_dict(self): return asdict(self)

def _evidence(id,capability,quantity,unit,source,computed,reference,tolerance=.03):
    if not source.strip(): raise ValueError("benchmark source is required")
    if reference == 0: raise ValueError("zero-reference checks must be registered as invariants")
    if not 0 < tolerance <= .03: raise ValueError("tolerance must be in (0, 3%]")
    error=abs(float(computed)-float(reference))/abs(float(reference))
    return BenchmarkEvidence(id,capability,quantity,unit,source,float(computed),float(reference),error,tolerance,error<tolerance)

def run_registered_benchmarks():
    from .benchmarks import run_single_bar
    from .frame_benchmarks import cantilever_tip_load
    from .continuum_benchmarks import run_cook_membrane
    from .advanced_beam import exact_tip_deflection,solve_timoshenko_cantilever
    from .advanced_buckling import column_buckling,exact_critical_load
    from .dynamics import newmark_linear
    from .solid3d_benchmarks import longitudinal_bar_frequency_benchmark
    from .plate import kirchhoff_sine_center_deflection,solve_simply_supported_sine_square
    from .nonlinear_truss import two_bar_shallow_arch_reaction
    from .contact import solve_rigid_plane_contact
    out=[]
    r=run_single_bar(); out.append(_evidence("truss.bar","truss","tip displacement","m","u=PL/(EA)",r.computed,r.reference))
    r,e=cantilever_tip_load(); out.append(_evidence("frame.cantilever","frame","tip displacement","m","Euler-Bernoulli v=PL^3/(3EI)",r.displacement[-2],e[0]))
    r=run_cook_membrane(16); out.append(_evidence("continuum.cook","Q4 continuum",r.quantity,"length","Simo & Rifai (1990), Int. J. Numer. Meth. Eng., DOI:10.1002/nme.1620290802; value 23.96",r.computed,r.reference))
    E,nu,b,h,L,P=210e9,.3,.1,.2,.4,-1e4; G=E/(2*(1+nu)); A=b*h; I=b*h**3/12
    out.append(_evidence("beam.timoshenko","Timoshenko beam","deep-beam tip displacement","m","Timoshenko bending-plus-shear solution",solve_timoshenko_cantilever(L,4,E,I,G,A,P).displacement[-2],exact_tip_deflection(L,E,I,G,A,P)))
    L,EI=3.,1.2e6; out.append(_evidence("buckling.euler","buckling","critical load","N","Euler effective-length solution",column_buckling(L,16,EI,"fixed-pinned").load_factors[0],exact_critical_load(L,EI,"fixed-pinned")))
    w=7.; T=2*math.pi/w; dt=T/100; t=torch.arange(0.,T+dt/2,dt,dtype=torch.float64); M=torch.tensor([[2.]],dtype=torch.float64)
    r=newmark_linear(M,M*w*w,torch.zeros((len(t),1),dtype=torch.float64),t,torch.tensor([.02]),torch.tensor([0.])); out.append(_evidence("dynamics.newmark","dynamics","one-period displacement","m","SDOF u=u0 cos(omega*t)",r.displacement[-1,0],.02))
    c,r,_=longitudinal_bar_frequency_benchmark(6); out.append(_evidence("solid.hex8","HEX8 solid","axial frequency","rad/s","omega=pi/(2L)sqrt(E/rho)",c,r))
    n=16; r=solve_simply_supported_sine_square(n,young=1e7,poisson=.3,thickness=.01,q0=1.); i=(n//2)*(n+1)+n//2
    out.append(_evidence("plate.mindlin","Mindlin plate","centre deflection","m","Navier/Kirchhoff sine-load solution",r.displacement[3*i],kirchhoff_sine_center_deflection(1.,1e7,.3,.01,1.)))
    a,h,E,A=1.,.2,2000.,.01; v=torch.linspace(0.,.35,10001,dtype=torch.float64); y=h/math.sqrt(3); ref=E*A*(h*h-y*y)*y/(a*a+h*h)**1.5
    out.append(_evidence("nonlinear.arch","nonlinear truss","limit load","force","Two-bar arch closed form",two_bar_shallow_arch_reaction(a,h,E,A,v).max(),ref))
    r=solve_rigid_plane_contact(torch.tensor([[1000.]],dtype=torch.float64),torch.tensor([-20.],dtype=torch.float64),torch.tensor([[1.]],dtype=torch.float64),torch.tensor([.01],dtype=torch.float64),method="penalty",penalty=1e5)
    out.append(_evidence("contact.spring","contact","reaction","N","Closed-form penalty spring",r.contact_force[0],1e5*10/101000.))
    from .modal import cantilever_beam_modes,cantilever_exact_angular_frequency
    c=cantilever_beam_modes(1.,8,210e9*8e-8,7850.*4e-4,1).angular_frequencies[0]; r=cantilever_exact_angular_frequency(1.,210e9*8e-8,7850.*4e-4)
    out.append(_evidence("modal.cantilever","beam modal","first angular frequency","rad/s","Euler-Bernoulli cantilever beta1=1.875104",c,r))
    from .shell4 import shell4_stiffness
    x=torch.tensor([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.]],dtype=torch.float64); E,nu,t=70e9,.25,.03; k=shell4_stiffness(x,E,nu,t); ex,ey,g=1.2e-4,-.4e-4,.7e-4; q=torch.zeros(24,dtype=torch.float64); q.reshape(4,6)[:,0]=ex*x[:,0]+.5*g*x[:,1]; q.reshape(4,6)[:,1]=ey*x[:,1]+.5*g*x[:,0]; s=torch.tensor([ex,ey,g],dtype=torch.float64); C=E/(1-nu**2)*torch.tensor([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]],dtype=torch.float64)
    out.append(_evidence("shell.patch_energy","flat shell","constant-strain energy","J","Plane-stress constant-strain patch analytical energy",.5*torch.dot(q,torch.mv(k,q)),.5*torch.dot(s,torch.mv(C,s))*t*2.))
    from .cohesive import BilinearCohesiveLaw
    law=BilinearCohesiveLaw(1000.,10.,.5); openings=torch.linspace(0.,law.failure_opening,10001,dtype=torch.float64); energy=torch.trapezoid(law.envelope_traction(openings),openings)
    out.append(_evidence("cohesive.fracture_energy","cohesive law","traction-separation area","energy/area","Bilinear cohesive triangular-area identity",energy,law.fracture_energy))
    from .plasticity import J2State,update_j2
    z=torch.zeros((3,3),dtype=torch.float64); stress,state,_=update_j2(torch.diag(torch.tensor([.004,-.002,-.002],dtype=torch.float64)),210000.,.3,300.,2000.,J2State(z,torch.tensor(0.,dtype=torch.float64))); dev=stress-torch.trace(stress)/3*torch.eye(3,dtype=torch.float64); seq=torch.sqrt(1.5*torch.sum(dev*dev)); radius=300.+2000.*state.alpha
    out.append(_evidence("plasticity.j2","J2 plasticity","yield-surface equivalent stress","stress","J2 radial-return consistency equation",seq,radius))
    return tuple(out)

def verification_report():
    r=run_registered_benchmarks(); return {"schema_version":1,"policy":"error < tolerance <= 0.03; zero references use invariants","passed":all(x.passed for x in r),"summary":{"total":len(r),"passed":sum(x.passed for x in r)},"results":[x.to_dict() for x in r]}
