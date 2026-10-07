"""Small public verification problems with closed-form reference values."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class FieldSample:
    x: float; y: float
    displacement: tuple[float, float, float]
    stress: tuple[float, float, float]

@dataclass(frozen=True)
class MeshPoint:
    elements: int; value: float; relative_error: float

def cantilever_beam(length=1.0, young=210e9, inertia=1e-6, load=100.0):
    tip=load*length**3/(3*young*inertia)
    return {"name":"cantilever_beam_tip_load","tip_displacement":tip,
            "root_moment":load*length,"reference":"Euler-Bernoulli closed form"}

def cantilever_beam_field(x,y,length=1.,young=210e9,inertia=1e-6,load=100.):
    u=load*x*x*(3*length-x)/(6*young*inertia)
    sigma=-load*(length-x)*y/inertia
    return FieldSample(x,y,(0.,u,0.),(sigma,0.,0.))

def simply_supported_plate_center(a=1.,b=1.,thickness=.01,young=210e9,poisson=.3,pressure=1.):
    d=young*thickness**3/(12*(1-poisson**2)); w=0.
    for m in range(1,80,2):
        for n in range(1,80,2):
            w += 16*pressure/(math.pi**6*d*m*n*((m/a)**2+(n/b)**2)**2)
    return {"name":"simply_supported_plate_uniform_pressure","center_deflection":w,
            "reference":"Navier thin-plate series"}

def hertz_contact_force(displacement,radius=.01,young=210e9,poisson=.3):
    if displacement<0: raise ValueError("contact displacement must be non-negative")
    return 4/3*(young/(1-poisson**2))*math.sqrt(radius)*displacement**1.5

def convergence(values,reference):
    if reference==0: raise ValueError("reference must be nonzero")
    return tuple(MeshPoint(n,float(v),abs(float(v)-reference)/abs(reference)) for n,v in values)
