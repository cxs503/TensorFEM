"""Small public verification problems with closed-form reference values."""
from dataclasses import dataclass
import math

REPORT_SCHEMA = "tensorfem.standard-benchmark-report/1.0"
ERROR_GATE = 0.03

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


def benchmark_case_catalog():
    """Return portable problem specifications used by report generators.

    The catalog deliberately contains no project-specific data.  It is also the
    contract used by HTML/PDF/Word renderers: a renderer can present these
    fields without knowing anything about the solver implementation.
    """
    return {
        "cantilever_beam_tip_load": {
            "title": "Euler--Bernoulli cantilever beam",
            "problem": "Prismatic beam under a transverse tip point load.",
            "conditions": {"length_m": 1.0, "young_pa": 210e9,
                           "second_moment_m4": 1e-6, "tip_load_n": 100.0,
                           "boundary": "clamped at x=0; free at x=L",
                           "units": "SI"},
            "procedure": ["assemble beam stiffness", "apply clamp and tip load",
                          "solve linear equilibrium", "recover displacement and bending stress"],
            "result_fields": ["displacement_y", "bending_stress_xx", "root_reaction_moment"],
            "visualizations": ["displacement_contour", "bending_stress_contour",
                               "mesh_convergence_plot"],
            "reference_type": "closed_form",
        },
        "simply_supported_plate_uniform_pressure": {
            "title": "Navier simply-supported square plate",
            "problem": "Thin isotropic plate under uniform transverse pressure.",
            "conditions": {"length_a_m": 1.0, "width_b_m": 1.0,
                           "thickness_m": 0.01, "young_pa": 210e9,
                           "poisson": 0.3, "pressure_pa": 1.0,
                           "boundary": "simply supported on all four edges", "units": "SI"},
            "procedure": ["compute plate rigidity", "evaluate Navier odd-mode series",
                          "recover center deflection and bending moments"],
            "result_fields": ["transverse_displacement", "bending_stress_xx",
                              "bending_stress_yy", "twisting_stress_xy"],
            "visualizations": ["deflection_contour", "von_mises_stress_contour",
                               "mesh_convergence_plot"],
            "reference_type": "Navier_series",
        },
        "hertz_spherical_contact": {
            "title": "Hertz spherical elastic contact",
            "problem": "Elastic sphere pressed into an elastic half-space.",
            "conditions": {"radius_m": 0.01, "young_pa": 210e9,
                           "poisson": 0.3, "control": "prescribed indentation", "units": "SI"},
            "procedure": ["prescribe indentation", "evaluate Hertz contact law",
                          "recover contact force and pressure field",
                          "compare FE force and contact pressure"],
            "result_fields": ["contact_force", "contact_pressure", "contact_area",
                              "indentation", "von_mises_stress"],
            "visualizations": ["contact_pressure_contour", "von_mises_stress_contour",
                               "force_indentation_curve", "mesh_convergence_plot"],
            "reference_type": "Hertz_closed_form",
        },
    }
