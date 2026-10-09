"""Closed-form von Mises shallow-arch branch for arc-length qualification."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .arc_length import ArcLengthProblem,solve_arc_length


@dataclass(frozen=True)
class ArchReference:
    displacement: float
    load: float


def arch_load(w, *, half_span=1., rise=.2, axial_rigidity=1000.):
    y=rise-w;l0=math.sqrt(half_span**2+rise**2);l=torch.sqrt(w.new_tensor(half_span**2)+y*y)
    return 2*axial_rigidity/l0*(l0/l-1)*y


def arch_limit_reference(*,half_span=1.,rise=.2,axial_rigidity=1000.):
    l0=math.sqrt(half_span**2+rise**2)
    y=math.sqrt((l0*half_span**2)**(2/3)-half_span**2);w=rise-y
    l=math.sqrt(half_span**2+y*y);p=2*axial_rigidity/l0*(l0/l-1)*y
    return ArchReference(w,p)


def trace_von_mises_arch(step_size=.01,steps=35,**kwargs):
    def response(u):
        x=u.detach().clone().requires_grad_(True);p=arch_load(x[0]);k=torch.autograd.grad(p,x,create_graph=False)[0]
        return p.detach().reshape(1),k.detach().reshape(1,1)
    problem=ArcLengthProblem(response,torch.ones(1,dtype=torch.float64))
    k0=float(response(torch.zeros(1,dtype=torch.float64))[1][0,0])
    return solve_arc_length(problem,torch.zeros(1,dtype=torch.float64),steps=steps,
                            step_size=step_size,load_scale=1/k0,**kwargs)
