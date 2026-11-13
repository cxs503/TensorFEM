"""Locking-free two-node Timoshenko beam bending.

Loads
-----
``solve_timoshenko_cantilever`` takes a tip transverse force (``load``) and,
optionally, a uniform transverse distributed load (``distributed_load``) whose
consistent nodal vector is ``timoshenko_fixed_end_forces``::

    [ q L / 2 , q L^2 / 12 , q L / 2 , -q L^2 / 12 ]

for dof order ``(w_i, theta_i, w_j, theta_j)`` and ``q`` positive in the same
sense as a positive ``load``.  That vector is *identical* to the Euler-Bernoulli
one: solving the exact clamped-clamped Timoshenko element problem symbolically
(``EI theta'' = V``, ``V = kappa G A (w' - theta)``, ``V' = -q``, four zero end
displacements) returns ``theta'(0) = -q L^2/(12 EI)``, ``V(0) = q L/2`` and
**no** shear-flexibility ``phi = 12 EI/(kappa G A L^2)`` in either, i.e. shear
deformation does not change the fixed-end forces of a uniformly loaded member.
The shear flexibility therefore shows up in the *response* (the extra
``q L^2/(2 kappa G A)`` tip deflection), not in the load vector -- which is what
``tests/test_cantilever_distributed_load.py`` gates.
"""
from dataclasses import dataclass
from typing import Sequence

import torch

@dataclass(frozen=True)
class TimoshenkoResult:
    displacement: torch.Tensor
    reaction: torch.Tensor

def timoshenko_stiffness(E: float, I: float, G: float, A: float, kappa: float,
                         length: float, *, dtype=torch.float64) -> torch.Tensor:
    """Exact-prismatic stiffness; shear-flexibility avoids equal-order locking."""
    if min(E, I, G, A, kappa, length) <= 0:
        raise ValueError("beam properties must be positive")
    phi = 12*E*I/(kappa*G*A*length**2)
    return E*I/(length**3*(1+phi))*torch.tensor([
        [12, 6*length, -12, 6*length],
        [6*length, (4+phi)*length**2, -6*length, (2-phi)*length**2],
        [-12, -6*length, 12, -6*length],
        [6*length, (2-phi)*length**2, -6*length, (4+phi)*length**2],
    ], dtype=dtype)


def timoshenko_fixed_end_forces(distributed_load: float, length: float, *,
                                dtype=torch.float64) -> torch.Tensor:
    """Consistent nodal loads of a uniform transverse load ``q`` on one element.

    ``q`` is positive in the same sense as a positive tip ``load`` in
    :func:`solve_timoshenko_cantilever`.  The vector is the negative of the
    clamped-clamped end forces, i.e. the work-equivalent nodal load; it carries
    no ``phi`` term (see the module docstring for the derivation).
    """
    q = float(distributed_load)
    if length <= 0:
        raise ValueError("length must be positive")
    return torch.tensor(
        [q*length/2, q*length**2/12, q*length/2, -q*length**2/12], dtype=dtype)


def _element_distributed_loads(distributed_load, elements: int,
                               dtype=torch.float64) -> torch.Tensor:
    """Per-element uniform loads, shape ``[elements]``."""
    if isinstance(distributed_load, (int, float)):
        q = torch.full((elements,), float(distributed_load), dtype=dtype)
    else:
        q = torch.as_tensor(distributed_load, dtype=dtype).reshape(-1)
        if q.numel() == 1:
            q = q.expand(elements).clone()
        elif q.numel() != elements:
            raise ValueError("distributed_load must be scalar or per-element")
    return q


def solve_timoshenko_cantilever(length: float, elements: int, E: float, I: float,
                                G: float, A: float, load: float,
                                kappa: float=5/6, distributed_load=None) -> TimoshenkoResult:
    """Clamped-free Timoshenko beam: tip force ``load`` plus optional ``q``.

    ``distributed_load`` may be a scalar (uniform over the whole span) or one
    value per element (so a partially loaded span is representable, provided the
    load boundary coincides with a node).
    """
    if elements < 1:
        raise ValueError("elements must be positive")
    ndof = 2*(elements+1)
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    le = length/elements
    ke = timoshenko_stiffness(E, I, G, A, kappa, le)
    for e in range(elements):
        dofs = torch.tensor([2*e, 2*e+1, 2*e+2, 2*e+3])
        K[dofs[:, None], dofs] += ke
    f = torch.zeros(ndof, dtype=torch.float64); f[-2] = load
    if distributed_load is not None:
        q = _element_distributed_loads(distributed_load, elements)
        fe = timoshenko_fixed_end_forces(1.0, le)*q[:, None]
        for e in range(elements):
            f[2*e:2*e+4] += fe[e]
    free = torch.arange(2, ndof)
    u = torch.zeros_like(f); u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    return TimoshenkoResult(u, K@u-f)

def exact_tip_deflection(length: float, E: float, I: float, G: float, A: float,
                         load: float, kappa: float=5/6) -> float:
    return load*(length**3/(3*E*I) + length/(kappa*G*A))


def exact_distributed_tip_deflection(length: float, q: float, E: float, I: float,
                                     G: float, A: float, kappa: float=5/6) -> float:
    """Exact Timoshenko tip deflection of a cantilever under uniform load ``q``.

    Bending term ``q L^4/(8 EI)`` plus the shear term ``q L^2/(2 kappa G A)``
    (Castigliano unit load, ``int M m/(EI) + int V v/(kappa G A)`` with
    ``m = L - x`` and unit shear ``v = 1``).
    """
    return q*length**4/(8*E*I) + q*length**2/(2*kappa*G*A)


def exact_partial_tip_deflection(length: float, a: float, q: float, E: float,
                                 I: float, G: float, A: float,
                                 kappa: float=5/6) -> float:
    """Exact tip deflection for a uniform load over ``[0, a]`` (``a <= L``).

    ``q a^3 (4 L - a)/(24 EI) + q a^2/(2 kappa G A)``.
    """
    if not 0.0 <= a <= length:
        raise ValueError("load extent a must lie in [0, L]")
    return q*a**3*(4*length - a)/(24*E*I) + q*a**2/(2*kappa*G*A)