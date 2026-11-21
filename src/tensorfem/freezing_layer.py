"""Mass-lumped LINE2 enthalpy solution for freezing fresh water at a cold plane."""
from dataclasses import dataclass
import math

import torch

from .thermal import ThermalModel, assemble_thermal


@dataclass
class LayerResult:
    depth_m: torch.Tensor
    temperature_c: torch.Tensor
    ice_fraction: torch.Tensor
    enthalpy_J_kg: torch.Tensor
    history: list[dict]


def freeze_layer(*, cells=32, depth=.002, end_s=2., plate_c=-20., initial_c=0.,
                 rho=1000., cp_ice=2100., cp_water=4218., latent=334000.,
                 k_ice=2.2, k_water=.56, safety=.8):
    """Return a unit-area layer cooled from z=0 and insulated at z=depth.

    The melting point is 0 °C. Enthalpy is zero in liquid at the melting point
    and equals -latent in ice at that temperature. Coexistence is isothermal.
    The initial water is at or above its melting point. The fixed mass density
    excludes volume expansion. Lengths, energy and time use SI units.
    """
    scalars = (depth, end_s, plate_c, initial_c, rho, cp_ice, cp_water, latent,
               k_ice, k_water, safety)
    if (not isinstance(cells, int) or cells < 2 or not all(map(math.isfinite, scalars))
            or min(depth, end_s, rho, cp_ice, cp_water, latent, k_ice, k_water) <= 0
            or plate_c > 0 or initial_c < 0 or not 0 < safety <= 1):
        raise ValueError("invalid water layer, material properties, temperatures or step")
    dx = depth/cells
    z = (torch.arange(cells, dtype=torch.float64)+.5)*dx
    ids = torch.arange(cells)
    model = ThermalModel(z[:, None], torch.stack((ids[:-1], ids[1:]), dim=1),
                         conductivity=1., density=rho, specific_heat=1.,
                         fixed_nodes=ids[:0], fixed_temperatures=z[:0])
    matrices = assemble_thermal(model)
    mass = matrices.capacity.sum(1)
    # LINE2 spans the cell centres. Complete the two end control volumes.
    mass[[0, -1]] += rho*dx/2
    conductance = -matrices.conductivity.diagonal(offset=1)
    dt_limit = safety*rho*dx**2 / (3*max(k_ice/cp_ice, k_water/cp_water))
    steps = math.ceil(end_s/dt_limit)
    dt = end_s/steps
    h = torch.full_like(z, cp_water*initial_c)
    cold_potential = k_ice*plate_c
    removed = 0.
    history = []
    for step in range(steps+1):
        temp = h.clamp_min(0)/cp_water + (h+latent).clamp_max(0)/cp_ice
        solid = (-h/latent).clamp(0, 1)
        if step % max(1, steps//50) == 0 or step == steps:
            history.append(dict(t_s=step*dt, front_m=float((mass*solid).sum()/rho),
                                removed_J_m2=removed,
                                enthalpy_change_J_m2=float((mass*(h-cp_water*initial_c)).sum())))
        if step == steps:
            break
        # Kirchhoff potential integrates conductivity exactly across 0 °C.
        potential = k_ice*temp.clamp_max(0) + k_water*temp.clamp_min(0)
        flux = conductance*(potential[1:]-potential[:-1])
        wall = (2/dx)*(cold_potential-potential[0])
        net = torch.zeros_like(h)
        net[:-1] += flux
        net[1:] -= flux
        net[0] += wall
        h += dt*net/mass
        removed -= dt*float(wall)
    return LayerResult(z, temp, solid, h, history)
