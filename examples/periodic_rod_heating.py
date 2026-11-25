"""Rod with zero end temperatures and volumetric heating sin(pi*x) cos(omega*t)."""
import math

import torch

from tensorfem.thermal import ThermalModel, solve_harmonic_thermal, solve_transient_thermal


def rod(n):
    h = 1/n
    x = torch.arange(n+1, dtype=torch.float64)*h
    nodes = torch.arange(n)
    conn = torch.stack((nodes, nodes+1), dim=1)
    fixed = torch.tensor([0, n])
    # Integrate the sinusoidal source against each linear nodal hat analytically.
    load = h*math.sin(math.pi*h/2)**2/(math.pi*h/2)**2*torch.sin(math.pi*x)
    load[fixed] = (1-math.sin(math.pi*h)/(math.pi*h))/math.pi
    model = ThermalModel(x[:, None], conn, 2., 1., 3., fixed, x.new_zeros(2))
    return model, load


def measure(n, steps):
    model, load = rod(n)
    k, capacity, omega = 2., 3., 15.
    x = model.nodes[:, 0]
    mode = torch.sin(math.pi*x)
    h = 1/n
    # Uniform LINE2 stiffness and consistent capacity eigenvalues at interior nodes.
    stiffness = 4*k*math.sin(math.pi*h/2)**2/h
    mass = capacity*h*(2+math.cos(math.pi*h))/3
    discrete = float(load[n//2])/(stiffness+1j*omega*mass)*mode
    gain = 1/(k*math.pi**2+1j*omega*capacity)
    time = torch.arange(steps+1, dtype=x.dtype)*(2*math.pi/(omega*steps))
    phase = torch.exp(1j*omega*time)
    t = solve_transient_thermal(model, time, discrete.real, .5, sparse=True,
                                nodal_heat_history=phase.real[:, None]*load)
    recovered = 2*(phase[:-1].conj() @ t.temperature[:-1].to(phase))/steps
    harmonic_model = ThermalModel(model.nodes, model.elements, k, 1., capacity, model.fixed_nodes,
                                  model.fixed_temperatures, nodal_heat=load)
    harmonic = solve_harmonic_thermal(harmonic_model, omega, sparse=True)
    torch.testing.assert_close(harmonic.temperature, discrete, atol=1e-13, rtol=0)
    # Net heat enters through each end at -k*pi*gain for unit area.
    ratio = recovered[n//2]/discrete[n//2]
    return (float((harmonic.temperature-gain*mode).abs().max()),
            float((harmonic.reaction_power[model.fixed_nodes]+k*math.pi*gain).abs().max()),
            float(abs(abs(ratio)-1)), float(abs(torch.angle(ratio))))


if __name__ == "__main__":
    for study in ("space", "time"):
        previous = None
        for level in (8, 16, 32, 64, 128):
            n, steps = (level, 32) if study == "space" else (24, level)
            values = measure(n, steps)
            errors = values[:2] if study == "space" else values[2:]
            orders = [math.log2(a/b) for a, b in zip(previous, errors)] if previous else []
            print(f"{study} n={n} steps={steps} errors={errors} orders={orders}")
            if not all(map(math.isfinite, errors)) or any(order <= 0 for order in orders):
                raise SystemExit("rod heating convergence failed")
            if level >= 64 and min(orders) < 1.9:
                raise SystemExit("finest grids must approach second order")
            previous = errors
