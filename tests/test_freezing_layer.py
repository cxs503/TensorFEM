import math

import pytest
import torch

from tensorfem.freezing_layer import freeze_layer


def test_stefan_front_refines_and_energy_closes():
    # One-phase Neumann root for Ste=2100*20/334000, computed independently.
    lo, hi = 0., 1.
    for _ in range(60):
        m = (lo+hi)/2
        if math.sqrt(math.pi)*m*math.exp(m*m)*math.erf(m) < 2100*20/334000:
            lo = m
        else:
            hi = m
    ref = 2*m*math.sqrt(2.2/(1000*2100)*2.)
    errors = []
    for n in [16, 32, 64]:
        result = freeze_layer(cells=n)
        errors.append(abs(result.history[-1]["front_m"]/ref-1))
        for r in result.history:
            assert abs(r["removed_J_m2"] + r["enthalpy_change_J_m2"]) < 1e-6
        assert torch.all(result.temperature_c[1:] >= result.temperature_c[:-1]-1e-12)
    assert max(errors) < .04 and errors[-1] < .01


def test_latent_plateau_and_insulated_equilibrium():
    steady = freeze_layer(plate_c=0, cells=8, end_s=.1)
    assert steady.ice_fraction.max() == 0 and steady.history[-1]["removed_J_m2"] == 0
    cold = freeze_layer(cells=12, end_s=.1)
    coexist = (cold.ice_fraction > 0) & (cold.ice_fraction < 1)
    assert coexist.any() and torch.all(cold.temperature_c[coexist] == 0)
    with pytest.raises(ValueError):
        freeze_layer(initial_c=-2)
