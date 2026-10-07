"""Run the verified plastic-bar/contact coupled load step."""
import torch

from tensorfem.coupled_nonlinear import (
    PlasticBarContact, coupled_bar_contact_problem, coupled_response,
    monotonic_closed_form,
)
from tensorfem.nonlinear_step import solve_adaptive

model = PlasticBarContact(2., .01, 200000., 250., 10000., .004, 500., 8.)
problem, initial = coupled_bar_contact_problem(model)
result = solve_adaptive(problem, initial, initial_increment=.1,
                        maximum_increment=.2)
response = coupled_response(model, result.displacement, result.material_state)
reference = monotonic_closed_form(model, model.reference_force)
print({
    "displacement": float(result.displacement[0]),
    "reference": reference,
    "relative_error": abs(float(result.displacement[0])-reference)/reference,
    "bar_force": float(response.bar_force),
    "contact_force": float(response.contact_force),
    "plastic_alpha": float(result.material_state.alpha),
    "increments": len(result.history),
})
