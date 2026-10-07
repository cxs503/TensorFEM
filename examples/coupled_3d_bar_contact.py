"""Reproducible geometric/material/contact coupling demonstration."""
from tensorfem.coupled_3d_demo import (
    coupled_3d_problem, coupled_3d_response, default_coupled_3d_model,
    monotonic_reference_displacement,
)
from tensorfem.nonlinear_step import solve_adaptive

model=default_coupled_3d_model()
problem,initial=coupled_3d_problem(model)
result=solve_adaptive(problem,initial,initial_increment=.1,
                      maximum_increment=.2,tolerance=1e-11)
response=coupled_3d_response(model,result)
reference=monotonic_reference_displacement(model,model.reference_force)
print(f"displacement       {float(result.displacement[0]):.12g}")
print(f"independent oracle {reference:.12g}")
print(f"relative error     {abs(float(result.displacement[0])-reference)/reference:.6e}")
print(f"equilibrium error  {abs(float(response.residual))/model.reference_force:.6e}")
print(f"plastic alpha      {float(result.material_state.plastic.alpha):.12g}")
print(f"contact active     {result.material_state.contact.active}")
print(f"accepted increments {len(result.history)}")

