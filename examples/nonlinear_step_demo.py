"""Adaptive global solve of a bilinear elastoplastic bar."""
from tensorfem.nonlinear_step import elastoplastic_bar_problem, solve_adaptive

problem, initial = elastoplastic_bar_problem(2., .01, 200000., 250., 10000., 4.)
result = solve_adaptive(problem, initial, initial_increment=.1)
print({"load_factor": result.load_factor,
       "displacement": float(result.displacement[0]),
       "accepted_increments": len(result.history)})
