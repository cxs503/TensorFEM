"""Adaptive global solve of a bilinear elastoplastic bar."""

from tensorfem.nonlinear_step import elastoplastic_bar_problem, solve_adaptive

problem, initial = elastoplastic_bar_problem(2.0, 0.01, 200000.0, 250.0, 10000.0, 4.0)
result = solve_adaptive(problem, initial, initial_increment=0.1)
print(
    {
        "load_factor": result.load_factor,
        "displacement": float(result.displacement[0]),
        "accepted_increments": len(result.history),
    }
)
