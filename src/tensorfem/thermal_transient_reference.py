"""Closed-form references and verification studies for transient heat conduction.

The public transient path :func:`tensorfem.thermal.solve_transient_thermal`
(theta-method, ``C Tdot + K T = f``) was, until this module, only checked
against a *semi-discrete* single-element answer
(``T = T0*exp(-t/tau)`` with ``tau = rho*cp*L**2/(3*k)``).  That check
quantifies the consistent-mass single element, not the governing PDE, so it
cannot detect a wrong capacity assembly, a wrong element length, or a
mislabelled time-integration order.

This module supplies independent, literature-standard references and the
convergence studies that tie the solver to them:

``A``  Slab with a temperature step at both faces (Carslaw & Jaeger,
       *Conduction of Heat in Solids*, 2nd ed., Sec. 3.4)::

           T(x,t) = sum_{n odd} (4*T0/(n*pi)) * sin(n*pi*x/L)
                    * exp(-n**2*pi**2*alpha*t/L**2)

``B``  Slab with a single sine initial mode: ``T = sin(pi*x/L)*
       exp(-pi**2*alpha*t/L**2)`` -- smooth initial data with no initial
       layer, which isolates the spatial and temporal error contributions.

``C``  Adiabatic slab: with the consistent capacity matrix the discrete mean
       temperature is conserved exactly while the peak relaxes towards it.

All studies use ``float64`` and report relative ``L2`` errors against the
closed-form value evaluated at the same nodes and time, so no hand-copied
numbers enter the evidence files.

Run the full study with ``python benchmarks/thermal_transient_reference.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tensorfem.thermal import ThermalModel, solve_transient_thermal

FLOAT = torch.float64


# ---------------------------------------------------------------------------
# closed-form references
# ---------------------------------------------------------------------------
def exact_slab_step(x: torch.Tensor, t: float, length: float = 1.0,
                    alpha: float = 1.0, T0: float = 1.0,
                    n_terms: int = 400) -> torch.Tensor:
    """Exact solution of the Dirichlet step problem (Carslaw & Jaeger 3.4).

    ``T(x,0) = T0`` and ``T(0,t) = T(L,t) = 0`` for ``t > 0``.
    ``n_terms`` counts the *odd* modes retained (mode indices ``1,3,...``).
    """
    if t <= 0.0:
        raise ValueError("t must be positive for the series representation")
    if length <= 0.0 or alpha <= 0.0:
        raise ValueError("length and alpha must be positive")
    xi = torch.as_tensor(x, dtype=FLOAT)
    total = torch.zeros_like(xi)
    for k in range(n_terms):
        n = 2 * k + 1
        decay = math.exp(-(n * math.pi / length) ** 2 * alpha * t)
        if decay < 1e-300 and k > 2:
            break  # remaining modes are dead in float64
        total = total + (4.0 * T0 / (n * math.pi)) * torch.sin(n * math.pi * xi / length) * decay
    return total


def exact_slab_mode(x: torch.Tensor, t: float, length: float = 1.0,
                    alpha: float = 1.0, amplitude: float = 1.0) -> torch.Tensor:
    """Exact solution for the sine initial mode ``sin(pi*x/L)`` with zero ends."""
    xi = torch.as_tensor(x, dtype=FLOAT)
    return amplitude * torch.sin(math.pi * xi / length) * \
        math.exp(-(math.pi / length) ** 2 * alpha * t)


def exact_slab_initial_step(x: torch.Tensor, T0: float = 1.0) -> torch.Tensor:
    """Initial temperature field ``T0`` for the step problem (interior nodes)."""
    return torch.full_like(torch.as_tensor(x, dtype=FLOAT), T0)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def line_mesh(n: int, length: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``n`` equal LINE2 elements spanning ``[0, length]``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x = torch.linspace(0.0, length, n + 1, dtype=FLOAT)
    conn = torch.stack([torch.arange(n), torch.arange(1, n + 1)], dim=1)
    return x.reshape(-1, 1), conn


def q4_strip(n: int, length: float = 1.0, height: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """``n`` Q4 elements forming a strip of height ``height`` (1 element thick)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    xs = torch.linspace(0.0, length, n + 1, dtype=FLOAT)
    bottom = torch.stack([xs, torch.zeros_like(xs)], dim=1)
    top = torch.stack([xs, torch.full_like(xs, height)], dim=1)
    nodes = torch.cat([bottom, top], dim=0)
    conn = []
    for i in range(n):
        conn.append([i, i + 1, n + 1 + i + 1, n + 1 + i])
    return nodes, torch.tensor(conn, dtype=torch.long)


def _relative_l2(numeric: torch.Tensor, reference: torch.Tensor) -> float:
    ref = torch.as_tensor(reference, dtype=FLOAT)
    num = torch.as_tensor(numeric, dtype=FLOAT)
    return float(torch.linalg.vector_norm(num - ref) / torch.linalg.vector_norm(ref))


def _max_abs(numeric: torch.Tensor, reference: torch.Tensor) -> float:
    return float((torch.as_tensor(numeric, dtype=FLOAT)
                  - torch.as_tensor(reference, dtype=FLOAT)).abs().max())


def observed_order(coarse_error: float, fine_error: float, ratio: float = 2.0) -> float:
    """Log-log observed order between two refinements of factor ``ratio``."""
    if coarse_error <= 0.0 or fine_error <= 0.0:
        raise ValueError("errors must be positive")
    return math.log(coarse_error / fine_error) / math.log(ratio)


# ---------------------------------------------------------------------------
# solver wrappers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransientCase:
    """Result of one transient solve, reduced to the final-time profile."""

    final_temperature: torch.Tensor
    nodes: torch.Tensor
    n_steps: int
    dt: float


def run_slab(n: int, t_end: float, n_steps: int, theta: float,
             initial: str = "mode", element: str = "line2",
             height: float = 0.1, conductivity: float = 1.0,
             density: float = 1.0, specific_heat: float = 1.0,
             thickness: float = 1.0) -> TransientCase:
    """Solve the unit slab with both faces held at zero temperature."""
    if initial not in {"mode", "step"}:
        raise ValueError("initial must be 'mode' or 'step'")
    if element == "line2":
        nodes, conn = line_mesh(n)
        kwargs = {"thickness": thickness, "element_type": "line2"}
    elif element == "q4":
        nodes, conn = q4_strip(n, height=height)
        kwargs = {"thickness": thickness, "element_type": "q4"}
    else:
        raise ValueError("element must be 'line2' or 'q4'")
    if element == "line2":
        faces = torch.tensor([0, n], dtype=torch.long)
    else:
        # both ends, both through-thickness levels: 2-D faces, not just corners
        faces = torch.tensor([0, n, n + 1, 2 * n + 1], dtype=torch.long)
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(conductivity, dtype=FLOAT),
        density=torch.tensor(density, dtype=FLOAT),
        specific_heat=torch.tensor(specific_heat, dtype=FLOAT),
        fixed_nodes=faces,
        fixed_temperatures=torch.zeros(faces.numel(), dtype=FLOAT),
        **kwargs,
    )
    time = torch.linspace(0.0, t_end, n_steps + 1, dtype=FLOAT)
    if initial == "mode":
        t0 = exact_slab_mode(nodes[:, 0], 0.0)
    else:
        t0 = torch.ones(nodes.shape[0], dtype=FLOAT)
    t0 = t0.clone()
    t0[faces] = 0.0
    result = solve_transient_thermal(model, time, t0, theta=theta)
    return TransientCase(result.temperature[-1].clone(), nodes, n_steps, float(t_end / n_steps))


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
def study_step_profile(n: int = 32, t_end: float = 0.1, n_steps: int = 1000,
                       theta: float = 1.0) -> dict:
    """Rigid gate: Dirichlet step problem at ``Fo = alpha*t/L**2 = 0.1``."""
    case = run_slab(n=n, t_end=t_end, n_steps=n_steps, theta=theta, initial="step")
    x = case.nodes[:, 0]
    exact = exact_slab_step(x, t_end)
    return {
        "anchor": "dirichlet_step_fourier_series",
        "description": "slab, T(x,0)=1, T(0,t)=T(L,t)=0, Fo=%.2f, %d LINE2 elements, "
                       "backward Euler, %d steps" % (t_end, n, n_steps),
        "reference": "T(x,t)=sum_{n odd} 4/(n*pi)*sin(n*pi*x/L)*exp(-n^2*pi^2*alpha*t/L^2)",
        "relative_l2_error": _relative_l2(case.final_temperature, exact),
        "max_absolute_error": _max_abs(case.final_temperature, exact),
        "gate": 0.03,
        "values": {"n": n, "t_end": t_end, "n_steps": n_steps, "theta": theta},
        "profile": {"x": x.tolist(),
                    "numeric": case.final_temperature.tolist(),
                    "exact": exact.tolist()},
    }


def study_spatial_order(t_end: float = 0.1, n_steps: int = 3000,
                        theta: float = 0.5, levels=(4, 8, 16, 32)) -> dict:
    """h-refinement on the smooth sine mode; time error is orders below."""
    errors = {}
    for n in levels:
        case = run_slab(n=n, t_end=t_end, n_steps=n_steps, theta=theta, initial="mode")
        exact = exact_slab_mode(case.nodes[:, 0], t_end)
        errors[n] = _relative_l2(case.final_temperature, exact)
    orders = [observed_order(errors[a], errors[b]) for a, b in zip(levels[:-1], levels[1:])]
    return {
        "anchor": "spatial_order_sine_mode",
        "description": "sine initial mode, Fo=0.10, Crank-Nicolson (%d steps), LINE2 h-refinement"
                       % n_steps,
        "reference": "T=sin(pi*x/L)*exp(-pi^2*alpha*t/L^2)",
        "levels": list(levels),
        "errors": [errors[n] for n in levels],
        "orders": orders,
        "observed_order": orders[-1],
        "expected_order": 2.0,
    }


def study_q4_strip_consistency(t_end: float = 0.1, n_steps: int = 1000,
                               theta: float = 1.0, n: int = 16) -> dict:
    """Q4 strip must reproduce the LINE2 answer for a 1-D heat flow field."""
    line = run_slab(n=n, t_end=t_end, n_steps=n_steps, theta=theta,
                    initial="step", element="line2")
    quad = run_slab(n=n, t_end=t_end, n_steps=n_steps, theta=theta,
                    initial="step", element="q4")
    q4_bottom = quad.final_temperature[:n + 1]
    top = quad.final_temperature[n + 1:]
    return {
        "anchor": "q4_strip_matches_line2",
        "description": "same Dirichlet step problem discretised with %d Q4 elements "
                       "(1 element thick) vs %d LINE2 elements" % (n, n),
        "reference": "LINE2 solution on the same node set",
        "relative_l2_error": _relative_l2(q4_bottom, line.final_temperature),
        "through_thickness_relative_l2": _relative_l2(top, q4_bottom),
        "max_absolute_error": _max_abs(q4_bottom, line.final_temperature),
        "values": {"n": n, "n_steps": n_steps, "theta": theta},
    }


def study_temporal_order(t_end: float = 0.1, n: int = 256,
                         scheme_steps=((1.0, (4, 8, 16, 32)), (0.5, (2, 4, 8)))) -> dict:
    """dt-refinement on a fine mesh: backward Euler ~1, Crank-Nicolson ~2.

    ``scheme_steps`` pairs each theta with the step counts that stay above the
    fixed O(h^2) spatial floor (``n = 256`` gives ~1.3e-6 relative), so the
    measured slope isolates the time integration only.
    """
    out = {}
    for theta, steps in scheme_steps:
        errors = {}
        for ns in steps:
            case = run_slab(n=n, t_end=t_end, n_steps=ns, theta=theta, initial="mode")
            exact = exact_slab_mode(case.nodes[:, 0], t_end)
            errors[ns] = _relative_l2(case.final_temperature, exact)
        orders = [observed_order(errors[a], errors[b]) for a, b in zip(steps[:-1], steps[1:])]
        out["theta=%.1f" % theta] = {
            "steps": list(steps), "errors": [errors[s] for s in steps],
            "orders": orders, "observed_order": orders[-1],
            "expected_order": 1.0 if theta == 1.0 else 2.0,
        }
    return {
        "anchor": "temporal_order_theta_method",
        "description": "sine initial mode on %d LINE2 elements (spatial floor ~1e-6), "
                       "dt refinement by factor 2" % n,
        "reference": "T=sin(pi*x/L)*exp(-pi^2*alpha*t/L^2)",
        "schemes": out,
    }


def study_adiabatic_conservation(n: int = 20, t_end: float = 0.5,
                                 n_steps: int = 50, theta: float = 0.5) -> dict:
    """No flux anywhere: the discrete mean temperature must be invariant."""
    nodes, conn = line_mesh(n)
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(1.0, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=torch.tensor([], dtype=torch.long),
        fixed_temperatures=torch.tensor([], dtype=FLOAT),
        thickness=torch.tensor(1.0, dtype=FLOAT),
    )
    x = nodes[:, 0]
    t0 = 1.0 + 0.5 * torch.cos(math.pi * x)  # zero flux at both ends, non-zero mean
    time = torch.linspace(0.0, t_end, n_steps + 1, dtype=FLOAT)
    result = solve_transient_thermal(model, time, t0, theta=theta)
    cap = result.matrices.capacity
    mean0 = float((cap @ result.temperature[0]).sum() / cap.sum())
    mean_end = float((cap @ result.temperature[-1]).sum() / cap.sum())
    return {
        "anchor": "adiabatic_mean_temperature_conservation",
        "description": "insulated slab, cos(pi*x) initial field, %d LINE2 elements, "
                       "Crank-Nicolson, %d steps" % (n, n_steps),
        "reference": "1^T C T is invariant when every boundary flux is zero",
        "initial_mean": mean0,
        "final_mean": mean_end,
        "relative_drift": abs(mean_end - mean0) / abs(mean0),
        "gate": 1e-10,
    }


def study_capacity_and_stiffness_scaling(length: float = 1.0, n: int = 7,
                                         conductivity: float = 2.5, density: float = 3.0,
                                         specific_heat: float = 4.0,
                                         thickness: float = 5.0) -> dict:
    """Absolute scaling of ``C`` and the constant-temperature null space of ``K``.

    A capacity scaled by the wrong volume reproduces the *same* decay curves as
    the reference (both branches of ``C Tdot + K T = 0`` would scale together)
    only if ``K`` is scaled identically; comparing against a closed-form total
    heat capacity removes that blind spot.
    """
    from tensorfem.thermal import ThermalModel, assemble_thermal

    nodes, conn = line_mesh(n, length=length)
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(conductivity, dtype=FLOAT),
        density=torch.tensor(density, dtype=FLOAT),
        specific_heat=torch.tensor(specific_heat, dtype=FLOAT),
        fixed_nodes=torch.tensor([], dtype=torch.long),
        fixed_temperatures=torch.tensor([], dtype=FLOAT),
        thickness=torch.tensor(thickness, dtype=FLOAT),
    )
    m = assemble_thermal(model)
    total_capacity = float(m.capacity.sum())
    analytic_capacity = density * specific_heat * thickness * length
    ones = torch.ones(nodes.shape[0], dtype=FLOAT)
    return {
        "anchor": "capacity_scaling_and_null_space",
        "description": "sum of the assembled capacity matrix vs rho*cp*A*L "
                       "(conductivity %.2f, rho %.2f, cp %.2f, A %.2f, L %.2f, %d LINE2 elements)"
                       % (conductivity, density, specific_heat, thickness, length, n),
        "reference": "sum(C) = rho*cp*A*L and K @ 1 = 0",
        "total_capacity": total_capacity,
        "analytic_capacity": analytic_capacity,
        "relative_error": abs(total_capacity - analytic_capacity) / analytic_capacity,
        "null_space_max_abs": float((m.conductivity @ ones).abs().max()),
        "gate": 1e-12,
    }


def study_step_convergence(t_end: float = 0.1, n_steps: int = 2000,
                           theta: float = 1.0, levels=(8, 16, 32, 64)) -> dict:
    """h-convergence of the Dirichlet step problem against the Fourier series.

    Also reports the Richardson-extrapolated limit error, i.e. the residual
    that is *not* attributable to the spatial discretisation.
    """
    errors = {}
    for n in levels:
        case = run_slab(n=n, t_end=t_end, n_steps=n_steps, theta=theta, initial="step")
        exact = exact_slab_step(case.nodes[:, 0], t_end)
        errors[n] = _relative_l2(case.final_temperature, exact)
    orders = [observed_order(errors[a], errors[b]) for a, b in zip(levels[:-1], levels[1:])]
    p_obs = orders[-1]
    extrapolated = errors[levels[-1]] / (2.0 ** p_obs - 1.0)
    return {
        "anchor": "step_problem_h_convergence",
        "description": "Dirichlet step problem, Fo=0.10, backward Euler with %d steps, "
                       "LINE2 h-refinement" % n_steps,
        "reference": "Fourier series (Carslaw & Jaeger 3.4)",
        "levels": list(levels),
        "errors": [errors[n] for n in levels],
        "orders": orders,
        "observed_order": p_obs,
        "richardson_limit_error": extrapolated,
    }


def run_all() -> dict:
    """Run every study and return the evidence payload (JSON-serialisable)."""
    payload = {
        "schema": "tensorfem.thermal-transient-reference/1.0",
        "generated_by": "benchmarks/thermal_transient_reference.py",
        "units": {"length": "m", "conductivity": "W/(m*K)", "alpha": "m^2/s"},
        "properties": {"k": 1.0, "rho": 1.0, "cp": 1.0, "alpha": 1.0,
                       "L": 1.0, "thickness": 1.0},
        "anchors": [],
    }
    payload["anchors"].append(study_step_profile())
    payload["anchors"].append(study_step_convergence())
    payload["anchors"].append(study_spatial_order())
    payload["anchors"].append(study_temporal_order())
    payload["anchors"].append(study_q4_strip_consistency())
    payload["anchors"].append(study_adiabatic_conservation())
    payload["anchors"].append(study_capacity_and_stiffness_scaling())
    return payload


def verdicts(payload: dict) -> dict:
    """Pass/fail per anchor, derived from the measured numbers only."""
    checks = {a["anchor"]: a for a in payload["anchors"]}
    out = {}
    step = checks["dirichlet_step_fourier_series"]
    out[step["anchor"]] = {
        "criterion": "relative L2 error vs Fourier series < 3%",
        "value": step["relative_l2_error"], "passed": step["relative_l2_error"] < step["gate"],
    }
    sp = checks["spatial_order_sine_mode"]
    out[sp["anchor"]] = {
        "criterion": "observed L2 order >= 1.85 (expected 2)",
        "value": sp["observed_order"], "passed": sp["observed_order"] >= 1.85,
    }
    tm = checks["temporal_order_theta_method"]
    be = tm["schemes"]["theta=1.0"]
    cn = tm["schemes"]["theta=0.5"]
    out[tm["anchor"] + "/theta=1.0"] = {
        "criterion": "observed order in [0.85, 1.15] (expected 1)",
        "value": be["observed_order"], "passed": 0.85 <= be["observed_order"] <= 1.15,
    }
    out[tm["anchor"] + "/theta=0.5"] = {
        "criterion": "observed order in [1.8, 2.2] (expected 2)",
        "value": cn["observed_order"], "passed": 1.8 <= cn["observed_order"] <= 2.2,
    }
    q4 = checks["q4_strip_matches_line2"]
    out[q4["anchor"]] = {
        "criterion": "Q4 strip vs LINE2 relative L2 < 1e-8",
        "value": q4["relative_l2_error"], "passed": q4["relative_l2_error"] < 1e-8,
    }
    sc = checks["step_problem_h_convergence"]
    out[sc["anchor"]] = {
        "criterion": "observed L2 order >= 1.85 and Richardson limit error < 1e-4",
        "value": {"order": sc["observed_order"], "limit": sc["richardson_limit_error"]},
        "passed": sc["observed_order"] >= 1.85 and sc["richardson_limit_error"] < 1e-4,
    }
    cap = checks["capacity_scaling_and_null_space"]
    out[cap["anchor"]] = {
        "criterion": "sum(C) matches rho*cp*A*L within 1e-12 and K@1 = 0",
        "value": {"capacity": cap["relative_error"], "null": cap["null_space_max_abs"]},
        "passed": cap["relative_error"] < cap["gate"] and cap["null_space_max_abs"] < 1e-10,
    }
    ad = checks["adiabatic_mean_temperature_conservation"]
    out[ad["anchor"]] = {
        "criterion": "relative drift of mean temperature < 1e-10",
        "value": ad["relative_drift"], "passed": ad["relative_drift"] < ad["gate"],
    }
    return out


__all__ = [
    "TransientCase", "exact_slab_step", "exact_slab_mode", "exact_slab_initial_step",
    "line_mesh", "q4_strip", "observed_order", "run_slab",
    "study_step_profile", "study_spatial_order", "study_temporal_order",
    "study_q4_strip_consistency", "study_adiabatic_conservation",
    "study_capacity_and_stiffness_scaling", "study_step_convergence", "run_all", "verdicts",
]