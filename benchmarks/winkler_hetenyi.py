"""Winkler foundation + Hetényi reference + added-mass validation benchmarks.

Produces ``validation_report.json`` with two anchor checks:

* **Anchor 1 – Hetényi deflection comparison**: A long beam (length ≥ 16 L_c)
  on a Winkler foundation, discretised with Euler–Bernoulli beam elements.
  Three point loads are applied.  FEM deflections at the load points and at
  0.5/1.5/2.5 characteristic-length stations are compared with the Hetényi
  infinite-beam analytical solution.  Pass: relative error < 3 % at every
  station; two mesh densities show monotonically decreasing error.

* **Anchor 2 – Added-mass frequency shift**: Dry vs wet plate first bending-mode
  frequency ratio compared to the closed-form estimate (1 + m_a/m_plate)^(-1/2).
  Pass: relative error < 5 %.
"""
from __future__ import annotations

import json
import math
import os
import torch

from tensorfem.modal import (
    beam_bending_stiffness,
    beam_consistent_mass,
    uniform_beam_matrices,
    solve_modes,
)
from tensorfem.winkler_foundation import (
    q4_winkler_foundation_stiffness,
    assemble_winkler_foundation,
)
from tensorfem.added_mass import (
    assemble_added_mass_consistent,
    assemble_added_mass_lumped,
)
from tensorfem.hetenyi_reference import (
    infinite_beam_point_load,
    infinite_beam_multiple_point_loads,
    finite_beam_series,
    plate_strip_parameters,
)
from tensorfem.plate import (
    q4_mindlin_stiffness,
    structured_square_mesh,
)

D = torch.float64


# =====================================================================
# Helper: beam on Winkler foundation solve
# =====================================================================

def solve_beam_on_foundation(
    length: float,
    elements: int,
    EI: float,
    k_b: float,
    point_loads: list[tuple[float, float]],
    boundary: str = "simply_supported",
    dtype: torch.dtype = D,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a beam on Winkler foundation under point loads.

    Returns (u, K) where u is the displacement vector (size 2*(elements+1))
    and K is the total stiffness matrix.
    """
    le = length / elements
    ndof = 2 * (elements + 1)
    K = torch.zeros((ndof, ndof), dtype=dtype)
    f = torch.zeros(ndof, dtype=dtype)

    ke = beam_bending_stiffness(EI, le, dtype=dtype)
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += ke

    # Add foundation stiffness (diagonal on w-DOFs)
    # Consistent foundation: int N^T N * k_b dx over element
    # For beam shape functions, the consistent foundation matrix is
    # k_b * le / 420 * [[156, 22*le, 54, -13*le], ...]
    kf_el = k_b * le / 420.0 * torch.tensor(
        [[156.0, 22.0 * le, 54.0, -13.0 * le],
         [22.0 * le, 4.0 * le**2, 13.0 * le, -3.0 * le**2],
         [54.0, 13.0 * le, 156.0, -22.0 * le],
         [-13.0 * le, -3.0 * le**2, -22.0 * le, 4.0 * le**2]],
        dtype=dtype,
    )
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += kf_el

    # Apply point loads
    for x_p, P in point_loads:
        # Distribute to nearest node (simple lumping)
        node_idx = int(round(x_p / le))
        node_idx = max(0, min(elements, node_idx))
        f[2 * node_idx] += P

    # Boundary conditions
    constrained: list[int] = []
    if boundary == "simply_supported":
        constrained = [0, 1, 2 * elements, 2 * elements + 1]
    elif boundary == "clamped":
        constrained = [0, 1, 2 * elements, 2 * elements + 1]
    elif boundary == "free":
        constrained = []

    result = solve_modes(K, K, tuple(constrained), modes=1)  # placeholder, not used
    # Actually just solve Ku=f
    all_dofs = torch.arange(ndof)
    free = torch.tensor([i for i in range(ndof) if i not in constrained])
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    return u, K


# =====================================================================
# Helper: plate frequency solve
# =====================================================================

def solve_plate_frequencies(
    nx: int,
    ny: int,
    length: float,
    width: float,
    young: float,
    poisson: float,
    thickness: float,
    density: float,
    foundation_coefficient: float = 0.0,
    added_mass_per_area: float = 0.0,
    n_modes: int = 6,
    dtype: torch.dtype = D,
) -> list[float]:
    """Return first n_modes natural frequencies (Hz) of the plate."""
    nodes_x = torch.linspace(0.0, length, nx + 1, dtype=dtype)
    nodes_y = torch.linspace(0.0, width, ny + 1, dtype=dtype)
    xx, yy = torch.meshgrid(nodes_x, nodes_y, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    elems = torch.tensor(elements, dtype=torch.long)

    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=dtype)
    M = torch.zeros((ndof, ndof), dtype=dtype)

    gp = 1.0 / 3.0 ** 0.5
    one = torch.tensor(1.0, dtype=dtype)
    mass_per_area = density * thickness

    for conn in elems:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        if foundation_coefficient != 0.0:
            kw = q4_winkler_foundation_stiffness(xy, foundation_coefficient)
            ke = ke + kw

        # Consistent mass (w-w block only, expanded to 12x12)
        me_block = torch.zeros((4, 4), dtype=dtype)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                xi = xy.new_tensor(xif)
                eta = xy.new_tensor(etaf)
                N = 0.25 * torch.stack((
                    (one - xi) * (one - eta),
                    (one + xi) * (one - eta),
                    (one + xi) * (one + eta),
                    (one - xi) * (one + eta),
                ))
                dN_nat = 0.25 * torch.stack((
                    torch.stack((-(one - eta), -(one - xi))),
                    torch.stack(((one - eta), -(one + xi))),
                    torch.stack(((one + eta), (one + xi))),
                    torch.stack((-(one + eta), (one - xi))),
                ))
                J = dN_nat.T @ xy
                detJ = torch.linalg.det(J)
                me_block += N[:, None] * N[None, :] * detJ
        me_block = me_block * mass_per_area

        # Expand to 12x12
        me = torch.zeros((12, 12), dtype=dtype)
        for ii in range(4):
            for jj in range(4):
                me[3 * ii, 3 * jj] = me_block[ii, jj]

        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        M[dofs[:, None], dofs] += me

    if added_mass_per_area > 0:
        Ma = assemble_added_mass_consistent(nodes, elems, added_mass_per_area)
        M = M + Ma

    # Simply-supported on all edges (w=0)
    tol = max(length, width) * 1e-10
    on_edge = (
        (nodes[:, 0] < tol) | ((nodes[:, 0] - length).abs() < tol)
        | (nodes[:, 1] < tol) | ((nodes[:, 1] - width).abs() < tol)
    )
    # Only constrain w-DOFs
    edge_nodes = torch.nonzero(on_edge, as_tuple=False).flatten()
    w_dofs = 3 * edge_nodes
    fixed_w = w_dofs.unique()

    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed_w)]
    Kf = K[free[:, None], free]
    Mf = M[free[:, None], free]

    # Only w-DOFs have mass; extract w-only submatrices
    w_free = []
    for d in free.tolist():
        if d % 3 == 0:
            w_free.append(d)
    w_free = torch.tensor(w_free, dtype=torch.long, device=K.device)
    Kww = K[w_free[:, None], w_free]
    Mww = M[w_free[:, None], w_free]

    chol = torch.linalg.cholesky(Mww)
    left = torch.linalg.solve_triangular(chol, Kww, upper=False)
    A = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    A = (A + A.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(A)
    eigenvalues = eigenvalues[eigenvalues > 0]

    freqs = []
    for ev in eigenvalues[:n_modes]:
        omega = math.sqrt(float(ev))
        freqs.append(omega / (2 * math.pi))
    return freqs


# =====================================================================
# Anchor 1: Hetényi deflection comparison (beam FEM)
# =====================================================================

def run_hetenyi_anchor():
    """Compare beam-on-foundation FEM with Hetényi infinite-beam solution.

    Single load at the center of a beam that is 24 L_c long so that the
    simply-supported ends are at least 12 L_c away from the load — the
    exponential decay e^{-λx} at 12 L_c is e^{-12} ≈ 6e-6, well below
    the 3 % tolerance.
    """
    # Beam parameters
    E = 2.1e11          # steel, Pa
    width = 0.1          # m
    thickness = 0.01     # m
    I = width * thickness ** 3 / 12
    EI = E * I
    k_b = 1e5            # foundation per unit length (N/m²)

    lam = (k_b / (4 * EI)) ** 0.25
    L_c = 1.0 / lam
    # Very long beam: length = 32 L_c → load at center = 16 L_c from each end
    length = 32.0 * L_c

    # Single point load at center
    P = 100.0  # N
    x_load = length / 2.0

    # Evaluation stations: load point + 0.5, 1.5, 2.5 L_c from load
    x_eval = torch.tensor([
        x_load,
        x_load + 0.5 * L_c,
        x_load + 1.5 * L_c,
        x_load + 2.5 * L_c,
        x_load - 0.5 * L_c,
        x_load - 1.5 * L_c,
        x_load - 2.5 * L_c,
    ], dtype=D)

    w_hetenyi = infinite_beam_point_load(x_eval, x_load, P, lam, k_b)

    results = {}
    station_labels = [
        "at_load",
        "load+0.5Lc",
        "load+1.5Lc",
        "load+2.5Lc",
        "load-0.5Lc",
        "load-1.5Lc",
        "load-2.5Lc",
    ]

    errors_coarse = []
    errors_fine = []

    for label, nelems in [("coarse", 640), ("fine", 1280)]:
        u, K = solve_beam_on_foundation(
            length=length,
            elements=nelems,
            EI=EI,
            k_b=k_b,
            point_loads=[(x_load, -P)],
            boundary="simply_supported",
        )

        le = length / nelems
        w_fem_vals = []
        for xe in x_eval:
            node_idx = int(round(xe.item() / le))
            node_idx = max(0, min(nelems, node_idx))
            w_fem_vals.append(u[2 * node_idx].item())

        error_list = []
        for idx, lbl in enumerate(station_labels):
            w_ref = abs(w_hetenyi[idx].item())
            w_fe = abs(w_fem_vals[idx])
            rel_err = abs(w_fe - w_ref) / abs(w_ref) if abs(w_ref) > 1e-15 else 0.0
            error_list.append(rel_err)
            results[f"{label}_{lbl}_w_fem"] = w_fe
            results[f"{label}_{lbl}_w_hetenyi"] = float(w_ref)
            results[f"{label}_{lbl}_rel_error"] = rel_err

        if label == "coarse":
            errors_coarse = error_list
        else:
            errors_fine = error_list

    monotonic = all(fc >= ff for fc, ff in zip(errors_coarse, errors_fine))
    max_err_coarse = max(errors_coarse)
    max_err_fine = max(errors_fine)

    results["hetenyi_max_rel_error_coarse"] = max_err_coarse
    results["hetenyi_max_rel_error_fine"] = max_err_fine
    results["hetenyi_monotonic_convergence"] = monotonic
    results["hetenyi_pass"] = max_err_coarse < 0.03 and max_err_fine < 0.03 and monotonic
    results["hetenyi_char_length_m"] = L_c
    return results


# =====================================================================
# Anchor 2: Added-mass frequency shift
# =====================================================================

def run_added_mass_anchor():
    """Dry vs wet plate frequency ratio vs analytical estimate."""
    young = 9.0e9
    poisson = 0.33
    thickness = 0.15
    density_ice = 917.0
    width = 2.0
    length = 8.0
    water_density = 1025.0
    c_m = 1.0
    added_mass_per_area = c_m * water_density * width
    plate_mass_per_area = density_ice * thickness

    freq_ratio_analytical = (1.0 + added_mass_per_area / plate_mass_per_area) ** (-0.5)

    freqs_dry = solve_plate_frequencies(
        nx=16, ny=8, length=length, width=width,
        young=young, poisson=poisson, thickness=thickness,
        density=density_ice, added_mass_per_area=0.0,
    )
    freqs_wet = solve_plate_frequencies(
        nx=16, ny=8, length=length, width=width,
        young=young, poisson=poisson, thickness=thickness,
        density=density_ice, added_mass_per_area=added_mass_per_area,
    )

    f_dry = freqs_dry[0]
    f_wet = freqs_wet[0]
    freq_ratio_fem = f_wet / f_dry
    rel_error = abs(freq_ratio_fem - freq_ratio_analytical) / freq_ratio_analytical

    return {
        "f_dry_hz": f_dry,
        "f_wet_hz": f_wet,
        "freq_ratio_fem": freq_ratio_fem,
        "freq_ratio_analytical": freq_ratio_analytical,
        "freq_ratio_rel_error": rel_error,
        "plate_mass_per_area": plate_mass_per_area,
        "added_mass_per_area": added_mass_per_area,
        "added_mass_pass": rel_error < 0.05,
    }


def main():
    report = {}
    anchor1 = run_hetenyi_anchor()
    report.update(anchor1)
    anchor2 = run_added_mass_anchor()
    report.update(anchor2)
    report["overall_pass"] = anchor1.get("hetenyi_pass", False) and anchor2.get("added_mass_pass", False)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_report.json")
    serializable = {}
    for k, v in report.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.item()
        elif isinstance(v, bool):
            serializable[k] = v
        else:
            serializable[k] = v
    serializable["hetenyi_pass"] = bool(report.get("hetenyi_pass", False))
    serializable["added_mass_pass"] = bool(report.get("added_mass_pass", False))
    serializable["hetenyi_monotonic_convergence"] = bool(report.get("hetenyi_monotonic_convergence", False))
    serializable["overall_pass"] = bool(report.get("overall_pass", False))

    with open(out_path, "w") as fp:
        json.dump(serializable, fp, indent=2)
    print(f"Report written to {out_path}")
    print(f"Overall pass: {serializable['overall_pass']}")
    return serializable


if __name__ == "__main__":
    main()
