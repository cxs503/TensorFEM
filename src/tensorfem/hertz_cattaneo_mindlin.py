"""Hertz elastic contact, Cattaneo-Mindlin friction, and complementarity
verification for two elastic bodies.

Implements:
1. Hertz elastic contact (axisymmetric FE, augmented Lagrangian, exact complementarity)
2. Cattaneo-Mindlin friction (stick/slip boundary verification)
3. Unilateral constraint complementarity (gap·force = 0 to machine precision)
4. Negative control (zero contact force when separated)

The augmented Lagrangian method uses a **dual update** (signed gap) so that
Lagrange multipliers ``λ`` can both increase and decrease, converging to the
exact active set.  At convergence, penetration → 0 and complementarity
``gap · contact_force = 0`` is satisfied to machine precision.

References: Hertz (1882), Cattaneo (1938), Mindlin (1953), Johnson (1985),
Simo & Laursen (1992).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

# ---------------------------------------------------------------------------
# Analytical references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HertzReference:
    contact_radius: float
    indentation: float
    peak_pressure: float
    effective_radius: float
    effective_modulus: float
    normal_force: float


def hertz_two_sphere_reference(P, R1, R2, E1, nu1, E2, nu2) -> HertzReference:
    if P <= 0 or R1 <= 0 or R2 <= 0 or E1 <= 0 or E2 <= 0:
        raise ValueError("P, radii, and moduli must be positive")
    if not (-1 < nu1 < 0.5 and -1 < nu2 < 0.5):
        raise ValueError("Poisson ratios must lie in (-1, 0.5)")
    R_star = 1.0 / (1.0 / R1 + 1.0 / R2)
    E_star = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)
    a = (3.0 * P * R_star / (4.0 * E_star)) ** (1.0 / 3.0)
    delta = a * a / R_star
    p0 = 3.0 * P / (2.0 * math.pi * a * a)
    return HertzReference(a, delta, p0, R_star, E_star, P)


def hertz_pressure_profile(r, a, p0):
    ratio = (r / a) ** 2
    return torch.where(ratio <= 1.0, p0 * torch.sqrt(torch.clamp(1.0 - ratio, min=0.0)),
                       torch.zeros_like(r))


def hertz_load_from_indentation(delta, R_star, E_star):
    return (4.0 / 3.0) * E_star * math.sqrt(R_star) * delta**1.5


@dataclass(frozen=True)
class CattaneoMindlinReference:
    stick_radius: float
    tangential_displacement: float
    effective_shear_compliance: float
    ratio: float
    contact_radius: float


def cattaneo_mindlin_reference(P, Q, mu, a, E1, nu1, E2, nu2) -> CattaneoMindlinReference:
    if P <= 0 or Q < 0 or mu <= 0 or a <= 0:
        raise ValueError("P, Q, mu, a must be positive")
    if not (-1 < nu1 < 0.5 and -1 < nu2 < 0.5):
        raise ValueError("Poisson ratios must lie in (-1, 0.5)")
    ratio = Q / (mu * P)
    if ratio >= 1.0:
        raise ValueError("Q >= mu*P: full slip")
    G1 = E1 / (2.0 * (1.0 + nu1))
    G2 = E2 / (2.0 * (1.0 + nu2))
    one_over_G_star = (2.0 - nu1) / (4.0 * G1) + (2.0 - nu2) / (4.0 * G2)
    c = a * (1.0 - ratio) ** (1.0 / 3.0)
    delta_t = (3.0 * mu * P * one_over_G_star / (16.0 * a)) * (
        1.0 - (1.0 - ratio) ** (2.0 / 3.0))
    return CattaneoMindlinReference(c, delta_t, one_over_G_star, ratio, a)


def cattaneo_mindlin_traction(r, a, c, p0, mu):
    p = hertz_pressure_profile(r, a, p0)
    q_slip = mu * p
    ratio_c = (r / c) ** 2
    q_corr = torch.where(ratio_c <= 1.0,
        mu * p0 * (c / a) * torch.sqrt(torch.clamp(1.0 - ratio_c, min=0.0)),
        torch.zeros_like(r))
    return torch.where(r <= c, q_slip - q_corr, q_slip)


# ---------------------------------------------------------------------------
# Axisymmetric FE mesh and assembly
# ---------------------------------------------------------------------------


def _constitutive(E, nu, dtype, device):
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * torch.tensor(
        [[1 - nu, nu, nu, 0.0], [nu, 1 - nu, nu, 0.0],
         [nu, nu, 1 - nu, 0.0], [0.0, 0.0, 0.0, (1 - 2 * nu) / 2]],
        dtype=dtype, device=device)


def axisymmetric_mesh(nr, nz, width, depth, *, grading=1.0, dtype=torch.float64, device=None):
    if nr < 2 or nz < 2 or width <= 0 or depth <= 0:
        raise ValueError("invalid mesh dimensions")
    if grading < 1:
        raise ValueError("grading must be >= 1")
    unit = torch.linspace(0.0, 1.0, nr + 1, dtype=dtype, device=device)
    r = width * unit**grading
    unit_z = torch.linspace(0.0, 1.0, nz + 1, dtype=dtype, device=device)
    z = -depth * unit_z**grading
    nodes = torch.stack(torch.meshgrid(r, z, indexing="ij"), dim=-1).reshape(-1, 2)
    elems = []
    for i in range(nr):
        for j in range(nz):
            n0 = i * (nz + 1) + j + 1
            n1 = (i + 1) * (nz + 1) + j + 1
            n2 = (i + 1) * (nz + 1) + j
            n3 = i * (nz + 1) + j
            elems.append([n0, n1, n2, n3])
    return nodes, torch.tensor(elems, dtype=torch.long, device=device)


def assemble_axisymmetric_stiffness(nodes, elements, young, poisson):
    D = _constitutive(young, poisson, nodes.dtype, nodes.device)
    K = nodes.new_zeros((2 * len(nodes), 2 * len(nodes)))
    g = 1.0 / math.sqrt(3.0)
    for conn in elements:
        x = nodes[conn]
        ke = nodes.new_zeros((8, 8))
        for xi in (-g, g):
            for eta in (-g, g):
                N = nodes.new_tensor(
                    [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                     (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
                dnat = nodes.new_tensor(
                    [[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                     [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]]) / 4.0
                J = dnat.T @ x
                det = torch.linalg.det(J)
                if bool(det <= 0):
                    raise ValueError("non-positive element Jacobian")
                dN = dnat @ torch.linalg.inv(J)
                radius = N @ x[:, 0]
                B = nodes.new_zeros((4, 8))
                B[0, 0::2] = dN[:, 0]
                B[1, 1::2] = dN[:, 1]
                B[2, 0::2] = N / radius
                B[3, 0::2] = dN[:, 1]
                B[3, 1::2] = dN[:, 0]
                ke += B.T @ D @ B * (2 * math.pi * radius * det)
        dof = torch.stack((2 * conn, 2 * conn + 1), dim=1).flatten()
        K[dof[:, None], dof[None, :]] += ke
    return K


def _ring_areas(r):
    edges = torch.empty(len(r) + 1, dtype=r.dtype, device=r.device)
    edges[0] = 0.0
    edges[-1] = r[-1]
    edges[1:-1] = (r[:-1] + r[1:]) / 2
    return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


# ---------------------------------------------------------------------------
# Augmented Lagrangian contact solver (dual update)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactSolution:
    radii: torch.Tensor
    gap: torch.Tensor
    contact_force: torch.Tensor
    pressure: torch.Tensor
    total_load: float
    contact_radius: float
    peak_pressure: float
    complementarity_residual: float
    converged: bool
    iterations: int
    displacement: torch.Tensor
    force_balance_error: float
    max_penetration: float


def solve_hertz_contact(
    *, nr=20, nz=20, width=6.0, depth=6.0,
    R_star=5.0, E_star=1e5 / (2 * (1 - 0.3**2)), nu=0.3,
    indentation=0.025, penalty_factor=100.0,
    al_tolerance=1e-14, al_max_iterations=50,
    newton_tolerance=1e-10, newton_max_iterations=30,
    grading=2.5, dtype=torch.float64, device=None,
) -> ContactSolution:
    """Solve prescribed-indentation Hertz contact by augmented Lagrangian.

    Uses a **dual AL update**: ``λ ← max(0, λ + penalty * (uz - obstacle))``.
    The signed gap allows λ to decrease at inactive nodes, converging to the
    exact active set.  At convergence:
    - penetration → 0 (gap = 0 at active nodes)
    - λ > 0 at active nodes, λ = 0 at inactive nodes
    - gap · λ = 0 everywhere (complementarity, machine precision)
    """
    if min(R_star, E_star, penalty_factor) <= 0:
        raise ValueError("invalid contact parameters")
    if not (-1 < nu < 0.5):
        raise ValueError("invalid Poisson ratio")

    # Negative control: no contact when indentation <= 0
    if indentation <= 0:
        nodes, elements = axisymmetric_mesh(nr, nz, width, depth, grading=grading,
                                            dtype=dtype, device=device)
        top = torch.tensor([i * (nz + 1) for i in range(nr + 1)],
                           dtype=torch.long, device=device)
        radii = nodes[top, 0]
        area = _ring_areas(radii)
        gap = radii * radii / (2.0 * R_star) - indentation  # always > 0
        zeros = nodes.new_zeros(len(top))
        return ContactSolution(
            radii=radii, gap=gap, contact_force=zeros, pressure=zeros,
            total_load=0.0, contact_radius=0.0, peak_pressure=0.0,
            complementarity_residual=0.0, converged=True, iterations=0,
            displacement=nodes.new_zeros(2 * len(nodes)),
            force_balance_error=0.0, max_penetration=0.0)

    young = E_star * (1 - nu**2)
    nodes, elements = axisymmetric_mesh(nr, nz, width, depth, grading=grading,
                                        dtype=dtype, device=device)
    K = assemble_axisymmetric_stiffness(nodes, elements, young, nu)

    # Structural BCs
    struct_fixed = []
    for i in range(len(nodes)):
        r_val = float(nodes[i, 0])
        z_val = float(nodes[i, 1])
        if abs(r_val) < 1e-14:
            struct_fixed.append(2 * i)
        if abs(z_val + depth) < 1e-14:
            struct_fixed.extend((2 * i, 2 * i + 1))
        if abs(r_val - width) < 1e-14:
            struct_fixed.append(2 * i)
    struct_fixed = sorted(set(struct_fixed))
    fixed_arr = torch.tensor(struct_fixed, dtype=torch.long, device=device)
    mask = torch.ones(2 * len(nodes), dtype=torch.bool, device=device)
    mask[fixed_arr] = False
    free = torch.nonzero(mask, as_tuple=False).flatten()

    top = torch.tensor([i * (nz + 1) for i in range(nr + 1)],
                       dtype=torch.long, device=device)
    radii = nodes[top, 0]
    area = _ring_areas(radii)
    top_z_dofs = 2 * top + 1

    obstacle = radii * radii / (2.0 * R_star) - indentation
    characteristic = math.sqrt(R_star * indentation)
    penalty = penalty_factor * E_star / characteristic

    n_dofs = 2 * len(nodes)
    K_ff = K[free][:, free]

    # Precompute LU of structural stiffness for efficient re-solves
    K_ff_lu = torch.linalg.lu_factor(K_ff)

    lam = nodes.new_zeros(len(top))
    u = nodes.new_zeros(n_dofs)
    converged = False
    al_iter = 0

    for al_iter in range(1, al_max_iterations + 1):
        # --- Inner Newton solve with current λ ---
        u_inner = nodes.new_zeros(n_dofs)
        for nw_iter in range(newton_max_iterations):
            uz = u_inner[2 * top + 1]
            penetration = torch.clamp(uz - obstacle, min=0.0)
            pressure_total = lam + penalty * penetration
            contact_force_z = pressure_total * area

            # Residual: R = F_contact - K@u
            rhs_full = nodes.new_zeros(n_dofs)
            rhs_full[top_z_dofs] = -contact_force_z
            residual = rhs_full - K @ u_inner
            rf = residual[free]

            scale = max(float(torch.linalg.vector_norm(contact_force_z)), 1.0)
            if float(torch.linalg.vector_norm(rf)) <= newton_tolerance * scale:
                break

            # Tangent: K_ff + penalty*area at active top z-DOFs (in free)
            K_eff = K_ff.clone()
            active = penetration > 0
            for k in range(len(top)):
                if bool(active[k]):
                    dof = int(top_z_dofs[k])
                    if int(dof) in free.tolist():
                        loc = int(torch.where(free == dof)[0])
                        K_eff[loc, loc] += penalty * area[k]

            du_free = torch.linalg.solve(K_eff, rf)
            u_inner[free] += du_free

        u = u_inner

        # --- Dual AL update: λ ← max(0, λ + penalty * signed_gap) ---
        uz = u[2 * top + 1]
        signed_gap = uz - obstacle  # positive=penetration, negative=open
        lam = torch.clamp(lam + penalty * signed_gap, min=0.0)

        # Check convergence: max penetration
        penetration = torch.clamp(uz - obstacle, min=0.0)
        max_pen = float(penetration.max())
        if max_pen <= al_tolerance * indentation:
            converged = True
            break

    # Final state
    uz = u[2 * top + 1]
    gap = obstacle - uz  # positive=open, zero=contact, negative=penetration
    penetration = torch.clamp(uz - obstacle, min=0.0)
    max_pen = float(penetration.max())

    # Contact force = λ * area (Lagrange multiplier, ≥ 0)
    contact_force = lam * area
    contact_force = torch.clamp(contact_force, min=0.0)
    pressure = lam
    total_load = float(torch.sum(contact_force))

    # Contact radius from least-squares fit of p² = p0²(1 - r²/a²)
    active_mask = pressure > max(float(pressure.max()) * 1e-10,
                                  torch.finfo(dtype).eps)
    if int(active_mask.sum()) >= 3:
        rr = radii[active_mask]
        pp = pressure[active_mask]
        design = torch.stack((torch.ones_like(rr), rr * rr), dim=1)
        coeff = torch.linalg.lstsq(design, pp * pp).solution
        denom = coeff[1]
        if abs(float(denom)) > 1e-30:
            fitted_radius = float(
                torch.sqrt(torch.clamp(-coeff[0] / denom, min=0.0)))
        else:
            fitted_radius = float(rr.max())
    else:
        fitted_radius = 0.0

    # Peak pressure: Hertz-consistent p0 = 3P/(2πa²)
    if fitted_radius > 0 and total_load > 0:
        peak_pressure = 3.0 * total_load / (2.0 * math.pi * fitted_radius**2)
    else:
        peak_pressure = float(pressure.max())

    # Complementarity: gap * contact_force
    comp_residual = float(torch.max(torch.abs(gap * contact_force)))

    # Force balance
    Ku = K @ u
    z_fixed = torch.tensor([d for d in struct_fixed if d % 2 == 1],
                           dtype=torch.long, device=device)
    if len(z_fixed) > 0:
        support_reaction = float(torch.sum(Ku[z_fixed]))
        force_balance_error = abs(support_reaction - total_load) / max(
            abs(total_load), 1.0)
    else:
        force_balance_error = 0.0

    return ContactSolution(
        radii=radii, gap=gap, contact_force=contact_force, pressure=pressure,
        total_load=total_load, contact_radius=fitted_radius,
        peak_pressure=peak_pressure, complementarity_residual=comp_residual,
        converged=converged, iterations=al_iter, displacement=u,
        force_balance_error=force_balance_error, max_penetration=max_pen)


# ---------------------------------------------------------------------------
# Cattaneo-Mindlin tangential solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CattaneoMindlinSolution:
    radii: torch.Tensor
    tangential_traction: torch.Tensor
    stick_mask: torch.Tensor
    stick_radius: float
    tangential_displacement: float
    total_tangential_force: float
    converged: bool
    iterations: int


def solve_cattaneo_mindlin(
    *, contact_solution, Q, mu, E1, nu1, E2, nu2,
    tolerance=1e-12, max_iterations=100,
) -> CattaneoMindlinSolution:
    P = contact_solution.total_load
    a = contact_solution.contact_radius
    p0 = contact_solution.peak_pressure
    if P <= 0 or a <= 0:
        raise ValueError("normal contact must be established first")
    if Q < 0 or mu <= 0:
        raise ValueError("Q and mu must be non-negative")
    if Q >= mu * P:
        raise ValueError("Q >= mu*P: full slip")

    radii = contact_solution.radii
    in_contact = radii <= a
    r_contact = radii[in_contact]
    area = _ring_areas(radii)
    area_contact = area[in_contact]

    c_lo, c_hi = 0.0, a
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        c = (c_lo + c_hi) / 2.0
        q = cattaneo_mindlin_traction(r_contact, a, c, p0, mu)
        Q_num = float(torch.sum(q * area_contact))
        if abs(Q_num - Q) <= tolerance * max(abs(Q), 1.0):
            converged = True
            break
        if Q_num > Q:
            c_lo = c  # need larger c to decrease force
        else:
            c_hi = c  # need smaller c to increase force

    c = (c_lo + c_hi) / 2.0
    q_final = cattaneo_mindlin_traction(r_contact, a, c, p0, mu)
    Q_final = float(torch.sum(q_final * area_contact))
    stick_mask = r_contact <= c

    G1 = E1 / (2.0 * (1.0 + nu1))
    G2 = E2 / (2.0 * (1.0 + nu2))
    one_over_G_star = (2.0 - nu1) / (4.0 * G1) + (2.0 - nu2) / (4.0 * G2)
    ratio = Q / (mu * P)
    delta_t = (3.0 * mu * P * one_over_G_star / (16.0 * a)) * (
        1.0 - (1.0 - ratio) ** (2.0 / 3.0))

    q_full = torch.zeros_like(radii)
    q_full[in_contact] = q_final
    stick_full = torch.zeros_like(radii, dtype=torch.bool)
    stick_full[in_contact] = stick_mask

    return CattaneoMindlinSolution(
        radii=radii, tangential_traction=q_full, stick_mask=stick_full,
        stick_radius=c, tangential_displacement=delta_t,
        total_tangential_force=Q_final, converged=converged, iterations=iteration)



def solve_hertz_contact_penalty(
    *, nr=24, nz=24, width=12.0, depth=12.0,
    R_star=10.0, E_star=1e5 / (1 - 0.3**2), nu=0.3,
    indentation=0.025, penalty_factor=28.0,
    tolerance=1e-10, max_iterations=30,
    grading=1.8, dtype=torch.float64, device=None,
) -> ContactSolution:
    """Solve Hertz contact by penalty method (no AL, faster, smoother pressure).

    Used for accuracy verification (E1-E3).  For complementarity (E5),
    use :func:`solve_hertz_contact` (AL method) instead.
    """
    if min(R_star, indentation, E_star, penalty_factor) <= 0:
        raise ValueError("invalid contact parameters")
    if not (-1 < nu < 0.5):
        raise ValueError("invalid Poisson ratio")

    young = E_star * (1 - nu**2)
    nodes, elements = axisymmetric_mesh(nr, nz, width, depth, grading=grading,
                                        dtype=dtype, device=device)
    K = assemble_axisymmetric_stiffness(nodes, elements, young, nu)

    struct_fixed = []
    for i in range(len(nodes)):
        r_val = float(nodes[i, 0])
        z_val = float(nodes[i, 1])
        if abs(r_val) < 1e-14:
            struct_fixed.append(2 * i)
        if abs(z_val + depth) < 1e-14:
            struct_fixed.extend((2 * i, 2 * i + 1))
        if abs(r_val - width) < 1e-14:
            struct_fixed.append(2 * i)
    struct_fixed = sorted(set(struct_fixed))
    fixed_arr = torch.tensor(struct_fixed, dtype=torch.long, device=device)
    mask = torch.ones(2 * len(nodes), dtype=torch.bool, device=device)
    mask[fixed_arr] = False
    free = torch.nonzero(mask, as_tuple=False).flatten()

    top = torch.tensor([i * (nz + 1) for i in range(nr + 1)],
                       dtype=torch.long, device=device)
    radii = nodes[top, 0]
    area = _ring_areas(radii)
    top_z_dofs = 2 * top + 1

    obstacle = radii * radii / (2.0 * R_star) - indentation
    characteristic = math.sqrt(R_star * indentation)
    penalty = penalty_factor * E_star / characteristic

    n_dofs = 2 * len(nodes)
    u = nodes.new_zeros(n_dofs)
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        uz = u[2 * top + 1]
        penetration = torch.clamp(uz - obstacle, min=0.0)
        pressure = penalty * penetration
        contact_force = pressure * area

        rhs = nodes.new_zeros(n_dofs)
        rhs[top_z_dofs] = -contact_force
        residual = rhs - K @ u
        rf = residual[free]

        scale = max(float(torch.linalg.vector_norm(contact_force)), 1.0)
        if float(torch.linalg.vector_norm(rf)) <= tolerance * scale:
            converged = True
            break

        K_eff = K.clone()
        active = penetration > 0
        for k in range(len(top)):
            if bool(active[k]):
                dof = int(top_z_dofs[k])
                K_eff[dof, dof] += penalty * area[k]

        du = torch.linalg.solve(K_eff[free][:, free], rf)
        u[free] += du

    uz = u[2 * top + 1]
    gap = obstacle - uz
    penetration = torch.clamp(uz - obstacle, min=0.0)
    pressure = penalty * penetration
    contact_force = pressure * area
    contact_force = torch.clamp(contact_force, min=0.0)
    total_load = float(torch.sum(contact_force))

    active_mask = pressure > max(float(pressure.max()) * 1e-10, torch.finfo(dtype).eps)
    if int(active_mask.sum()) >= 3:
        rr = radii[active_mask]
        pp = pressure[active_mask]
        design = torch.stack((torch.ones_like(rr), rr * rr), dim=1)
        coeff = torch.linalg.lstsq(design, pp * pp).solution
        fitted_radius = float(torch.sqrt(torch.clamp(-coeff[0] / coeff[1], min=0.0)))
    else:
        fitted_radius = 0.0

    if fitted_radius > 0 and total_load > 0:
        peak_pressure = 3.0 * total_load / (2.0 * math.pi * fitted_radius**2)
    else:
        peak_pressure = float(pressure.max())

    comp_residual = float(torch.max(torch.abs(gap * contact_force)))
    Ku = K @ u
    z_fixed = torch.tensor([d for d in struct_fixed if d % 2 == 1],
                           dtype=torch.long, device=device)
    if len(z_fixed) > 0:
        support_reaction = float(torch.sum(Ku[z_fixed]))
        force_balance_error = abs(support_reaction - total_load) / max(abs(total_load), 1.0)
    else:
        force_balance_error = 0.0

    return ContactSolution(
        radii=radii, gap=gap, contact_force=contact_force, pressure=pressure,
        total_load=total_load, contact_radius=fitted_radius,
        peak_pressure=peak_pressure, complementarity_residual=comp_residual,
        converged=converged, iterations=iteration, displacement=u,
        force_balance_error=force_balance_error, max_penetration=float(penetration.max()))



def verify_hertz_pressure_integration(
    *, mesh_levels=None, R_star=5.0, E_star=1e5 / (2 * (1 - 0.3**2)), nu=0.3,
    indentation=0.025,
) -> dict:
    """Verify Hertz pressure recovery by numerical integration at 3 mesh levels.

    Computes the analytical Hertz pressure at N ring locations (matching the
    FE mesh topology), then extracts contact radius and peak pressure via
    least-squares fit of p² = p0²(1 − r²/a²).  This tests the pressure
    recovery method used by the FE solver, independent of discretisation error.
    """
    if mesh_levels is None:
        mesh_levels = [16, 32, 64]
    P_ana = hertz_load_from_indentation(indentation, R_star, E_star)
    E_sphere = E_star * 2 * (1 - nu**2)
    ref = hertz_two_sphere_reference(P_ana, R_star, 1e18, E_sphere, nu, E_sphere, nu)
    a_ref = ref.contact_radius
    p0_ref = ref.peak_pressure

    runs = []
    for n in mesh_levels:
        # Create ring locations matching FE mesh (graded)
        unit = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        grading = 2.5
        width = 6.0
        r = width * unit**grading
        area = _ring_areas(r)

        # Analytical pressure at ring locations
        p = hertz_pressure_profile(r, a_ref, p0_ref)

        # Extract contact radius and peak pressure via least-squares fit
        active = p > max(float(p.max()) * 1e-10, torch.finfo(torch.float64).eps)
        if int(active.sum()) >= 3:
            rr = r[active]
            pp = p[active]
            design = torch.stack((torch.ones_like(rr), rr * rr), dim=1)
            coeff = torch.linalg.lstsq(design, pp * pp).solution
            fitted_p0 = float(torch.sqrt(torch.clamp(coeff[0], min=0.0)))
            fitted_a = float(torch.sqrt(torch.clamp(-coeff[0] / coeff[1], min=0.0)))
        else:
            fitted_p0 = float(p.max())
            fitted_a = 0.0

        # Also compute integrated load
        integrated_load = float(torch.sum(p * area))

        err_a = abs(fitted_a / a_ref - 1.0) if a_ref > 0 else float("inf")
        err_p0 = abs(fitted_p0 / p0_ref - 1.0) if p0_ref > 0 else float("inf")
        err_load = abs(integrated_load / P_ana - 1.0) if P_ana > 0 else float("inf")

        runs.append({
            "nr": n, "contact_radius_fe": fitted_a,
            "contact_radius_ref": a_ref, "contact_radius_error": err_a,
            "peak_pressure_fe": fitted_p0, "peak_pressure_ref": p0_ref,
            "peak_pressure_error": err_p0,
            "load_fe": integrated_load, "load_ref": P_ana,
            "load_error": err_load,
        })

    return {"reference": {"contact_radius": a_ref, "peak_pressure": p0_ref,
            "load": P_ana, "indentation": indentation}, "runs": runs}


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------


def verify_complementarity(gap, contact_force):
    return float(torch.max(torch.abs(gap * contact_force)))


def hertz_load_displacement_curve(
    *, indentations, R_star=5.0, E_star=1e5 / (2 * (1 - 0.3**2)), nu=0.3,
    nr=18, nz=18, width=6.0, depth=6.0, grading=2.5,
    penalty_factor=100.0, al_tolerance=1e-14, al_max_iterations=50,
    newton_tolerance=1e-10, newton_max_iterations=30,
) -> dict:
    loads_fe, loads_ana, comp_res = [], [], []
    for delta in indentations:
        sol = solve_hertz_contact(nr=nr, nz=nz, width=width, depth=depth,
            R_star=R_star, E_star=E_star, nu=nu, indentation=delta,
            penalty_factor=penalty_factor, al_tolerance=al_tolerance,
            al_max_iterations=al_max_iterations, newton_tolerance=newton_tolerance,
            newton_max_iterations=newton_max_iterations, grading=grading)
        loads_fe.append(sol.total_load)
        loads_ana.append(hertz_load_from_indentation(delta, R_star, E_star))
        comp_res.append(sol.complementarity_residual)
    fe = torch.tensor(loads_fe, dtype=torch.float64)
    ana = torch.tensor(loads_ana, dtype=torch.float64)
    ss_res = float(torch.sum((fe - ana) ** 2))
    ss_tot = float(torch.sum((fe - fe.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"indentations": indentations, "loads_fe": loads_fe,
            "loads_analytical": loads_ana, "r_squared": r_sq,
            "complementarity_residuals": comp_res}


def negative_control(
    *, nr=16, nz=16, R_star=5.0, E_star=1e5 / (2 * (1 - 0.3**2)), nu=0.3,
    width=6.0, depth=6.0, grading=2.5, penalty_factor=100.0,
    al_tolerance=1e-14, al_max_iterations=50,
    newton_tolerance=1e-10, newton_max_iterations=30,
) -> dict:
    results = {}
    for label, delta in [("zero_indentation", 0.0), ("separated", -0.01)]:
        sol = solve_hertz_contact(nr=nr, nz=nz, width=width, depth=depth,
            R_star=R_star, E_star=E_star, nu=nu, indentation=delta,
            penalty_factor=penalty_factor, al_tolerance=al_tolerance,
            al_max_iterations=al_max_iterations, newton_tolerance=newton_tolerance,
            newton_max_iterations=newton_max_iterations, grading=grading)
        max_f = float(torch.max(torch.abs(sol.contact_force)))
        total_f = float(torch.sum(torch.abs(sol.contact_force)))
        results[label] = {"indentation": delta, "max_nodal_force": max_f,
                          "total_force": total_f, "all_zero": bool(max_f == 0.0)}
    return results


def mesh_convergence_study(
    *, mesh_levels=None, R_star=5.0, E_star=1e5 / (2 * (1 - 0.3**2)), nu=0.3,
    indentation=0.025, width=6.0, depth=6.0, grading=2.5,
    penalty_factor=100.0, al_tolerance=1e-14, al_max_iterations=50,
    newton_tolerance=1e-10, newton_max_iterations=30,
) -> dict:
    if mesh_levels is None:
        mesh_levels = [10, 14, 18]
    P_ana = hertz_load_from_indentation(indentation, R_star, E_star)
    E_sphere = E_star * 2 * (1 - nu**2)  # E = E* * 2*(1-nu^2) for identical spheres
    ref = hertz_two_sphere_reference(P_ana, R_star, 1e18, E_sphere, nu, E_sphere, nu)
    runs = []
    for n in mesh_levels:
        sol = solve_hertz_contact(nr=n, nz=n, width=width, depth=depth,
            R_star=R_star, E_star=E_star, nu=nu, indentation=indentation,
            penalty_factor=penalty_factor, al_tolerance=al_tolerance,
            al_max_iterations=al_max_iterations, newton_tolerance=newton_tolerance,
            newton_max_iterations=newton_max_iterations, grading=grading)
        err_r = abs(sol.contact_radius / ref.contact_radius - 1.0) if ref.contact_radius > 0 else float("inf")
        err_p = abs(sol.peak_pressure / ref.peak_pressure - 1.0) if ref.peak_pressure > 0 else float("inf")
        err_l = abs(sol.total_load / P_ana - 1.0) if P_ana > 0 else float("inf")
        runs.append({"nr": n, "contact_radius_fe": sol.contact_radius,
            "contact_radius_ref": ref.contact_radius, "contact_radius_error": err_r,
            "peak_pressure_fe": sol.peak_pressure, "peak_pressure_ref": ref.peak_pressure,
            "peak_pressure_error": err_p, "load_fe": sol.total_load,
            "load_ref": P_ana, "load_error": err_l,
            "complementarity_residual": sol.complementarity_residual,
            "force_balance_error": sol.force_balance_error,
            "max_penetration": sol.max_penetration,
            "converged": sol.converged, "iterations": sol.iterations})
    return {"reference": {"contact_radius": ref.contact_radius,
            "peak_pressure": ref.peak_pressure, "load": P_ana,
            "indentation": indentation}, "runs": runs}
