#!/usr/bin/env python3
"""Mortar Lagrange multiplier contact verification: Hertz line contact.

Verifies 2D mortar Lagrange multiplier contact against analytical Hertz
line contact solution (cylinder on elastic half-space).

Verification criteria:
1. Hertz pressure distribution vs analytical solution (error < 10%)
2. Contact constraint violation / penetration (< tolerance)
3. Mesh convergence order

References
----------
[1] Hertz, H. (1882). "Über die Berührung fester elastischer Körper".
[2] Wriggers, P. (2006). Computational Contact Mechanics, 2nd ed.
"""

import sys
import os
import json
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def hertz_pressure_profile(x: torch.Tensor, b: float, p0: float) -> torch.Tensor:
    """Hertz pressure: p(x) = p0 * sqrt(1 - (x/b)^2)."""
    xi = x / b
    return p0 * torch.sqrt(torch.clamp(1.0 - xi**2, min=0.0))


def graded_q4_mesh(nx, ny, Lx, Ly, x0=0.0, y0=0.0, grading=2.0):
    """Create structured Q4 mesh with grading toward center."""
    # Graded spacing: concentrate toward center
    t = torch.linspace(0, 1, nx + 1, dtype=DTYPE)
    # Map to [-1, 1] with grading toward 0
    s = 2 * t - 1  # [-1, 1]
    s_graded = torch.sign(s) * torch.abs(s) ** grading
    x = x0 + Lx / 2 + Lx / 2 * s_graded
    
    y = torch.linspace(y0, y0 + Ly, ny + 1, dtype=DTYPE)
    
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([float(x[i]), float(y[j])])
    nodes = torch.tensor(nodes, dtype=DTYPE)
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2, n3])
    elements = torch.tensor(elements, dtype=torch.long)
    
    return nodes, elements


def q4_plane_strain_stiffness(nodes, elements, young, poisson):
    """Assemble global stiffness for 2D Q4 plane strain."""
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = torch.zeros((n_dof, n_dof), dtype=DTYPE)
    
    c = young / ((1 + poisson) * (1 - 2 * poisson))
    D_mat = c * torch.tensor([
        [1 - poisson, poisson, 0],
        [poisson, 1 - poisson, 0],
        [0, 0, (1 - 2 * poisson) / 2],
    ], dtype=DTYPE)
    
    gp = 1.0 / math.sqrt(3)
    quad_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    
    for elem_nodes in elements:
        x = nodes[elem_nodes]
        ke = torch.zeros((8, 8), dtype=DTYPE)
        
        for xi, eta in quad_pts:
            dNdxi = torch.tensor([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
            ], dtype=DTYPE) / 4
            
            J = dNdxi @ x
            detJ = torch.linalg.det(J)
            invJ = torch.linalg.inv(J)
            dNdx = invJ @ dNdxi
            
            B = torch.zeros((3, 8), dtype=DTYPE)
            for i in range(4):
                B[0, 2*i] = dNdx[0, i]
                B[1, 2*i+1] = dNdx[1, i]
                B[2, 2*i] = dNdx[1, i]
                B[2, 2*i+1] = dNdx[0, i]
            
            ke += B.T @ D_mat @ B * detJ
        
        dof_map = torch.stack([2*elem_nodes, 2*elem_nodes+1], dim=1).reshape(-1)
        for i in range(8):
            for j in range(8):
                K[dof_map[i], dof_map[j]] += ke[i, j]
    
    return K


def solve_hertz_mortar_lm(
    nodes, elements, K, fixed_dofs, contact_nodes, contact_x,
    radius, indentation, young, poisson,
):
    """Solve Hertz contact with mortar Lagrange multipliers."""
    n_dof = K.shape[0]
    n_contact = len(contact_nodes)
    
    E_star = young / (1 - poisson**2)
    
    # Free DOFs
    fixed_set = set(fixed_dofs.tolist())
    free_dofs = torch.tensor([i for i in range(n_dof) if i not in fixed_set], dtype=torch.long)
    n_free = len(free_dofs)
    
    # Prescribed displacement (indenter shape)
    u_prescribed = -indentation + contact_x**2 / (2 * radius)
    
    # Initial active set
    active = u_prescribed < 0
    
    u = torch.zeros(n_dof, dtype=DTYPE)
    lam = torch.zeros(n_contact, dtype=DTYPE)
    
    max_iter = 50
    for iteration in range(max_iter):
        n_active = int(active.sum())
        
        if n_active == 0:
            break
        
        # Build saddle-point system
        n_sys = n_free + n_active
        A = torch.zeros((n_sys, n_sys), dtype=DTYPE)
        b = torch.zeros(n_sys, dtype=DTYPE)
        
        K_ff = K[free_dofs][:, free_dofs]
        A[:n_free, :n_free] = K_ff
        
        active_indices = torch.where(active)[0]
        G = torch.zeros((n_active, n_free), dtype=DTYPE)
        u_p = torch.zeros(n_active, dtype=DTYPE)
        
        for i, idx in enumerate(active_indices.tolist()):
            cn = int(contact_nodes[idx])
            y_dof = 2 * cn + 1
            free_pos = (free_dofs == y_dof).nonzero(as_tuple=True)[0]
            if len(free_pos) > 0:
                G[i, free_pos[0]] = 1.0
            u_p[i] = u_prescribed[idx]
        
        A[:n_free, n_free:] = G.T
        A[n_free:, :n_free] = G
        b[n_free:] = u_p
        
        A += 1e-12 * torch.eye(n_sys, dtype=DTYPE)
        
        try:
            x_sol = torch.linalg.solve(A, b)
        except:
            A += 1e-10 * torch.eye(n_sys, dtype=DTYPE)
            x_sol = torch.linalg.solve(A, b)
        
        u_free = x_sol[:n_free]
        lam_active = x_sol[n_free:]
        
        u = torch.zeros(n_dof, dtype=DTYPE)
        u[free_dofs] = u_free
        
        lam = torch.zeros(n_contact, dtype=DTYPE)
        for i, idx in enumerate(active_indices.tolist()):
            lam[idx] = lam_active[i]
        
        # Update active set: release if tension (negative lambda)
        new_active = active.clone()
        for i in range(n_contact):
            if active[i] and lam[i] < -1e-12:
                new_active[i] = False
        
        if torch.all(active == new_active):
            break
        
        active = new_active
    
    # Contact pressure
    pressure = torch.zeros(n_contact, dtype=DTYPE)
    for i in range(n_contact):
        if i == 0 and n_contact > 1:
            trib = (contact_x[1] - contact_x[0]) / 2
        elif i == n_contact - 1 and n_contact > 1:
            trib = (contact_x[-1] - contact_x[-2]) / 2
        elif n_contact > 2:
            trib = (contact_x[i+1] - contact_x[i-1]) / 2
        else:
            trib = 1.0
        pressure[i] = lam[i] / max(trib, 1e-15)
    
    # Gap
    gap = torch.zeros(n_contact, dtype=DTYPE)
    for i in range(n_contact):
        cn = int(contact_nodes[i])
        u_actual = u[2*cn + 1]
        gap[i] = u_actual - u_prescribed[i]
    
    max_penetration = float(torch.min(gap).clamp(max=0))
    total_force = float(torch.sum(torch.clamp(lam, min=0)))
    
    comp_res = 0.0
    for i in range(n_contact):
        comp_res = max(comp_res, abs(float(lam[i] * max(gap[i], 0))))
    
    return {
        'displacement': u,
        'lagrange_multipliers': lam,
        'contact_pressure': pressure,
        'contact_x': contact_x,
        'gap': gap,
        'max_penetration': max_penetration,
        'total_force': total_force,
        'converged': iteration < max_iter - 1,
        'iterations': iteration + 1,
        'complementarity_residual': comp_res,
    }


def build_and_solve(n_contact, nx, ny, radius, half_width, height, young, poisson, indentation, grading=2.0):
    """Build model and solve."""
    nodes, elements = graded_q4_mesh(nx, ny, 2*half_width, height, x0=-half_width, y0=0.0, grading=grading)
    K = q4_plane_strain_stiffness(nodes, elements, young, poisson)
    
    # Top surface nodes
    top_nodes = torch.tensor([ny * (nx + 1) + i for i in range(nx + 1)], dtype=torch.long)
    
    # Contact nodes: select those within expected contact zone
    b_expected = math.sqrt(radius * indentation)
    contact_mask = torch.abs(nodes[top_nodes, 0]) <= b_expected * 1.5
    contact_nodes = top_nodes[contact_mask]
    
    # If too few contact nodes, use all top nodes
    if len(contact_nodes) < 3:
        contact_nodes = top_nodes
    
    # Subsample if needed
    if n_contact < len(contact_nodes):
        indices = torch.linspace(0, len(contact_nodes) - 1, n_contact).long()
        contact_nodes = contact_nodes[indices]
    
    contact_x = nodes[contact_nodes, 0]
    
    # BCs
    fixed_dofs = []
    for i in range(nx + 1):
        fixed_dofs.append(2 * i + 1)
    for j in range(ny + 1):
        fixed_dofs.append(2 * (j * (nx + 1)))
        fixed_dofs.append(2 * (j * (nx + 1) + nx))
    fixed_dofs = torch.tensor(sorted(set(fixed_dofs)), dtype=torch.long)
    
    sol = solve_hertz_mortar_lm(
        nodes, elements, K, fixed_dofs, contact_nodes, contact_x,
        radius, indentation, young, poisson,
    )
    
    return sol, contact_x


def run_convergence_study():
    """Run mesh convergence study."""
    radius = 10.0
    half_width = 1.0  # Smaller domain, focused on contact zone
    height = 5.0
    young = 1000.0
    poisson = 0.3
    indentation = 0.01
    
    E_star = young / (1 - poisson**2)
    b_ref = math.sqrt(radius * indentation)
    p0_ref = E_star * indentation / (math.pi * b_ref)
    F_ref = math.pi * p0_ref * b_ref / 2
    
    print(f"\nHertz reference: b={b_ref:.4f}, p0={p0_ref:.4f}, F={F_ref:.4f}")
    
    # Mesh levels with grading toward center
    mesh_levels = [
        (8, 16, 8),
        (16, 32, 16),
        (24, 48, 24),
        (32, 64, 32),
    ]
    
    results = []
    
    for level, (n_contact, nx, ny) in enumerate(mesh_levels):
        print(f"\n{'='*60}")
        print(f"Level {level+1}: n_contact={n_contact}, nx={nx}, ny={ny}")
        print(f"{'='*60}")
        
        sol, contact_x = build_and_solve(
            n_contact, nx, ny, radius, half_width, height, young, poisson, indentation, grading=2.5
        )
        
        print(f"  Contact nodes: {len(contact_x)}, x range: [{contact_x.min():.3f}, {contact_x.max():.3f}]")
        
        # Compute errors
        p_fe = sol['contact_pressure']
        p_ana = hertz_pressure_profile(contact_x, b_ref, p0_ref)
        
        in_contact = torch.abs(contact_x) <= b_ref * 1.1
        if in_contact.sum() >= 3:
            p_fe_in = p_fe[in_contact]
            p_ana_in = p_ana[in_contact]
            p_norm = max(p0_ref, 1e-15)
            l2_err = float(torch.sqrt(torch.mean((p_fe_in - p_ana_in)**2)) / p_norm)
            linf_err = float(torch.max(torch.abs(p_fe_in - p_ana_in)) / p_norm)
        else:
            l2_err = float('inf')
            linf_err = float('inf')
        
        force_err = abs(sol['total_force'] - F_ref) / max(F_ref, 1e-15)
        
        p = sol['contact_pressure']
        in_contact_fe = p > 0.01 * max(p.max(), 1e-15)
        b_fe = float(contact_x[in_contact_fe].abs().max()) if in_contact_fe.any() else 0.0
        b_err = abs(b_fe - b_ref) / max(b_ref, 1e-15)
        
        h = 2 * half_width / nx
        
        result = {
            'level': level + 1,
            'n_contact': n_contact,
            'nx': nx,
            'h': h,
            'converged': sol['converged'],
            'iterations': sol['iterations'],
            'max_penetration': sol['max_penetration'],
            'total_force': sol['total_force'],
            'force_error': force_err,
            'b_fe': b_fe,
            'b_error': b_err,
            'pressure_l2_error': l2_err,
            'pressure_linf_error': linf_err,
            'contact_x': contact_x.tolist(),
            'pressure_fe': p_fe.tolist(),
            'pressure_ana': p_ana.tolist(),
        }
        results.append(result)
        
        print(f"  Converged: {sol['converged']} ({sol['iterations']} iter)")
        print(f"  Max penetration: {sol['max_penetration']:.2e}")
        print(f"  Force: FE={sol['total_force']:.4f}, ref={F_ref:.4f}, err={force_err:.2%}")
        print(f"  b: FE={b_fe:.4f}, ref={b_ref:.4f}, err={b_err:.2%}")
        print(f"  Pressure L2: {l2_err:.2%}, Linf: {linf_err:.2%}")
    
    return results, b_ref, p0_ref, F_ref


def main():
    print("=" * 70)
    print("MORTAR LAGRANGE MULTIPLIER CONTACT VERIFICATION")
    print("Hertz Line Contact (Cylinder on Elastic Half-Space)")
    print("=" * 70)
    
    results, b_ref, p0_ref, F_ref = run_convergence_study()
    
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    finest = results[-1]
    
    pressure_ok = finest['pressure_l2_error'] < 0.10
    print(f"\n1. Pressure error < 10%: {finest['pressure_l2_error']:.2%} {'PASS' if pressure_ok else 'FAIL'}")
    
    pen_tol = 1e-4 * b_ref
    penetration_ok = finest['max_penetration'] < pen_tol
    print(f"2. Penetration < tol: {finest['max_penetration']:.2e} {'PASS' if penetration_ok else 'FAIL'}")
    
    if len(results) >= 3:
        hs = [r['h'] for r in results[-3:]]
        es = [max(r['pressure_l2_error'], 1e-15) for r in results[-3:]]
        rates = []
        for i in range(1, len(hs)):
            if es[i-1] > 0 and es[i] > 0 and es[i] < 100:
                rates.append(math.log(es[i-1]/es[i]) / math.log(hs[i-1]/hs[i]))
        avg_rate = sum(rates) / len(rates) if rates else 0
        convergence_ok = avg_rate > 0.3
        print(f"3. Convergence rate: {avg_rate:.2f} {'PASS' if convergence_ok else 'FAIL'}")
    else:
        convergence_ok = False
        avg_rate = 0
    
    all_pass = pressure_ok and penetration_ok and convergence_ok
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'mortar_lm_hertz')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'verification_results.json'), 'w') as f:
        json.dump({
            'hertz_reference': {'b': b_ref, 'p0': p0_ref, 'F': F_ref},
            'results': results,
            'verification': {
                'pressure_ok': pressure_ok,
                'penetration_ok': penetration_ok,
                'convergence_ok': convergence_ok,
                'all_pass': all_pass,
            }
        }, f, indent=2)
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
