"""Full 3-D HEX8 hollow sphere under internal pressure, checked against Lame.

This is a thick-wall solid discretisation (inner/outer radii 8/10 m), not a
point-load thin-shell stress oracle. Every stored stress is recovered from FE
DOFs before comparison with the independent spherical elasticity solution.
"""
from __future__ import annotations

import math
import warnings
import torch
from .plate import _shape
from .solid3d import elasticity_matrix_3d, hex8_stiffness, _SIGNS
from .sparse_core import assemble_coo, solve_sparse_static, sparse_mv, conjugate_gradient, SparseStaticResult

PROBLEM = {'inner_radius_m': 8., 'outer_radius_m': 10., 'young_modulus_pa': 210e9,
           'poisson_ratio': .3, 'internal_pressure_pa': 1e6, 'external_pressure_pa': 0.}


def cube_sphere_surface(n):
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError('even positive cube-face divisions required')
    nodes, index, cells = [], {}, []
    for axis, sign, a, b in [(0, 1, 1, 2), (0, -1, 2, 1), (1, 1, 2, 0),
                              (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 1, 0)]:
        grid = []
        for j in range(n+1):
            row = []
            for i in range(n+1):
                key = [0, 0, 0]
                key[axis], key[a], key[b] = sign*n, -n+2*i, -n+2*j
                key = tuple(key)
                if key not in index:
                    x = torch.tensor(key, dtype=torch.float64)
                    index[key] = len(nodes)
                    nodes.append(x/torch.linalg.vector_norm(x))
                row.append(index[key])
            grid.append(row)
        for j in range(n):
            for i in range(n):
                cells.append([grid[j][i], grid[j][i+1], grid[j+1][i+1], grid[j+1][i]])
    return torch.stack(nodes), torch.tensor(cells, dtype=torch.long)


def sphere_volume_mesh(n, radial_layers, problem=PROBLEM):
    if not isinstance(radial_layers, int) or radial_layers < 1:
        raise ValueError('positive radial layers required')
    surface, faces = cube_sphere_surface(n)
    radii = torch.linspace(problem['inner_radius_m'], problem['outer_radius_m'],
                           radial_layers+1, dtype=torch.float64)
    nodes = (radii[:, None, None]*surface[None, :, :]).reshape(-1, 3)
    ns = len(surface)
    elements = torch.cat([torch.cat((faces+i*ns, faces+(i+1)*ns), 1)
                          for i in range(radial_layers)], 0)
    return nodes, elements, faces, surface


def lame_reference(x, problem=PROBLEM):
    a, b, E, nu, p = [problem[k] for k in ('inner_radius_m', 'outer_radius_m',
                    'young_modulus_pa', 'poisson_ratio', 'internal_pressure_pa')]
    r = torch.linalg.vector_norm(x, dim=1)
    direction = x/r[:, None]
    A = p*a**3/(b**3-a**3)
    B = p*a**3*b**3/(b**3-a**3)
    ur = ((1-2*nu)*A*r+(1+nu)*B/(2*r*r))/E
    sr, st = A-B/r**3, A+B/(2*r**3)
    stress = st[:, None, None]*torch.eye(3, dtype=x.dtype)
    stress += (sr-st)[:, None, None]*direction[:, :, None]*direction[:, None, :]
    return ur[:, None]*direction, stress, sr, st, (st-sr).abs()


def _tensor_stress(s):
    out = s.new_zeros((len(s), 3, 3))
    out[:, 0, 0], out[:, 1, 1], out[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    out[:, 0, 1] = out[:, 1, 0] = s[:, 3]
    out[:, 1, 2] = out[:, 2, 1] = s[:, 4]
    out[:, 0, 2] = out[:, 2, 0] = s[:, 5]
    return out


def _system(n, radial_layers, problem=PROBLEM):
    nodes, elements, faces, surface = sphere_volume_mesh(n, radial_layers, problem)
    ne = len(elements)
    E, nu, p = [problem[k] for k in ('young_modulus_pa', 'poisson_ratio', 'internal_pressure_pa')]
    D = elasticity_matrix_3d(nodes.new_full((ne,), E), nodes.new_full((ne,), nu))
    ke, B = hex8_stiffness(nodes[elements], D)
    edofs = (3*elements[:, :, None]+torch.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, 3*len(nodes))
    force = nodes.new_zeros(3*len(nodes))
    face_force = nodes.new_zeros((len(faces), 4, 3))
    volume = nodes.new_zeros(ne)
    x = nodes[elements]
    signs = x.new_tensor(_SIGNS)
    g = 1/math.sqrt(3)
    for xi in (-g, g):
        for eta in (-g, g):
            N, dn = _shape(nodes.new_tensor(xi), nodes.new_tensor(eta))
            tangents = torch.einsum('eia,ib->eab', nodes[faces], dn)
            normal_area = torch.linalg.cross(tangents[:, :, 0], tangents[:, :, 1], dim=1)
            face_force += p*N[None, :, None]*normal_area[:, None, :]
            for zeta in (-g, g):
                aa, bb, cc = signs[:, 0], signs[:, 1], signs[:, 2]
                gradient = torch.stack((aa*(1+bb*eta)*(1+cc*zeta),
                    bb*(1+aa*xi)*(1+cc*zeta), cc*(1+aa*xi)*(1+bb*eta)), 1)/8
                volume += torch.linalg.det(torch.einsum('eia,ib->eab', x, gradient))
    force.index_add_(0, (3*faces[:, :, None]+torch.arange(3)).flatten(), face_force.flatten())
    fixed = {}
    for axis in range(3):
        point = nodes.new_zeros(3); point[axis] = 1
        node = int(torch.linalg.vector_norm(surface-point, dim=1).argmin())
        if float(torch.linalg.vector_norm(surface[node]-point)) > 1e-12:
            raise ValueError('axis gauge point absent')
        # Six tangential constraints remove rigid motion and leave radial
        # expansion unconstrained. No analytical displacement is prescribed.
        for component in range(3):
            if component != axis:
                fixed[3*node+component] = 0.
    return nodes, elements, K, force, fixed, B, D, edofs, volume


def _metrics(nodes, elements, K, force, fixed, B, D, edofs, volume, u):
    centres = nodes[elements].mean(1)
    strain = torch.einsum('eij,ej->ei', B, u.flatten()[edofs])
    stress = torch.einsum('eij,ej->ei', D, strain)
    tensor = _tensor_stress(stress)
    ref_u = lame_reference(nodes)[0]
    _, ref_tensor, ref_sr, ref_st, ref_vm = lame_reference(centres)
    normal = centres/torch.linalg.vector_norm(centres, dim=1)[:, None]
    sr = torch.einsum('ei,eij,ej->e', normal, tensor, normal)
    st = (tensor.diagonal(dim1=1, dim2=2).sum(1)-sr)/2
    dev = tensor-tensor.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]*torch.eye(3)/3
    vm = torch.sqrt(1.5*(dev*dev).sum((1, 2)))
    def relative(a, b, weights=None):
        square_error = (a-b).reshape(len(a), -1).square().sum(1)
        square_ref = b.reshape(len(b), -1).square().sum(1)
        weight = torch.ones_like(square_ref) if weights is None else weights
        return float(torch.sqrt((weight*square_error).sum()/(weight*square_ref).sum()))
    errors = {'displacement_vector_l2': relative(u, ref_u),
              'displacement_max_relative': float((torch.linalg.vector_norm(u-ref_u, dim=1)/
                                                  torch.linalg.vector_norm(ref_u, dim=1)).max()),
              'stress_tensor_volume_l2': relative(tensor, ref_tensor, volume),
              'radial_stress_volume_l2': relative(sr, ref_sr, volume),
              'radial_stress_pointwise_max': float(((sr-ref_sr).abs()/ref_sr.abs()).max()),
              'stress_tensor_pointwise_max': float((torch.linalg.vector_norm(tensor-ref_tensor, dim=(1,2))/torch.linalg.vector_norm(ref_tensor, dim=(1,2))).max()),
              'hoop_stress_pointwise_max': float(((st-ref_st).abs()/ref_st.abs()).max()),
              'von_mises_pointwise_max': float(((vm-ref_vm).abs()/ref_vm.abs()).max()),
              'hoop_stress_volume_l2': relative(st, ref_st, volume),
              'von_mises_volume_l2': relative(vm, ref_vm, volume),
              'radial_stress_max_over_pressure': float((sr-ref_sr).abs().max()/PROBLEM['internal_pressure_pa'])}
    residual = sparse_mv(K, u.flatten())-force
    free = torch.ones_like(force, dtype=torch.bool); free[list(fixed)] = False
    scale = float(torch.linalg.vector_norm(force))
    free_residual = float(torch.linalg.vector_norm(residual[free]))/scale
    gauges = float(torch.linalg.vector_norm(residual[list(fixed)]))/scale
    applied = force.reshape(-1, 3)
    force_balance = float(torch.linalg.vector_norm(applied.sum(0)))/scale
    moment_balance = float(torch.linalg.vector_norm(torch.linalg.cross(nodes, applied, dim=1).sum(0)))/(scale*10)
    passed = max(errors.values()) < .03 and max(free_residual, gauges, force_balance, moment_balance) < 1e-8
    check = {'errors': errors, 'free_residual_relative': free_residual,
             'gauge_reaction_relative': gauges, 'applied_force_balance_relative': force_balance,
             'applied_moment_balance_relative': moment_balance, 'passed': passed,
             'scope': 'all nodes and all HEX8 centres; volume L2 and pointwise tensor/radial/hoop/von Mises errors; radial absolute max also normalized by pressure'}
    return stress, vm, sr, st, check


def solve_sphere_pressure(n, radial_layers, *, preconditioned=None):
    system = _system(n, radial_layers)
    nodes, elements, K, force, fixed, B, D, edofs, volume = system
    preconditioned = n >= 32 if preconditioned is None else preconditioned
    if preconditioned:
        # Zero-valued gauge constraints: symmetric row/column elimination,
        # with an identity block on constrained DOFs. No dense reduced matrix.
        ids = torch.tensor(list(fixed), dtype=torch.long)
        diagonal = force.new_zeros(len(force))
        index = K.indices(); ondiag = index[0] == index[1]
        diagonal.index_add_(0, index[0, ondiag], K.values()[ondiag])
        diagonal[ids] = 1.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Sparse CSR tensor support is in beta state')
            csr = K.to_sparse_csr()
        def operator(q):
            constrained = q.clone(); constrained[ids] = 0.
            result = sparse_mv(csr, constrained); result[ids] = q[ids]
            return result
        rhs = force.clone(); rhs[ids] = 0.
        iterative = conjugate_gradient(operator, rhs, diagonal=diagonal, rtol=1e-11, maxiter=20000)
        if not iterative.converged:
            raise RuntimeError('sphere PCG did not converge')
        solution = SparseStaticResult(iterative.x, sparse_mv(K, iterative.x)-force, iterative)
    else:
        solution = solve_sparse_static(K, force, dirichlet=fixed, rtol=1e-11, maxiter=20000)
    u = solution.displacement.reshape(-1, 3)
    stress, vm, sr, st, check = _metrics(*system, u)
    return {'problem': PROBLEM.copy(), 'mesh': {'face_divisions': n, 'radial_layers': radial_layers,
                'nodes': len(nodes), 'elements': len(elements)},
            'nodes': nodes.tolist(), 'elements': elements.tolist(), 'displacement': u.tolist(),
            'element_stress': stress.tolist(), 'von_mises': vm.tolist(),
            'radial_stress': sr.tolist(), 'hoop_stress': st.tolist(), 'element_volume': volume.tolist(),
            'verification': check, 'status': 'qualified' if check['passed'] else 'blocked',
            'solver': {'algorithm': 'Jacobi-PCG (CSR)' if preconditioned else 'sparse CG', 'iterations': solution.iterative.iterations,
                       'relative_residual': solution.iterative.relative_residual}}


def verify_sphere_pressure(case):
    if case.get('problem') != PROBLEM:
        raise ValueError('sphere reference specification mismatch')
    system = _system(case['mesh']['face_divisions'], case['mesh']['radial_layers'])
    nodes, elements = system[:2]
    for key, expected in [('nodes', nodes), ('elements', elements)]:
        stored = torch.tensor(case[key], dtype=expected.dtype)
        if stored.shape != expected.shape or not torch.equal(stored, expected):
            raise ValueError('sphere mesh mismatch: '+key)
    u = torch.tensor(case['displacement'], dtype=torch.float64)
    if u.shape != nodes.shape or not bool(torch.isfinite(u).all()):
        raise ValueError('incomplete/nonfinite sphere displacement')
    stress, vm, sr, st, check = _metrics(*system, u)
    for key, expected in [('element_stress', stress), ('von_mises', vm),
                          ('radial_stress', sr), ('hoop_stress', st), ('element_volume', system[-1])]:
        stored = torch.tensor(case[key], dtype=torch.float64)
        if stored.shape != expected.shape or not torch.allclose(stored, expected, rtol=1e-11, atol=1e-8):
            raise ValueError('sphere field not recovered from FE DOFs: '+key)
    if torch.linalg.vector_norm(u.flatten()[list(system[4])]) > 1e-12:
        raise ValueError('sphere gauge constraints violated')
    return check


def sphere_mesh_study(cases):
    """Retain coarse maxima; qualify convergence on the declared fine window.

    8/12 are coarse geometry trials; 20 and finer form the refinement window.
    Every L2 error must decrease across all levels. All metrics (including
    pointwise maxima) must decrease across at least three fine-window levels.
    """
    checks = [c['verification'] for c in cases]
    fine = [c for c in cases if c['mesh']['face_divisions'] >= 20]
    def decreases(rows, keys):
        return all(b['errors'][key] < a['errors'][key]
                   for a,b in zip(rows,rows[1:]) for key in keys)
    alltrend = decreases(checks, checks[0]['errors'])
    l2trend = decreases(checks, [k for k in checks[0]['errors'] if 'pointwise' not in k])
    finetrend = len(fine) >= 3 and decreases([c['verification'] for c in fine], checks[0]['errors'])
    refined = all(b['mesh']['face_divisions'] > a['mesh']['face_divisions'] and
                  b['mesh']['radial_layers'] >= a['mesh']['radial_layers']
                  for a,b in zip(cases,cases[1:]))
    passed = refined and l2trend and finetrend and checks[-1]['passed']
    return {'all_declared_errors_decrease':alltrend,
            'l2_and_original_errors_decrease_all_levels':l2trend,
            'all_declared_errors_decrease_fine_window':finetrend,
            'fine_window_face_divisions':[c['mesh']['face_divisions'] for c in fine],
            'passed':bool(passed)}


def sphere_pressure_report():
    cases = [solve_sphere_pressure(n, nr) for n, nr in ((8, 1), (12, 2), (20, 4), (32, 4), (40, 4), (64, 6))]
    study = sphere_mesh_study(cases)
    return {'schema': 'tensorfem.sphere-pressure-field-report/2.0',
            'physical_case_id': 'sphere-pressure-hex8-lame', 'cases': cases,
            'qualification_scope': 'All nodal displacement and all HEX8 centre tensor/radial/hoop/von Mises pointwise maxima plus L2 against Lame',
            'mesh_convergence': study,
            'status': 'qualified' if study['passed'] else 'blocked'}
