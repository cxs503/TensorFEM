"""Independent full-span beam-field checks for a 120 m idealised hull girder.

Stress is recovered from FE Hermite curvature, not from the reference moment.
The depth axis represents beam fibres; it is not a 3-D hull shell stress field.
"""
import math
import torch
from .marine_structures import solve_hull_girder_uniform_load

PROBLEM = {'length_m': 120., 'young_modulus_pa': 210e9, 'area_m2': 5.,
           'inertia_m4': 180., 'fibre_half_depth_m': 6.,
           'still_water_load_n_m': 1.4e6, 'wave_load_n_m': .9e6}


def recover_field(case):
    p = case['problem']; n = case['mesh']['elements']
    L, E, I = (p[k] for k in ('length_m', 'young_modulus_pa', 'inertia_m4'))
    raw = torch.tensor(case['nodal_dofs'], dtype=torch.float64)
    if raw.shape != (n+1, 3) or not bool(torch.isfinite(raw).all()):
        raise ValueError('incomplete/nonfinite frame DOFs')
    le = L/n
    # Eight fixed interior samples per member: no duplicated boundary points.
    s = (torch.arange(8, dtype=torch.float64)+.5)/8
    x = (torch.arange(n, dtype=torch.float64)[:, None]+s)*le
    v0, t0, v1, t1 = [z[:, None] for z in
                      (raw[:-1, 1], raw[:-1, 2], raw[1:, 1], raw[1:, 2])]
    v = (1-3*s*s+2*s**3)*v0 + le*(s-2*s*s+s**3)*t0
    v += (3*s*s-2*s**3)*v1 + le*(-s*s+s**3)*t1
    curvature = (-6+12*s)/le**2*v0 + (-4+6*s)/le*t0
    curvature += (6-12*s)/le**2*v1 + (-2+6*s)/le*t1
    moment = E*I*curvature
    y = torch.linspace(-p['fibre_half_depth_m'], p['fibre_half_depth_m'], 25,
                       dtype=torch.float64)
    sigma = -moment.flatten()[None, :]*y[:, None]/I
    return x.flatten(), v.flatten(), moment.flatten(), y, sigma


def verify_hull_case(case):
    p = case['problem']
    if p != PROBLEM:
        raise ValueError('unexpected hull reference problem')
    n = case['mesh']['elements']
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError('even frame mesh required')
    x, v, moment, y, sigma = recover_field(case)
    L, E, I = (p[k] for k in ('length_m', 'young_modulus_pa', 'inertia_m4'))
    q = p['still_water_load_n_m']+p['wave_load_n_m']
    ref_v = -q*x*(L**3-2*L*x*x+x**3)/(24*E*I)
    ref_m = q*x*(L-x)/2
    ref_s = -ref_m[None, :]*y[:, None]/I
    def error(a, b):
        return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
    errors = {'displacement_field_l2': error(v, ref_v),
              'moment_field_l2': error(moment, ref_m),
              'sigma_x_field_l2': error(sigma, ref_s)}
    ref_mid = -5*q*L**4/(384*E*I)
    mid = float(case['nodal_dofs'][n//2][1])
    errors['midship_displacement'] = abs(mid/ref_mid-1)
    errors['moment_and_sigma_x_pointwise_max'] = float(((moment-ref_m).abs()/ref_m.abs()).max())
    reactions = case['support_reactions_n']
    if len(reactions) != 2 or not all(math.isfinite(z) for z in reactions):
        raise ValueError('invalid support reactions')
    balance = abs(sum(reactions)-q*L)/(q*L)
    # Compare stored samples with the actual DOF recovery to detect stale fields.
    field = case['field']
    for key, computed in [('x', x), ('uy', v), ('moment', moment), ('y', y),
                          ('sigma_x', sigma)]:
        stored = torch.tensor(field[key], dtype=torch.float64)
        if stored.shape != computed.shape or not torch.allclose(stored, computed,
                                                                rtol=1e-12, atol=1e-10):
            raise ValueError('stored hull field does not match FE DOFs: '+key)
    return {'scope': 'full span, Euler-Bernoulli fibres at y=-6..6 m; not shell stress',
            'errors': errors, 'force_balance_relative_error': balance,
            'passed': max(errors.values()) < .03 and balance < 1e-8}


def solve_hull_case(elements):
    p = PROBLEM.copy()
    r = solve_hull_girder_uniform_load(length=p['length_m'], young=p['young_modulus_pa'],
            area=p['area_m2'], inertia=p['inertia_m4'],
            still_water_load=p['still_water_load_n_m'], wave_load=p['wave_load_n_m'],
            elements=elements)
    case = {'problem': p, 'mesh': {'elements': elements, 'nodes': elements+1},
            'nodal_dofs': r.frame.displacement.reshape(-1, 3).tolist(),
            'element_end_forces_local': r.frame.element_end_forces_local.tolist(),
            'support_reactions_n': [float(r.frame.reaction[1]), float(r.frame.reaction[-2])]}
    x, v, m, y, sigma = recover_field(case)
    case['field'] = {k: z.tolist() for k, z in
                     [('x', x), ('uy', v), ('moment', m), ('y', y), ('sigma_x', sigma)]}
    case['relative_error'] = r.maximum_relative_error
    case['verification'] = verify_hull_case(case)
    case['status'] = 'qualified' if case['verification']['passed'] else 'blocked'
    return case


def hull_report():
    cases = [solve_hull_case(n) for n in (6, 12, 24, 48, 96)]
    checks = [c['verification'] for c in cases]
    converged = all(b['errors'][k] < a['errors'][k]
                    for a, b in zip(checks, checks[1:])
                    for k in ('displacement_field_l2', 'moment_field_l2', 'sigma_x_field_l2', 'moment_and_sigma_x_pointwise_max'))
    passed = checks[-1]['passed'] and converged
    return {'schema': 'tensorfem.hull-girder-field-report/3.0',
            'title': '120 m simply supported hull girder under combined uniform line load',
            'physical_case_id': 'hull-girder-simply-supported-uniform-load',
            'independent_case': True, 'problem': PROBLEM.copy(), 'cases': cases,
            'mesh_convergence': {'mesh_levels': len(cases), 'checks': checks,
                                 'error_decreases_or_roundoff': converged, 'passed': passed},
            'status': 'qualified' if passed else 'blocked'}
