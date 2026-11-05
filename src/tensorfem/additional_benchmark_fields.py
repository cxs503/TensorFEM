"""Real FE fields for Navier Mindlin plate and response-only Cook membrane."""
import math
import torch
from .plate import structured_square_mesh, solve_simply_supported_sine_square, _shape
from .continuum_benchmarks import cook_membrane
from .continuum import elasticity_matrix, q4_stiffness
from .sparse_core import assemble_coo, solve_sparse_static
from .stress_recovery import (
    interpolate_nodal_field,
    probe_errors,
    q4_gauss_stresses,
    recover_nodal_stress,
)


def relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


def mindlin_fields(n):
    nodes, cells = structured_square_mesh(n)
    result = solve_simply_supported_sine_square(n)
    u = result.displacement.reshape(-1, 3)
    E, nu, t, q = 1e7, .3, .01, 1.
    k = math.pi; lam = 2*k*k
    D = E*t**3/(12*(1-nu*nu)); Ds = 5/6*E/(2*(1+nu))*t
    Wb, Ws = q/(D*lam*lam), q/(Ds*lam)
    def exact(x):
        sx, sy = torch.sin(k*x[:,0]), torch.sin(k*x[:,1])
        cx, cy = torch.cos(k*x[:,0]), torch.cos(k*x[:,1])
        disp = torch.stack(((Wb+Ws)*sx*sy, -k*Wb*cx*sy, -k*Wb*sx*cy), 1)
        kap = torch.stack((k*k*Wb*sx*sy, k*k*Wb*sx*sy, -2*k*k*Wb*cx*cy), 1)
        shear = torch.stack((k*q/lam*cx*sy, k*q/lam*sx*cy), 1)
        return disp, kap, shear
    xy = nodes[cells]; centers = xy.mean(1)
    N, dn = _shape(nodes.new_tensor(0.), nodes.new_tensor(0.))
    J = torch.einsum('eia,ib->eab', xy, dn)
    grad = torch.einsum('ib,ebc->eic', dn, torch.linalg.inv(J))
    ue = u[cells]
    deriv = torch.einsum('eic,eia->eca', ue, grad)
    kap = torch.stack((deriv[:,1,0], deriv[:,2,1], deriv[:,1,1]+deriv[:,2,0]), 1)
    shear = Ds*(deriv[:,0,:]+torch.einsum('i,eic->ec', N, ue[:,:,1:]))
    C = nodes.new_tensor([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])*D
    moment = kap@C.T
    # z=+t/2 under the element's curvature convention.
    stress = 6*moment/t**2
    vm = torch.sqrt(stress[:,0]**2-stress[:,0]*stress[:,1]+stress[:,1]**2+3*stress[:,2]**2)
    eu, _, _ = exact(nodes); _, ek, eq = exact(centers)
    em = ek@C.T; es = 6*em/t**2
    evm = torch.sqrt(es[:,0]**2-es[:,0]*es[:,1]+es[:,1]**2+3*es[:,2]**2)
    errors = {'w_L2':relative(u[:,0],eu[:,0]), 'rotation_L2':relative(u[:,1:],eu[:,1:]),
              'moment_L2':relative(moment,em), 'shear_L2':relative(shear,eq),
              'top_stress_L2':relative(stress,es), 'von_mises_L2':relative(vm,evm)}
    edge = (nodes==0).any(1)|(nodes==1).any(1)
    xedge = (nodes[:,0]==0)|(nodes[:,0]==1)
    yedge = (nodes[:,1]==0)|(nodes[:,1]==1)
    fixed = torch.cat((3*torch.where(edge)[0],3*torch.where(yedge)[0]+1,3*torch.where(xedge)[0]+2)).unique()
    free = ~torch.isin(torch.arange(result.displacement.numel()),fixed)
    residual = float(torch.linalg.vector_norm(result.reaction[free])/torch.linalg.vector_norm(result.force))
    balance = float(abs(result.reaction[0::3].sum()+result.force[0::3].sum())/result.force[0::3].sum())
    errors['total_load'] = abs(float(result.force[0::3].sum())-4/math.pi**2)/(4/math.pi**2)
    return {'mesh':n,'nodes':nodes.tolist(),'elements':cells.tolist(),'displacement':u.tolist(),
            'centers':centers.tolist(),'moment':moment.tolist(),'shear':shear.tolist(),
            'top_stress':stress.tolist(),'von_mises':vm.tolist(),
            'verification':{'errors':errors,'free_residual':residual,'force_balance':balance,
                            'passed':max(errors.values())<.03 and max(residual,balance)<1e-8}}


# Cook membrane material (plane stress, unit thickness) as assembled by ``cook_membrane``.
COOK_YOUNG = 1.0
COOK_POISSON = 1.0/3.0


def cook_fields(n, reference=None, *, include_recovery=False):
    """Production Cook membrane payload: raw field plus both audited recoveries.

    ``stress``/``von_mises`` remain the raw constant-per-element field (the historical,
    fail-closed ``stress_accuracy_qualified=False`` protocol).  The payload now also
    carries the recovered nodal field:

    * ``nodal_stress`` — Gauss-point (Barlow) extrapolation plus area-weighted nodal
      averaging, i.e. a continuous single-valued stress field;
    * ``stress_recovery`` — the well-posed pointwise criterion at the reference probe
      sites when an independent reference is supplied, otherwise the field only.

    ``include_recovery=True`` additionally exposes the audited 2x2-Gauss to nodal
    stress recovery (:mod:`tensorfem.cook_stress_recovery`) as
    ``recovered_nodal_stress`` / ``recovered_von_mises`` / ``recovered_stress_protocol``;
    the default ``False`` keeps earlier callers and their hash manifests unchanged.

    Nothing here relaxes the fail-closed flag of the raw protocol: both recovered
    fields are post-processing of the same displacement solution and never certify
    the raw element-centre field.
    """
    model, probe = cook_membrane(n)
    nodes, cells = model.nodes, model.elements
    ne = len(cells)
    C = elasticity_matrix(nodes.new_ones(ne), nodes.new_full((ne,),1/3),'stress')
    ke, B = q4_stiffness(nodes[cells], C, nodes.new_ones(ne))
    edofs = (2*cells[:,:,None]+torch.arange(2)).reshape(ne,8)
    K = assemble_coo(ke,edofs,2*len(nodes))
    fixed = {int(d):0. for d in model.fixed_dofs}
    result = solve_sparse_static(K,model.forces,dirichlet=fixed,rtol=1e-11,maxiter=30000)
    u = result.displacement
    stress = (C@B@u[edofs].unsqueeze(-1)).squeeze(-1)
    vm = torch.sqrt(stress[:,0]**2-stress[:,0]*stress[:,1]+stress[:,1]**2+3*stress[:,2]**2)
    response = float(u[probe]); error = abs(response-23.96)/23.96
    gauss = q4_gauss_stresses(nodes, cells, u, COOK_YOUNG, COOK_POISSON)
    nodal = recover_nodal_stress(nodes, cells, gauss)
    recovery = {
        'method': '2x2 Gauss (Barlow) extrapolation + area-weighted nodal averaging',
        'gauss_points_per_element': 4,
        'nodal_von_mises_scale': float(
            torch.sqrt(nodal[:,0]**2-nodal[:,0]*nodal[:,1]+nodal[:,1]**2+3*nodal[:,2]**2).max()
        ),
        'single_valued_on_interfaces': True,
        'passed': None,
    }
    if reference is not None:
        points = [
            {'point': point['point'], 'von_mises_reference': point['von_mises']['extrapolated']}
            for point in reference['points']
        ]
        scale = max(point['von_mises_reference'] for point in points)
        block = probe_errors(nodes, cells, nodal, points, scale)
        recovery.update(block)
        recovery['passed'] = bool(block['worst_site_error'] < 0.03)
        recovery['criterion'] = 'worst probe-site normalised von Mises error < 3% on the recovered field'
    payload = {'mesh':n,'nodes':nodes.tolist(),'elements':cells.tolist(),
            'displacement':u.reshape(-1,2).tolist(),'centers':nodes[cells].mean(1).tolist(),
            'stress':stress.tolist(),'von_mises':vm.tolist(), 'reference':23.96,
            'nodal_stress':nodal.tolist(),'stress_recovery':recovery,
            'verification':{'response':response,'response_relative_error':error,
                            'relative_residual':result.iterative.relative_residual,
                            'passed':error<.03},'stress_accuracy_qualified':False}
    if include_recovery:
        from .cook_stress_recovery import recovered_nodal_stress
        recovered = recovered_nodal_stress(
            payload['nodes'], payload['elements'], payload['displacement'])
        payload['recovered_nodal_stress'] = recovered['field'].tolist()
        payload['recovered_von_mises'] = recovered['von_mises'].tolist()
        payload['recovered_stress_protocol'] = recovered['protocol']
    return payload


def build_additional_reports():
    plate = [mindlin_fields(n) for n in (8,16,32)]
    monotone = all(b['verification']['errors'][k]<a['verification']['errors'][k]
                   for a,b in zip(plate,plate[1:]) for k in a['verification']['errors'])
    cook = [cook_fields(n) for n in (8,16,32)]
    return {'mindlin-navier':{'physical_case_id':'mindlin-navier','cases':plate,
             'scope':'displacement, rotation, moment, shear and top bending stress against exact Mindlin solution',
             'monotone':monotone,'status':'qualified' if monotone and plate[-1]['verification']['passed'] else 'blocked'},
            'cook-membrane':{'physical_case_id':'cook-membrane','cases':cook,
             'scope':'published scalar displacement reference; stress fields have no independent accuracy oracle',
             'status':'response-qualified' if cook[-1]['verification']['passed'] and all(
                 b['verification']['response_relative_error']<a['verification']['response_relative_error']
                 for a,b in zip(cook,cook[1:])) else 'blocked'}}