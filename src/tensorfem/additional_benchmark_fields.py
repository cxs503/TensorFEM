"""Real FE fields for Navier Mindlin plate and response-only Cook membrane."""
import math
import torch
from .plate import structured_square_mesh, solve_simply_supported_sine_square, _shape
from .continuum_benchmarks import cook_membrane
from .continuum import elasticity_matrix, q4_stiffness
from .sparse_core import assemble_coo, solve_sparse_static


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


def cook_fields(n):
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
    return {'mesh':n,'nodes':nodes.tolist(),'elements':cells.tolist(),
            'displacement':u.reshape(-1,2).tolist(),'centers':nodes[cells].mean(1).tolist(),
            'stress':stress.tolist(),'von_mises':vm.tolist(), 'reference':23.96,
            'verification':{'response':response,'response_relative_error':error,
                            'relative_residual':result.iterative.relative_residual,
                            'passed':error<.03},'stress_accuracy_qualified':False}


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
