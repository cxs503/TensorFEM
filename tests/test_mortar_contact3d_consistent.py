import torch

from tensorfem.mortar_contact3d import assemble_mortar_contact


D = torch.float64


def patch(z):
    x=torch.tensor([[0.,0.,z],[1.,0.,z],[1.,1.,z],[0.,1.,z]],dtype=D)
    return x,torch.tensor([[0,1,2,3]],dtype=torch.long)


def triangle(z):
    x=torch.tensor([[0.,0.,z],[1.,0.,z],[0.,1.,z]],dtype=D)
    return x,torch.tensor([[0,1,2]],dtype=torch.long)


def assembly(us=None,um=None,kn=2.e5):
    slave,sf=patch(-1.e-3); master,mf=patch(0.)
    us=torch.zeros_like(slave) if us is None else us
    um=torch.zeros_like(master) if um is None else um
    return assemble_mortar_contact(slave,sf,master,mf,us,um,normal_penalty=kn)


def test_two_sided_residual_tangent_and_rigid_body_balance():
    a=assembly()
    force=torch.cat((a.slave_forces,a.master_forces))
    points=torch.cat((patch(-1.e-3)[0],patch(0.)[0]))
    assert torch.linalg.vector_norm(force.sum(0)) < 1.e-11
    assert torch.linalg.vector_norm(torch.linalg.cross(points,force).sum(0)) < 1.e-11
    assert torch.allclose(a.tangent,a.tangent.T,atol=2.e-8,rtol=2.e-10)
    assert a.active_quadrature_points==4

    # Exact active-branch linearisation against an independent centered
    # difference of the assembled residual.
    # A normal closing mode stays within the same facet-search branch (the
    # coplanar QUAD4 diagonal is deliberately a non-smooth search boundary).
    direction=torch.zeros((8,3),dtype=D)
    direction[:4,2]=-1.; direction[4:,2]=1.
    direction=direction/torch.linalg.vector_norm(direction)
    h=1.e-7
    plus=assembly(h*direction[:4],h*direction[4:]).residual
    minus=assembly(-h*direction[:4],-h*direction[4:]).residual
    numerical=(plus-minus)/(2*h)
    assert torch.allclose(a.tangent@direction.reshape(-1),numerical,rtol=2.e-5,atol=2.e-5)


def test_two_elastic_surface_patch_compression_newton_equilibrium():
    # Two independently compliant surface patches are driven through their
    # initial clearance.  Contact is part of the monolithic Newton matrix, so
    # both bodies deform and carry equal and opposite interface resultants.
    slave,sf=triangle(.01); master,mf=triangle(0.)
    target_s=torch.zeros_like(slave); target_s[:,2]=-.006
    target_m=torch.zeros_like(master); target_m[:,2]= .006
    target=torch.cat((target_s.reshape(-1),target_m.reshape(-1)))
    q=target.clone(); stiffness=1.e4; kn=1.e7; ndof=slave.numel()
    for _ in range(12):
        a=assemble_mortar_contact(slave,sf,master,mf,q[:ndof].reshape_as(slave),q[ndof:].reshape_as(master),
                                  normal_penalty=kn)
        residual=stiffness*(q-target)+a.residual
        if float(torch.linalg.vector_norm(residual))<1.e-8: break
        q=q-torch.linalg.solve(stiffness*torch.eye(q.numel(),dtype=D)+a.tangent,residual)
    else: raise AssertionError("two-body contact Newton iteration did not converge")
    final=assemble_mortar_contact(slave,sf,master,mf,q[:ndof].reshape_as(slave),q[ndof:].reshape_as(master),
                                  normal_penalty=kn)
    total=torch.cat((final.slave_forces,final.master_forces)).sum(0)
    assert final.active_quadrature_points==3
    assert torch.linalg.vector_norm(total)<1.e-8
    assert torch.linalg.vector_norm(residual)<1.e-8
    assert torch.mean(q[:ndof].reshape_as(slave)[:,2]-target_s[:,2])>0
    assert torch.mean(q[ndof:].reshape_as(master)[:,2]-target_m[:,2])<0
    assert 0<float(final.maximum_penetration)<2.e-3


def test_open_active_set_has_exact_zero_residual_and_tangent():
    slave,sf=patch(.01); master,mf=patch(0.)
    z=torch.zeros_like(slave)
    a=assemble_mortar_contact(slave,sf,master,mf,z,z,normal_penalty=1.e6)
    assert a.active_quadrature_points==0
    assert torch.count_nonzero(a.residual)==0
    assert torch.count_nonzero(a.tangent)==0


def test_common_finite_rigid_motion_is_objective_and_moment_balanced():
    slave,sf=triangle(-1.e-3); master,mf=triangle(0.)
    zero_s=torch.zeros_like(slave); zero_m=torch.zeros_like(master)
    base=assemble_mortar_contact(slave,sf,master,mf,zero_s,zero_m,normal_penalty=2.e5)
    angle=.71; c=torch.cos(torch.tensor(angle,dtype=D)); s=torch.sin(torch.tensor(angle,dtype=D))
    rotation=torch.tensor([[c,0.,s],[0.,1.,0.],[-s,0.,c]],dtype=D)
    shift=torch.tensor([.4,-.2,.7],dtype=D)
    current_s=slave@rotation.T+shift; current_m=master@rotation.T+shift
    moved=assemble_mortar_contact(slave,sf,master,mf,current_s-slave,current_m-master,
                                  normal_penalty=2.e5)
    assert torch.allclose(moved.slave_forces,base.slave_forces@rotation.T,rtol=2.e-11,atol=2.e-11)
    forces=torch.cat((moved.slave_forces,moved.master_forces))
    points=torch.cat((current_s,current_m))
    assert torch.linalg.vector_norm(forces.sum(0))<1.e-10
    assert torch.linalg.vector_norm(torch.linalg.cross(points,forces).sum(0))<1.e-10
