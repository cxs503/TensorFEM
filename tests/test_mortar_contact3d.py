import math
import torch

from tensorfem.mortar_contact3d import integrate_mortar_contact,self_contact_candidates

D=torch.float64


def grid(n,zfun=lambda x,y: -.001):
    nodes=torch.tensor([[i/n,j/n,zfun(i/n,j/n)] for i in range(n+1) for j in range(n+1)],dtype=D)
    faces=[]
    ix=lambda i,j:i*(n+1)+j
    for i in range(n):
        for j in range(n): faces.extend(([ix(i,j),ix(i+1,j),ix(i+1,j+1)],
                                         [ix(i,j),ix(i+1,j+1),ix(i,j+1)]))
    return nodes,torch.tensor(faces,dtype=torch.long)


def master_plane(n=1,z=0.,reverse=False):
    x,f=grid(n,lambda _x,_y:z)
    if reverse: f=torch.flip(f,dims=(1,))
    return x,f


def test_constant_pressure_patch_consistent_force_and_moment():
    slave,sf=grid(3); master,mf=master_plane(2)
    r=integrate_mortar_contact(slave,sf,master,mf,normal_penalty=2e5)
    expected=200.
    assert abs(float(r.slave_forces[:,2].sum())-expected)/expected < .03
    assert torch.linalg.vector_norm(r.slave_forces.sum(0)+r.master_forces.sum(0))/expected < 1e-12
    sm=torch.linalg.cross(slave,r.slave_forces).sum(0)
    mm=torch.linalg.cross(master,r.master_forces).sum(0)
    assert torch.linalg.vector_norm(sm+mm)/expected < 1e-12
    assert abs(float(r.integrated_area)-1.) < 1e-13


def test_master_slave_exchange_has_roundoff_sensitivity_on_patch():
    bottom,bf=grid(3); top,tf=master_plane(2)
    first=integrate_mortar_contact(bottom,bf,top,tf,normal_penalty=2e5)
    bottom_master,bmf=master_plane(3,z=-.001,reverse=True)
    top_slave,tsf=master_plane(2,z=0.)
    second=integrate_mortar_contact(top_slave,tsf,bottom_master,bmf,normal_penalty=2e5)
    f1=torch.linalg.vector_norm(first.slave_forces.sum(0)); f2=torch.linalg.vector_norm(second.slave_forces.sum(0))
    assert abs(float(f1-f2))/float(f1) < .03
    assert abs(float(f1-f2))/float(f1) < 1e-10


def curved_error(n):
    delta=.002
    slave,sf=grid(n,lambda x,y:-delta*(1-.5*x*x))
    master,mf=master_plane(3)
    r=integrate_mortar_contact(slave,sf,master,mf,normal_penalty=1e5)
    # Independent high-order scalar quadrature of p(x) over curved area.
    q=torch.linspace(0.,1.,20001,dtype=D); zprime=delta*q
    exact=torch.trapezoid(1e5*delta*(1-.5*q*q)*torch.sqrt(1+zprime*zprime),q)
    return abs(float(r.slave_forces[:,2].sum()-exact))/float(exact)


def test_nonmatching_curved_surface_converges_below_three_percent():
    errors=[curved_error(n) for n in (1,2,4)]
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert errors[-1] < .03


def test_self_contact_filter_finds_only_nonadjacent_fold_pair():
    # Connected four-triangle strip folded back so faces 0 and 3 approach.
    v=torch.tensor([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],
                    [.05,0.,.02],[.05,1.,.02]],dtype=D)
    f=torch.tensor([[0,2,1],[1,2,3],[2,4,3],[3,4,5]],dtype=torch.long)
    pairs=self_contact_candidates(v,f,search_distance=.08)
    assert (0,3) in pairs
    assert all(not (set(f[i].tolist())&set(f[j].tolist())) for i,j in pairs)


def test_degenerate_self_contact_and_bad_penalty_fail_closed():
    v=torch.zeros((3,3),dtype=D); f=torch.tensor([[0,1,2]])
    try: self_contact_candidates(v,f,search_distance=1.)
    except ValueError: pass
    else: raise AssertionError("degenerate face accepted")
    slave,sf=grid(1); master,mf=master_plane()
    try: integrate_mortar_contact(slave,sf,master,mf,normal_penalty=0.)
    except ValueError: pass
    else: raise AssertionError("zero penalty accepted")
