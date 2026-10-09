import torch
import pytest

from tensorfem.quadratic_solid import (
    Tet10Model, tet10_shape, tet10_stiffness, tet4_to_tet10, solve_tet10,
    cantilever_tet10_benchmark, cantilever_tet4_benchmark)
from tensorfem.solid3d import elasticity_matrix_3d, structured_hex_mesh, hex_to_tet_mesh

D=torch.float64


def unit_tet10():
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
    return tet4_to_tet10(v,torch.tensor([[0,1,2,3]]))[0]


def test_partition_unity_kronecker_and_degree_three_rule():
    x=unit_tet10()
    natural=x[:,[0,1,2]]
    for i,p in enumerate(natural):
        n,dn=tet10_shape(p)
        assert torch.allclose(n,torch.eye(10,dtype=D)[i],atol=2e-15)
        assert torch.allclose(n.sum(),torch.tensor(1.,dtype=D))
        assert torch.allclose(dn.sum(0),torch.zeros(3,dtype=D),atol=2e-15)


def test_quadratic_displacement_patch_reproduces_exact_strain():
    x=unit_tet10(); _,dn=tet10_shape(torch.tensor([.2,.3,.1],dtype=D))
    j=x.T@dn; g=dn@torch.linalg.inv(j)
    # A genuinely quadratic displacement field represented exactly by TET10.
    u=torch.stack((x[:,0]**2+x[:,1]*x[:,2], x[:,1]**2, x[:,2]**2),1)
    from tensorfem.solid3d import _b
    strain=(_b(g[None])[0]@u.reshape(-1))
    p=torch.tensor([.2,.3,.1],dtype=D)
    exact=torch.tensor([2*p[0],2*p[1],2*p[2],p[2],0.,p[1]],dtype=D)
    assert torch.allclose(strain,exact,atol=2e-14,rtol=2e-14)


def test_affine_patch_energy_and_global_solver():
    nodes=unit_tet10(); conn=torch.arange(10)[None]; E=torch.tensor(210e9,dtype=D); nu=torch.tensor(.29,dtype=D)
    eps=torch.tensor([1.2e-4,-.4e-4,.7e-4,.3e-4,-.2e-4,.5e-4],dtype=D)
    u=torch.stack((eps[0]*nodes[:,0]+eps[3]*nodes[:,1]/2+eps[5]*nodes[:,2]/2,
                   eps[1]*nodes[:,1]+eps[3]*nodes[:,0]/2+eps[4]*nodes[:,2]/2,
                   eps[2]*nodes[:,2]+eps[5]*nodes[:,0]/2+eps[4]*nodes[:,1]/2),1).reshape(-1)
    d=elasticity_matrix_3d(E[None],nu[None]); ke,_=tet10_stiffness(nodes[None],d)
    exact=.5*(eps@(d[0]@eps))/6
    assert abs(float(.5*u@ke[0]@u-exact))/float(exact) < 2e-13
    # Prescribed patch through the public assembly/solve path.
    m=Tet10Model(nodes,conn,E,nu,torch.zeros(30,dtype=D),torch.arange(30),u)
    result=solve_tet10(m)
    assert torch.allclose(result.strain[0],eps,atol=2e-13)


def test_jacobian_guard_checks_integration_points():
    x=unit_tet10(); x[4]=torch.tensor([2.,0.,0.],dtype=D)
    d=elasticity_matrix_3d(torch.tensor([1.],dtype=D),torch.tensor([.25],dtype=D))
    with pytest.raises(ValueError,match="Jacobian"):
        tet10_stiffness(x[None],d)


def test_conforming_upgrade_shares_midside_nodes():
    nodes,hexes=structured_hex_mesh(2.,1.,1.,2,1,1,dtype=D); _,tet=hex_to_tet_mesh(nodes,hexes)
    qnodes,q=tet4_to_tet10(nodes,tet)
    assert q.shape==(12,10)
    edges=set()
    for c in tet.tolist():
        for a,b in ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)): edges.add(tuple(sorted((c[a],c[b]))))
    assert len(qnodes)==len(nodes)+len(edges)
    assert q[:,4:].unique().numel()==len(edges)


def test_tet10_cantilever_formal_error_below_three_percent():
    computed,reference=cantilever_tet10_benchmark(10)
    assert abs(float(computed/reference)-1) < .03
    linear,_=cantilever_tet4_benchmark(10)
    assert abs(float(computed/reference)-1) < abs(float(linear/reference)-1)/20
