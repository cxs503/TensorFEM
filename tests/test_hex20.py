import torch
import pytest
from tensorfem.hex20 import (Hex20Model,hex20_shape_gradients,
    solve_hex20,structured_hex20_mesh,upgrade_hex8_to_hex20,
    cantilever_hex20_benchmark,cantilever_hex8_benchmark)
from tensorfem.solid3d import structured_hex_mesh

D=torch.float64

def test_shape_partition_and_kronecker():
    dummy=torch.zeros(1,dtype=D)
    n,dn=hex20_shape_gradients(.17,-.31,.22,like=dummy)
    assert abs(float(n.sum())-1)<1e-14
    assert torch.linalg.vector_norm(dn.sum(0))<1e-14
    nodes,e=structured_hex20_mesh(2.,1.,.5,1,1,1)
    # Physical nodal locations map exactly from the parent coordinates.
    for a in range(20):
        p=2*nodes[a]/torch.tensor([2.,1.,.5],dtype=D)-1
        shape,_=hex20_shape_gradients(*p,like=dummy)
        assert torch.allclose(shape,torch.eye(20,dtype=D)[a],atol=2e-14)

def test_upgrade_shares_edges():
    n,h=structured_hex_mesh(2.,1.,1.,2,1,1)
    nq,q=upgrade_hex8_to_hex20(n,h)
    assert q.shape==(2,20) and len(nq)==32
    assert len(set(q[0].tolist()) & set(q[1].tolist()))==8

def test_affine_patch_machine_precision():
    nodes,e=structured_hex20_mesh(2.,1.,.8,2,2,2)
    A=torch.tensor([[.012,.007,-.003],[-.002,-.009,.005],[.004,.006,.015]],dtype=D)
    exact=(nodes@A.T+torch.tensor([.1,-.2,.05],dtype=D)).reshape(-1)
    boundary=((nodes[:,0]==0)|(nodes[:,0]==2)|(nodes[:,1]==0)|(nodes[:,1]==1)|(nodes[:,2]==0)|(nodes[:,2]==.8))
    ids=torch.where(boundary)[0]; fixed=torch.stack((3*ids,3*ids+1,3*ids+2),1).reshape(-1)
    r=solve_hex20(Hex20Model(nodes,e,nodes.new_tensor(70e9),nodes.new_tensor(.27),nodes.new_zeros(3*len(nodes)),fixed,exact[fixed]))
    assert torch.linalg.vector_norm(r.displacement-exact)/torch.linalg.vector_norm(exact)<1e-11

def test_uniaxial_traction_exact():
    nodes,e=structured_hex20_mesh(3.,1.,.5,2,1,1); E=210e9; sigma=12e6
    exact=torch.stack((sigma/E*nodes[:,0],torch.zeros(len(nodes)),torch.zeros(len(nodes))),1).reshape(-1)
    fixed=torch.arange(3*len(nodes))
    r=solve_hex20(Hex20Model(nodes,e,nodes.new_tensor(E),nodes.new_tensor(0.),nodes.new_zeros(3*len(nodes)),fixed,exact))
    ref=nodes.new_tensor([sigma,0,0,0,0,0]).expand_as(r.stress)
    assert torch.linalg.vector_norm(r.stress-ref)/torch.linalg.vector_norm(ref)<1e-11

def test_jacobian_gate():
    nodes,e=structured_hex20_mesh(1.,1.,1.,1,1,1); nodes[:,2]=0
    with pytest.raises(ValueError,match="Jacobian"):
        solve_hex20(Hex20Model(nodes,e,nodes.new_tensor(1.),nodes.new_tensor(.2),nodes.new_zeros(60),torch.arange(60)))

def test_cantilever_qualifies_and_beats_hex8():
    h20,reference=cantilever_hex20_benchmark(8)
    h8,_=cantilever_hex8_benchmark(8)
    e20=abs((h20-reference)/reference); e8=abs((h8-reference)/reference)
    assert e20 < .03
    assert e20 < e8/10
