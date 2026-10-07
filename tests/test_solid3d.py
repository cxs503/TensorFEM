import pytest
import torch
from tensorfem.solid3d import SolidModel,hex_to_tet_mesh,solve_solid,structured_hex_mesh

D=torch.float64
def affine(nodes):
    a=torch.tensor([[.012,.007,-.003],[-.002,-.009,.005],[.004,.006,.015]],dtype=D)
    return (nodes@a.T+torch.tensor([.1,-.2,.05],dtype=D)).reshape(-1)

@pytest.mark.parametrize("kind",["hex8","tet4"])
def test_affine_patch(kind):
    nodes,hexes=structured_hex_mesh(2.,1.,.8,2,2,2); interior=torch.tensor([13]); nodes[interior]+=torch.tensor([.08,-.04,.03])
    elements=hexes if kind=="hex8" else hex_to_tet_mesh(nodes,hexes)[1]; exact=affine(nodes)
    boundary=((nodes[:,0]==0)|(nodes[:,0]==2)|(nodes[:,1]==0)|(nodes[:,1]==1)|(nodes[:,2]==0)|(nodes[:,2]==.8))
    bn=torch.where(boundary)[0]; fixed=torch.stack((3*bn,3*bn+1,3*bn+2),1).reshape(-1)
    r=solve_solid(SolidModel(nodes,elements,torch.tensor(70e9,dtype=D),torch.tensor(.27,dtype=D),torch.zeros(3*len(nodes),dtype=D),fixed,kind,exact[fixed]))
    error=torch.linalg.vector_norm(r.displacement-exact)/torch.linalg.vector_norm(exact)
    assert error.item()<1e-11

@pytest.mark.parametrize("kind",["hex8","tet4"])
def test_uniaxial_analytical(kind):
    nodes,hexes=structured_hex_mesh(2.,1.,.6,2,2,2); elements=hexes if kind=="hex8" else hex_to_tet_mesh(nodes,hexes)[1]
    e,nu,sigma=210e9,.3,15e6
    exact=torch.stack((sigma/e*nodes[:,0],-nu*sigma/e*nodes[:,1],-nu*sigma/e*nodes[:,2]),1).reshape(-1)
    fixed=torch.arange(3*len(nodes)); r=solve_solid(SolidModel(nodes,elements,torch.tensor(e,dtype=D),torch.tensor(nu,dtype=D),torch.zeros(3*len(nodes),dtype=D),fixed,kind,exact))
    ref=torch.tensor([sigma,0.,0.,0.,0.,0.],dtype=D)
    error=torch.linalg.vector_norm(r.stress-ref)/torch.linalg.vector_norm(ref.expand_as(r.stress))
    assert error.item()<1e-11
    right=torch.where(nodes[:,0]==2)[0]
    assert abs(r.reaction[3*right].sum().item()-sigma*.6)/(sigma*.6)<1e-11

def test_autograd_young_modulus():
    nodes,elements=structured_hex_mesh(1.,1.,1.,1,1,1); e=torch.tensor(100.,dtype=D,requires_grad=True); exact=affine(nodes)
    r=solve_solid(SolidModel(nodes,elements,e,torch.tensor(.25,dtype=D),torch.zeros(24,dtype=D),torch.arange(24),"hex8",exact))
    r.strain_energy.backward(); assert e.grad is not None and e.grad.item()>0

@pytest.mark.parametrize("kind",["hex8","tet4"])
def test_degenerate_rejected(kind):
    nodes,hexes=structured_hex_mesh(1.,1.,1.,1,1,1); elements=hexes if kind=="hex8" else hex_to_tet_mesh(nodes,hexes)[1]; nodes[:,2]=0
    with pytest.raises(ValueError,match="degenerate"):
        solve_solid(SolidModel(nodes,elements,torch.tensor(1.),torch.tensor(.2),torch.zeros(24),torch.arange(24),kind,torch.zeros(24)))
