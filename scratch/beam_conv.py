import torch, sys
sys.path.insert(0,'src')
from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness, structured_hex_mesh
D=torch.float64
E,nu=70e9,0.3; L,b,h=0.02,0.003,0.003
P=1.0
for nz in (1,2,3,4,6,8):
  nodes,hexes=structured_hex_mesh(L,b,h,20,1,nz,dtype=D)
  d=elasticity_matrix_3d(torch.tensor([E],dtype=D),torch.tensor([nu],dtype=D))
  ke,_=hex8_stiffness(nodes[hexes],d)
  ne,nn=hexes.shape; ndof=3*len(nodes)
  ed=torch.stack(tuple(3*hexes+i for i in range(3)),2).reshape(ne,24).long()
  K=torch.zeros((ndof,ndof),dtype=D)
  rows=ed[:,:,None].expand(-1,-1,24).reshape(-1); cols=ed[:,None,:].expand(-1,24,-1).reshape(-1)
  K=K.index_put((rows,cols),ke.reshape(-1),accumulate=True)
  tipn=(nodes[:,0]==L).nonzero().reshape(-1)
  f=torch.zeros(ndof,dtype=D); f[(3*tipn+2)]=P/tipn.numel()
  fixed=(nodes[:,0]==0).nonzero().reshape(-1)
  fd=(3*fixed[:,None]+torch.arange(3)).reshape(-1)
  mask=torch.ones(ndof,dtype=torch.bool); mask[fd]=False
  free=mask.nonzero().reshape(-1); u=torch.zeros(ndof,dtype=D)
  u[free]=torch.linalg.solve(K[free][:,free],f[free]-K[free][:,fd]@u[fd])
  tip=float(u[2::3][tipn].mean())
  ref=P*L**3/(3*E*b*h**3/12)+P*L/(5/6*E/(2*(1+nu))*b*h)
  print(f"nz={nz} tip={tip:.6e} ref={ref:.6e} rel={100*(tip-ref)/ref:+.3f}%")
