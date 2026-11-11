import sys, time, torch
sys.path.insert(0,'src')
from tensorfem.cohesive3d import BKCohesive3DLaw, cohesive_hex8_response
law=BKCohesive3DLaw(2.4e14,1.0e14,120e6,80e6,300.0,800.0,2.2)
torch.set_num_threads(1)
c=torch.tensor([[0,0,0],[0.5,0,0],[0.5,0.5,0],[0,0.5,0],[0,0,0],[0.5,0,0],[0.5,0.5,0],[0,0.5,0]],dtype=torch.float64)*1e-3
u=torch.zeros((8,3),dtype=torch.float64); u[4:,2]=1e-7
t=time.time()
for _ in range(100):
    f,k,h,d=cohesive_hex8_response(c,u,law)
print("per call ms", (time.time()-t)/100*1e3)
