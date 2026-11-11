import math, numpy as np, torch
from scipy.linalg import expm
import sys
sys.path.insert(0,'src')
from tensorfem.hex20 import Hex20Model, solve_hex20, structured_hex20_mesh, hex20_shape_gradients

def exact(a,b,h,E,nu,q0,m=1,n=1):
    lam=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))
    al=m*math.pi/a; be=n*math.pi/b
    A=np.zeros((6,6))
    A[0,3]=1; A[1,4]=1; A[2,5]=1
    A[3,0]=((lam+2*mu)*al**2+mu*be**2)/mu
    A[3,1]=(lam+mu)*al*be/mu
    A[3,5]=-(lam+mu)*al/mu
    A[4,0]=(lam+mu)*al*be/mu
    A[4,1]=((lam+2*mu)*be**2+mu*al**2)/mu
    A[4,5]=-(lam+mu)*be/mu
    A[5,2]=mu*(al**2+be**2)/(lam+2*mu)
    A[5,3]=(lam+mu)*al/(lam+2*mu)
    A[5,4]=(lam+mu)*be/(lam+2*mu)
    r1=np.array([0,0,al,1,0,0.0]); r2=np.array([0,0,be,0,1,0.0])
    rz=np.array([-lam*al,-lam*be,0,0,0,(lam+2*mu)])
    Ph=expm(A*h)
    rows=np.vstack([r1,r2,rz,r1@Ph,r2@Ph,rz@Ph]); rhs=np.array([0,0,0,0,0,-q0])
    y0=np.linalg.solve(rows,rhs)
    def at(z):
        return expm(A*(z+h/2))@y0
    return dict(A=A,y0=y0,al=al,be=be,lam=lam,mu=mu,h=h,z=at)

def top_load(nodes,cells,a,b,h,q0,al,be):
    nn=len(nodes); f=torch.zeros(3*nn,dtype=torch.float64)
    g=1/math.sqrt(3); pts=[(g*math.sqrt(3/5),0) for _ in range(1)]
    # 3x3 Gauss on the 8-node quad face (top): use hex20 shape funcs at t=+1
    gp=[-math.sqrt(3/5),0.0,math.sqrt(3/5)]; gw=[5/9,8/9,5/9]
    top_face=[4,5,6,7,16,17,18,19]
    for c in cells.tolist():
        if abs(nodes[c[4],2]-h)>1e-12: continue
        xyz=nodes[c]
        for i,r in enumerate(gp):
            for j,s in enumerate(gp):
                N,_=hex20_shape_gradients(r,s,1.0,like=xyz)
                N=N.detach().numpy()
                xs=(N[:,None]*xyz.numpy()).sum(0)
                q=q0*math.sin(al*xs[0])*math.sin(be*xs[1])
                # jacobian of the face map (r,s) at t=1
                hh=1e-6
                def P(rr,ss):
                    Nn,_=hex20_shape_gradients(rr,ss,1.0,like=xyz); return (Nn.detach().numpy()[:,None]*xyz.numpy()).sum(0)
                dr=(P(r+hh,s)-P(r-hh,s))/(2*hh); ds=(P(r,s+hh)-P(r,s-hh))/(2*hh)
                J=np.linalg.norm(np.cross(dr,ds))
                w=q*J*gw[i]*gw[j]
                for k,gi in enumerate(top_face):
                    f[3*c[gi]+2]+= -w*N[k]
    return f

E=200e9; nu=0.3; a=1.0; q0=1e6
for ah,nxy,nz in [(10,8,2),(10,12,3),(10,16,4),(5,12,3),(20,8,2),(100,8,2)]:
    h=a/ah
    nodes,cells=structured_hex20_mesh(a,a,h,nxy,nxy,nz)
    ex=exact(a,a,h,E,nu,q0)
    f=top_load(nodes,cells,a,a,h,q0,ex['al'],ex['be'])
    tol=a*1e-12
    xe=(nodes[:,0].abs()<tol)|((nodes[:,0]-a).abs()<tol)
    ye=(nodes[:,1].abs()<tol)|((nodes[:,1]-a).abs()<tol)
    fixed=torch.cat((torch.nonzero(xe).flatten()*3+1, torch.nonzero(xe).flatten()*3+2,
                     torch.nonzero(ye).flatten()*3, torch.nonzero(ye).flatten()*3+2)).unique()
    res=solve_hex20(Hex20Model(nodes,cells,nodes.new_tensor(E),nodes.new_tensor(nu),f,fixed))
    u=res.displacement
    cen=((nodes[:,0]-a/2).abs()<a/(2*nxy)+1e-12)&((nodes[:,1]-a/2).abs()<a/(2*nxy)+1e-12)&((nodes[:,2]-h).abs()<1e-12)
    idx=torch.nonzero(cen).flatten()
    wf=float(u[3*idx[0]+2]) if len(idx) else float('nan')
    sd=ex['z'](h/2); wex=sd[2]*math.sin(ex['al']*a/2)*math.sin(ex['be']*a/2)
    D=E*h**3/(12*(1-nu**2)); R=2*math.pi**2/a**2; wk=q0/(D*R**2)
    print(f"a/h={ah:5.1f} nxy={nxy:2d} nz={nz} ndof={3*len(nodes)}  wFE={wf:.6e} wEX={wex:.6e} err={(wf-wex)/wex*100:6.2f}%  wEX/wK={wex/wk:.5f}")