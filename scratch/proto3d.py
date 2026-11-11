import numpy as np
from scipy.linalg import expm

def exact(a,b,h,E,nu,q0,m=1,n=1):
    lam=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))
    al=m*np.pi/a; be=n*np.pi/b
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
    r1=np.array([0,0,al,1,0,0.0])
    r2=np.array([0,0,be,0,1,0.0])
    rz=np.array([-lam*al,-lam*be,0,0,0,(lam+2*mu)])
    Phalf=expm(A*h)
    rows=np.vstack([r1, r2, rz, r1@Phalf, r2@Phalf, rz@Phalf])
    rhs=np.array([0,0,0,0,0,-q0])
    y0=np.linalg.solve(rows,rhs)
    return dict(A=A,y0=y0,al=al,be=be,lam=lam,mu=mu,h=h)

E=200e9; nu=0.3; a=1.0; q0=1e6
for ah in [100,50,20,10,5,4]:
    h=a/ah
    s=exact(a,a,h,E,nu,q0)
    Y=expm(s['A']*h)@s['y0']
    Wtop=Y[2]
    f=np.sin(s['al']*a/2)*np.sin(s['be']*a/2)
    wc=Wtop*f
    D=E*h**3/(12*(1-nu**2)); R=2*np.pi**2/a**2
    w_k=q0/(D*R**2)
    G=E/(2*(1+nu)); k=5/6
    w_f=w_k+q0/(k*G*h*R)
    print(f"a/h={ah:5.1f} w3D/wKirch={wc/w_k:.6f} fsdt/ Kirch={w_f/w_k:.6f}  dw={(wc-w_k)/w_k*100:6.2f}%")