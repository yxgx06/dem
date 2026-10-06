import numpy as np, sys
dt=0.01;T=75.0;t=np.arange(0,T+dt/2,dt);N=len(t)
import os
m,Iz,lf,lr,h,Cf,Cr,Vx,g,dmax=1500.,3000.,1.2,1.6,0.5,80000.,80000.,float(os.environ.get("VX",10)),9.81,0.5
L=lf+lr
slope=np.zeros(N);mu=np.zeros(N)
for k,tk in enumerate(t):
    if tk<15: slope[k]=0;mu[k]=.85
    elif tk<30: slope[k]=10*(1-np.exp(-(tk-15)/1.5));mu[k]=.85
    elif tk<45: slope[k]=10;mu[k]=.85-.35*(1-np.exp(-(tk-30)/1.5))
    elif tk<60:
        s=(tk-45)/3; slope[k]=10-20*s if s<=1 else -10; mu[k]=.5-.25*(1-np.exp(-(tk-45)/1.5))
    else: slope[k]=-10+10*min(1,(tk-60)/2); mu[k]=.25+.35*min(1,(tk-60)/2)
dref=np.zeros(N)
for k,tk in enumerate(t):
    if tk<=60: dref[k]=0.05*np.sin(2*np.pi/30*tk)
    else:
        l=tk-60
        if l<2.5: dref[k]=0
        elif l<6.5: dref[k]=0.08*np.sin(2*np.pi/4*(l-2.5))
        elif l<10.5: dref[k]=-0.08*np.sin(2*np.pi/4*(l-6.5))
        else: dref[k]=0
Xr=np.zeros(N);Yr=np.zeros(N);pr=np.zeros(N);rr=np.zeros(N)
for k in range(1,N):
    rr[k-1]=Vx/L*np.tan(dref[k-1]);pr[k]=pr[k-1]+rr[k-1]*dt
    Xr[k]=Xr[k-1]+Vx*np.cos(pr[k-1])*dt;Yr[k]=Yr[k-1]+Vx*np.sin(pr[k-1])*dt
rr[N-1]=Vx/L*np.tan(dref[N-1])
W1=np.array([[0.85,0.42,1.10,0.15,0.30,-0.65],[0.12,0.78,0.25,0.60,-0.45,-0.80],[0.95,0.10,1.45,0.05,0.55,-0.50],[0.05,0.85,0.10,0.90,-0.30,-0.95],[0.40,0.35,0.60,0.40,0.80,0.20],[-0.30,0.50,-0.20,0.70,-0.85,0.10],[0.60,0.20,0.80,0.30,0.10,-0.70],[0.10,0.65,0.15,0.55,0.25,-0.40]])
b1=np.array([.1,.25,.05,.3,-.1,.15,.05,.2])
W2=np.array([[0.45,0.10,0.55,0.05,0.20,-0.15,0.30,0.10],[0.05,0.15,0.02,0.10,0.05,-0.02,0.08,0.05],[0.10,0.65,0.05,0.75,-0.10,0.30,0.25,0.45],[0.50,0.10,0.65,0.05,0.35,-0.20,0.40,0.15]])
b2=np.array([.8,.05,.08,1.0])

from scipy.linalg import solve_discrete_are
QW=[0.05,0.0,1.0,0.0];RW=1.0
_cache={}
def lqr_gain(mm,v=None,Q=None,tau=0.0):
    v=v or Vx;Q=Q or QW
    key=(round(mm),v,tuple(Q),tau)
    if key in _cache:return _cache[key]
    A=np.array([[0,1,0,0],[0,-(Cf+Cr)/(mm*v),(Cf+Cr)/mm,(-Cf*lf+Cr*lr)/(mm*v)],[0,0,0,1],[0,-(Cf*lf-Cr*lr)/(Iz*v),(Cf*lf-Cr*lr)/Iz,-(Cf*lf**2+Cr*lr**2)/(Iz*v)]])
    B=np.array([[0],[Cf/mm],[0],[Cf*lf/Iz]])
    from scipy.linalg import expm
    M=np.zeros((5,5));M[:4,:4]=A;M[:4,4:]=B;E=expm(M*dt);Ad=E[:4,:4];Bd=E[:4,4:]
    P=solve_discrete_are(Ad,Bd,np.diag(Q),np.array([[RW]]))
    K=np.linalg.solve(RW+Bd.T@P@Bd,Bd.T@P@Ad)[0]
    _cache[key]=K;return K
def lqr_u(e,de,dp,dyaw,kap_rate,mm):
    K=lqr_gain(mm)
    x=np.array([-e,-de,-dp,dyaw])
    kap=kap_rate/Vx
    kv=lr*mm/(Cf*L)-lf*mm/(Cr*L)
    ff=L*kap+kv*Vx**2*kap-K[2]*(lr*kap-lf*mm*Vx**2*kap/(Cr*L))
    return -K@x+ff
def run(c,mu_est_err=0.0,mass_scale=1.0,noise=0.0,delay=0,seed=0):
    rng=np.random.default_rng(seed)
    X=Y=psi=Vy=r=0.;ey=np.zeros(N);prev=0.;integ=0.;pe=0.;buf=[0.]*(delay+1)
    mm=m*mass_scale
    for k in range(N-1):
        exg=Xr[k]-X;eyg=Yr[k]-Y
        e=eyg*np.cos(psi)-exg*np.sin(psi)+rng.normal()*noise
        ey[k]=e
        dp=np.arctan2(np.sin(pr[k]-psi),np.cos(pr[k]-psi))
        de=(e-pe)/dt if k>0 else 0.;pe=e
        th=np.radians(slope[k]);mk=mu[k]
        Fzf=mm*g*(lr*np.cos(th)-h*np.sin(th))/L;Fzr=mm*g*(lf*np.cos(th)+h*np.sin(th))/L
        Ff=max(mk*Fzf,100);Fr=max(mk*Fzr,100)
        mue=np.clip(mk+mu_est_err,0.05,1.0)  # controller's belief
        if c=='pid':
            integ=np.clip(integ+e*dt,-1.5,1.5);u=.8*e+.05*integ+.05*de+1.0*dp
        elif c=='mpc':
            u=.95*e+.12*de+1.15*dp+(L/Vx)*rr[k]
            if mue<.5:u*=.85
        elif c=='lqr':
            u=lqr_u(e,de,dp,r-rr[k],rr[k],mm)
        else:
            obs=np.array([e,de,dp,r-rr[k],slope[k]/10,(mue-.55)/.35])
            gg=W2@np.tanh(W1@obs+b1)+b2
            Kp=np.clip(gg[0],.4,2.2);Ki=np.clip(gg[1],.01,.15);Kd=np.clip(gg[2],.02,.45);Kh=np.clip(gg[3],.6,1.8)
            if mue<.6:Kd*=1+1.2*(.6-mue);Ki*=mue/.85
            if slope[k]>2:Kp*=1+.025*slope[k];Kh*=1+.02*slope[k]
            integ=np.clip(integ+e*dt,-1.5,1.5)
            u=Kp*e+Ki*integ+Kd*de+Kh*dp
            if abs(Vx)>.1:u+=L*rr[k]/Vx
        buf.append(u);u=buf.pop(0)
        mr=.6*dt;d=prev+np.clip(u-prev,-mr,mr);d=np.clip(d,-dmax,dmax);prev=d
        af=d-np.arctan2(Vy+lf*r,Vx);ar=-np.arctan2(Vy-lr*r,Vx)
        Fyf=np.clip(Cf*af,-Ff,Ff);Fyr=np.clip(Cr*ar,-Fr,Fr)
        ay=(Fyf*np.cos(d)+Fyr)/mm
        dVy=ay-Vx*r;dr=(lf*Fyf*np.cos(d)-lr*Fyr)/Iz
        X+=(Vx*np.cos(psi)-Vy*np.sin(psi))*dt;Y+=(Vx*np.sin(psi)+Vy*np.cos(psi))*dt
        Vy+=dVy*dt;r+=dr*dt;psi+=r*dt
    ey[N-1]=ey[N-2];return ey
def rep(name,ey):
    print(f"{name:40s} max={np.max(abs(ey))*100:7.2f} rms={np.sqrt(np.mean(ey**2))*100:6.2f} ice={np.max(abs(ey[4500:6000]))*100:7.2f} rain={np.max(abs(ey[3000:4500]))*100:7.2f} lc={np.max(abs(ey[6000:]))*100:7.2f}")
