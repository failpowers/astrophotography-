import cv2,numpy as np
exec(open('stab2.py').read().split('ref=480')[0])
def match(P0,P1,guess,tol=5):
    pred=cv2.transform(P0[None],guess)[0]
    d=np.hypot(pred[:,None,0]-P1[None,:,0],pred[:,None,1]-P1[None,:,1]); j=d.argmin(1); ok=d[np.arange(len(P0)),j]<tol
    if ok.sum()<12: return None,0
    M,inl=cv2.estimateAffinePartial2D(P0[ok],P1[j[ok]],method=cv2.RANSAC,ransacReprojThreshold=1.5)
    return (M,int(inl.sum())) if M is not None else (None,0)
def step(Pa,Pb):
    I=np.float32([[1,0,0],[0,1,0]]); M,n=match(Pa,Pb,I)
    if n<20:
        dif=(Pb[:80,None,:]-Pa[None,:80,:]).reshape(-1,2)
        Hh,xe,ye=np.histogram2d(dif[:,0],dif[:,1],bins=[np.arange(-600,600,4),np.arange(-400,400,4)])
        k=np.unravel_index(Hh.argmax(),Hh.shape); g=I.copy(); g[:,2]=[xe[k[0]]+2,ye[k[1]]+2]; M2,n2=match(Pa,Pb,g,8)
        if n2>n: M,n=M2,n2
    return M,n
def comp(B,A):  # apply A then B (2x3)
    A3=np.vstack([A,[0,0,1]]); B3=np.vstack([B,[0,0,1]]); return (B3@A3)[:2].astype(np.float32)
P=[pts(f) for f in F]
T=[None]*N; T[0]=np.float32([[1,0,0],[0,1,0]]); last=0; nin=[0]*N
for i in range(1,N):
    M,n=step(P[last],P[i]); nin[i]=n
    if M is not None and n>=20: T[i]=comp(M,T[last]); last=i
good=[i for i in range(N) if T[i] is not None]; print('chained',len(good),'of',N)
par=np.full((N,4),np.nan)
for i in good:
    M=T[i]; par[i]=[np.arctan2(M[1,0],M[0,0]),np.hypot(M[0,0],M[1,0]),M[0,2],M[1,2]]
for k in range(4):
    v=par[:,k]; ok=~np.isnan(v); par[:,k]=np.interp(np.arange(N),np.where(ok)[0],v[ok])
np.save('met/tl_par.npy',par); np.save('met/tl_good.npy',np.array([T[i] is not None for i in range(N)]))
print('rot deg',np.degrees(par[:,0]).min().round(2),np.degrees(par[:,0]).max().round(2),'scale',par[:,1].min().round(3),par[:,1].max().round(3))
bad=[i for i in range(N) if T[i] is None]; print('unregistered sample',bad[:40], '... tail', bad[-10:])
