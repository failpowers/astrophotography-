import cv2,numpy as np
cap=cv2.VideoCapture('met/timelapse.mp4'); F=[]
while True:
    ok,f=cap.read()
    if not ok: break
    F.append(f)
N=len(F)
def pts(f,n=300):
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
    th=g-cv2.morphologyEx(g,cv2.MORPH_OPEN,np.ones((9,9),np.uint8)); th=cv2.GaussianBlur(th,(0,0),1.0)
    mx=cv2.dilate(th,np.ones((7,7),np.uint8)); s=np.median(np.abs(th-np.median(th)))*1.48+1e-3
    ys,xs=np.where((th==mx)&(th>6*s)); v=th[ys,xs]; o=np.argsort(-v)[:n]
    return np.stack([xs[o],ys[o]],1).astype(np.float32)
def img(p,shape=(810,1440)):
    m=np.zeros(shape,np.float32)
    for x,y in p[:150].astype(int): cv2.circle(m,(x,y),2,1,-1)
    return cv2.GaussianBlur(m,(0,0),2)
ref=480; RP=pts(F[ref])
A=[None]*N; ninl=[0]*N
def solve(i,guess):
    P=pts(F[i])
    if len(P)<15: return None,0
    pred=cv2.transform(RP[None],guess)[0]
    d=np.hypot(pred[:,None,0]-P[None,:,0],pred[:,None,1]-P[None,:,1]); j=d.argmin(1); ok=d[np.arange(len(RP)),j]<6
    if ok.sum()<12: return None,int(ok.sum())
    M,inl=cv2.estimateAffinePartial2D(RP[ok],P[j[ok]],method=cv2.RANSAC,ransacReprojThreshold=2.0)
    return M,int(inl.sum()) if inl is not None else 0
for rng in (range(ref,N),range(ref-1,-1,-1)):
    prev=np.float32([[1,0,0],[0,1,0]])
    for i in rng:
        M,n=solve(i,prev)
        if M is None or n<15:   # re-acquire after a jump: translation from star-map phase correlation
            (dx,dy),_=cv2.phaseCorrelate(np.float64(img(cv2.transform(RP[None],prev)[0])),np.float64(img(pts(F[i]))))
            g=prev.copy(); g[:,2]+=[dx,dy]; M2,n2=solve(i,g)
            if M2 is not None and n2>n: M,n=M2,n2
        A[i]=M if (M is not None and n>=15) else None; ninl[i]=n
        if A[i] is not None: prev=A[i]
good=[i for i in range(N) if A[i] is not None]; print('registered',len(good),'of',N,'median inliers',int(np.median([ninl[i] for i in good])))
# fill gaps (cloudy frames) by interpolating parameters
par=np.full((N,4),np.nan)
for i in good:
    M=A[i]; s=np.hypot(M[0,0],M[1,0]); par[i]=[np.arctan2(M[1,0],M[0,0]),s,M[0,2],M[1,2]]
for k in range(4):
    v=par[:,k]; ok=~np.isnan(v); par[:,k]=np.interp(np.arange(N),np.where(ok)[0],v[ok])
np.save('met/tl_par.npy',par); np.save('met/tl_ninl.npy',np.array(ninl))
print('rotation range deg',np.degrees(par[:,0]).min().round(2),np.degrees(par[:,0]).max().round(2),' shift x',par[:,2].min().round(1),par[:,2].max().round(1),' y',par[:,3].min().round(1),par[:,3].max().round(1))
d=np.hypot(np.diff(par[:,2]),np.diff(par[:,3])); print('frame-to-frame jumps >3px:',int((d>3).sum()),'max',d.max().round(1))
