import numpy as np, cv2, subprocess, sys
from scipy import ndimage
exec(open('sun.py').read().split('# ---- pass 1')[0])
meta=np.load('sun/meta.npy'); q=meta[:,4]
FRAC=float(sys.argv[1]) if len(sys.argv)>1 else 0.12
sel=set(np.argsort(q)[::-1][:int(len(q)*FRAC)].tolist()); ref_set=set(np.argsort(q)[::-1][:60].tolist())
O=820; Rm=float(np.median(meta[:,3]))
def center(f,k):
    cx,cy=meta[k,1],meta[k,2]
    M=np.float32([[1,0,O/2-cx],[0,1,O/2-cy]])
    return cv2.warpAffine(f.astype(np.float32),M,(O,O),flags=cv2.INTER_CUBIC)
def hp(g): return g-cv2.GaussianBlur(g,(0,0),8)
# reference from the 60 sharpest (centroid aligned)
ref=np.zeros((O,O,3),np.float32); n=0
for k,f in enumerate(frames()):
    if k in ref_set: ref+=center(f,k); n+=1
ref/=n; refg=hp(ref.mean(2))
yy,xx=np.mgrid[:O,:O]; rr=np.hypot(xx-O/2,yy-O/2); inner=rr<0.93*Rm
# multipoint grid
P=96; step=64; cents=[(y,x) for y in range(P//2,O-P//2,step) for x in range(P//2,O-P//2,step) if np.hypot(x-O/2,y-O/2)<0.88*Rm]
win=cv2.createHanningWindow((P,P),cv2.CV_64F)
acc=np.zeros((O,O,3),np.float64); cnt=0; gshift=[]
for k,f in enumerate(frames()):
    if k not in sel: continue
    c=center(f,k); g=hp(c.mean(2))
    (sx,sy),r=cv2.phaseCorrelate(refg.astype(np.float64)*inner,g.astype(np.float64)*inner)
    if abs(sx)<5 and abs(sy)<5:
        c=cv2.warpAffine(c,np.float32([[1,0,-sx],[0,1,-sy]]),(O,O),flags=cv2.INTER_CUBIC); g=hp(c.mean(2))
    gshift.append((sx,sy))
    # local shifts
    dx=np.full((len(cents),),np.nan); dy=dx.copy()
    for i,(y,x) in enumerate(cents):
        a=refg[y-P//2:y+P//2,x-P//2:x+P//2].astype(np.float64); b=g[y-P//2:y+P//2,x-P//2:x+P//2].astype(np.float64)
        (lx,ly),lr=cv2.phaseCorrelate(a,b,win)
        if lr>0.08 and abs(lx)<4 and abs(ly)<4: dx[i],dy[i]=lx,ly
    ok=~np.isnan(dx)
    if ok.sum()<10: fx=np.zeros((O,O),np.float32); fy=fx.copy()
    else:
        gy=np.array([c_[0] for c_ in cents]); gx=np.array([c_[1] for c_ in cents])
        # smooth field: Gaussian-weighted interpolation of valid local shifts
        sig=70.0; fx=np.zeros((O,O),np.float32); fy=fx.copy(); ws=np.zeros((O,O),np.float32)
        Ys,Xs=np.mgrid[0:O:8,0:O:8]; fxs=np.zeros(Ys.shape); fys=fxs.copy(); wss=fxs.copy()
        for i in np.where(ok)[0]:
            w=np.exp(-((Ys-gy[i])**2+(Xs-gx[i])**2)/(2*sig**2)); fxs+=w*dx[i]; fys+=w*dy[i]; wss+=w
        fx=cv2.resize((fxs/np.maximum(wss,1e-6)).astype(np.float32),(O,O),interpolation=cv2.INTER_CUBIC)
        fy=cv2.resize((fys/np.maximum(wss,1e-6)).astype(np.float32),(O,O),interpolation=cv2.INTER_CUBIC)
    mapx=(xx+fx).astype(np.float32); mapy=(yy+fy).astype(np.float32)
    c=cv2.remap(c,mapx,mapy,cv2.INTER_CUBIC)
    acc+=c; cnt+=1
st=(acc/cnt).astype(np.float32); np.save('sun/stack.npy',st)
gs=np.array(gshift); print('stacked',cnt,'global residual shift median',np.median(np.abs(gs),0).round(2))
ref8=np.clip(ref,0,255).astype(np.uint8); st8=np.clip(st,0,255).astype(np.uint8)
cv2.imwrite('sun/ref_vs_stack.png',np.hstack([ref8[200:520,380:780],st8[200:520,380:780]]))
