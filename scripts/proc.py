"""Shared deep-sky pipeline (linear DWARF stacks): crop to fully-covered area, 2-pass grid background
extraction, star-based color calibration, linear NR, color-preserving stretch, star color, dithered 8-bit."""
import numpy as np, cv2, glob
from scipy import ndimage
import fitsio
SCALE=0.7
def find(obj):
    return [f for f in glob.glob('/mnt/user-data/uploads/DWARF*/stacked*.fits') if f.split('/')[-1].startswith(f'stacked-16_{obj}_')]
def load(path):
    a,h=fitsio.read(path); lin=np.stack([a[2],a[1],a[0]],-1).astype(np.float32); return lin,h
def valid_mask(h,shape,margin=45):
    H,W=shape[:2]
    q1=np.array([[int(h['CRTL1X']),int(h['CRTL1Y'])],[int(h['CRTR1X']),int(h['CRTR1Y'])],[int(h['CRBR1X']),int(h['CRBR1Y'])],[int(h['CRBL1X']),int(h['CRBL1Y'])]],np.int32)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[q1],1)
    r=np.zeros((H,W),np.uint8); r[int(h['CRTL2Y']):int(h['CRBL2Y']),int(h['CRTL2X']):int(h['CRTR2X'])]=1
    m=m&r
    return cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_RECT,(2*margin+1,2*margin+1)))
def max_rect(mask,step=8,min_aspect=1.45):
    m=mask[::step,::step].astype(bool); R,C=m.shape; best=(0,None); hts=np.zeros(C,int)
    for i in range(R):
        hts=np.where(m[i],hts+1,0)
        st=[]
        for j in range(C+1):
            hh=hts[j] if j<C else 0; s=j
            while st and st[-1][1]>=hh:
                s,ph=st.pop(); w=j-s
                # constrain aspect: shrink width or height to satisfy min_aspect
                for hgt in (ph,):
                    wd=w; ht=hgt
                    if wd<min_aspect*ht: ht=int(wd/min_aspect)
                    if wd*ht>best[0]: best=(wd*ht,(i-ht+1,s,ht,wd))
            st.append((s,hh))
    i0,j0,ht,wd=best[1]
    return i0*step,j0*step,ht*step,wd*step
def downs(lin): return cv2.resize(lin,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_AREA)
def star_mask(L):
    H,W=L.shape
    hp=L-cv2.GaussianBlur(L,(0,0),8); nz=1.4826*np.median(np.abs(hp-np.median(hp)))
    det=(cv2.GaussianBlur(hp,(0,0),1.0)>3.0*nz).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(det); sm=np.zeros((H,W),np.uint8)
    for i in range(1,n): cv2.circle(sm,(int(cen[i][0]),int(cen[i][1])),int(3+1.8*np.sqrt(st[i,4])),1,-1)
    return sm,nz
def grid_bg(ch,smask,T,valid=None):
    H,W=ch.shape; gy,gx=H//T,W//T; g=np.full((gy,gx),np.nan,np.float32)
    for j in range(gy):
        for i in range(gx):
            t=ch[j*T:(j+1)*T,i*T:(i+1)*T]; ok=smask[j*T:(j+1)*T,i*T:(i+1)*T]==0
            if valid is not None: ok&=valid[j*T:(j+1)*T,i*T:(i+1)*T]>0
            v=t[ok]
            if v.size<T*T*0.25: continue
            for _ in range(3):
                md=np.median(v); s=1.4826*np.median(np.abs(v-md))+1e-6; v=v[np.abs(v-md)<2.5*s]
            g[j,i]=np.median(v)
    bad=np.isnan(g)
    if bad.all(): return np.zeros_like(ch)
    idx=ndimage.distance_transform_edt(bad,return_distances=False,return_indices=True); g=g[tuple(idx)]
    md=ndimage.median_filter(g,5,mode='nearest'); dev=g-md; s=1.4826*np.median(np.abs(dev))+1e-6
    g=np.where(np.abs(dev)>2.5*s,md,g); g=ndimage.gaussian_filter(g,1.2,mode='nearest')
    up=cv2.resize(g,(gx*T,gy*T),interpolation=cv2.INTER_CUBIC)
    up=cv2.copyMakeBorder(up,0,H-gy*T,0,W-gx*T,cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(up,(0,0),T*0.6)
def dbe(lin,smask,valid=None,T1=48,T2=28):
    for T in (T1,T2):
        bg=np.dstack([grid_bg(lin[...,c],smask,T,valid) for c in range(3)]); lin=lin-bg
    return lin
def color_cal(lin,smask,nz):
    L=lin.mean(2); st=(smask>0)&(L>20*nz)&(L<np.percentile(L,99.995))
    cw=np.array([lin[...,c][st].sum() for c in range(3)]); cw=cw.mean()/cw; lin=lin*cw
    for c in range(3): lin[...,c]-=np.median(lin[...,c][smask==0])
    return lin,cw
def finish(lin,smask,BGT=0.075,SSAT=1.6,K=260,lift=0.32,nr_h=2.2,neb=None):
    """lin: calibrated linear BGR, bg~0. Returns float rgb 0..1 and soft star mask."""
    L=lin.mean(2)
    hp=L-cv2.GaussianBlur(L,(0,0),2); nz=1.4826*np.median(np.abs(hp[smask==0]))
    lo,hi=-8*nz,np.percentile(L,99.999)
    u=np.clip((L-lo)/(hi-lo)*65535,0,65535).astype(np.uint16); nzu=nz/(hi-lo)*65535
    Ld=cv2.fastNlMeansDenoising(u,h=[float(nzu*nr_h)],templateWindowSize=7,searchWindowSize=21,normType=cv2.NORM_L1).astype(np.float32)/65535*(hi-lo)+lo
    th=L-cv2.morphologyEx(L,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    th=cv2.GaussianBlur(th,(0,0),1.0)
    sm=np.clip((th-2.0*nz)/(6*nz),0,1); sm=cv2.dilate(sm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))); sm=cv2.GaussianBlur(sm,(0,0),0.8)
    Lnr=L*sm+(0.85*Ld+0.15*L)*(1-sm)
    Lpos=np.maximum(L,0)
    rs=cv2.GaussianBlur(lin,(0,0),6)/np.maximum(cv2.GaussianBlur(Lpos,(0,0),6),1e-3)[...,None]
    starratio=cv2.GaussianBlur(lin,(0,0),1.0)/np.maximum(cv2.GaussianBlur(Lpos,(0,0),1.0),1e-3)[...,None]
    skyw=0.3 if neb is None else neb      # keep a bit more large-scale color where there is nebulosity
    rat=starratio*sm[...,None]+(1+skyw*(np.clip(rs,0.6,1.4)-1))*(1-sm[...,None]); rat=np.clip(rat,0.2,2.5)
    nz2=1.4826*np.median(np.abs((Lnr-cv2.GaussianBlur(Lnr,(0,0),8))[smask==0]))
    top=np.percentile(Lnr,99.995); f=lambda x:np.arcsinh(K*x)/np.arcsinh(K)
    skyL=np.median(Lnr[smask==0]); bps=np.linspace(skyL-12*nz2,skyL,400)
    bp=bps[np.argmin(np.abs(np.array([f(max(skyL-b,1e-7)/(top-b)) for b in bps])-BGT))]
    Lt=f(np.clip((Lnr-bp)/(top-bp),0,1))
    Lt=np.clip(Lt+lift*sm*Lt*(1-Lt),0,1)
    satw=1+(SSAT-1)*np.clip(sm*1.2,0,1)*np.clip((Lt-0.12)/0.25,0,1)
    rat2=1+(rat-1)*satw[...,None]; rat2=rat2/rat2.mean(2,keepdims=True)
    rgb=np.clip(Lt[...,None]*rat2,0,1)
    b_,g_,r_=rgb[...,0],rgb[...,1],rgb[...,2]
    rgb[...,1]=np.minimum(g_,np.maximum((r_+b_)/2,np.minimum(r_,b_))+0.01)
    rgb=np.where(rgb>0.85,0.85+(1-np.exp(-(rgb-0.85)/0.15))*0.15,rgb)
    er=cv2.erode(rgb,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    rgb=rgb*(1-0.25*sm[...,None])+er*(0.25*sm[...,None])
    return rgb,sm
def save8(rgb,path,seed=7):
    rng=np.random.default_rng(seed)
    out=np.clip(rgb*255+rng.uniform(-0.5,0.5,rgb.shape),0,255).astype(np.uint8); cv2.imwrite(path,out); return out
def max_rect2(mask,step=8,amin=1.5,amax=1.85):
    m=mask[::step,::step].astype(bool); R,C=m.shape; best=(0,None); hts=np.zeros(C,int)
    for i in range(R):
        hts=np.where(m[i],hts+1,0); st=[]
        for j in range(C+1):
            hh=hts[j] if j<C else 0; s=j
            while st and st[-1][1]>=hh:
                s,ph=st.pop(); w=j-s; ht=ph
                if w>amax*ht: w2=int(amax*ht); ht2=ht
                elif w<amin*ht: w2=w; ht2=int(w/amin)
                else: w2,ht2=w,ht
                if w2*ht2>best[0]: best=(w2*ht2,(i-ht2+1,s+(w-w2)//2,ht2,w2))
            st.append((s,hh))
    i0,j0,ht,wd=best[1]; return i0*step,j0*step,ht*step,wd*step
def coverage(lin,T=48,thr=0.45):
    """relative frame coverage from local noise (inverse variance), smoothed; returns full-res weight map and mask"""
    L=lin.mean(2); hp=L-cv2.GaussianBlur(L,(0,0),1.5); H,W=L.shape; gy,gx=H//T,W//T
    t=np.abs(hp[:gy*T,:gx*T]).reshape(gy,T,gx,T).transpose(0,2,1,3).reshape(gy,gx,-1)
    nm=np.median(t,axis=2)*1.4826
    z=(L[:gy*T,:gx*T].reshape(gy,T,gx,T).transpose(0,2,1,3).reshape(gy,gx,-1)==0).mean(2)
    nm=ndimage.median_filter(nm,5,mode='nearest'); nm[z>0.2]=np.inf
    fin=np.isfinite(nm); mn=np.percentile(nm[fin],5); rel=np.where(fin,(mn/nm)**2,0).astype(np.float32)
    up=cv2.resize(rel,(gx*T,gy*T),interpolation=cv2.INTER_LINEAR)
    up=cv2.copyMakeBorder(up,0,H-gy*T,0,W-gx*T,cv2.BORDER_REPLICATE)
    up=cv2.GaussianBlur(up,(0,0),T/2)
    return np.clip(up,0,1),(up>=thr).astype(np.uint8)

def color_cal_robust(lin,smask,nz):
    L=lin.mean(2); hp=L-cv2.GaussianBlur(L,(0,0),6)
    det=((cv2.GaussianBlur(hp,(0,0),1)>15*nz)&(L<np.percentile(L,99.99))).astype(np.uint8)
    k,lab,st,cen=cv2.connectedComponentsWithStats(det); rat=[]
    for i in range(1,k):
        if not 4<=st[i,4]<=200: continue
        m=lab==i; f=lin[m].sum(0)
        if (f>0).all(): rat.append(f/f.mean())
    rat=np.array(rat); cw=1/np.median(rat,0); cw/=cw.mean(); lin=lin*cw
    for c in range(3): lin[...,c]-=np.median(lin[...,c][smask==0])
    return lin,cw
