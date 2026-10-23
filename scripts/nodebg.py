import numpy as np, cv2
from scipy import ndimage
def node_bg(ch,smask,T):
    H,W=ch.shape; ny=int(np.ceil(H/T))+1; nx=int(np.ceil(W/T))+1
    ys=np.linspace(0,H-1,ny); xs=np.linspace(0,W-1,nx); g=np.full((ny,nx),np.nan,np.float32)
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            y0,y1=int(max(0,y-T)),int(min(H,y+T)); x0,x1=int(max(0,x-T)),int(min(W,x+T))
            t=ch[y0:y1,x0:x1]; ok=smask[y0:y1,x0:x1]==0; v=t[ok]
            if v.size<50: continue
            for _ in range(3):
                md=np.median(v); s=1.4826*np.median(np.abs(v-md))+1e-6; v=v[(v-md<2.0*s)&(v-md>-3*s)]
            g[j,i]=np.median(v)
    bad=np.isnan(g)
    if bad.any():
        idx=ndimage.distance_transform_edt(bad,return_distances=False,return_indices=True); g=g[tuple(idx)]
    md=ndimage.median_filter(g,3,mode='nearest'); dev=g-md; s=1.4826*np.median(np.abs(dev))+1e-6
    g=np.where(np.abs(dev)>3*s,md,g)
    # bicubic surface through the nodes (nodes sit exactly on image borders)
    up=cv2.resize(g,(W,H),interpolation=cv2.INTER_CUBIC) if False else None
    yy=np.linspace(0,ny-1,H); xx=np.linspace(0,nx-1,W)
    from scipy.interpolate import RectBivariateSpline
    sp=RectBivariateSpline(np.arange(ny),np.arange(nx),g,kx=3,ky=3,s=0)
    return sp(yy,xx).astype(np.float32)
def dbe2(lin,smask,Ts=(140,70)):
    for T in Ts:
        bg=np.dstack([node_bg(lin[...,c],smask,T) for c in range(3)]); lin=lin-bg
    return lin
