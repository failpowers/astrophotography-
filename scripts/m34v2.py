import numpy as np,cv2
from scipy import ndimage
a=np.load('m34_lin.npy')[:,310:1840,215:3600].astype(np.float32)      # stay clear of stacking edges
a=np.stack([cv2.resize(c,None,fx=0.7,fy=0.7,interpolation=cv2.INTER_AREA) for c in a])
lin=np.stack([a[2],a[1],a[0]],-1)                                      # BGR
H,W=lin.shape[:2]
L=lin.mean(2)
# ---------- robust star mask (multi-scale, size-dependent dilation) ----------
hp=L-cv2.GaussianBlur(L,(0,0),8)
nz=1.4826*np.median(np.abs(hp-np.median(hp)))
det=(cv2.GaussianBlur(hp,(0,0),1.0)>3.0*nz).astype(np.uint8)
n,lab,st,cen=cv2.connectedComponentsWithStats(det)
smask=np.zeros((H,W),np.uint8)
for i in range(1,n):
    r=int(3+1.8*np.sqrt(st[i,4]))
    cv2.circle(smask,(int(cen[i][0]),int(cen[i][1])),r,1,-1)
print('stars',n-1,'mask %',round(smask.mean()*100,1))
# ---------- DBE-style grid background (per channel, sigma-clipped tiles) ----------
T=48
gy,gx=H//T,W//T
def grid_bg(ch):
    g=np.full((gy,gx),np.nan,np.float32)
    for j in range(gy):
        for i in range(gx):
            t=ch[j*T:(j+1)*T,i*T:(i+1)*T]; m=smask[j*T:(j+1)*T,i*T:(i+1)*T]==0
            v=t[m]
            if v.size<T*T*0.25: continue
            for _ in range(3):
                md=np.median(v); s=1.4826*np.median(np.abs(v-md)); v=v[np.abs(v-md)<2.5*s]
            g[j,i]=np.median(v)
    # fill holes, reject outlier tiles, smooth
    bad=np.isnan(g); idx=ndimage.distance_transform_edt(bad,return_distances=False,return_indices=True)
    g=g[tuple(idx)]
    md=ndimage.median_filter(g,5,mode='nearest'); dev=g-md; s=1.4826*np.median(np.abs(dev))
    g=np.where(np.abs(dev)>2.5*s,md,g)
    g=ndimage.gaussian_filter(g,1.2,mode='nearest')
    up=cv2.resize(g,(gx*T,gy*T),interpolation=cv2.INTER_CUBIC)
    up=cv2.copyMakeBorder(up,0,H-gy*T,0,W-gx*T,cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(up,(0,0),T*0.6)
bg=np.dstack([grid_bg(lin[...,c]) for c in range(3)])
print('gradient removed (ADU per channel):',[round(float(np.ptp(bg[...,c])),1) for c in range(3)])
lin=lin-bg
# ---------- color calibration: average star = white; background neutral ----------
L=lin.mean(2)
stars=(smask>0)&(L>20*nz)&(L<np.percentile(L,99.995))
cw=np.array([lin[...,c][stars].sum() for c in range(3)]); cw=cw.mean()/cw
lin*=cw; print('star color weights',cw.round(3))
for c in range(3): lin[...,c]-=np.median(lin[...,c][smask==0])
np.save('m34v2_lin.npy',lin); np.save('m34v2_mask.npy',smask)
cv2.imwrite('m34v2_bgmodel.jpg',(cv2.resize((bg-bg.min())/(np.ptp(bg)),None,fx=0.4,fy=0.4)*255).astype(np.uint8))
# check flatness: residual large-scale structure in starless background
Lb=lin.mean(2).copy(); Lb[smask>0]=np.nan
tiles=np.nanmedian(Lb[:gy*T,:gx*T].reshape(gy,T,gx,T),axis=(1,3))
print('residual bg tile std / pixel noise:',round(float(np.nanstd(tiles)),2),'/',round(float(nz),2))
# ---------- second, finer pass to catch edge roll-off (stacking edge / glow) ----------
T=28; gy,gx=H//T,W//T
bg2=np.dstack([grid_bg(lin[...,c]) for c in range(3)])
lin=lin-bg2
for c in range(3): lin[...,c]-=np.median(lin[...,c][smask==0])
lin=lin[10:-14,10:-16]; smask=smask[10:-14,10:-16]
np.save('m34v2_lin.npy',lin); np.save('m34v2_mask.npy',smask)
Lb=np.where(smask>0,np.nan,lin.mean(2)); H2,W2=Lb.shape
print('edge check bottom/right/top/left:',[round(float(np.nanmedian(x)),2) for x in (Lb[-40:],Lb[:,-40:],Lb[:40],Lb[:,:40])],'center',round(float(np.nanmedian(Lb[H2//3:2*H2//3,W2//3:2*W2//3])),2))
