import numpy as np,cv2,sys
BGT=float(sys.argv[1]) if len(sys.argv)>1 else 0.085   # target sky level
SSAT=float(sys.argv[2]) if len(sys.argv)>2 else 1.5    # star color saturation
lin=np.load('m34v2_lin.npy'); smask=np.load('m34v2_mask.npy'); H,W=lin.shape[:2]
L=lin.mean(2)
hp=L-cv2.GaussianBlur(L,(0,0),2); nz=1.4826*np.median(np.abs(hp[smask==0]))
# ---------- noise reduction in linear (background only; stars protected) ----------
lo,hi=-8*nz,np.percentile(L,99.999)
u=np.clip((L-lo)/(hi-lo)*65535,0,65535).astype(np.uint16)
nzu=nz/(hi-lo)*65535
Ld=cv2.fastNlMeansDenoising(u,h=[float(nzu*2.2)],templateWindowSize=7,searchWindowSize=21,normType=cv2.NORM_L1).astype(np.float32)/65535*(hi-lo)+lo
th=L-cv2.morphologyEx(L,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
th=cv2.GaussianBlur(th,(0,0),1.0)
sm=np.clip((th-2.0*nz)/(6*nz),0,1)
sm=cv2.dilate(sm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
sm=cv2.GaussianBlur(sm,(0,0),0.8)
print('tight star mask mean',round(float(sm.mean()),3))
Lnr=L*sm+(0.85*Ld+0.15*L)*(1-sm)
# chroma: ratios smoothed strongly in the sky, kept sharp on stars
ratio=lin/np.maximum(np.abs(L),1e-3)[...,None]*np.sign(L)[...,None]
Lpos=np.maximum(L,0)
rs=cv2.GaussianBlur(lin,(0,0),6)/np.maximum(cv2.GaussianBlur(Lpos,(0,0),6),1e-3)[...,None]
starratio=cv2.GaussianBlur(lin,(0,0),1.0)/np.maximum(cv2.GaussianBlur(Lpos,(0,0),1.0),1e-3)[...,None]
rat=starratio*sm[...,None]+(1+0.3*(np.clip(rs,0.6,1.4)-1))*(1-sm[...,None])
rat=np.clip(rat,0.2,2.5)
nz2=1.4826*np.median(np.abs((Lnr-cv2.GaussianBlur(Lnr,(0,0),8))[smask==0]))
print('noise before/after NR',round(float(nz),2),round(float(nz2),2))
# ---------- stretch (color preserving arcsinh) ----------
K=float(sys.argv[3]) if len(sys.argv)>3 else 220
top=np.percentile(Lnr,99.995)
f=lambda x:np.arcsinh(K*x)/np.arcsinh(K)
skyL=np.median(Lnr[smask==0])
bps=np.linspace(skyL-12*nz2,skyL,400)
vals=[f(max(skyL-bp,1e-7)/(top-bp)) for bp in bps]
bp=bps[np.argmin(np.abs(np.array(vals)-BGT))]
x=np.clip((Lnr-bp)/(top-bp),0,1); Lt=f(x)
print('K',K,'black point in sky sigmas',round(float((skyL-bp)/nz2),2),'sky level',round(float(np.median(Lt[smask==0])),3))
Lt=np.clip(Lt+0.32*sm*Lt*(1-Lt),0,1)                       # lift star bodies, sky untouched
# star color: amplify chroma ratios on stars only (saturation in linear ratio space)
satw=1+(SSAT-1)*np.clip(sm*1.2,0,1)*np.clip((Lt-0.12)/0.25,0,1)   # faint/noisy stars get less boost
rat2=1+(rat-1)*satw[...,None]
rat2=rat2/rat2.mean(2,keepdims=True)
rgb=np.clip(Lt[...,None]*rat2,0,1)
b_,g_,r_=rgb[...,0],rgb[...,1],rgb[...,2]
rgb[...,1]=np.minimum(g_,np.maximum((r_+b_)/2,np.minimum(r_,b_))+0.01)   # SCNR: no green stars
# keep cores from clipping to flat white: soft shoulder
rgb=np.where(rgb>0.85,0.85+(1-np.exp(-(rgb-0.85)/0.15))*0.15,rgb)
# ---------- gentle star reduction (faint halos), stars only ----------
er=cv2.erode(rgb,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
rgb=rgb*(1-0.25*sm[...,None])+er*(0.25*sm[...,None])
# ---------- crop 16:9 around the cluster ----------
Lb=cv2.GaussianBlur((L>40*nz).astype(np.float32),(0,0),60)
cy,cx=np.unravel_index(np.argmax(Lb[H//4:3*H//4,W//4:3*W//4]),(H//2,W//2)); cy+=H//4; cx+=W//4
ch=H; cw_=int(round(ch*16/9)); x0=int(np.clip(cx-cw_//2,0,W-cw_)); print('cluster center',cx,cy,'crop x0',x0,'w',cw_)
rgb=rgb[:,x0:x0+cw_]
# ---------- 8-bit with dithering (prevents banding in smooth sky) ----------
rng=np.random.default_rng(7)
out8=np.clip(rgb*255+rng.uniform(-0.5,0.5,rgb.shape),0,255).astype(np.uint8)
cv2.imwrite('m34v2.png',out8)
cv2.imwrite('v_m34v2.jpg',cv2.resize(out8,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA))
h2,w2=out8.shape[:2]; cv2.imwrite('v_m34v2c.jpg',out8[h2//2-220:h2//2+220,w2//2-330:w2//2+330])
