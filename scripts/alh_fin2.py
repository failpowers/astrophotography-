import sys,os; sys.path.insert(0,'.')
import numpy as np, cv2, proc
from scipy import ndimage
lin=np.load('alh_lin0.npy'); raw=np.load('alh_rawcrop.npy'); sm0,nz=proc.star_mask(lin.mean(2))
P=dict(a.split('=') for a in sys.argv[2:]); g=lambda k,d: float(P.get(k,d))
SATL=g('satl',2200)
# --- robust star colour calibration on unsaturated stars
L=lin.mean(2); hp=L-cv2.GaussianBlur(L,(0,0),5)
det=(cv2.GaussianBlur(hp,(0,0),1)>15*nz).astype(np.uint8)
k,lab,st,cen=cv2.connectedComponentsWithStats(det); rats=[]; wts=[]
satm=(raw.max(2)>SATL)
for i in range(1,k):
    if not 4<=st[i,4]<=150: continue
    x0,y0,w0,h0=st[i,:4]; 
    if satm[y0:y0+h0,x0:x0+w0].any(): continue
    m=lab==i; m=cv2.dilate(m.astype(np.uint8),np.ones((3,3),np.uint8))>0
    f=lin[m].sum(0)
    if (f>0).all(): rats.append(f/f.mean()); wts.append(f.mean())
rats=np.array(rats); cw=1/np.median(rats,0); cw/=cw.mean(); print('stars used',len(rats),'cw (BGR)',cw.round(3))
lin=lin*cw
for c in range(3): lin[...,c]-=np.median(lin[...,c][sm0==0])
# --- repair saturated cores: take colour from the unsaturated wing
sat=cv2.dilate(satm.astype(np.uint8),np.ones((3,3),np.uint8))
n,lab2,st2,_=cv2.connectedComponentsWithStats(sat)
L=lin.mean(2)
for i in range(1,n):
    x0,y0,w0,h0,a=st2[i]; pad=8
    ys,xs=slice(max(0,y0-pad),y0+h0+pad),slice(max(0,x0-pad),x0+w0+pad)
    core=(lab2[ys,xs]==i).astype(np.uint8)
    ring=(cv2.dilate(core,np.ones((9,9),np.uint8))-cv2.dilate(core,np.ones((3,3),np.uint8)))>0
    sub=lin[ys,xs]; Ls=sub.mean(2)
    rr=ring&(Ls>12*nz)
    if rr.sum()<4: rat=np.ones(3)
    else:
        rat=np.median(sub[rr]/np.maximum(Ls[rr],1e-3)[:,None],0)
    rat=np.clip(rat,0.7,1.4); rat=g('ringw',0.4)*rat+(1-g('ringw',0.4)); rat/=rat.mean()
    cm=cv2.dilate(core,np.ones((3,3),np.uint8))>0
    sub[cm]=Ls[cm][:,None]*rat[None]
    lin[ys,xs]=sub
np.save('alh_lincal.npy',lin)
rgb,sm=proc.finish(lin,sm0,BGT=g('bgt',0.08),K=g('k',120),SSAT=g('ssat',1.3),lift=g('lift',0.25),nr_h=g('nr',2.2))
# --- chroma noise control: faint stars keep only part of their (noisy) colour
Lr=lin.mean(2); snr=cv2.GaussianBlur(Lr,(0,0),1.0)/nz
Y=rgb.mean(2,keepdims=True); chroma=rgb/np.maximum(Y,1e-4)
chs=cv2.GaussianBlur(chroma,(0,0),g('cblur',1.0))
wt=np.clip((snr-g('snr0',8))/g('snrw',40),0,1)[...,None]
chroma=1+(chs-1)*(g('cmin',0.35)+(1-g('cmin',0.35))*wt)
rgb=np.clip(Y*chroma,0,1)
# --- remove non-physical magenta fringing (longitudinal CA halos): stars are never both redder and bluer than green
b_,g_,r_=rgb[...,0],rgb[...,1],rgb[...,2]; m=np.clip(np.minimum(r_,b_)-g_,0,None)*g('mag',1.0)
rgb[...,0]-=m; rgb[...,2]-=m
# --- bright cores: pull chroma toward white (saturated + CA-tinted halos are not real star colour)
Yb=rgb.mean(2,keepdims=True); chb=rgb/np.maximum(Yb,1e-4); dsat=np.clip((Yb-g('d0',0.45))/0.35,0,1)*g('dsat',0.6)
rgb=np.clip(Yb*(1+(chb-1)*(1-dsat)),0,1)
# --- gentle softon-style glow on the brightest stars so the constellation pattern reads
Lb=rgb.mean(2); bright=np.clip((Lb-g('gthr',0.55))/0.3,0,1)[...,None]*rgb
glow=cv2.GaussianBlur(bright,(0,0),g('gsig',5))*g('glow',0.0)
rgb=1-(1-rgb)*(1-np.clip(glow,0,1))
cov=np.load('alh_cov.npy')
a8=np.clip(rgb*255,0,255).astype(np.uint8); dn=cv2.fastNlMeansDenoisingColored(a8,None,g('nlh',5),g('nlc',7),5,15).astype(np.float32)/255
al=(np.clip((g('cov1',0.75)-cov)/0.5,0,1)*g('covnr',0.85))[...,None]
starp=np.clip((rgb.mean(2)-0.35)/0.3,0,1)[...,None]   # keep bright stars crisp
al=al*(1-starp)
rgb=rgb*(1-al)+dn*al
out=proc.save8(rgb,sys.argv[1]); H,W=out.shape[:2]; print(out.shape)
cv2.imwrite('met/v_alh.jpg',cv2.resize(out,(W*2//3,H*2//3),interpolation=cv2.INTER_AREA))
cv2.imwrite('met/v_alh_c.jpg',out[H//2-200:H//2+200,W//2-300:W//2+300])
