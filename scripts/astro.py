import cv2, numpy as np, sys, time
T=time.time()
def log(*a): print(f'[{time.time()-T:5.1f}s]',*a,flush=True)
src,tag=sys.argv[1],sys.argv[2]
HIM=float(sys.argv[3]) if len(sys.argv)>3 else 1.12
NLH=float(sys.argv[4]) if len(sys.argv)>4 else 4
BASES=float(sys.argv[5]) if len(sys.argv)>5 else 4.0
if src.endswith('.png'):
    img=cv2.imread(src,cv2.IMREAD_UNCHANGED).astype(np.float32)/65535.0
else:
    img=cv2.imread(src)[90:, :-60].astype(np.float32)/255.0      # crop stacking-edge artifacts
H,W=img.shape[:2]
L=img.mean(2)
# ---------- star mask ----------
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
th=cv2.morphologyEx(L,cv2.MORPH_TOPHAT,k)
noise=np.median(np.abs(th-np.median(th)))*1.4826
smask=(th>6*noise).astype(np.uint8)
smask=cv2.dilate(smask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
big=(cv2.GaussianBlur(L,(0,0),2)>np.percentile(L,99.7)).astype(np.uint8)
smask=np.maximum(smask,cv2.dilate(big,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))))
log('star mask coverage',round(smask.mean()*100,1),'%')
# ---------- star removal: normalized-convolution fill from surrounding sky/nebula ----------
m=np.maximum(smask,cv2.dilate(big,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))).astype(np.float32)
keepw=1-m
fill=np.zeros_like(img)
for sig in (3,7,15):                       # multi-radius so big stars fill too
    num=cv2.GaussianBlur(img*keepw[...,None],(0,0),sig); den=cv2.GaussianBlur(keepw,(0,0),sig)[...,None]
    f=num/np.maximum(den,1e-4)
    fill=np.where((fill.sum(2,keepdims=True)==0)&(den>0.25),f,fill)
fill=np.where(fill.sum(2,keepdims=True)==0,cv2.GaussianBlur(img,(0,0),20),fill)
# add matching grain so filled spots don't look smooth
rng=np.random.default_rng(1); sd=np.std((img-cv2.GaussianBlur(img,(0,0),2))[keepw>0],0)
fill=fill+rng.normal(0,1,img.shape).astype(np.float32)*sd*0.8
mf=cv2.GaussianBlur(m,(0,0),1.2)[...,None]
starless=img*(1-mf)+fill*mf
log('starless done')
# ---------- background (light pollution / gradient) model ----------
sm=cv2.resize(starless,(W//16,H//16),interpolation=cv2.INTER_AREA)
hh,ww=sm.shape[:2]; ys,xs=np.mgrid[0:hh,0:ww]; xn=xs/ww-0.5; yn=ys/hh-0.5
A=np.stack([np.ones_like(xn),xn,yn,xn*yn,xn**2,yn**2],-1).reshape(-1,6)
bgfull=np.zeros_like(img)
Yf,Xf=np.mgrid[0:H,0:W]; Af=np.stack([np.ones((H,W)),Xf/W-0.5,Yf/H-0.5,(Xf/W-0.5)*(Yf/H-0.5),(Xf/W-0.5)**2,(Yf/H-0.5)**2],-1)
lum_s=sm.mean(2).ravel(); keep=np.ones_like(lum_s,bool)
for it in range(6):                       # iteratively reject nebula (positive outliers)
    c,*_=np.linalg.lstsq(A[keep],lum_s[keep],rcond=None); r=lum_s-A@c
    s=np.std(r[keep]); keep=r<0.6*s
for ch in range(3):
    c,*_=np.linalg.lstsq(A[keep],sm[:,:,ch].ravel()[keep],rcond=None)
    bgfull[:,:,ch]=(Af@c).astype(np.float32)
ped=0.02
img=img-bgfull+ped; starless=starless-bgfull+ped
log('bg fit kept',round(keep.mean()*100),'% samples; bg range',[round(float(bgfull[:,:,i].max()-bgfull[:,:,i].min()),3) for i in range(3)])
# ---------- star-based color calibration ----------
stars=np.clip(img-starless,0,None)
m=(smask>0)&(stars.sum(2)>0.15)
cw=stars[m].mean(0); cw=cw.mean()/cw; cw=1+(cw-1)*0.7
img=(img-ped)*cw+ped; starless=(starless-ped)*cw+ped; stars=stars*cw
log('color weights',cw.round(3))
# ---------- noise reduction on starless ----------
s8=np.clip(starless*255,0,255).astype(np.uint8)
lab=cv2.cvtColor(s8,cv2.COLOR_BGR2LAB).astype(np.float32)
Lc=cv2.fastNlMeansDenoising(lab[:,:,0].astype(np.uint8),None,NLH,7,21).astype(np.float32)
lab[:,:,0]=Lc*0.65+lab[:,:,0]*0.35                         # keep a touch of natural grain
for c in (1,2): lab[:,:,c]=cv2.GaussianBlur(cv2.medianBlur(lab[:,:,c].astype(np.uint8),7).astype(np.float32),(0,0),6)
starless=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)/255
log('denoise done')
# ---------- multiscale stretch: stretch large-scale structure, keep fine detail un-amplified ----------
base=cv2.GaussianBlur(starless,(0,0),BASES)
detail=starless-base
Lb=base.mean(2); med=np.median(Lb); mad=np.median(np.abs(Lb-med))*1.4826
lo=med-0.8*mad; hi=np.percentile(Lb,99.98)*HIM
x=np.clip((base-lo)/(hi-lo),0,1)
for c in range(3): x[:,:,c]=np.clip(x[:,:,c]-np.median(x[:,:,c])+np.median(x),0,1)   # neutral bg
x[:,:,1]=np.minimum(x[:,:,1],(x[:,:,0]+x[:,:,2])/2)                                # SCNR green
nb=np.arcsinh(1.4*x)/np.arcsinh(1.4)
nb=np.clip((nb-0.03)/0.97,0,1)**1.1
gain=np.clip(nb.mean(2,keepdims=True)/np.maximum(x.mean(2,keepdims=True),1e-3),1,3.0)
neb=np.clip(nb+detail.mean(2,keepdims=True)*gain*0.9,0,1)
lab=cv2.cvtColor((neb*255).astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
Lr=lab[:,:,0]; w=np.clip((Lr-np.percentile(Lr,55))/(np.percentile(Lr,99.5)-np.percentile(Lr,55)),0,1)
sat=0.65+0.35*w
lab[:,:,1]=128+(lab[:,:,1]-128)*sat*(1+0.18*w)   # a: toward crimson
lab[:,:,2]=128+(lab[:,:,2]-128)*sat*(1-0.22*w)   # b: less yellow
lab[:,:,0]=lab[:,:,0]*(1-0.06*w)
neb=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)/255
# ---------- stars: reduce & recombine ----------
stars=np.clip(stars-0.012,0,None)
st=np.clip(stars/np.percentile(stars.max(2)[m>0],99.0),0,1)
st=np.arcsinh(1.5*st)/np.arcsinh(1.5)
st=cv2.erode(st,np.ones((2,2),np.uint8))*0.5+st*0.5        # gentle star reduction
st=cv2.GaussianBlur(st,(0,0),0.4)
def screen(a,b,k): return 1-(1-a)*(1-b*k)
out=screen(neb,st,0.85)
cv2.imwrite(f'{tag}_natural.png',(np.clip(out,0,1)*255).astype(np.uint8))
cv2.imwrite(f'{tag}_starless.png',(np.clip(neb,0,1)*255).astype(np.uint8))
np.save(f'{tag}_neb.npy',neb); np.save(f'{tag}_stars.npy',st)
# ---------- Hubble-style (artistic remap from broadband, labeled as such) ----------
Bc,Gc,Rc=[neb[:,:,i] for i in range(3)]
Hp=Rc; Op=(Gc+Bc)/2
r=np.clip(0.85*Hp+0.25*Op,0,1); g=np.clip(0.62*Hp+0.38*Op,0,1); b=np.clip(0.15*Hp+0.95*Op,0,1)
hub=np.dstack([b,g,r])
lab=cv2.cvtColor((hub*255).astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
lab[:,:,1:]=128+(lab[:,:,1:]-128)*1.25
hub=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)/255
st_n=np.dstack([st.max(2)]*3)*0.6+st*0.4              # desaturate stars a bit, Hubble-style
out2=screen(hub,st_n,0.75)
cv2.imwrite(f'{tag}_hubble.png',(np.clip(out2,0,1)*255).astype(np.uint8))
log('wrote',tag)
