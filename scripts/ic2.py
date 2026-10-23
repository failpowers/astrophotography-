import numpy as np,cv2,sys
STR=float(sys.argv[1]) if len(sys.argv)>1 else 120   # stretch strength
SAT=float(sys.argv[2]) if len(sys.argv)>2 else 1.35
R,G,B=np.load('ic_lin.npy'); lin=np.stack([B,G,R],-1)          # BGR linear, bg~0
H,W=lin.shape[:2]
L=lin.mean(2); nz=1.4826*np.median(np.abs(L-np.median(L)))
# ---- star mask (linear) ----
th=L-cv2.morphologyEx(L,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
core=(cv2.GaussianBlur(th,(0,0),1)>3.5*nz).astype(np.uint8)
n,lab,st,_=cv2.connectedComponentsWithStats(core)
mask=np.zeros((H,W),np.float32)
for i in range(1,n):                      # grow mask with star size
    x,y,w,h,a=st[i]; r=int(3+1.2*np.sqrt(a)+ (0.06*a if a>60 else 0)); cx,cy=x+w//2,y+h//2
    cv2.circle(mask,(int(cx),int(cy)),r,1,-1)
print('stars',n-1,'mask',round(mask.mean()*100,1),'%')
# ---- starless via normalized convolution fill (linear) ----
keep=1-mask; fill=np.zeros_like(lin); got=np.zeros((H,W,1),bool)
for s in (3,6,12,24):
    num=cv2.GaussianBlur(lin*keep[...,None],(0,0),s); den=cv2.GaussianBlur(keep,(0,0),s)[...,None]
    ok=(den>0.2)&~got; fill=np.where(ok,num/np.maximum(den,1e-4),fill); got|=ok
rng=np.random.default_rng(3); sd=np.array([np.std(lin[...,c][keep>0]-cv2.GaussianBlur(lin[...,c],(0,0),2)[keep>0]) for c in range(3)])
fill=fill+rng.normal(0,1,lin.shape).astype(np.float32)*sd*0.9
mf=cv2.GaussianBlur(mask,(0,0),1.0)[...,None]
starless=lin*(1-mf)+fill*mf
stars=np.clip(lin-starless,0,None)
# ---- linear-domain NR on starless: NLM on luminance (16-bit), heavy chroma smoothing ----
Lst=starless.mean(2); lo_,hi_=np.percentile(Lst,0.01),np.percentile(Lst,99.999)
u=np.clip((Lst-lo_)/(hi_-lo_)*65535,0,65535).astype(np.uint16)
nzu=1.4826*np.median(np.abs(u.astype(np.float32)-np.median(u)))
Ld=cv2.fastNlMeansDenoising(u,h=[float(nzu*1.6)],templateWindowSize=7,searchWindowSize=21,normType=cv2.NORM_L1).astype(np.float32)/65535*(hi_-lo_)+lo_
Ld=0.85*Ld+0.15*cv2.GaussianBlur(Lst,(0,0),1.0)
chrom=cv2.GaussianBlur(starless,(0,0),8)/np.maximum(cv2.GaussianBlur(Lst,(0,0),8),1e-3)[...,None]
starless=chrom*Ld[...,None]
print('NR done')
# ---- stretch starless (color-preserving asinh) ----
top=np.percentile(starless.mean(2),99.98)*1.15
def stretch(x,k): return np.arcsinh(k*x)/np.arcsinh(k)
Ls=starless.mean(2); med=np.median(Ls); nzs=1.4826*np.median(np.abs(Ls-med)); bp=med-1.3*nzs
top=np.percentile(Ls,99.97); print('med',med,'nzs',nzs,'top',top)
xs=np.clip((Ls-bp)/(top-bp),1e-6,1)
xb=np.median(xs); ks=np.logspace(-1,4,500); kk=ks[np.argmin(np.abs(stretch(xb,ks)-STR/1000))]
print('bg x',round(float(xb),4),'k',round(float(kk),1)); Lt=stretch(xs,kk)
ratio=(Lt/xs)[...,None]
neb=np.clip((starless-bp)/(top-bp)*ratio,0,1)
neb=np.clip((neb-0.03)/0.97,0,1)
# ---- noise reduction: luminance NLM (mild), chroma heavy ----
labi=cv2.cvtColor((neb*255).astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
Lr=labi[:,:,0]

for c in (1,2): labi[:,:,c]=cv2.GaussianBlur(labi[:,:,c],(0,0),5)
# local contrast on large scales (gentle)
Lc=labi[:,:,0]; big=cv2.GaussianBlur(Lc,(0,0),25)
labi[:,:,0]=np.clip(Lc+0.25*(Lc-big),0,255)
labi[:,:,0]=255*np.clip((labi[:,:,0]/255-0.035)/0.965,0,1)**1.08
# saturation: lift on nebula, hold background neutral; nudge Ha toward crimson
w=np.clip((labi[:,:,0]-np.percentile(labi[:,:,0],50))/(np.percentile(labi[:,:,0],99.5)-np.percentile(labi[:,:,0],50)),0,1)
bga=np.median(labi[:,:,1]); bgb=np.median(labi[:,:,2])
labi[:,:,1]=128+(labi[:,:,1]-bga)*(0.2+SAT*w)
labi[:,:,2]=128+(labi[:,:,2]-bgb)*(0.2+SAT*0.8*w)
neb=cv2.cvtColor(np.clip(labi,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)/255
# ---- stars: separate gentler stretch, slight reduction ----
stop=np.percentile(stars.max(2),99.99); sx=np.clip(stars/stop,0,1); sl=stars.mean(2); lx=np.clip(sl/stop,1e-6,1)
sr=(stretch(lx,90)/lx)[...,None]; st=np.clip(sx*sr,0,1)
st=0.75*st+0.25*cv2.erode(st,np.ones((2,2),np.uint8))
st=np.clip(st*1.1,0,1)
out=1-(1-neb)*(1-st)
cv2.imwrite('D_natural.png',(np.clip(out,0,1)*255).astype(np.uint8))
cv2.imwrite('D_starless.png',(np.clip(neb,0,1)*255).astype(np.uint8))
cv2.imwrite('v_D.jpg',cv2.resize((out*255).astype(np.uint8),None,fx=0.42,fy=0.42,interpolation=cv2.INTER_AREA))
cv2.imwrite('v_Dcrop.jpg',(out[350:750,950:1550]*255).astype(np.uint8))
