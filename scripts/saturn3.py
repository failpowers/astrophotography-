import numpy as np, cv2
from skimage.restoration import richardson_lucy
img=np.load('sat/stack.npy')          # 4x upsampled sigma-clipped stack (BGR), from saturn2.py 0.12
bg=np.median(img[:25,:25].reshape(-1,3),0); img=np.clip(img-bg,0,None)
# --- RGB channel alignment (atmospheric dispersion) ---
G=img[...,1].astype(np.float64); win=cv2.createHanningWindow(G.shape[::-1],cv2.CV_64F)
for c in (0,2):
    (sx,sy),r=cv2.phaseCorrelate(G,img[...,c].astype(np.float64),win)
    img[...,c]=cv2.warpAffine(img[...,c],np.float32([[1,0,-sx],[0,1,-sy]]),G.shape[::-1],flags=cv2.INTER_CUBIC)
    print('channel',c,'shift (4x px)',round(sx,2),round(sy,2))
L=img.mean(2); top=np.percentile(L,99.9)
m=L>0.35*top; cw=img[m].mean(0); cw=cw.mean()/cw; img=img*cw*np.array([0.88,0.98,1.06])   # neutral globe -> pale gold
L=img.mean(2)/top
yy,xx=np.mgrid[-14:15,-14:15]; psf=np.exp(-(xx**2+yy**2)/(2*2.8**2)); psf/=psf.sum()
dec=richardson_lucy(np.clip(L,1e-5,1.5),psf,num_iter=40,clip=False).astype(np.float32)
dec=cv2.bilateralFilter(dec,5,0.03,2)
ratio=cv2.GaussianBlur(img/top,(0,0),3.0)/np.maximum(cv2.GaussianBlur(L,(0,0),3.0),1e-3)[...,None]
w=np.clip((cv2.GaussianBlur(L,(0,0),3)-0.05)/0.25,0,1)[...,None]       # faint edges -> neutral (no purple fringe)
ratio=1+(ratio-1)*w; ratio/=ratio.mean(2,keepdims=True)
out=dec[...,None]*ratio
out=out/np.percentile(out,99.9)*0.84
out=np.where(out>0.78,0.78+(1-np.exp(-(out-0.78)/0.22))*0.22,out)
out=np.clip(out,0,1)**(1/1.15)
fade=np.clip((cv2.GaussianBlur(L,(0,0),3)-0.015)/0.05,0,1)[...,None]; out=out*fade
# frame: 1350x1080 (4:5 portrait-friendly), Saturn upscaled to ~560 px across the rings
Lb=cv2.GaussianBlur(out.mean(2),(0,0),6); y,x=np.unravel_index(np.argmax(Lb),Lb.shape)
ys,xs=np.where(out.mean(2)>0.08); ringw=xs.max()-xs.min(); print('ring span (4x px)',ringw)
sc=430/ringw; big=cv2.resize(out,None,fx=sc,fy=sc,interpolation=cv2.INTER_LANCZOS4); big=np.clip(big,0,1)
Hc,Wc=1080,1350; can=np.zeros((Hc,Wc,3),np.float32)
cy,cx=int(y*sc),int(x*sc); y0=cy-Hc//2+40; x0=cx-Wc//2
ys0,xs0=max(0,-y0),max(0,-x0); y1=min(Hc,big.shape[0]-y0); x1=min(Wc,big.shape[1]-x0)
can[ys0:y1,xs0:x1]=big[y0+ys0:y0+y1,x0+xs0:x0+x1]
rng=np.random.default_rng(3)
cv2.imwrite('saturn_final.png',np.clip(can*255+rng.uniform(-0.5,0.5,can.shape),0,255).astype(np.uint8))
cv2.imwrite('sat/v_final.jpg',cv2.resize((can*255).astype(np.uint8),None,fx=0.5,fy=0.5))
