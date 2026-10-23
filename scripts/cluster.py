import numpy as np,cv2
a=np.load('m34_lin.npy')[:,300:1870,200:3620]
a=np.stack([cv2.resize(c,None,fx=0.7,fy=0.7,interpolation=cv2.INTER_AREA) for c in a]); H,W=a.shape[1:]
def bgmodel(c):
    o=cv2.morphologyEx(c,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    sm=cv2.resize(o,(W//24,H//24),interpolation=cv2.INTER_AREA); hh,ww=sm.shape
    def A(x,y): return np.stack([np.ones_like(x),x,y,x*y,x*x,y*y,x**3,y**3,x*x*y,x*y*y],-1)
    ys,xs=np.mgrid[0:hh,0:ww]; M=A(xs/ww-.5,ys/hh-.5).reshape(-1,10); v=sm.ravel(); k=np.ones_like(v,bool)
    for _ in range(8):
        cf,*_=np.linalg.lstsq(M[k],v[k],rcond=None); r=v-M@cf; s=1.4826*np.median(np.abs(r[k]-np.median(r[k]))); k=np.abs(r)<2.5*s
    Y,X=np.mgrid[0:H,0:W]; return (A(X/W-.5,Y/H-.5)@cf).astype(np.float32)
lin=np.stack([c-bgmodel(c) for c in a]); R,G,B=lin; lin=np.stack([B,G,R],-1)
L=lin.mean(2); nz=1.4826*np.median(np.abs(L-np.median(L)))
# star color calibration (average star = white, slightly warm overall), background neutral
th=L-cv2.morphologyEx(L,cv2.MORPH_OPEN,np.ones((9,9)))
m=(th>20*nz)&(L<np.percentile(L,99.99))
cw=lin[m].sum(0); cw=cw.mean()/cw; lin=lin*cw; print('cw',cw.round(3))
for c in range(3): lin[...,c]-=np.median(lin[...,c])
# NR on background (linear, luminance-guided), stars untouched
L=lin.mean(2); top=np.percentile(L,99.97)
smooth=cv2.GaussianBlur(lin,(0,0),1.6); starm=cv2.GaussianBlur((th>6*nz).astype(np.float32),(0,0),1.5)[...,None]
starm=np.clip(starm*3,0,1); lin=lin*starm+smooth*(1-starm)
# stretch (color preserving)
L=lin.mean(2); x=np.clip((L+2.0*nz)/top,1e-6,1); k=55
Lt=np.arcsinh(k*x)/np.arcsinh(k); rgb=np.clip((lin+2.0*nz)/top*(Lt/x)[...,None],0,1)
rgb=np.clip((rgb-0.05)/0.95,0,1)
# star reduction + saturation
rgb=0.7*rgb+0.3*cv2.erode(rgb,np.ones((2,2),np.uint8))
lab=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_BGR2LAB); lab[...,1:]*=1.35
lab[...,1:]=lab[...,1:]*np.clip(lab[...,0:1]/30,0,1)            # keep sky neutral
out=np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),0,1)
cv2.imwrite('m34.png',(out*255).astype(np.uint8)); print(out.shape)
cv2.imwrite('v_m34.jpg',cv2.resize((out*255).astype(np.uint8),None,fx=0.42,fy=0.42,interpolation=cv2.INTER_AREA))
c=out.shape[0]//2-200,out.shape[1]//2-300; cv2.imwrite('v_m34c.jpg',(out[c[0]:c[0]+400,c[1]:c[1]+600]*255).astype(np.uint8))
