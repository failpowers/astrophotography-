import fitsio,glob,numpy as np,cv2
a,h=fitsio.read(glob.glob('/mnt/user-data/uploads/DWARF*IC*/*.fits')[0])
a=a[:,250:2025,105:3700]
a=np.stack([cv2.resize(c,None,fx=0.7,fy=0.7,interpolation=cv2.INTER_AREA) for c in a])
H,W=a.shape[1:]
# --- linear background extraction (per channel, stars & nebula rejected) ---
def bgmodel(c):
    o=cv2.morphologyEx(c,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    sm=cv2.resize(o,(W//24,H//24),interpolation=cv2.INTER_AREA)
    hh,ww=sm.shape; ys,xs=np.mgrid[0:hh,0:ww]; x=xs/ww-.5; y=ys/hh-.5
    A=np.stack([np.ones_like(x),x,y,x*y,x*x,y*y,x**3,y**3,x*x*y,x*y*y],-1).reshape(-1,10)
    v=sm.ravel(); k=np.ones_like(v,bool)
    for _ in range(8):
        cf,*_=np.linalg.lstsq(A[k],v[k],rcond=None); r=v-A@cf; s=1.4826*np.median(np.abs(r[k]-np.median(r[k]))); k=(r<1.0*s)&(r>-3*s)
    Y,X=np.mgrid[0:H,0:W]; x=X/W-.5; y=Y/H-.5
    return (np.stack([np.ones_like(x),x,y,x*y,x*x,y*y,x**3,y**3,x*x*y,x*y*y],-1)@cf).astype(np.float32), k.mean()
out=[]
for i,c in enumerate(a):
    bg,kf=bgmodel(c); print('ch',i,'bg range',round(float(bg.max()-bg.min()),1),'kept',round(kf,2))
    out.append(c-bg)
lin=np.stack(out)                       # background ~0
np.save('ic_lin.npy',lin)
noise=[1.4826*np.median(np.abs(c-np.median(c))) for c in lin]; print('noise',np.round(noise,2))
