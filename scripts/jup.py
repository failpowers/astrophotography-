import numpy as np,cv2,glob,fitsio
from skimage.restoration import richardson_lucy
fs=sorted(glob.glob('/mnt/user-data/uploads/DWARF*Jupiter*/Jupiter_*.fits'))
crops=[];sh=[]
for f in fs:
    b,_=fitsio.read(f)
    rgb=cv2.cvtColor(np.clip(b,0,65535).astype(np.uint16),cv2.COLOR_BayerRG2BGR).astype(np.float32)
    rgb-=np.median(rgb.reshape(-1,3),0)
    L=cv2.GaussianBlur(rgb.mean(2),(0,0),3); y,x=np.unravel_index(np.argmax(L),L.shape)
    c=rgb[y-40:y+40,x-40:x+40]
    if c.shape[:2]!=(80,80): continue
    Lc=c.mean(2); m=Lc>0.25*Lc.max()
    sh.append(cv2.Laplacian(cv2.GaussianBlur(Lc,(0,0),0.7),cv2.CV_32F)[m].var()/Lc.max()**2); crops.append(c)
crops=np.array(crops); sh=np.array(sh); print(len(sh),'frames; sharpness spread',np.round(np.percentile(sh,[10,50,90])/np.median(sh),2))
best=np.argsort(sh)[::-1][:max(8,len(sh)//4)]
U=3; ref=cv2.resize(crops[best[0]].mean(2),None,fx=U,fy=U,interpolation=cv2.INTER_CUBIC)
acc=np.zeros((80*U,80*U,3),np.float32)
for i in best:
    up=cv2.resize(crops[i],None,fx=U,fy=U,interpolation=cv2.INTER_CUBIC)
    (dx,dy),_=cv2.phaseCorrelate(ref.astype(np.float64),up.mean(2).astype(np.float64))
    M=np.float32([[1,0,-dx],[0,1,-dy]]); acc+=cv2.warpAffine(up,M,(80*U,80*U),flags=cv2.INTER_CUBIC)
st=acc/len(best)
np.save('jup_lucky.npy',st)
dw=np.load('jup_dw.npy')[581-40:581+40,894-40:894+40]
dwu=cv2.resize(dw,None,fx=U,fy=U,interpolation=cv2.INTER_CUBIC)
def proc(x,it):
    L=x.mean(2); top=L.max()
    wb=x[L>0.4*top].mean(0); x=x*(wb.mean()/wb)*np.array([0.92,1.0,1.08])   # neutral then Jupiter's natural cream/tan
    L=x.mean(2)/top
    yy,xx=np.mgrid[-12:13,-12:13]; psf=np.exp(-(xx**2+yy**2)/(2*(1.3*U)**2)); psf/=psf.sum()
    d=richardson_lucy(np.clip(L,1e-4,1),psf,num_iter=it,clip=False)
    d=cv2.GaussianBlur(d.astype(np.float32),(0,0),0.8)
    out=np.clip(x/top*(d/np.maximum(L,1e-3))[...,None],0,1.2)
    return np.clip(out/np.percentile(out,99.95)*0.95,0,1)**(1/1.4)
a=proc(dwu,25); b=proc(st,25)
cv2.imwrite('jup_cmp.png',(np.hstack([a,b])*255).astype(np.uint8))
