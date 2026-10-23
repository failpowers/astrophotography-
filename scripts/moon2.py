import numpy as np,cv2
img=np.load('moon_lin.npy')[:,308:1650]                    # BGR linear, same framing as your earlier edit
sky=np.median(img[:200,:200].reshape(-1,3),0); img=np.clip(img-sky,0,None)
L=img.mean(2); m=L>np.percentile(L,85)
wb=img[m].mean(0); wb=wb.mean()/wb; wb*=np.array([0.985,1.0,1.012])    # neutral grey, hint of natural warmth
img=img*wb
top=np.percentile(img[m].mean(1),99.9)
x=np.clip(img/top,0,1.2)
# gentle tone curve (mild gamma; moon is high-SNR so little stretch needed)
L=x.mean(2); Lt=np.clip(L,0,None)**1.0; Lt=Lt/np.percentile(Lt[m],99.9)*0.90
rgb=x*(Lt/np.maximum(L,1e-5))[...,None]
# multiscale sharpening on luminance (restrained), noise-aware
lab=cv2.cvtColor(np.clip(rgb,0,1).astype(np.float32),cv2.COLOR_BGR2LAB)
Lc=lab[:,:,0]
from skimage.restoration import richardson_lucy
Ln=Lc/100.0
yy,xx=np.mgrid[-7:8,-7:8]; psf=np.exp(-(xx**2+yy**2)/(2*1.6**2)); psf/=psf.sum()
Ld=cv2.bilateralFilter(Ln.astype(np.float32),5,0.012,3)
dec=richardson_lucy(np.clip(Ld,1e-4,1),psf,num_iter=35,clip=False)
dec=np.clip(dec,0,1.1)
d2=dec-cv2.GaussianBlur(dec,(0,0),3.0)
Ls=(dec+0.25*d2)*100
# gentle S-curve for natural local contrast
t=np.clip(Ls/100,0,1); Ls=100*(t+0.10*np.sin(np.pi*(t-0.08))*t*(1-t)*4)
mk=cv2.GaussianBlur(cv2.erode((Lc>4).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32),(0,0),2.5)        # only on the moon, sky stays black
lab[:,:,0]=np.clip(Lc*(1-mk)+Ls*mk,0,100)
lab[:,:,1:]=cv2.GaussianBlur(lab[:,:,1:],(0,0),2)*0.6           # calm chroma noise, near-neutral
out=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
out[lab[:,:,0]<2.5]=0
cv2.imwrite('moon2.png',(np.clip(out,0,1)*255).astype(np.uint8))
o=cv2.imread('/mnt/user-data/outputs/moon_enhanced.jpg')
cv2.imwrite('moon_cmp.jpg',np.hstack([o[250:880,480:980],(np.clip(out,0,1)*255).astype(np.uint8)[250:880,480:980]]))
