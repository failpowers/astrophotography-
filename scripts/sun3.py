import numpy as np, cv2, sys
from scipy import ndimage
st=np.load('sun/stack.npy').astype(np.float32)
O=st.shape[0]
print('chan means center', st[O//2-50:O//2+50,O//2-50:O//2+50].reshape(-1,3).mean(0))
L=st.mean(2)
p=np.percentile(L,99.5); m=L>0.5*p
m=ndimage.binary_fill_holes(ndimage.binary_opening(m,iterations=2))
lab,n=ndimage.label(m); sz=ndimage.sum(m,lab,range(1,n+1)); m=lab==(1+np.argmax(sz))
cy,cx=ndimage.center_of_mass(m); R=np.sqrt(m.sum()/np.pi)
print('center',cx,cy,'R',R)
yy,xx=np.mgrid[:O,:O]; rr=np.hypot(xx-cx,yy-cy)
# radial profile (median in 0.5px bins) -> smooth limb darkening model
bins=(rr*2).astype(int); nb=bins.max()+1
prof=ndimage.median(L,bins,index=np.arange(nb))
prof=np.nan_to_num(prof); prof=ndimage.gaussian_filter1d(prof,1.0)
I=prof[bins].astype(np.float32)
np.save('sun/geom.npy',np.array([cx,cy,R]))
inside=rr<R-6
flat=np.ones_like(L); flat[inside]=L[inside]/np.maximum(I[inside],1)
# extend flat smoothly outside disk to avoid edge effects
flat=np.where(inside,flat,1.0).astype(np.float32)
np.save('sun/flat.npy',flat); np.save('sun/I.npy',I); np.save('sun/L.npy',L)
print('flat std inside',flat[rr<0.8*R].std())
