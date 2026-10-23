import sys; sys.path.insert(0,'.')
import numpy as np, cv2, proc
lin=np.load('alh_lin.npy'); sm=np.load('alh_sm.npy')
kw=dict(BGT=float(sys.argv[1]),K=float(sys.argv[2]),SSAT=float(sys.argv[3]),lift=float(sys.argv[4]),nr_h=float(sys.argv[5]))
rgb,s=proc.finish(lin,sm,**kw)
out=proc.save8(rgb,sys.argv[6]); print(out.shape)
H,W=out.shape[:2]
cv2.imwrite('met/v_alh.jpg',cv2.resize(out,(W*2//3,H*2//3),interpolation=cv2.INTER_AREA))
cv2.imwrite('met/v_alh_c.jpg',out[H//2-200:H//2+200,W//2-300:W//2+300])
