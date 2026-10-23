import sys; sys.path.insert(0,'.')
import numpy as np, cv2, proc
name=sys.argv[1]; tag=sys.argv[2]
kw=dict(BGT=float(sys.argv[3]) if len(sys.argv)>3 else 0.075, K=float(sys.argv[4]) if len(sys.argv)>4 else 260, SSAT=float(sys.argv[5]) if len(sys.argv)>5 else 1.6)
if name=='DC':
    lin=np.load('dc_comb.npy'); y,x,h,w=np.load('dc_crop.npy'); lin=lin[y:y+h,x:x+w]; pre_dbe=True
else:
    lin,hd=proc.load(proc.find(name)[0]); cov,m=proc.coverage(lin)
    sm0,_=proc.star_mask(lin.mean(2)); lin=proc.dbe(lin,sm0,m,T1=64,T2=40)      # full-frame gradient removal first
    y,x,h,w=proc.max_rect2(cv2.erode(m,np.ones((25,25))),amin=1.6,amax=1.85); lin=lin[y:y+h,x:x+w]; print('crop',y,x,h,w)
lin=proc.downs(lin)
sm,nz=proc.star_mask(lin.mean(2))
lin=proc.dbe(lin,sm,None,T1=48,T2=28)        # fine residual pass at final scale
lin,cw=(proc.color_cal_robust if name=='DC' else proc.color_cal)(lin,sm,nz); print('color weights',cw.round(3))
np.save(f'{tag}_lin.npy',lin); np.save(f'{tag}_sm.npy',sm)
rgb,_=proc.finish(lin,sm,**kw)
out=proc.save8(rgb,f'{tag}.png'); print(tag,out.shape)
cv2.imwrite(f'v_{tag}.jpg',cv2.resize(out,None,fx=0.45,fy=0.45,interpolation=cv2.INTER_AREA))
H,W=out.shape[:2]; cv2.imwrite(f'v_{tag}_c.jpg',out[H//2-220:H//2+220,W//2-330:W//2+330])
