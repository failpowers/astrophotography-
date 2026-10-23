import sys; sys.path.insert(0,'.')
import numpy as np, cv2, proc
U='/mnt/user-data/uploads/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan/'
st=np.load(U+'alhena_stack2.npy'); cnt=np.load(U+'alhena_count2.npy').astype(np.float32).min(2)
print('count max',cnt.max())
valid=((cnt>=float(__import__("os").environ.get("CT","0.45"))*cnt.max())&np.isfinite(st).all(2)).astype(np.uint8)
valid=cv2.erode(valid,np.ones((9,9),np.uint8))
y,x,h,w=proc.max_rect2(valid,step=4,amin=1.7,amax=1.80); y,x,h,w=y+12,x+12,h-24,w-24; print('crop',y,x,h,w)
np.save('alh_crop.npy',np.array([y,x,h,w]))
lin=np.nan_to_num(st[y:y+h,x:x+w]).astype(np.float32)
sm,nz=proc.star_mask(lin.mean(2))
T1,T2=int(sys.argv[1]) if len(sys.argv)>1 else 96,int(sys.argv[2]) if len(sys.argv)>2 else 64
import nodebg
lin=nodebg.dbe2(lin,sm,(T1,T2))
sm,nz=proc.star_mask(lin.mean(2))
np.save('alh_lin0.npy',lin); np.save('alh_cov.npy',(cnt[y:y+h,x:x+w]/cnt.max()).astype(np.float32)); np.save('alh_rawcrop.npy',np.nan_to_num(st[y:y+h,x:x+w]).astype(np.float32))
lin,cw=proc.color_cal(lin,sm,nz); print('color w',cw.round(3),'nz',nz)
np.save('alh_lin.npy',lin); np.save('alh_sm.npy',sm)
