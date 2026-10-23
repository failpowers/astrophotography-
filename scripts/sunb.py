import numpy as np, cv2, subprocess
exec(open('sun.py').read().split('# ---- pass 1')[0])
meta=np.load('sun/meta.npy'); q=meta[:,4]; ok=q>0
k=int(meta[ok][np.argsort(q[ok])[len(q[ok])//2],0]); print('median-sharpness frame',k)
p=subprocess.run(['ffmpeg','-v','error','-i',V,'-vf',f'select=eq(n\\,{k}),crop={C}:{C}:{X0}:{Y0}','-vframes','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True)
f=np.frombuffer(p.stdout,np.uint8).reshape(C,C,3).astype(np.float32)
O=820; cx,cy=meta[k,1],meta[k,2]
f=cv2.warpAffine(f,np.float32([[1,0,O/2-cx],[0,1,O/2-cy]]),(O,O),flags=cv2.INTER_CUBIC)
exec(open('sun5.py').read().split("if __name__=='__main__'")[0])
L1=f.mean(2); Lc=np.percentile(L1[rr<0.2*R],50); v=L1/Lc
rgb=colorize(v); rgb=rgb*(1-np.clip((rr-(R+3))/4,0,1)[...,None])
up=cv2.resize(rgb,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
out=np.zeros((1640,1640,3),np.float32); ox=int(round(820-cx_*2)) if False else int(round(820-410*2)); out[:up.shape[0],:up.shape[1]]=up
a=np.clip(out,0,1)*255; cv2.imwrite('sun_before2x.png',a.astype(np.uint8)[:,:,::-1]); print('ok')
