import numpy as np, cv2
exec(open('sun5.py').read().split("if __name__=='__main__'")[0])
v=build(sig=1.6,it=20,den=300,boost=1.25)
rgb=colorize(v)            # float RGB, disk on black
# soft black outside the limb (stack background is ~0 already); clean residual glow
bg=np.clip((rr-(R+3))/4,0,1)[...,None]
rgb=rgb*(1-bg)
np.save('sun/disk_rgb.npy',rgb.astype(np.float32))
def frame(S,W,H,cxF,cyF):
    up=cv2.resize(rgb,None,fx=S,fy=S,interpolation=cv2.INTER_LANCZOS4)
    out=np.zeros((H,W,3),np.float32)
    ox=int(round(cxF-cx*S)); oy=int(round(cyF-cy*S))
    h,w=up.shape[:2]
    xs,ys=max(0,ox),max(0,oy); xe,ye=min(W,ox+w),min(H,oy+h)
    out[ys:ye,xs:xe]=up[ys-oy:ye-oy,xs-ox:xe-ox]
    return out
def save8(rgb,path):
    a=np.clip(rgb,0,1)*255+np.random.default_rng(1).uniform(-0.5,0.5,rgb.shape)
    cv2.imwrite(path,np.clip(a,0,255).astype(np.uint8)[:,:,::-1])
f=frame(1.2,1350,1080,720,532); save8(f,'sun_final.png')
m=frame(2.0,1640,1640,820,820); save8(m,'sun_master2x.png')
print('ok')
