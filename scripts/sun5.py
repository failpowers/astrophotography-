import numpy as np, cv2, sys
exec(open('sun4.py').read().split("if __name__")[0])
from scipy import ndimage
def dering(d,ref,k=5,marg=0.004):
    lo=ndimage.minimum_filter(ref,k)-marg; hi=ndimage.maximum_filter(ref,k)+marg
    return np.clip(d,lo,hi)
def colorize(v,gs=(0.45,1.1,2.2),sc=(1.0,0.74,0.40)):
    v=np.clip(v,0,1.2)
    rgb=np.stack([sc[i]*np.power(np.maximum(v,0),gs[i]) for i in range(3)],-1)
    return rgb
def build(sig=1.6,it=20,den=300,boost=1.25,k=5):
    f0=proc(0,0,den); d=rl(f0,sig,it) if it>0 else f0
    d=dering(d,f0,k)
    d=1+boost*(d-1)
    o=recombine(d)
    Lc=np.percentile(o[rr<0.2*R],50)
    v=o/Lc
    return v
if __name__=='__main__':
    tiles=[]
    for args in [dict(it=0,boost=1.0),dict(sig=1.6,it=20),dict(sig=1.8,it=30,boost=1.35)]:
        v=build(**args); rgb=colorize(v)
        c=rgb[200:400,380:780][:,:,::-1]
        c=np.clip(c*255,0,255).astype(np.uint8); c=cv2.resize(c,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(c,str(args),(10,30),0,0.7,(255,255,255),2); tiles.append(c)
    cv2.imwrite('sun/cmp3.png',np.vstack(tiles))
