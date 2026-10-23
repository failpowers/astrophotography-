import numpy as np, cv2, sys
flat=np.load('sun/flat.npy'); I=np.load('sun/I.npy'); L=np.load('sun/L.npy')
cx,cy,R=np.load('sun/geom.npy'); O=L.shape[0]
yy,xx=np.mgrid[:O,:O]; rr=np.hypot(xx-cx,yy-cy)
def G(a,s): return cv2.GaussianBlur(a,(0,0),s)
def rl(img,s,it):
    est=img.copy()
    for _ in range(it):
        ratio=img/np.maximum(G(est,s),1e-6); est=est*G(ratio,s)
    return est
def proc(sig,it,den):
    f=flat.copy()
    if den>0:
        # light edge-preserving denoise of compression residue
        f8=np.clip((f-0.6)/0.6*65535,0,65535).astype(np.uint16)
        f8=cv2.fastNlMeansDenoising(f8,h=[den],templateWindowSize=5,searchWindowSize=15,normType=cv2.NORM_L1)
        f=f8.astype(np.float32)/65535*0.6+0.6
    d=rl(f,sig,it) if it>0 else f
    return d
def recombine(d):
    out=L.copy()
    w=np.clip((R-6-rr)/10,0,1).astype(np.float32)  # feather to original near limb
    out=w*(d*I)+(1-w)*L
    return out
if __name__=='__main__':
    tiles=[]
    for sig,it,den in [(0,0,0),(1.6,15,0),(1.6,25,300),(2.0,25,300),(2.0,40,500)]:
        o=recombine(proc(sig,it,den))
        c=o[200:400,380:780]
        c=np.clip(c/190*255,0,255).astype(np.uint8); c=cv2.resize(c,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(c,f's{sig} it{it} nr{den}',(10,30),0,0.9,255,2); tiles.append(c)
    cv2.imwrite('sun/cmp.png',np.vstack(tiles[:3])); cv2.imwrite('sun/cmp2.png',np.vstack([tiles[0]]+tiles[3:]))
