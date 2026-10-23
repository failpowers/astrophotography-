# Robust detector for the OHN keyed watermark: tolerates screenshot cropping, rescaling and re-compression
import cv2, numpy as np, sys
from wm import pattern, SIZES, detect as detect_fast
def _upat(W,H):
    h2=((H+1)//2+1)//2; w2=((W+1)//2+1)//2
    P=pattern((h2,w2)).astype(np.float32)
    return np.kron(P,np.ones((4,4),np.float32))[:H,:W]
def _corr(Ys,U):
    H,W=U.shape; h,w=Ys.shape
    hp=Ys-cv2.GaussianBlur(Ys,(0,0),4); hp=hp-cv2.GaussianBlur(hp,(0,0),12)
    m=(cv2.GaussianBlur(Ys,(0,0),12)>25).astype(np.float32)
    a=hp*m
    if m.sum()<4000: return 0.0
    sh=(cv2.getOptimalDFTSize(H+h),cv2.getOptimalDFTSize(W+w))
    A=np.fft.rfft2(a,sh); B=np.fft.rfft2(U,sh)
    c=np.fft.irfft2(np.conj(A)*B,sh)   # c[dy,dx] = sum a(x) U(x+d)
    # valid offsets: screenshot placed inside original with offset 0..(H-h),(W-w) (allow small negatives via wrap)
    sub=np.concatenate([c[-24:],c[:max(H-h,0)+25]],0); sub=np.concatenate([sub[:,-24:],sub[:,:max(W-w,0)+25]],1)
    s=sub.std()
    return float((sub.max()-sub.mean())/s) if s>0 else 0.0
def detect_robust(bgr,maxcrop=0.08,step=0.004):
    h,w=bgr.shape[:2]; Y=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
    best=(0.0,None)
    for W,H in SIZES:
        if abs((W/H)/(w/h)-1)>0.12: continue
        U=_upat(W,H); U=U-U.mean()
        for c in np.arange(0,maxcrop+1e-9,step):
            f=W*(1-c)/w
            if h*f>H*1.001: continue
            Ys=cv2.resize(Y,(max(8,int(round(w*f))),max(8,int(round(h*f)))),interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_CUBIC)
            z=_corr(Ys,U)
            if z>best[0]: best=(z,(W,H,round(c,3)))
    return best
def check(path):
    im=cv2.imread(path); zf=detect_fast(im)
    if zf>6: return zf,'exact'
    zr,info=detect_robust(im)
    return zr,info
if __name__=='__main__':
    z,info=check(sys.argv[1])
    thr=6 if info=='exact' else 9
    print(f"score z={z:.1f} ({info}) -> {'WATERMARK PRESENT (Oh Hail Nah!!)' if z>thr else 'not detected'}")
