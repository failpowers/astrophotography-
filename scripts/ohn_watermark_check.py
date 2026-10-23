# OH HAIL NAH!! invisible-watermark checker. Exact check first, then a crop/rescale search for screenshots.
# Keyed spread-spectrum watermark in DWT LL band (Y channel). Survives screenshots, resizing, JPEG.
import cv2, numpy as np, pywt, sys
import os
KEY=int(os.environ['OHN_WM_KEY'],0)  # secret: set OHN_WM_KEY in the environment, never commit it
SIZES=[(1342,1080),(2516,1366),(2516,1242),(1200,900),(2140,1060),(1861,1047),(2038,1274),(2430,1316),(2201,1193),(1350,1080)]   # known originals (moon, IC 1318)
def pattern(shape):
    r=np.random.default_rng(KEY); return r.choice([-1.0,1.0],size=shape)
def embed(bgr,strength=9.0):
    ycc=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb).astype(np.float32)
    LL,(LH,HL,HH)=pywt.dwt2(ycc[:,:,0],'haar')
    LL2,d2=pywt.dwt2(LL,'haar')
    p=pattern(LL2.shape)
    m=(cv2.GaussianBlur(LL2,(0,0),3)>4*25).astype(np.float32)
    m=cv2.GaussianBlur(m,(0,0),2)
    LL2=LL2+strength*p*m
    LL=pywt.idwt2((LL2,d2),'haar')[:LH.shape[0],:LH.shape[1]]
    Y=pywt.idwt2((LL,(LH,HL,HH)),'haar')[:ycc.shape[0],:ycc.shape[1]]
    ycc[:,:,0]=np.clip(Y,0,255)
    return cv2.cvtColor(ycc.astype(np.uint8),cv2.COLOR_YCrCb2BGR)
def _score(bgr,W,H):
    img=cv2.resize(bgr,(W,H),interpolation=cv2.INTER_AREA)
    Y=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
    LL,_=pywt.dwt2(Y,'haar'); LL2,_=pywt.dwt2(LL,'haar')
    hp=LL2-cv2.GaussianBlur(LL2,(0,0),1)
    p=pattern(LL2.shape)
    m=cv2.GaussianBlur(LL2,(0,0),3)>4*25
    if m.sum()<500: return 0.0
    return (hp[m]*p[m]).sum()/(hp[m].std()*np.sqrt(m.sum()))
def detect(bgr):
    h,w=bgr.shape[:2]
    cands=[(W,H) for W,H in SIZES if abs(W/H-w/h)<0.03] or SIZES
    return max(_score(bgr,W,H) for W,H in cands)

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
def detect_robust(bgr,maxcrop=0.08,step=0.005):
    h,w=bgr.shape[:2]; Y=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
    best=(0.0,None)
    for W,H in SIZES:
        if abs((W/H)/(w/h)-1)>0.07: continue
        U=_upat(W,H); U=U-U.mean()
        for c in np.arange(0,maxcrop+1e-9,step):
            f=W*(1-c)/w
            if h*f>H*1.001: continue
            Ys=cv2.resize(Y,(max(8,int(round(w*f))),max(8,int(round(h*f)))),interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_CUBIC)
            z=_corr(Ys,U)
            if z>best[0]: best=(z,(W,H,round(c,3)))
    return best

if __name__=='__main__':
    # usage: python3 ohn_watermark_check.py image.jpg [image2.png ...]
    for path in sys.argv[1:]:
        im=cv2.imread(path)
        if im is None: print(path,': cannot read'); continue
        z=detect(im)
        if z>6: print(f"{path}: z={z:.1f} exact match -> WATERMARK PRESENT (Oh Hail Nah!!)"); continue
        zr,info=detect_robust(im)
        verdict='WATERMARK PRESENT (Oh Hail Nah!!)' if zr>6 else ('inconclusive' if zr>5 else 'not detected')
        print(f"{path}: z={zr:.1f} screenshot/crop search -> {verdict}")
