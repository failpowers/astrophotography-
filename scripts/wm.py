# Keyed spread-spectrum watermark in DWT LL band (Y channel). Survives screenshots, resizing, JPEG.
import cv2, numpy as np, pywt, sys
import os
KEY=int(os.environ['OHN_WM_KEY'],0)  # secret: set OHN_WM_KEY in the environment, never commit it
SIZES=[(1342,1080),(2516,1366),(2516,1242),(1200,900),(2140,1060),(1861,1047),(2038,1274),(2430,1316),(2201,1193),(1350,1080),(1780,996)]   # known originals (moon, IC 1318)
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
if __name__=='__main__':
    z=detect(cv2.imread(sys.argv[1]))
    print(f"score z={z:.1f} -> {'WATERMARK PRESENT (Oh Hail Nah!!)' if z>6 else 'not detected'}")
