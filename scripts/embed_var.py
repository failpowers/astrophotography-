import cv2,numpy as np,pywt,sys
from wm import pattern
def embed_var(bgr,smap_full):
    ycc=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb).astype(np.float32)
    LL,(LH,HL,HH)=pywt.dwt2(ycc[:,:,0],'haar'); LL2,d2=pywt.dwt2(LL,'haar')
    p=pattern(LL2.shape)
    m=(cv2.GaussianBlur(LL2,(0,0),3)>4*25).astype(np.float32); m=cv2.GaussianBlur(m,(0,0),2)
    s=cv2.resize(smap_full,(LL2.shape[1],LL2.shape[0]),interpolation=cv2.INTER_AREA)
    LL2=LL2+s*p*m
    LL=pywt.idwt2((LL2,d2),'haar')[:LH.shape[0],:LH.shape[1]]
    Y=pywt.idwt2((LL,(LH,HL,HH)),'haar')[:ycc.shape[0],:ycc.shape[1]]
    ycc[:,:,0]=np.clip(Y,0,255); return cv2.cvtColor(ycc.astype(np.uint8),cv2.COLOR_YCrCb2BGR)
src,out=sys.argv[1],sys.argv[2]; base,boost=float(sys.argv[3]),float(sys.argv[4])
img=cv2.imread(src); H,W=img.shape[:2]
smap=np.full((H,W),base,np.float32); smap[int(H*0.62):,:int(W*0.45)]=boost   # signature corner
cv2.imwrite(out,embed_var(img,smap),[cv2.IMWRITE_JPEG_QUALITY,95])
