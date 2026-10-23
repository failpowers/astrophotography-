import cv2, numpy as np, sys
src=sys.argv[1]; out=sys.argv[2]
img=cv2.imread(src).astype(np.float32)/255
# white balance: neutralize moon surface (pixels in bright disc)
lum=img.mean(2); m=lum>0.25
means=img[m].mean(0); g=means.mean()
wb=g/means; wb=1+(wb-1)*0.8
img=np.clip(img*wb,0,1)
# black point / exposure from lit-surface percentiles
p_lo=np.percentile(img.mean(2),40); p_hi=np.percentile(img.mean(2)[m],99.7)
img=np.clip((img-p_lo*0.9)/(min(p_hi*1.0,1)-p_lo*0.9)*0.92,0,1)
# denoise
u8=(img*255).astype(np.uint8)
den=cv2.fastNlMeansDenoisingColored(u8,None,2,2,5,13)
lab=cv2.cvtColor(den,cv2.COLOR_BGR2LAB)
L=lab[:,:,0]
# local contrast (gentle CLAHE blended)
cl=cv2.createCLAHE(clipLimit=1.3,tileGridSize=(8,8)).apply(L)
L2=cv2.addWeighted(L,0.8,cl,0.2,0).astype(np.float32)
# fine detail: small-radius unsharp, restrained
bl=cv2.GaussianBlur(L2,(0,0),1.2)
L3=np.clip(L2+0.35*(L2-bl),0,255)
# keep sky black
L3[L<8]=L[L<8]
lab[:,:,0]=L3.astype(np.uint8)
res=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
cv2.imwrite(out,res,[cv2.IMWRITE_JPEG_QUALITY,96])
print(wb,p_lo,p_hi)
