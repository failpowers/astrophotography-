import numpy as np, cv2, sys
from skimage.restoration import richardson_lucy
crops=np.load('sat/crops.npy').astype(np.float32); meta=np.load('sat/meta.npy')
vi=meta[:,0].astype(int); sharp=meta[:,4]; peak=meta[:,5]; flux=meta[:,6]
print('per-video frames',np.bincount(vi),'median sharpness',[round(float(np.median(sharp[vi==k]))*1e3,3) for k in range(3)])
# reject frames with abnormal flux (clouds/defocus), then rank by sharpness
fz=(flux-np.median(flux))/(1.4826*np.median(np.abs(flux-np.median(flux))))
ok=np.abs(fz)<3
FRAC=float(sys.argv[1]) if len(sys.argv)>1 else 0.08
order=np.argsort(sharp*ok)[::-1]; sel=order[:int(len(order)*FRAC)]
print('selected',len(sel),'from videos',np.bincount(vi[sel],minlength=3))
U=4; P=crops.shape[1]
def up(i):
    c=crops[i]; dx,dy=meta[i,2],meta[i,3]
    M=np.float32([[U,0,-dx*U+ (U-1)/2*0],[0,U,-dy*U]])
    return cv2.warpAffine(c,M,(P*U,P*U),flags=cv2.INTER_CUBIC)
# reference: centroid-aligned mean of top 40
ref=np.mean([up(i) for i in sel[:40]],0); refL=ref.mean(2).astype(np.float64)
S=10; c0=P*U//2; R=56
tmpl=refL[c0-R:c0+R,c0-2*R:c0+2*R].astype(np.float32) if 2*R<=c0 else refL.astype(np.float32)
stack=[]; scores=[]
for i in sel:
    u=up(i); g=u.mean(2).astype(np.float32)
    srch=g[c0-R-S:c0+R+S, max(0,c0-2*R-S):c0+2*R+S]
    res=cv2.matchTemplate(srch,tmpl,cv2.TM_CCOEFF_NORMED)
    _,mx,_,(px,py)=cv2.minMaxLoc(res)
    def sub(v,p):
        if 0<p<len(v)-1:
            d=v[p-1]-2*v[p]+v[p+1]; return p+(0.5*(v[p-1]-v[p+1])/d if d!=0 else 0)
        return p
    fx=sub(res[py,:],px)-S; fy=sub(res[:,px],py)-S
    if mx<0.85: continue
    u=cv2.warpAffine(u,np.float32([[1,0,-fx],[0,1,-fy]]),(P*U,P*U),flags=cv2.INTER_CUBIC)
    stack.append(u); scores.append(mx)
stack=np.array(stack); print('aligned',len(stack))
# sigma-clipped mean (removes compression blocks / outliers)
med=np.median(stack,0); sd=1.4826*np.median(np.abs(stack-med),0)+1
w=(np.abs(stack-med)<2.5*sd).astype(np.float32)
img=(stack*w).sum(0)/np.maximum(w.sum(0),1)
np.save('sat/stack.npy',img)
bg=np.median(img[:20,:20].reshape(-1,3),0); img=np.clip(img-bg,0,None)
L=img.mean(2); top=np.percentile(L,99.9)
# color: neutralize the globe, then Saturn's natural pale gold
m=L>0.35*top; cw=img[m].mean(0); cw=cw.mean()/cw; img=img*cw*np.array([0.88,0.98,1.06])
L=img.mean(2)/top
yy,xx=np.mgrid[-12:13,-12:13]
for s,it in ((float(sys.argv[2]) if len(sys.argv)>2 else 3.0, int(sys.argv[3]) if len(sys.argv)>3 else 30),):
    psf=np.exp(-(xx**2+yy**2)/(2*s*s)); psf/=psf.sum()
    dec=richardson_lucy(np.clip(L,1e-5,1.5),psf,num_iter=it,clip=False).astype(np.float32)
dec=cv2.bilateralFilter(dec,5,0.03,2)
ratio=cv2.GaussianBlur(img/top,(0,0),2.0)/np.maximum(cv2.GaussianBlur(L,(0,0),2.0),1e-3)[...,None]
out=np.clip(dec[...,None]*ratio,0,1.2)
out=out/np.percentile(out,99.9)*0.86
out=np.where(out>0.8,0.8+(1-np.exp(-(out-0.8)/0.2))*0.2,out)
out=np.clip(out,0,1)**(1/1.2)
fade=np.clip((cv2.GaussianBlur(L,(0,0),3)-0.01)/0.04,0,1)[...,None]; out=out*fade
cv2.imwrite('sat/saturn_stack.png',(out*255).astype(np.uint8))
raw=np.clip(img/top,0,1)**(1/1.25); cv2.imwrite('sat/saturn_raw.png',(raw*255).astype(np.uint8))
single=up(sel[0])/top
cv2.imwrite('sat/cmp.png',cv2.resize((np.hstack([np.clip(crops[order[len(order)//2]]/top,0,1)[...,:].repeat(U,0).repeat(U,1),np.clip(single,0,1),raw,out])*255).astype(np.uint8),None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC))
