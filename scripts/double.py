import sys; sys.path.insert(0,'.')
import numpy as np, cv2, proc
A,hA=proc.load(proc.find('NGC 869')[0]); B,hB=proc.load(proc.find('NGC 884')[0])
cA,mA=proc.coverage(A); cB,mB=proc.coverage(B)
H,W=A.shape[:2]
def wcs(h):
    cd=np.array([[float(h['CD1_1']),float(h['CD1_2'])],[float(h['CD2_1']),float(h['CD2_2'])]])
    return np.radians(float(h['CRVAL1'])),np.radians(float(h['CRVAL2'])),float(h['CRPIX1']),float(h['CRPIX2']),cd
def pix2sky(h,x,y):
    a0,d0,cx,cy,cd=wcs(h); u=x+1-cx; v=y+1-cy
    xi,eta=np.radians(cd@np.vstack([u,v]))
    den=np.cos(d0)-eta*np.sin(d0); a=a0+np.arctan2(xi,den); d=np.arctan2(np.sin(d0)+eta*np.cos(d0),np.hypot(xi,den))
    return a,d
def sky2pix(h,a,d):
    a0,d0,cx,cy,cd=wcs(h)
    cosc=np.sin(d0)*np.sin(d)+np.cos(d0)*np.cos(d)*np.cos(a-a0)
    xi=np.cos(d)*np.sin(a-a0)/cosc; eta=(np.cos(d0)*np.sin(d)-np.sin(d0)*np.cos(d)*np.cos(a-a0))/cosc
    uv=np.linalg.solve(cd,np.degrees(np.vstack([xi,eta]))); return uv[0]+cx-1,uv[1]+cy-1
def stars(img,mask,n=600):
    L=img.mean(2); hp=L-cv2.GaussianBlur(L,(0,0),6); nz=1.4826*np.median(np.abs(hp-np.median(hp)))
    det=((cv2.GaussianBlur(hp,(0,0),1.2)>8*nz)&(mask>0)).astype(np.uint8)
    k,lab,st,cen=cv2.connectedComponentsWithStats(det)
    fl=np.array([hp[lab==i].sum() for i in range(1,k)]) if k>1 else np.array([])
    ok=(st[1:,4]>=4)&(st[1:,4]<400); idx=np.argsort(-fl*ok)[:n]
    return cen[1:][idx],fl[idx]
sA,fA=stars(A,mA); sB,fB=stars(B,mB)
# predict B stars in A frame via WCS, match, then fit
a,d=pix2sky(hB,sB[:,0],sB[:,1]); px,py=sky2pix(hA,a,d); pB=np.stack([px,py],1)
from scipy.spatial import cKDTree
t=cKDTree(sA); dist,j=t.query(pB); good=dist<15
print('WCS-predicted matches',good.sum(),'median offset px',round(float(np.median(dist[good])),2))
M,inl=cv2.estimateAffine2D(sB[good].astype(np.float32),sA[j[good]].astype(np.float32),ransacReprojThreshold=1.5,maxIters=5000)
inl=inl.ravel()>0; pr=(M[:,:2]@sB[good][inl].T).T+M[:,2]; res=np.linalg.norm(pr-sA[j[good]][inl],axis=1)
print('affine inliers',inl.sum(),'residual rms px',round(float(np.sqrt((res**2).mean())),3))
print('rotation deg',round(float(np.degrees(np.arctan2(M[1,0],M[0,0]))),2),'scale',round(float(np.sqrt(abs(np.linalg.det(M[:,:2])))),5))
# flux scale B->A from matched stars
fs=np.median(fA[j[good]][inl]/fB[good][inl]); print('flux scale B->A',round(float(fs),3))
def chflux(img,pts):
    out=[]
    for (x,y) in pts:
        x,y=int(round(x)),int(round(y)); p=img[y-4:y+5,x-4:x+5]; b=np.median(img[y-12:y+13,x-12:x+13].reshape(-1,3),0)
        out.append((p-b).reshape(-1,3).sum(0))
    return np.array(out)
pa=sA[j[good]][inl]; pb=sB[good][inl]
okp=(pa[:,0]>20)&(pa[:,1]>20)&(pa[:,0]<W-20)&(pa[:,1]<H-20)&(pb[:,0]>20)&(pb[:,1]>20)&(pb[:,0]<B.shape[1]-20)&(pb[:,1]<B.shape[0]-20)
cfA=chflux(A,pa[okp]); cfB=chflux(B,pb[okp]); okf=(cfA>0).all(1)&(cfB>0).all(1)
fsc=np.median(cfA[okf]/cfB[okf],0); print('per-channel scale B->A (B,G,R)',fsc.round(3))
# background extraction in each native frame, then warp B
LB=B.mean(2); hpB=LB-cv2.GaussianBlur(LB,(0,0),5); nzb=1.4826*np.median(np.abs(hpB-np.median(hpB)))
cand=(cv2.GaussianBlur(hpB,(0,0),1.0)>3.5*nzb).astype(np.uint8)
k,lab,st,_=cv2.connectedComponentsWithStats(cand)
elong=np.zeros(k,bool); elong[1:]=(np.maximum(st[1:,2],st[1:,3])>60)          # keep only long thin pieces, drop stars
cand=(elong[lab]).astype(np.uint8)*255
ls=cv2.HoughLinesP(cand,1,np.pi/720,threshold=150,minLineLength=600,maxLineGap=80)
trail=np.zeros(LB.shape,np.uint8)
if ls is not None:
    for l in np.asarray(ls).reshape(-1,4): cv2.line(trail,(int(l[0]),int(l[1])),(int(l[2]),int(l[3])),1,15)
print('B trail segments',0 if ls is None else len(np.asarray(ls).reshape(-1,4)),'mask px',int(trail.sum()))
for c in range(3):
    ch=B[...,c].astype(np.float32); B[...,c]=cv2.inpaint(ch,trail,7,cv2.INPAINT_TELEA)
np.save('dc_trailB.npy',trail)
smA,nzA=proc.star_mask(A.mean(2)); smB,nzB=proc.star_mask(B.mean(2))
A=proc.dbe(A,smA,mA,T1=64,T2=40); B=proc.dbe(B,smB,mB,T1=64,T2=40)*fsc
Bw=cv2.warpAffine(B,M,(W,H),flags=cv2.INTER_LANCZOS4); yc,xc=H//2,W//2; ra=A[yc-600:yc+600,xc-900:xc+900]; rb=Bw[yc-600:yc+600,xc-900:xc+900]
hw=cv2.createHanningWindow((1800,1200),cv2.CV_64F)
for c in range(3):
    (sx,sy),r=cv2.phaseCorrelate(ra[...,c].astype(np.float64),rb[...,c].astype(np.float64),hw)
    if abs(sx)<2 and abs(sy)<2: Bw[...,c]=cv2.warpAffine(Bw[...,c],np.float32([[1,0,-sx],[0,1,-sy]]),(W,H),flags=cv2.INTER_LANCZOS4)
    print('  channel',c,'residual shift',round(sx,3),round(sy,3))
mBw=cv2.warpAffine(mB,M,(W,H),flags=cv2.INTER_NEAREST); cBw=cv2.warpAffine(cB,M,(W,H),flags=cv2.INTER_LINEAR)
def noise(img,m):
    L=img.mean(2); hp=L-cv2.GaussianBlur(L,(0,0),2); return 1.4826*np.median(np.abs(hp[m>0]))
sA_,sB_=noise(A,mA),noise(Bw,mBw); print('noise A,B',round(float(sA_),2),round(float(sB_),2))
# feathered weights
fA_=cv2.distanceTransform(mA,cv2.DIST_L2,5); fB_=cv2.distanceTransform(mBw,cv2.DIST_L2,5)
wA=cA*np.clip(fA_/60,0,1)/sA_**2; wB=cBw*np.clip(fB_/60,0,1)/sB_**2
# outlier rejection where both cover (satellite trails / hot spots): luminance difference test
both=(mA>0)&(mBw>0)
dl=cv2.GaussianBlur(Bw.mean(2)-A.mean(2),(0,0),0.8); sig=np.sqrt(sA_**2+sB_**2)/1.6
rejB=both&(dl>5*sig); rejA=both&(dl<-5*sig)
# ignore star cores (small misregistration / seeing differences): only reject where neither is a star
st=cv2.dilate(((smA>0)|(cv2.warpAffine(smB,M,(W,H),flags=cv2.INTER_NEAREST)>0)).astype(np.uint8),np.ones((3,3)))
rejB&=st==0; rejA&=st==0
def lines_only(cand):
    c=cand.astype(np.uint8)*255
    ls=cv2.HoughLinesP(c,1,np.pi/720,threshold=120,minLineLength=400,maxLineGap=60)
    lm=np.zeros_like(c)
    if ls is not None:
        for l in np.asarray(ls).reshape(-1,4): cv2.line(lm,(int(l[0]),int(l[1])),(int(l[2]),int(l[3])),255,13)
    print('  trail segments found',0 if ls is None else len(ls))
    return ((lm>0)&cand).astype(np.uint8)
rejB=cv2.dilate(lines_only(rejB),np.ones((7,7))); rejA=cv2.dilate(lines_only(rejA),np.ones((7,7)))
print('trail px rejected: B',int(rejB.sum()),'A',int(rejA.sum()))
wB=wB*(1-rejB); wA=wA*(1-rejA)
ws=wA+wB; comb=(A*wA[...,None]+Bw*wB[...,None])/np.maximum(ws,1e-30)[...,None]
# fallback where weights vanish: whichever session is present and not a trail
fa=((cA>0.05)&(rejA==0)).astype(np.float32); fb=((cBw>0.05)&(rejB==0)).astype(np.float32)
fb_=(A*fa[...,None]+Bw*fb[...,None])/np.maximum(fa+fb,1)[...,None]
comb=np.where((ws<=np.percentile(ws[ws>0],0.5)*0.01)[...,None],fb_,comb)
cov=((cA+cBw)>=0.6).astype(np.uint8)
np.save('dc_comb.npy',comb.astype(np.float32)); np.save('dc_cov.npy',cov); np.save('dc_both.npy',both.astype(np.uint8))
cv2.imwrite('dc_cov.jpg',cv2.resize(((mA>0)*120+(mBw>0)*120).astype(np.uint8),None,fx=0.25,fy=0.25))
y,x,hh,ww=proc.max_rect2(cv2.erode(cov,np.ones((25,25))),amin=1.5,amax=1.85); print('crop',y,x,hh,ww)
np.save('dc_crop.npy',np.array([y,x,hh,ww]))
