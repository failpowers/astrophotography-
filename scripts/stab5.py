import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
C1=np.load('met/tl_C.npy'); C2=np.load('met/tl_C2.npy'); H,W=810,1440; c=np.array([W/2,H/2]); N=len(C1)
T=np.array([C2[i]@C1[i] for i in range(N)])
th=np.arctan2(T[:,1,0],T[:,0,0]); t=T[:,:2,2]
R=np.stack([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]).transpose(2,0,1)
tc=t+np.einsum('nij,j->ni',R,c)-c          # translation expressed about the frame centre
SIG=float(__import__('sys').argv[1])
th_h=th-gaussian_filter1d(th,SIG,mode='nearest'); tc_h=tc-gaussian_filter1d(tc,SIG,axis=0,mode='nearest')
cap=cv2.VideoCapture('met/timelapse.mp4'); out=cv2.VideoWriter('met/timelapse_stab_raw.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(W,H))
cw,ch=912,810; x0=(W-cw)//2; bad=0; mx=0
for i in range(N):
    ok,f=cap.read()
    M=cv2.getRotationMatrix2D(tuple(c),-np.degrees(th_h[i]),1.0); M[:,2]+=tc_h[i]
    out.write(cv2.warpAffine(f,M,(W,H),flags=cv2.INTER_LINEAR,borderValue=(4,3,3)))
    Mi=cv2.invertAffineTransform(M); corners=np.array([[x0,0],[x0+cw,0],[x0,ch],[x0+cw,ch]],np.float32)
    s=cv2.transform(corners[None],Mi)[0]; o=max((-s[:,0]).max(),(s[:,0]-W).max(),(-s[:,1]).max(),(s[:,1]-H).max())
    if o>0: bad+=1; mx=max(mx,o)
out.release(); print('sigma',SIG,'frames with black corners in window',bad,'worst',round(float(mx),1))
