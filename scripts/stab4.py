import cv2,numpy as np
exec(open('stab3.py').read().split('P=[pts(f)')[0])
P=[pts(f) for f in F]; N=len(F)
def sim(M):  # force pure rotation+translation
    t=np.arctan2(M[1,0],M[0,0]); R=np.array([[np.cos(t),-np.sin(t),M[0,2]],[np.sin(t),np.cos(t),M[1,2]],[0,0,1]]); return R
C=np.eye(3); Cs=[C.copy()]; jumps=0
for i in range(1,N):
    M,n=step(P[i-1],P[i])
    if M is not None and n>=20:
        S=sim(M)
        if np.hypot(S[0,2],S[1,2])>1.5 or abs(np.degrees(np.arctan2(S[1,0],S[0,0])))>0.15:
            C=C@np.linalg.inv(S); jumps+=1          # cancel this jump: map frame i back onto frame i-1's view
    Cs.append(C.copy())
Cs=np.array(Cs); print('jumps cancelled',jumps,'max offset',np.abs(Cs[:,:2,2]).max().round(1),'max rot deg',np.degrees(np.abs(np.arctan2(Cs[:,1,0],Cs[:,0,0]))).max().round(2))
np.save('met/tl_C.npy',Cs)
H,W=810,1440; out=cv2.VideoWriter('met/timelapse_stab_raw.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(W,H))
for i in range(N): out.write(cv2.warpAffine(F[i],Cs[i][:2],(W,H),flags=cv2.INTER_LINEAR,borderValue=(4,3,3)))
out.release()
