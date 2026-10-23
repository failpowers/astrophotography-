import numpy as np, cv2, subprocess, sys
V='/root/.claude/uploads/8f6db876-c82a-5451-a2ee-c0831b19d297/cf782fe8-DWARF3_TELE_2026-10-20-12-11-40-754.mp4'
CX0,CY0,C=1919,1080,1000; X0,Y0=CX0-C//2,CY0-C//2
def frames():
    p=subprocess.Popen(['ffmpeg','-v','error','-i',V,'-vf',f'crop={C}:{C}:{X0}:{Y0}','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    while True:
        b=p.stdout.read(C*C*3)
        if len(b)<C*C*3: break
        yield np.frombuffer(b,np.uint8).reshape(C,C,3)
    p.wait()
# ---- pass 1: centroid + sharpness ----
meta=[]
for k,f in enumerate(frames()):
    g=f.mean(2).astype(np.float32); m=g>0.5*np.percentile(g,99)
    M=cv2.moments(m.astype(np.uint8),True)
    if M['m00']<1000: meta.append((k,0,0,0,0)); continue
    cx,cy=M['m10']/M['m00'],M['m01']/M['m00']; R=np.sqrt(M['m00']/np.pi)
    yy,xx=np.ogrid[:C,:C]; inner=(xx-cx)**2+(yy-cy)**2<(0.85*R)**2
    L=cv2.GaussianBlur(g,(0,0),1.0); lap=cv2.Laplacian(L,cv2.CV_32F)
    q=lap[inner].var()/max(g[inner].mean(),1)**2
    meta.append((k,cx,cy,R,q))
meta=np.array(meta); np.save('sun/meta.npy',meta)
q=meta[:,4]; print('frames',len(meta),'radius median',round(float(np.median(meta[:,3])),1),'q spread',np.percentile(q[q>0],[10,50,90]).round(6))
