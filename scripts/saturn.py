import numpy as np, cv2, subprocess, glob, sys
vids=sorted(glob.glob('/root/.claude/uploads/8f6db876-c82a-5451-a2ee-c0831b19d297/*DWARF3_TELE_2026-10-19-*.mp4'))+['/mnt/user-data/uploads/DWARF3_TELE_2026-10-19-22-40-13-195.mp4']
C=320; X0,Y0=1975-C//2,1100-C//2; P=48      # crop around Saturn; planet patch half-size
crops=[]; meta=[]
for vi,v in enumerate(vids):
    p=subprocess.Popen(['ffmpeg','-v','error','-i',v,'-vf',f'crop={C}:{C}:{X0}:{Y0}','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    n=0
    while True:
        b=p.stdout.read(C*C*3)
        if len(b)<C*C*3: break
        f=np.frombuffer(b,np.uint8).reshape(C,C,3).astype(np.float32); n+=1
        g=f.mean(2); gs=cv2.GaussianBlur(g,(0,0),2)
        y,x=np.unravel_index(np.argmax(gs),gs.shape)
        if gs.max()<40 or not(P<y<C-P and P<x<C-P): continue
        # intensity-weighted centroid in a window
        w=g[y-25:y+25,x-25:x+25]; w=np.clip(w-np.percentile(g,50)-10,0,None)
        if w.sum()<500: continue
        yy,xx=np.mgrid[-25:25,-25:25]; cy=y+(w*yy).sum()/w.sum(); cx=x+(w*xx).sum()/w.sum()
        iy,ix=int(round(cy)),int(round(cx)); patch=f[iy-P:iy+P,ix-P:ix+P]
        if patch.shape[:2]!=(2*P,2*P): continue
        L=cv2.GaussianBlur(patch.mean(2),(0,0),0.8)
        sharp=cv2.Laplacian(L,cv2.CV_32F).var()/max(L.max(),1)**2
        crops.append(patch.astype(np.uint8)); meta.append((vi,n,cx-ix,cy-iy,sharp,L.max(),w.sum()))
    p.wait(); print(v.split('/')[-1],'frames',n,flush=True)
crops=np.array(crops); meta=np.array(meta)
np.save('sat/crops.npy',crops); np.save('sat/meta.npy',meta)
print('usable',len(crops))
