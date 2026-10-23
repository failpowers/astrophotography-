import sys; sys.path.insert(0,'.')
import numpy as np, proc
exec(open('double.py').read().split("def stars(")[0].split("A,hA=proc.load")[0])
_,hA=proc.load(proc.find('NGC 869')[0])
exec("\n".join(l for l in open('double.py').read().split("\n") if l.startswith(('def wcs','    cd=','    return np.radians','def pix2sky','    a0,d0','    xi,eta','    den=','    return a,d','def sky2pix','    cosc=','    xi=np.cos','    uv=np.linalg'))))
cl={'NGC 869':(34.75,57.128),'NGC 884':(35.575,57.137)}
pts=[sky2pix(hA,np.radians(np.array([ra])),np.radians(np.array([de]))) for ra,de in cl.values()]
pts=[(float(p[1][0]),float(p[0][0])) for p in pts]; print('cluster centers (y,x):',[tuple(round(v) for v in p) for p in pts])
cov=np.load('dc_cov.npy'); H,W=cov.shape
cy=np.mean([p[0] for p in pts]); cx=np.mean([p[1] for p in pts]); best=None
for asp in (16/9,1.6,1.5):
    for h in range(2100,600,-20):
        w=int(h*asp); found=False
        for dy in range(-400,401,20):
            for dx in range(-500,501,20):
                y0=int(cy-h/2+dy); x0=int(cx-w/2+dx)
                if y0<0 or x0<0 or y0+h>H or x0+w>W: continue
                if not all(y0+200<=p[0]<=y0+h-200 and x0+250<=p[1]<=x0+w-250 for p in pts): continue
                if cov[y0:y0+h:8,x0:x0+w:8].mean()>=0.985:
                    if best is None or h*w>best[0]: best=(h*w,(y0,x0,h,w))
                    found=True; break
            if found: break
        if found: break
print('crop',best); np.save('dc_crop.npy',np.array(best[1]))
