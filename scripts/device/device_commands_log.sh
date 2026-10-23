# Commands run on the Mac (Cowork VM) during the Alhena meteor scan, in order.
# Scripts lived in $HOME/mnt/outputs/met/ ; recreate them by running these heredocs.

# ----
cd "$HOME/mnt" && ls; for d in "DWARF_RAW_TELE_M 34_EXP_15_GAIN_60_2026-10-22-02-02-08-161" "DWARF_RAW_TELE_NAME χ Persei_EXP_15_GAIN_60_2026-10-21-23-58-33-699" DWARF_RAW_WIDE_Alhena_*; do echo "== $d"; [ -d "$d" ] && (ls "$d" | head -5; echo "count: $(ls "$d" | wc -l)  size: $(du -sh "$d" | cut -f1)"); done

# ----
cd "$HOME/mnt"; which python3; python3 -c "import numpy, sys; print('numpy', numpy.__version__, sys.version)"; python3 -c "import cv2; print('cv2', cv2.__version__)" 2>&1 | tail -1; python3 -c "import astropy; print('astropy', astropy.__version__)" 2>&1 | tail -1; nproc; free -g | head -2; df -h $HOME | tail -1; for d in "DWARF_RAW_TELE_M 34_EXP_15_GAIN_60_2026-10-22-02-02-08-161" "DWARF_RAW_TELE_NAME χ Persei_EXP_15_GAIN_60_2026-10-21-23-58-33-699" DWARF_RAW_WIDE_Alhena_*; do echo "== $d"; ls -la "$d" | grep -v "_[0-9]*C.fits$" | grep -v "^total"; ls "$d" | grep -c "^failed"; done

# ----
cd "$HOME/mnt"; for d in "DWARF_RAW_TELE_M 34_EXP_15_GAIN_60_2026-10-22-02-02-08-161" "DWARF_RAW_TELE_NAME χ Persei_EXP_15_GAIN_60_2026-10-21-23-58-33-699" DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073 DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_40_2026-10-22-03-40-50-589; do echo "== $d"; cat "$d/shotsInfo.json"; echo; done; python3 - <<'EOF'
import glob,os
os.chdir(os.path.expanduser('~/mnt'))
def hdr(p):
    b=open(p,'rb').read(2880*6); out=[]
    for i in range(0,len(b),80):
        c=b[i:i+80].decode('latin1')
        out.append(c.rstrip())
        if c.startswith('END'): break
    return out
f=sorted(glob.glob('DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/Alhena_*.fits'))[0]
print('\n'.join(l for l in hdr(f) if l))
print('-----')
f=glob.glob('DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/stacked-16*.fits')[0]
print('\n'.join(l for l in hdr(f) if l and not l.startswith('COMMENT'))[:3000])
EOF

# ----
mkdir -p "$HOME/mnt/outputs/met" && cd "$HOME/mnt/outputs" && pwd && cat > met/scan.py <<'EOF'
import numpy as np, cv2, glob, os, json, sys, re
ROOT=os.path.expanduser('~/mnt')
OUT=os.path.expanduser('~/mnt/outputs/met')
def read(p):
    with open(p,'rb') as f:
        h={}; n=0
        while True:
            blk=f.read(2880); n+=1
            for i in range(0,2880,80):
                c=blk[i:i+80].decode('latin1')
                if '=' in c[:10]:
                    k=c[:8].strip(); v=c[10:].split('/')[0].strip().strip("'").strip(); h[k]=v
                if c.startswith('END'): break
            else: continue
            break
        W,H=int(h['NAXIS1']),int(h['NAXIS2'])
        a=np.frombuffer(f.read(W*H*2),'>i2').reshape(H,W).astype(np.float32)+float(h.get('BZERO',0))
    return a,h
def bin2(a):  # RGGB superpixel -> R,G,B half-res
    r=a[0::2,0::2]; g=(a[0::2,1::2]+a[1::2,0::2])*0.5; b=a[1::2,1::2]
    return np.stack([r,g,b],-1)
def tkey(p):
    m=re.search(r'_(\d{8}-\d{9})_',os.path.basename(p)); return m.group(1)
files=[]
for d in sorted(glob.glob(ROOT+'/DWARF_RAW_WIDE_Alhena_*')):
    for p in glob.glob(d+'/*.fits'):
        b=os.path.basename(p)
        if b.startswith('stacked'): continue
        files.append((os.path.basename(d),tkey(p),p))
files.sort(key=lambda x:x[1])
start=int(sys.argv[1]); end=min(int(sys.argv[2]),len(files))
if start==0: json.dump([[s,t,os.path.basename(p)] for s,t,p in files],open(OUT+'/files.json','w'))
print('total',len(files))
cache={}
def get(i):
    if i not in cache:
        a,h=read(files[i][2]); c=bin2(a); L=c.sum(-1)
        cache[i]=(c,L,h)
    return cache[i]
res=[]
for i in range(start,end):
    sess=files[i][0]
    nb=[j for j in (i-2,i-1,i+1,i+2) if 0<=j<len(files) and files[j][0]==sess]
    c,L,h=get(i)
    if len(nb)<2: res.append(dict(i=i,skip=1)); continue
    Ls=np.stack([get(j)[1] for j in nb])
    mx=Ls.max(0); med=np.median(Ls,0)
    D=L-mx                     # bright only in this frame
    Dm=L-med
    sig=1.4826*np.median(np.abs(Dm-np.median(Dm)))
    T=(D>4.5*sig).astype(np.uint8)
    T=cv2.morphologyEx(T,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    T=cv2.dilate(T,np.ones((3,3),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(T,8)
    cands=[]
    for k in range(1,n):
        x,y,w,hh,area=st[k]
        if max(w,hh)<10 or area<15: continue
        ys,xs=np.where(lab==k)
        pts=np.stack([xs,ys],1).astype(np.float32); m=pts.mean(0); cov=np.cov((pts-m).T)
        ev,evec=np.linalg.eigh(cov); length=4*np.sqrt(max(ev[1],1e-6)); width=4*np.sqrt(max(ev[0],1e-6))
        if length/max(width,1)<3.0: continue
        d=evec[:,1]; proj=(pts-m)@d
        flux=float(D[ys,xs].sum())
        # brightness profile along the trail (10 bins)
        bins=np.linspace(proj.min(),proj.max()+1e-3,11); prof=[float(D[ys,xs][(proj>=bins[q])&(proj<bins[q+1])].mean()) if ((proj>=bins[q])&(proj<bins[q+1])).any() else 0.0 for q in range(10)]
        cands.append(dict(x=int(x),y=int(y),w=int(w),h=int(hh),cx=float(m[0]),cy=float(m[1]),len=float(length),wid=float(width),ang=float(np.degrees(np.arctan2(d[1],d[0]))),flux=flux,peak=float(D[ys,xs].max()/sig),prof=prof,p0=[float(v) for v in (m+d*proj.min())],p1=[float(v) for v in (m+d*proj.max())]))
    res.append(dict(i=i,sess=sess,t=files[i][1],sig=float(sig),sky=float(np.median(L)),c=cands))
    for k in list(cache):
        if k<i-2: del cache[k]
json.dump(res,open(f'{OUT}/scan_{start:04d}.json','w'))
print('done',start,end,'frames with cands',sum(1 for r in res if r.get('c')))
EOF
time python3 met/scan.py 0 60

# ----
cd "$HOME/mnt/outputs" && python3 met/scan.py 60 260 && python3 met/scan.py 260 460

# ----
cd "$HOME/mnt/outputs" && python3 met/scan.py 460 669 && python3 - <<'EOF'
import json,glob,os
R=[]
for f in sorted(glob.glob(os.path.expanduser('~/mnt/outputs/met/scan_*.json'))): R+=json.load(open(f))
print('frames',len(R),'skipped',sum(1 for r in R if r.get('skip')))
from collections import Counter
c=Counter(); nc=Counter()
for r in R:
    if r.get('skip'): continue
    c[r['sess'][-23:]]+=1; nc[r['sess'][-23:]]+=len(r['c'])
print(c); print(nc)
# per frame summary of candidate counts
big=[(r['i'],r['t'],len(r['c']),round(r['sky']),round(r['sig'],1),[ (round(x['len']),round(x['peak'],1)) for x in sorted(r['c'],key=lambda x:-x['len'])[:4]]) for r in R if r.get('c')]
for b in big: print(b)
EOF

# ----
cd "$HOME/mnt/outputs" && cat > met/crops.py <<'EOF'
import numpy as np, cv2, json, glob, os, sys
exec(open(os.path.expanduser('~/mnt/outputs/met/scan.py')).read().split('files=[]')[0])
OUT=os.path.expanduser('~/mnt/outputs/met')
F=json.load(open(OUT+'/files.json')); ROOT=os.path.expanduser('~/mnt')
def path(i): return f"{ROOT}/{F[i][0]}/{F[i][2]}"
def lum(i): a,h=read(path(i)); return bin2(a).sum(-1)
req=json.loads(sys.argv[1]); name=sys.argv[2]; S=int(sys.argv[3]) if len(sys.argv)>3 else 120
tiles=[]
for i,cx,cy in req:
    L=lum(i); P=lum(i-1); N=lum(i+1)
    def cr(a):
        return cv2.getRectSubPix(a.astype(np.float32),(S,S),(cx,cy))
    a,p,n=cr(L),cr(P),cr(N)
    sky=np.median(a); top=np.percentile(a,99.7)
    def st(x): return np.clip((x-sky)/(top-sky+1e-3),0,1)**0.5*255
    d=np.clip((a-np.maximum(p,n))/(top-sky+1e-3)*2,0,1)*255
    row=np.hstack([st(p),st(a),st(n),d]).astype(np.uint8)
    row=cv2.resize(row,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    row=cv2.cvtColor(row,cv2.COLOR_GRAY2BGR)
    cv2.putText(row,f'{i} ({cx:.0f},{cy:.0f})',(5,20),0,0.6,(0,255,255),1)
    tiles.append(row)
cv2.imwrite(f'{OUT}/{name}.png',np.vstack(tiles)); print('ok',len(tiles))
EOF
python3 - <<'EOF'
import json,glob,os,subprocess
R=[]
for f in sorted(glob.glob(os.path.expanduser('~/mnt/outputs/met/scan_*.json'))): R+=json.load(open(f))
R={r['i']:r for r in R}
pick=[]
for i in [128,302,319,338,440,518,374,380,386,392,404,422]:
    for c in sorted(R[i]['c'],key=lambda x:-x['len'])[:1]: pick.append([i,c['cx'],c['cy']])
json.dump(pick,open(os.path.expanduser('~/mnt/outputs/met/pick1.json'),'w'))
print(pick)
EOF
python3 met/crops.py "$(cat met/pick1.json)" sheet1 120

# ----
cd "$HOME/mnt/outputs" && cat > met/classify.py <<'EOF'
import json,glob,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
R=[]
for f in sorted(glob.glob(OUT+'/scan_*.json')): R+=json.load(open(f))
R={r['i']:r for r in R if not r.get('skip')}
F=json.load(open(OUT+'/files.json'))
def angdiff(a,b):
    d=abs((a-b+90)%180-90); return d
def perp(c,o):  # distance of point c from line through o's endpoints
    p0=np.array(o['p0']); p1=np.array(o['p1']); d=p1-p0; n=np.linalg.norm(d)+1e-6
    return abs(np.cross(d,np.array([c['cx'],c['cy']])-p0))/n
rows=[]
for i,r in R.items():
    cs=r['c']
    if not cs: continue
    angs=[c['ang'] for c in cs if c['len']>12]
    for c in cs:
        flags=[]
        # persistent / stationary: something at same spot in i±1..3
        for j in range(i-3,i+4):
            if j==i or j not in R: continue
            for o in R[j]['c']:
                if np.hypot(o['cx']-c['cx'],o['cy']-c['cy'])<14: flags.append('persist'); break
            if 'persist' in flags: break
        # satellite/plane continuation: collinear trail in i±1, i±2
        for j in (i-2,i-1,i+1,i+2):
            if j not in R: continue
            for o in R[j]['c']:
                if o['len']>10 and angdiff(o['ang'],c['ang'])<4 and perp(o,c)<6: flags.append(f'collinear{j-i:+d}'); break
        # rigid-shift frame: >=2 other cands with same angle (+-8 deg) and similar length
        same=sum(1 for o in cs if o is not c and angdiff(o['ang'],c['ang'])<8 and abs(o['len']-c['len'])<0.35*c['len'])
        if same>=2: flags.append('shift')
        rows.append(dict(i=i,t=r['t'],sess=r['sess'][-23:],cx=round(c['cx'],1),cy=round(c['cy'],1),len=round(c['len'],1),wid=round(c['wid'],1),ang=round(c['ang'],1),peak=round(c['peak'],1),flux=round(c['flux']),prof=[round(p) for p in c['prof']],flags=flags,p0=c['p0'],p1=c['p1']))
json.dump(rows,open(OUT+'/rows.json','w'))
clean=[r for r in rows if not r['flags']]
print('all',len(rows),'clean',len(clean))
for r in sorted(clean,key=lambda r:-r['len']): print(r['i'],r['t'][9:15],r['cx'],r['cy'],'len',r['len'],'wid',r['wid'],'ang',r['ang'],'peak',r['peak'],r['prof'])
EOF
python3 met/classify.py

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF' 2>/dev/null
import json,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
rows=json.load(open(OUT+'/rows.json'))
# recurring locations: count frames having a candidate within 20px
pos=np.array([[r['cx'],r['cy'],r['i']] for r in rows])
keep=[]
for r in rows:
    d=np.hypot(pos[:,0]-r['cx'],pos[:,1]-r['cy']); fr=set(pos[d<22,2].astype(int))-{r['i']}
    r['recur']=len(fr)
    if r['wid']>=10: continue
    if r['recur']>=2: continue
    if any(f.startswith('collinear') for f in r['flags']): continue
    keep.append(r)
print('survivors',len(keep))
for r in sorted(keep,key=lambda r:-r['peak']*r['len']): print(r['i'],r['t'][9:15],r['cx'],r['cy'],'len',r['len'],'wid',r['wid'],'ang',r['ang'],'peak',r['peak'],r['flags'],r['prof'])
json.dump(keep,open(OUT+'/keep.json','w'))
EOF

# ----
cd "$HOME/mnt/outputs" && D="$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan" && mkdir -p "$D" && python3 - <<'EOF'
import json,os
OUT=os.path.expanduser('~/mnt/outputs/met')
k=json.load(open(OUT+'/keep.json')); k=sorted(k,key=lambda r:-r['peak']*r['len'])
req=[[r['i'],r['cx'],r['cy']] for r in k]
for s in range(0,len(req),14): json.dump(req[s:s+14],open(f'{OUT}/req{s//14}.json','w'))
print(len(req))
EOF
for n in 0 1 2 3; do python3 met/crops.py "$(cat met/req$n.json)" sheet_k$n 110 && cp met/sheet_k$n.png "$D/"; done; ls -la "$D"

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import json,os
OUT=os.path.expanduser('~/mnt/outputs/met')
F=json.load(open(OUT+'/files.json'))
fail=[i for i,f in enumerate(F) if f[2].startswith('failed_')]
print('failed idx',fail)
print('mod6 of failed',sorted(set(i%6 for i in fail)))
k=json.load(open(OUT+'/keep.json'))
nf=[r for r in k if r['i'] not in fail]
print('survivors not in failed frames:',len(nf))
for r in nf: print(r['i'],r['t'][9:15],r['cx'],r['cy'],r['len'],r['wid'],r['peak'])
EOF

# ----
cd "$HOME/mnt/outputs" && cat > met/scan2.py <<'EOF'
import numpy as np, cv2, json, os, sys
exec(open(os.path.expanduser('~/mnt/outputs/met/scan.py')).read().split('files=[]')[0])
OUT=os.path.expanduser('~/mnt/outputs/met'); ROOT=os.path.expanduser('~/mnt')
F=json.load(open(OUT+'/files.json'))
def path(i): return f"{ROOT}/{F[i][0]}/{F[i][2]}"
cache={}
def get(i):
    if i not in cache:
        a,h=read(path(i)); L=bin2(a).sum(-1)
        bg=cv2.medianBlur(cv2.resize(L,(240,135),interpolation=cv2.INTER_AREA).astype(np.float32),5)
        bg=cv2.resize(bg,(L.shape[1],L.shape[0]),interpolation=cv2.INTER_LINEAR)
        hp=L-bg
        cache[i]=(L,hp)
    return cache[i]
def sharp(hp):  # star peak index: mean of top 200 local maxima
    mx=cv2.dilate(hp,np.ones((5,5),np.uint8)); pk=hp[(hp==mx)&(hp>0)]
    return float(np.sort(pk)[-200:].mean()) if pk.size>200 else 0.0
start,end=int(sys.argv[1]),min(int(sys.argv[2]),len(F))
res=[]
for i in range(start,end):
    sess=F[i][0]
    nb=[j for j in (i-2,i-1,i+1,i+2) if 0<=j<len(F) and F[j][0]==sess]
    if len(nb)<2: res.append(dict(i=i,skip=1)); continue
    L,hp=get(i); H=np.stack([get(j)[1] for j in nb])
    D=hp-H.max(0)
    med=np.median(H,0)
    s=1.4826*np.median(np.abs((hp-med)-np.median(hp-med)))
    # star mask from static sky
    star=(med>6*s).astype(np.uint8); star=cv2.dilate(star,np.ones((5,5),np.uint8))
    # cloud mask: large-scale structure in the difference
    lg=cv2.GaussianBlur(hp-med,(0,0),6); cloud=(np.abs(lg)>1.2*s).astype(np.uint8)
    Dm=D.copy(); Dm[star>0]=0
    T=(Dm>2.5*s).astype(np.uint8)*255
    lines=cv2.HoughLinesP(T,1,np.pi/360,threshold=14,minLineLength=14,maxLineGap=3)
    out=[]
    if lines is not None:
        for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4):
            n=int(np.hypot(x1-x0,y1-y0))+1
            xs=np.linspace(x0,x1,n); ys=np.linspace(y0,y1,n)
            v=cv2.remap(D.astype(np.float32),xs.astype(np.float32)[None],ys.astype(np.float32)[None],cv2.INTER_LINEAR)[0]
            snr=float(v.mean()/s*np.sqrt(n))
            cf=float(cloud[ys.astype(int).clip(0,539),xs.astype(int).clip(0,959)].mean())
            out.append([int(x0),int(y0),int(x1),int(y1),round(snr,1),round(float(v.mean()/s),2),round(cf,2)])
    # merge near-duplicate segments: keep best per (angle,offset) cluster
    out.sort(key=lambda r:-r[4]); keep=[]
    for r in out:
        a=np.degrees(np.arctan2(r[3]-r[1],r[2]-r[0]))%180; c=((r[0]+r[2])/2,(r[1]+r[3])/2)
        dup=False
        for k in keep:
            ak=np.degrees(np.arctan2(k[3]-k[1],k[2]-k[0]))%180
            if min(abs(a-ak),180-abs(a-ak))<6 and np.hypot(c[0]-(k[0]+k[2])/2,c[1]-(k[1]+k[3])/2)<40: dup=True; break
        if not dup: keep.append(r)
    res.append(dict(i=i,s=round(float(s),1),sh=round(sharp(hp),1),cl=round(float(cloud.mean()),3),lines=keep[:12]))
    for k in list(cache):
        if k<i-2: del cache[k]
json.dump(res,open(f'{OUT}/scan2_{start:04d}.json','w'))
print('done',start,end,'frames w/ lines',sum(1 for r in res if r.get('lines')))
EOF
time python3 met/scan2.py 0 150

# ----
cd "$HOME/mnt/outputs" && python3 met/scan2.py 150 380 && python3 met/scan2.py 380 669

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import json,glob,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
R=[]
for f in sorted(glob.glob(OUT+'/scan2_*.json')): R+=json.load(open(f))
R={r['i']:r for r in R if not r.get('skip')}
idx=sorted(R); sh=np.array([R[i]['sh'] for i in idx])
# local median sharpness
jump=set()
for k,i in enumerate(idx):
    lo=max(0,k-4); hi=min(len(idx),k+5); m=np.median(sh[lo:hi])
    R[i]['shr']=round(R[i]['sh']/m,2) if m>0 else 1
    if R[i]['shr']<0.8: jump.add(i)
print('jump frames',len(jump),sorted(jump)[:60])
print('mod6',sorted({i%6 for i in jump}))
# collect all lines
allL=[]
for i in idx:
    for l in R[i]['lines']:
        allL.append(dict(i=i,x0=l[0],y0=l[1],x1=l[2],y1=l[3],snr=l[4],m=l[5],cf=l[6],jump=i in jump,cl=R[i]['cl']))
# recurring location filter (static artifacts): segment midpoint within 25px in >=3 other frames
mids=np.array([[(a['x0']+a['x1'])/2,(a['y0']+a['y1'])/2,a['i']] for a in allL])
for a in allL:
    mx,my=(a['x0']+a['x1'])/2,(a['y0']+a['y1'])/2
    d=np.hypot(mids[:,0]-mx,mids[:,1]-my); a['rec']=len(set(mids[d<25,2].astype(int))-{a['i']})
    a['len']=round(float(np.hypot(a['x1']-a['x0'],a['y1']-a['y0'])),1)
    a['ang']=round(float(np.degrees(np.arctan2(a['y1']-a['y0'],a['x1']-a['x0']))%180),1)
# satellites: collinear line in adjacent frames
def perp(a,b):
    p0=np.array([b['x0'],b['y0']],float); p1=np.array([b['x1'],b['y1']],float); d=p1-p0; n=np.linalg.norm(d)+1e-6
    q=np.array([(a['x0']+a['x1'])/2,(a['y0']+a['y1'])/2]); v=q-p0
    return abs(d[0]*v[1]-d[1]*v[0])/n
byi={}
for a in allL: byi.setdefault(a['i'],[]).append(a)
for a in allL:
    a['sat']=0
    for j in (a['i']-2,a['i']-1,a['i']+1,a['i']+2):
        for b in byi.get(j,[]):
            da=abs(a['ang']-b['ang']); da=min(da,180-da)
            if da<3 and perp(a,b)<5 and perp(b,a)<5: a['sat']=j-a['i']
json.dump(allL,open(OUT+'/lines.json','w'))
good=[a for a in allL if a['cf']<0.3 and a['rec']<2 and not a['sat'] and a['snr']>=10]
print('lines',len(allL),'satellite/plane (multi-frame collinear):',len(set((a['i'],a['x0']) for a in allL if a['sat'])),'  clean candidates',len(good))
for a in sorted(good,key=lambda a:-a['snr']): print(a['i'],'J' if a['jump'] else ' ',(a['x0'],a['y0'],a['x1'],a['y1']),'len',a['len'],'ang',a['ang'],'snr',a['snr'],'mean',a['m'],'cf',a['cf'],'cl',a['cl'])
sats=sorted({a['i'] for a in allL if a['sat']})
print('frames with multi-frame trails:',sats)
EOF

# ----
cd "$HOME/mnt/outputs" && D="$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan" && python3 - <<'EOF'
import json,os
OUT=os.path.expanduser('~/mnt/outputs/met')
L=json.load(open(OUT+'/lines.json'))
req=[]
for i in (413,449,450,451,501,503,573,592,600,617,629,657):
    c=[a for a in L if a['i']==i and a['sat']]
    if c:
        a=max(c,key=lambda a:a['snr']); req.append([i,(a['x0']+a['x1'])/2,(a['y0']+a['y1'])/2]); print(i,a['x0'],a['y0'],a['x1'],a['y1'],a['snr'],a['cf'])
req+= [[440,803,169],[589,610,110],[580,367,15]]
json.dump(req,open(OUT+'/req_s.json','w'))
EOF
python3 met/crops.py "$(cat met/req_s.json)" sheet_sat 110 && cp met/sheet_sat.png "$D/"

# ----
cd "$HOME/mnt/outputs" && cat > met/thin.py <<'EOF'
import numpy as np, cv2, json, os, sys
src=open(os.path.expanduser('~/mnt/outputs/met/scan2.py')).read().split('start,end=')[0]
exec(src)
L_=json.load(open(OUT+'/lines.json'))
byi={}
for a in L_: byi.setdefault(a['i'],[]).append(a)
start,end=int(sys.argv[1]),int(sys.argv[2])
res=[]
for i in sorted(byi):
    if not (start<=i<end): continue
    sess=F[i][0]; nb=[j for j in (i-2,i-1,i+1,i+2) if 0<=j<len(F) and F[j][0]==sess]
    L,hp=get(i); H=np.stack([get(j)[1] for j in nb]); D=(hp-H.max(0)).astype(np.float32)
    med=np.median(H,0); s=1.4826*np.median(np.abs((hp-med)-np.median(hp-med)))
    for a in byi[i]:
        p0=np.array([a['x0'],a['y0']],float); p1=np.array([a['x1'],a['y1']],float); d=p1-p0; n=np.linalg.norm(d)
        if n<1: continue
        u=d/n; v=np.array([-u[1],u[0]]); t=np.linspace(0,n,int(n)+1)
        prof=[]
        for o in range(-7,8):
            pts=p0[None]+t[:,None]*u[None]+o*v[None]
            val=cv2.remap(D,pts[:,0].astype(np.float32)[None],pts[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
            prof.append(float(np.median(val)))
        prof=np.array(prof)/s
        a['thin']=round(float(prof[6:9].max()-0.5*(prof[:3].mean()+prof[-3:].mean())),2)
        a['side']=round(float(max(prof[:3].mean(),prof[-3:].mean())),2)
        res.append(a)
    for k in list(cache):
        if k<i-2: del cache[k]
json.dump(res,open(f'{OUT}/thin_{start:04d}.json','w')); print('ok',len(res))
EOF
python3 met/thin.py 0 330

# ----
cd "$HOME/mnt/outputs" && python3 met/thin.py 330 700 && python3 - <<'EOF'
import json,glob,os
OUT=os.path.expanduser('~/mnt/outputs/met')
A=[]
for f in sorted(glob.glob(OUT+'/thin_*.json')): A+=json.load(open(f))
c=[a for a in A if a.get('thin',0)>2.0 and a['side']<1.5 and a['rec']<2 and not a['sat'] and not a['jump']]
print('thin, isolated, non-jump, non-satellite:',len(c))
for a in sorted(c,key=lambda a:-a['thin']*a['len']): print(a['i'],(a['x0'],a['y0'],a['x1'],a['y1']),'len',a['len'],'ang',a['ang'],'snr',a['snr'],'thin',a['thin'],'side',a['side'],'cf',a['cf'])
cj=[a for a in A if a.get('thin',0)>2.0 and a['side']<1.5 and a['rec']<2 and not a['sat'] and a['jump']]
print('jump-frame ones:',len(cj))
for a in sorted(cj,key=lambda a:-a['thin']*a['len'])[:15]: print(a['i'],(a['x0'],a['y0'],a['x1'],a['y1']),'len',a['len'],'snr',a['snr'],'thin',a['thin'])
EOF

# ----
cd "$HOME/mnt/outputs" && D="$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan" && python3 met/crops.py '[[289,585,421],[627,701,284],[288,556,325],[290,627,306],[663,638,375],[359,681,128],[358,722,319],[360,529,40],[587,716,223],[357,745,156],[295,670,305],[292,840,290],[506,643,222],[404,713,445]]' sheet_t 90 && cp met/sheet_t.png "$D/"

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import json,glob,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
F=json.load(open(OUT+'/files.json'))
R=[]
for f in sorted(glob.glob(OUT+'/scan2_*.json')): R+=json.load(open(f))
R={r['i']:r for r in R if not r.get('skip')}
idx=sorted(R); sh=np.array([R[i]['sh'] for i in idx])
for k,i in enumerate(idx):
    lo=max(0,k-4); hi=min(len(idx),k+5); m=np.median(sh[lo:hi]); R[i]['shr']=R[i]['sh']/m if m>0 else 1
for s in range(0,669,12):
    rows=[R[i] for i in range(s,s+12) if i in R]
    if not rows: print(s,'skip'); continue
    print(s,F[s][1][9:15],F[s][0][-23:-13],'sig',round(np.median([r['s'] for r in rows])),'sharp',round(np.median([r['sh'] for r in rows])),'cloud',round(max(r['cl'] for r in rows),3),'jumps',[r['i'] for r in rows if r['shr']<0.8])
EOF

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import json,glob,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
F=json.load(open(OUT+'/files.json'))
R=[]
for f in sorted(glob.glob(OUT+'/scan2_*.json')): R+=json.load(open(f))
R={r['i']:r for r in R if not r.get('skip')}
idx=sorted(R); sh=np.array([R[i]['sh'] for i in idx])
jump=set()
for k,i in enumerate(idx):
    lo=max(0,k-4); hi=min(len(idx),k+5); m=np.median(sh[lo:hi])
    if R[i]['sh']/m<0.8: jump.add(i)
jump|={i for i in range(116,576) if i%6==2}
clean=[i for i in range(0,576) if i not in jump and not (252<=i<=383) and not F[i][2].startswith('failed') and i in R]
print(len(clean), len([i for i in clean if i<116]))
paths=[f"/Volumes/Untitled/Astronomy/{F[i][0]}/{F[i][2]}" for i in clean]
json.dump(dict(clean=clean,paths=paths),open(OUT+'/clean.json','w'))
for b in range(0,len(paths),50): print(json.dumps(paths[b:b+50]))
EOF

# ----
python3 -c "import scipy; print(scipy.__version__)" 2>&1 | tail -1; python3 -c "import pywt; print('pywt ok')" 2>&1 | tail -1; python3 -c "import PIL; print('PIL', PIL.__version__)" 2>&1|tail -1; pip --version 2>&1 | tail -1

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import numpy as np, cv2, os, json, glob
exec(open(os.path.expanduser('~/mnt/outputs/met/scan.py')).read().split('files=[]')[0])
OUT=os.path.expanduser('~/mnt/outputs/met'); ROOT=os.path.expanduser('~/mnt')
F=json.load(open(OUT+'/files.json'))
p=f"{ROOT}/{F[480][0]}/{F[480][2]}"; a,h=read(p)
r=a[0::2,0::2]; g=(a[0::2,1::2]+a[1::2,0::2])/2; b=a[1::2,1::2]
print('subsample medians R G B',np.median(r),np.median(g),np.median(b))
u=np.clip(a,0,65535).astype(np.uint16)
for code,name in [(cv2.COLOR_BayerRG2BGR,'RG2BGR'),(cv2.COLOR_BayerBG2BGR,'BG2BGR')]:
    c=cv2.cvtColor(u,code); print(name,'B G R medians',[float(np.median(c[:,:,k])) for k in range(3)])
# DWARF stacked
sp=glob.glob(ROOT+'/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/stacked-16*.fits')[0]
with open(sp,'rb') as f:
    hd=b''
    while True:
        blk=f.read(2880); hd+=blk
        if b'END     ' in blk: 
            # check END card position
            if any(blk[i:i+3]==b'END' for i in range(0,2880,80)): break
    s=np.frombuffer(f.read(1920*1080*3*2),'>i2').reshape(3,1080,1920).astype(np.float32)+32768
print('stacked plane medians',[float(np.median(s[k][200:900,200:1700])) for k in range(3)])
print('raw min/max',a.min(),a.max(),'black guess p0.1',np.percentile(a,0.1))
EOF

# ----
mkdir -p "$HOME/mnt/outputs/met/stk" && cd "$HOME/mnt/outputs" && cat > met/stack.py <<'EOF'
import numpy as np, cv2, json, os, sys, time
exec(open(os.path.expanduser('~/mnt/outputs/met/scan.py')).read().split('files=[]')[0])
OUT=os.path.expanduser('~/mnt/outputs/met'); ST=OUT+'/stk'; ROOT=os.path.expanduser('~/mnt')
F=json.load(open(OUT+'/files.json')); C=json.load(open(OUT+'/clean.json'))['clean']
REF=480; W,H=1920,1080
def path(i): return f"{ROOT}/{F[i][0]}/{F[i][2]}"
def gain(i): return 40 if 'GAIN_40' in F[i][0] else 60
def raw(i):
    a,h=read(path(i))
    hp=ST+f'/hot{gain(i)}.npy'
    if os.path.exists(hp): a=a-np.load(hp)
    return a
def debayer(a): return cv2.cvtColor(np.clip(a,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2BGR).astype(np.float32)
def lum2(a): return (a[0::2,0::2]+a[0::2,1::2]+a[1::2,0::2]+a[1::2,1::2]).astype(np.float32)
def stars(a,n=400):
    L=lum2(a); bg=cv2.resize(cv2.medianBlur(cv2.resize(L,(120,68),interpolation=cv2.INTER_AREA),5),(L.shape[1],L.shape[0]),interpolation=cv2.INTER_LINEAR)
    hp=L-bg; s=1.4826*np.median(np.abs(hp-np.median(hp)))
    mx=cv2.dilate(hp,np.ones((5,5),np.uint8)); pk=(hp==mx)&(hp>8*s)
    pk[:4]=0;pk[-4:]=0;pk[:,:4]=0;pk[:,-4:]=0
    ys,xs=np.nonzero(pk); v=hp[ys,xs]; o=np.argsort(-v)[:n]; ys,xs,v=ys[o],xs[o],v[o]
    out=[]
    for y,x,p in zip(ys,xs,v):
        w=np.clip(hp[y-2:y+3,x-2:x+3],0,None); t=w.sum()
        if t<=0: continue
        gy,gx=np.mgrid[-2:3,-2:3]; cy=y+(w*gy).sum()/t; cx=x+(w*gx).sum()/t
        flux=float(hp[y-2:y+3,x-2:x+3].sum())
        out.append((2*cx+0.5,2*cy+0.5,flux,p))
    return np.array(out,np.float64),s
def pfeat(xy):
    x=(xy[:,0]-W/2)/(W/2); y=(xy[:,1]-H/2)/(W/2)
    return np.stack([np.ones_like(x),x,y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3],1)
def pfit(src,dst):
    A=pfeat(src); cx,_,_,_=np.linalg.lstsq(A,dst[:,0],rcond=None); cy,_,_,_=np.linalg.lstsq(A,dst[:,1],rcond=None); return np.stack([cx,cy])
def papply(P,xy): A=pfeat(xy); return np.stack([A@P[0],A@P[1]],1)
def match(ref,frm,pred,r=6.0):
    d=np.hypot(pred[:,None,0]-frm[None,:,0],pred[:,None,1]-frm[None,:,1]); j=d.argmin(1); ok=d[np.arange(len(pred)),j]<r
    # unique
    pairs={}
    for a_,(jj,o) in enumerate(zip(j,ok)):
        if o and (jj not in pairs or d[a_,jj]<d[pairs[jj],jj]): pairs[jj]=a_
    ia=np.array(list(pairs.values()),int); ib=np.array(list(pairs.keys()),int); return ia,ib
def search(ref,frm):
    R=ref[:60,:2]; Fm=frm[:60,:2]; best=(0,None)
    c=np.array([W/2,H/2])
    for th in np.arange(-60,61,1.0):
        t=np.radians(th); M=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        rr=(R-c)@M.T+c; dif=(Fm[:,None,:]-rr[None,:,:]).reshape(-1,2)
        Hh,xe,ye=np.histogram2d(dif[:,0],dif[:,1],bins=[np.arange(-1400,1400,8),np.arange(-900,900,8)])
        k=np.unravel_index(Hh.argmax(),Hh.shape)
        if Hh[k]>best[0]: best=(Hh[k],(th,xe[k[0]]+4,ye[k[1]]+4))
    th,tx,ty=best[1]; t=np.radians(th); M=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    def f(xy): return (xy-c)@M.T+c+np.array([tx,ty])
    return f,best[0]
def register(ref,frm,prevP=None):
    pred=papply(prevP,ref[:,:2]) if prevP is not None else None
    if pred is not None:
        ia,ib=match(ref,frm,pred,8.0)
    if pred is None or len(ia)<40:
        f,score=search(ref,frm); pred=f(ref[:,:2]); ia,ib=match(ref,frm,pred,10.0)
    if len(ia)<25: return None,0,99
    src=ref[ia,:2]; dst=frm[ib,:2]
    Maff,inl=cv2.estimateAffine2D(src.astype(np.float32),dst.astype(np.float32),method=cv2.RANSAC,ransacReprojThreshold=3.0)
    inl=inl.ravel().astype(bool); P=pfit(src[inl],dst[inl])
    for _ in range(2):
        pred=papply(P,ref[:,:2]); ia,ib=match(ref,frm,pred,3.0)
        src=ref[ia,:2]; dst=frm[ib,:2]; P=pfit(src,dst); res=np.hypot(*(papply(P,src)-dst).T)
        k=res<1.5; P=pfit(src[k],dst[k])
    res=np.hypot(*(papply(P,src[k])-dst[k]).T)
    fl=np.median(ref[ia[k],2]/np.maximum(frm[ib[k],2],1e-3))
    return P,int(k.sum()),float(np.sqrt((res**2).mean())),float(fl)
cmd=sys.argv[1]
if cmd=='hot':
    for g,sel in [(40,[i for i in C if gain(i)==40][::3][:40]),(60,[i for i in C if gain(i)==60][::6][:45])]:
        stack=np.stack([read(path(i))[0] for i in sel])
        med=np.median(stack,0); hot=np.zeros_like(med)
        for dy in (0,1):
            for dx in (0,1):
                s=med[dy::2,dx::2]; b=cv2.medianBlur(s.astype(np.float32),5); d=s-b
                sg=1.4826*np.median(np.abs(d)); hot[dy::2,dx::2]=np.where(d>5*sg,d,0)
        np.save(ST+f'/hot{g}.npy',hot.astype(np.float32)); print('hot',g,len(sel),'hot px',int((hot>0).sum()))
if cmd=='ref':
    a=raw(REF); s,_=stars(a,500); np.save(ST+'/refstars.npy',s); print('ref stars',len(s))
if cmd=='reg':
    ref=np.load(ST+'/refstars.npy'); a0,a1=int(sys.argv[2]),int(sys.argv[3])
    regf=ST+'/reg.json'; reg=json.load(open(regf)) if os.path.exists(regf) else {}
    sub=[i for i in C if a0<=i<a1]; prev=None; t0=time.time()
    for i in sub:
        # prev transform from nearest registered frame of same session
        cands=[int(k) for k in reg if F[int(k)][0]==F[i][0]]
        if cands: prev=np.array(reg[str(min(cands,key=lambda k:abs(k-i)))]['P'])
        else: prev=None
        a=raw(i); s,sg=stars(a,400)
        P,n,rms,fl=register(ref,s,prev) if True else None
        if P is None: print('fail',i); continue
        reg[str(i)]=dict(P=P.tolist(),n=n,rms=rms,k=fl,noise=float(sg))
    json.dump(reg,open(regf,'w')); print('reg',len(sub),'total',len(reg),'t',round(time.time()-t0,1))
EOF
python3 met/stack.py hot && python3 met/stack.py ref && python3 met/stack.py reg 470 490 && python3 -c "
import json,os; r=json.load(open(os.path.expanduser('~/mnt/outputs/met/stk/reg.json')))
for k,v in sorted(r.items(),key=lambda x:int(x[0])): print(k,v['n'],round(v['rms'],2),round(v['k'],3),round(v['noise'],1))"

# ----
cd "$HOME/mnt/outputs" && python3 met/stack.py reg 490 576 && python3 met/stack.py reg 384 470 && python3 met/stack.py reg 116 252 && python3 met/stack.py reg 0 116 && python3 -c "
import json,os,numpy as np; r=json.load(open(os.path.expanduser('~/mnt/outputs/met/stk/reg.json')))
ks=sorted(r,key=int); n=np.array([r[k]['n'] for k in ks]); rms=np.array([r[k]['rms'] for k in ks]); kk=np.array([r[k]['k'] for k in ks])
print('registered',len(ks),'n min/med',n.min(),np.median(n),'rms max/med',rms.max().round(2),np.median(rms).round(2))
for k in ks[::25]: print(k,r[k]['n'],round(r[k]['rms'],2),round(r[k]['k'],3),round(r[k]['noise'],1))
bad=[k for k in ks if r[k]['n']<120 or r[k]['rms']>1.0]; print('weak',bad)"

# ----
cd "$HOME/mnt/outputs" && cat >> met/stack.py <<'EOF'
def maps(P):
    gy,gx=np.mgrid[0:H+8:8,0:W+8:8].astype(np.float64)
    xy=np.stack([gx.ravel(),gy.ravel()],1); m=papply(P,xy)
    mx=m[:,0].reshape(gx.shape).astype(np.float32); my=m[:,1].reshape(gx.shape).astype(np.float32)
    mx=cv2.resize(mx,(W+8,H+8),interpolation=cv2.INTER_LINEAR)[:H,:W]; my=cv2.resize(my,(W+8,H+8),interpolation=cv2.INTER_LINEAR)[:H,:W]
    return mx,my
SKYREF=None
def frame_warped(i,reg):
    r=reg[str(i)]; a=raw(i); c=debayer(a)
    sky=np.median(c.reshape(-1,3)[::7],0)
    c=(c-sky)*r['k']
    mx,my=maps(np.array(r['P']))
    w=cv2.remap(c,mx,my,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(np.nan,np.nan,np.nan))
    return w
if cmd in ('acc1','acc2'):
    reg=json.load(open(ST+'/reg.json')); a0,a1=int(sys.argv[2]),int(sys.argv[3])
    fn=ST+f'/{cmd}.npz'
    if os.path.exists(fn) and a0>0:
        z=np.load(fn); S=z['S']; Q=z['Q']; N=z['N']; done=list(z['done'])
    else:
        S=np.zeros((H,W,3),np.float64); Q=np.zeros((H,W,3),np.float64); N=np.zeros((H,W,3),np.float32); done=[]
    if cmd=='acc2':
        z1=np.load(ST+'/acc1.npz'); n1=np.maximum(z1['N'],1); mu=(z1['S']/n1).astype(np.float32); sd=np.sqrt(np.maximum(z1['Q']/n1-mu.astype(np.float64)**2,0)).astype(np.float32)
        sd=np.maximum(sd,cv2.GaussianBlur(sd,(0,0),3)*0.7)
    t0=time.time(); sub=[i for i in C if a0<=i<a1 and str(i) in reg and i not in done]
    for i in sub:
        w=frame_warped(i,reg); ok=np.isfinite(w)
        if cmd=='acc1':
            w0=np.where(ok,w,0); S+=w0; Q+=w0.astype(np.float64)**2; N+=ok
        else:
            r=reg[str(i)]; wt=1.0/(r['k']*r['noise'])**2*1e4
            ok&=np.abs(w-mu)<3.0*sd
            w0=np.where(ok,w,0); S+=wt*w0; N+=wt*ok; Q+=ok
        done.append(i)
        if time.time()-t0>150: break
    np.savez(fn,S=S,Q=Q,N=N,done=np.array(done))
    print(cmd,'frames done',len(done),'of',len([i for i in C if str(i) in reg]),'t',round(time.time()-t0,1))
EOF
python3 met/stack.py acc1 0 700

# ----
cd "$HOME/mnt/outputs" && python3 met/stack.py acc2 0 700 && python3 - <<'EOF'
import numpy as np, os, cv2
ST=os.path.expanduser('~/mnt/outputs/met/stk')
z=np.load(ST+'/acc2.npz'); S,N,Q=z['S'],z['N'],z['Q']
st=(S/np.maximum(N,1e-9)).astype(np.float32); cnt=Q.astype(np.float32)
st[N<=0]=np.nan
print('stack stats',np.nanpercentile(st,[1,50,99,99.99]),'count min/med/max',cnt.min(),np.median(cnt),cnt.max())
z1=np.load(ST+'/acc1.npz'); print('rejected fraction', 1-np.median(cnt)/np.median(z1['N']))
D=os.path.expanduser('~/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan')
np.save(D+'/alhena_stack_389.npy',st); np.save(D+'/alhena_count_389.npy',cnt.astype(np.float16))
v=np.nan_to_num(st.sum(2)); v=np.clip((v-np.percentile(v,5))/(np.percentile(v,99.5)-np.percentile(v,5)),0,1)**0.4
cv2.imwrite(D+'/alhena_stack_preview.jpg',(v*255).astype(np.uint8))
print(os.listdir(D))
EOF

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import numpy as np, os, json
exec(open(os.path.expanduser('~/mnt/outputs/met/stack.py')).read().split("cmd=sys.argv[1]")[0])
def cfa_med(i):
    a,h=read(path(i)); return [float(np.median(a[dy::2,dx::2][100:440,200:760])) for dy,dx in ((0,0),(0,1),(1,0),(1,1))]
g40=np.array([cfa_med(i) for i in range(106,116)]); g60=np.array([cfa_med(i) for i in [117,118,119,120,121,123,124,125]])
print('G40 center sky R,G1,G2,B',g40.mean(0).round(1)); print('G60',g60.mean(0).round(1))
reg=json.load(open(ST+'/reg.json')); k40=np.median([reg[str(i)]['k'] for i in range(100,116) if str(i) in reg]); k60=np.median([reg[str(i)]['k'] for i in range(117,130) if str(i) in reg])
r=k40/k60; print('gain ratio (60/40)',round(r,3))
b=(r*g40.mean(0)-g60.mean(0))/(r-1); print('implied black per CFA',b.round(1))
# darkest pixels stats
a,h=read(path(480)); print('p0.01 per cfa',[float(np.percentile(a[dy::2,dx::2],0.01)) for dy,dx in ((0,0),(0,1),(1,0),(1,1))])
EOF

# ----
cd "$HOME/mnt/outputs" && cat >> met/stack.py <<'EOF'
BLACK=240.0
def superflat_fit(g):
    sel=[i for i in C if gain(i)==g][::max(1,len([i for i in C if gain(i)==g])//45)][:45]
    st=np.stack([read(path(i))[0]-np.load(ST+f'/hot{g}.npy')-BLACK for i in sel])
    med=np.median(st,0)
    subs=[med[0::2,0::2],(med[0::2,1::2]+med[1::2,0::2])/2,med[1::2,1::2]]
    gh,gw=54,96
    ys,xs=np.mgrid[0:gh,0:gw]; X=((xs+0.5)*20-960/2)/(960/2); Y=((ys+0.5)*20-540/2)/(960/2); R2=X**2+Y**2
    rows=[];rhs=[]
    cells=[cv2.resize(cv2.medianBlur(s.astype(np.float32),5),(gw,gh),interpolation=cv2.INTER_AREA) for s in subs]
    for c in range(3):
        v=np.log(np.maximum(cells[c],1)).ravel()
        A=np.zeros((v.size,9)); A[:,3*c:3*c+3]=np.stack([np.ones(v.size),X.ravel(),Y.ravel()],1)
        A=np.hstack([A[:,:9],np.stack([R2.ravel(),R2.ravel()**2,R2.ravel()**3],1)])
        rows.append(A); rhs.append(v)
    A=np.vstack(rows); b=np.concatenate(rhs); m=np.ones(b.size,bool)
    for _ in range(3):
        coef,*_=np.linalg.lstsq(A[m],b[m],rcond=None); res=b-A@coef; s=1.4826*np.median(np.abs(res[m])); m=np.abs(res)<3*s
    v1,v2,v3=coef[9:]
    yy,xx=np.mgrid[0:H,0:W]; Xf=(xx-W/2)/(W/2); Yf=(yy-H/2)/(W/2); r2=Xf**2+Yf**2
    V=np.exp(v1*r2+v2*r2**2+v3*r2**3).astype(np.float32)
    np.save(ST+f'/vig{g}.npy',V); print('vig',g,'coef',np.round(coef[9:],3),'corner/center',round(float(V[0,0]),3),'edge mid-left',round(float(V[540,0]),3),'fit rms',round(float(s),4))
    return V
def polybg(c,deg=2):
    L=cv2.resize(c,(240,135),interpolation=cv2.INTER_AREA)
    ys,xs=np.mgrid[0:135,0:240]; X=(xs-120)/120.0; Y=(ys-67.5)/120.0
    terms=[np.ones_like(X)]+[X**i*Y**j for d in range(1,deg+1) for i in range(d+1) for j in [d-i]]
    A=np.stack([t.ravel() for t in terms],1); out=np.zeros_like(c)
    yf,xf=np.mgrid[0:H,0:W]; Xf=(xf-W/2)/(W/2); Yf=(yf-H/2)/(W/2)
    Tf=[np.ones_like(Xf)]+[Xf**i*Yf**j for d in range(1,deg+1) for i in range(d+1) for j in [d-i]]
    for ch in range(3):
        v=L[:,:,ch].ravel(); valid=np.isfinite(v); m=valid.copy()
        for _ in range(4):
            coef,*_=np.linalg.lstsq(A[m],v[m],rcond=None); res=v-A@coef; s=1.4826*np.median(np.abs(res[m])); m=valid&(res<2.0*s)&(res>-3*s)
        out[:,:,ch]=sum(cf*t for cf,t in zip(coef,Tf))
    return out
def frame_warped(i,reg):
    r=reg[str(i)]; a=raw(i)-BLACK; V=np.load(ST+f'/vig{gain(i)}.npy') if gain(i) not in VC else VC[gain(i)]; VC[gain(i)]=V
    a=a/V; c=cv2.cvtColor(np.clip(a+1000,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2BGR).astype(np.float32)-1000
    c=(c-polybg(c,DEG))*r['k']
    mx,my=maps(np.array(r['P']))
    return cv2.remap(c,mx,my,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(np.nan,np.nan,np.nan))
VC={}; DEG=int(os.environ.get('DEG','2'))
if cmd=='flat':
    superflat_fit(40); superflat_fit(60)
EOF
sed -i 's/^if cmd in (.acc1.,.acc2.):/if cmd in ("acc1x","acc2x"):/' met/stack.py && grep -n 'if cmd in' met/stack.py && python3 met/stack.py flat

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import os
p=os.path.expanduser('~/mnt/outputs/met/stack.py'); s=open(p).read()
blk=s[s.index('if cmd in ("acc1x","acc2x"):'):s.index('BLACK=240.0')]
s=s+'\n'+blk.replace('if cmd in ("acc1x","acc2x"):','if cmd in ("acc1","acc2"):')
s=s.replace("np.savez(fn,S=S,Q=Q,N=N,done=np.array(done))","np.savez(fn,S=S,Q=Q,N=N,done=np.array(done,dtype=int))")
open(p,'w').write(s)
EOF
DEG=2 python3 met/stack.py acc1 0 700

# ----
cd "$HOME/mnt/outputs" && DEG=2 python3 met/stack.py acc2 0 700 && python3 - <<'EOF'
import numpy as np, os, cv2
ST=os.path.expanduser('~/mnt/outputs/met/stk')
z=np.load(ST+'/acc2.npz'); S,N,Q=z['S'],z['N'],z['Q']
st=(S/np.maximum(N,1e-9)).astype(np.float32); st[N<=0]=np.nan
D=os.path.expanduser('~/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan')
np.save(D+'/alhena_stack2.npy',st); np.save(D+'/alhena_count2.npy',Q.astype(np.float16))
v=np.nan_to_num(st.sum(2)); lo,hi=np.percentile(v,[2,99.7]); v=np.clip((v-lo)/(hi-lo),0,1)**0.45
cv2.imwrite(D+'/alhena_stack2_preview.jpg',(v*255).astype(np.uint8)); print('ok',np.nanpercentile(st,[1,50,99]))
EOF

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import numpy as np, cv2, os, json
exec(open(os.path.expanduser('~/mnt/outputs/met/stack.py')).read().split("cmd=sys.argv[1]")[0])
res=[]
for i in (200,480,530):
    a=raw(i)-240; c=cv2.cvtColor(np.clip(a,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2BGR).astype(np.float32)
    s,_=stars(a,300)
    for x,y,fl,p in s:
        xi,yi=int(round(x)),int(round(y))
        if xi<8 or yi<8 or xi>W-9 or yi>H-9: continue
        cent=[]
        for ch in range(3):
            w=c[yi-5:yi+6,xi-5:xi+6,ch]; w=w-np.median(c[yi-10:yi+11,xi-10:xi+11,ch]); w=np.clip(w,0,None)
            if w.max()>3500 or w.sum()<=0: cent=None; break
            gy,gx=np.mgrid[-5:6,-5:6]; cent.append((xi+(w*gx).sum()/w.sum(),yi+(w*gy).sum()/w.sum()))
        if cent is None: continue
        cent=np.array(cent); rx,ry=x-W/2,y-H/2; r=np.hypot(rx,ry)
        if r<50: continue
        u=np.array([rx,ry])/r
        res.append((r,float((cent[2]-cent[1])@u),float((cent[0]-cent[1])@u)))
res=np.array(res)
for lo,hi in ((0,300),(300,600),(600,800),(800,1000),(1000,1200)):
    m=(res[:,0]>=lo)&(res[:,0]<hi)
    if m.sum()>5: print(f'r {lo}-{hi}: n={m.sum()} radial shift R-G {np.median(res[m,1]):+.2f}px  B-G {np.median(res[m,2]):+.2f}px')
EOF

# ----
cd "$HOME/mnt/outputs" && cat > met/inject.py <<'EOF'
import numpy as np, cv2, json, os
src=open(os.path.expanduser('~/mnt/outputs/met/scan2.py')).read().split('start,end=')[0]
exec(src)
# reference star photometry in frame 480 (binned coords = full/2)
L,hp=get(480)
def apflux(x,y,r=4):
    xb,yb=x/2-0.25,y/2-0.25; xi,yi=int(round(xb)),int(round(yb))
    w=hp[yi-r:yi+r+1,xi-r:xi+r+1]; return float(w.sum())
ref={'nu Gem':(1089.5,433.4,4.1),'xi Ori':(1243.4,679.3,4.5),'13 Mon':(1024.8,948.5,4.5),'nu Ori':(1286.9,658.2,4.4),'eps Mon':(1111.9,1068.9,4.4),'lambda Gem':(610.4,542.1,3.6)}
zs=[]
for n,(x,y,m) in ref.items():
    f=apflux(x,y); zs.append(m+2.5*np.log10(max(f,1))); print(n,m,round(f),'zp',round(zs[-1],2))
ZP=float(np.median(zs)); print('ZP (10 s frame, binned lum units)',round(ZP,2))
def detect(i,addimg):
    sess=F[i][0]; nb=[j for j in (i-2,i-1,i+1,i+2) if F[j][0]==sess]
    L,hp=get(i); hp=hp+addimg; H_=np.stack([get(j)[1] for j in nb]); D=hp-H_.max(0); med=np.median(H_,0)
    s=1.4826*np.median(np.abs((hp-med)-np.median(hp-med)))
    star=cv2.dilate((med>6*s).astype(np.uint8),np.ones((5,5),np.uint8)); Dm=D.copy(); Dm[star>0]=0
    T=(Dm>2.5*s).astype(np.uint8)*255
    lines=cv2.HoughLinesP(T,1,np.pi/360,threshold=14,minLineLength=14,maxLineGap=3)
    return lines
rng=np.random.default_rng(3)
res={}
for m in [1.0,2.0,2.5,3.0,3.5,4.0]:
    hit=0; tot=0
    for i in [150,160,190,430,450,480,510,540]:
        for trial in range(3):
            tau=0.4; ell=rng.uniform(25,60); ang=rng.uniform(0,np.pi)
            x0,y0=rng.uniform(150,800),rng.uniform(100,440); x1,y1=x0+ell*np.cos(ang),y0+ell*np.sin(ang)
            img=np.zeros((540,960),np.float32); cv2.line(img,(int(x0*8),int(y0*8)),(int(x1*8),int(y1*8)),1.0,1,cv2.LINE_AA,3)
            img=cv2.GaussianBlur(img,(0,0),1.1); tot_flux=10**((ZP-m)/2.5)*tau/10
            img*=tot_flux/max(img.sum(),1e-6)
            ln=detect(i,img); found=False
            if ln is not None:
                for a,b,c,d in np.asarray(ln).reshape(-1,4):
                    mx,my=(a+c)/2,(b+d)/2
                    if abs(np.cross([x1-x0,y1-y0],[mx-x0,my-y0]))/ell<4 and min(np.hypot(mx-x0,my-y0),np.hypot(mx-x1,my-y1))<ell+5: found=True
            hit+=found; tot+=1
    res[m]=hit/tot; print('mag',m,'detected',hit,'/',tot)
EOF
python3 met/inject.py

# ----
cd "$HOME/mnt/outputs" && cat > met/mf.py <<'EOF'
import numpy as np, cv2, json, os, sys
src=open(os.path.expanduser('~/mnt/outputs/met/scan2.py')).read().split('start,end=')[0]
exec(src)
KL=15; KER=[]
for th in np.arange(0,180,10):
    k=np.zeros((KL+6,KL+6),np.float32); c=(KL+6)/2; t=np.radians(th)
    cv2.line(k,(int((c-KL/2*np.cos(t))*16),int((c-KL/2*np.sin(t))*16)),(int((c+KL/2*np.cos(t))*16),int((c+KL/2*np.sin(t))*16)),1.0,1,cv2.LINE_AA,4)
    k=cv2.GaussianBlur(k,(0,0),0.8); k/=k.sum(); KER.append((th,k,float(np.sqrt((k**2).sum()))))
def Dframe(i,add=None):
    sess=F[i][0]; nb=[j for j in (i-2,i-1,i+1,i+2) if 0<=j<len(F) and F[j][0]==sess]
    L,hp=get(i)
    if add is not None: hp=hp+add
    H_=np.stack([get(j)[1] for j in nb]); med=np.median(H_,0); D=hp-H_.max(0)
    s=1.4826*np.median(np.abs((hp-med)-np.median(hp-med)))
    star=cv2.dilate((med>5*s).astype(np.uint8),np.ones((7,7),np.uint8))
    lg=cv2.GaussianBlur(hp-med,(0,0),6); cloud=(np.abs(lg)>1.2*s).astype(np.uint8)
    D=np.where(star>0,0,D).astype(np.float32)
    R=np.max(np.stack([cv2.filter2D(D,-1,k)/(s*n) for _,k,n in KER]),0)
    return R,cloud,s
def cands(R,cloud,thr):
    T=(R>thr).astype(np.uint8); n,lab,st,cen=cv2.connectedComponentsWithStats(T,8); out=[]
    for k in range(1,n):
        x,y,w,h,a=st[k]
        if max(w,h)<12 or a<12: continue
        ys,xs=np.where(lab==k); pts=np.stack([xs,ys],1).astype(np.float32); m=pts.mean(0); cov=np.cov((pts-m).T); ev,evec=np.linalg.eigh(cov)
        ln=4*np.sqrt(max(ev[1],1e-6)); wd=4*np.sqrt(max(ev[0],1e-6))
        if ln/max(wd,1)<3: continue
        out.append(dict(cx=float(m[0]),cy=float(m[1]),len=float(ln),wid=float(wd),peak=float(R[ys,xs].max()),cf=float(cloud[ys,xs].mean()),ang=float(np.degrees(np.arctan2(evec[1,1],evec[0,1]))%180)))
    return out
if __name__=='__main__' and sys.argv[1]=='inj':
    ZP=14.69; rng=np.random.default_rng(5); thr=float(sys.argv[2])
    # false positives on real frames without injection
    fp=0
    for i in [150,160,190,430,450,480,510,540]:
        R,cl,s=Dframe(i); fp+=len([c for c in cands(R,cl,thr) if c['cf']<0.3])
    print('thr',thr,'false cands in 8 clean frames:',fp)
    for m in [-1.0,0.0,1.0,2.0,3.0]:
        hit=0;tot=0
        for i in [150,160,190,430,450,480,510,540]:
            for tr in range(2):
                tau=0.4; ell=rng.uniform(25,60); ang=rng.uniform(0,np.pi)
                x0,y0=rng.uniform(150,800),rng.uniform(100,440); x1,y1=x0+ell*np.cos(ang),y0+ell*np.sin(ang)
                img=np.zeros((540,960),np.float32); cv2.line(img,(int(x0*8),int(y0*8)),(int(x1*8),int(y1*8)),1.0,1,cv2.LINE_AA,3)
                img=cv2.GaussianBlur(img,(0,0),1.1); img*=10**((ZP-m)/2.5)*tau/10/max(img.sum(),1e-6)
                R,cl,s=Dframe(i,img); f=False
                for c in cands(R,cl,thr):
                    if np.hypot(c['cx']-(x0+x1)/2,c['cy']-(y0+y1)/2)<ell/2+5: f=True
                hit+=f; tot+=1
        print(' mag',m,'tau 0.4s len 25-60px: detected',hit,'/',tot)
EOF
python3 met/mf.py inj 5.0

# ----
cd "$HOME/mnt/outputs" && python3 met/mf.py inj 4.0

# ----
cd "$HOME/mnt/outputs" && cat >> met/mf.py <<'EOF'
if __name__=='__main__' and sys.argv[1]=='scan':
    a0,a1=int(sys.argv[2]),min(int(sys.argv[3]),len(F)); out=[]
    import time; t0=time.time()
    for i in range(a0,a1):
        sess=F[i][0]; nb=[j for j in (i-2,i-1,i+1,i+2) if 0<=j<len(F) and F[j][0]==sess]
        if len(nb)<2: continue
        R,cl,s=Dframe(i)
        for c in cands(R,cl,5.0): c['i']=i; out.append(c)
        for k in list(cache):
            if k<i-2: del cache[k]
    json.dump(out,open(f'{OUT}/mf_{a0:04d}.json','w')); print('done',a0,a1,len(out),round(time.time()-t0,1))
EOF
python3 met/mf.py scan 0 230

# ----
cd "$HOME/mnt/outputs" && python3 met/mf.py scan 230 460 && python3 met/mf.py scan 460 669

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF' 2>/dev/null
import json,glob,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
A=[]
for f in sorted(glob.glob(OUT+'/mf_*.json')): A+=json.load(open(f))
jump={i for i in range(116,669) if i%6==2}|{440}
print('total',len(A))
P=np.array([[a['cx'],a['cy'],a['i']] for a in A])
def angd(a,b): d=abs(a-b)%180; return min(d,180-d)
byi={}
for a in A: byi.setdefault(a['i'],[]).append(a)
keep=[]
for a in A:
    if a['cf']>=0.3: continue
    d=np.hypot(P[:,0]-a['cx'],P[:,1]-a['cy']); rec=len(set(P[d<20,2].astype(int))-{a['i']})
    if rec>=2: continue
    sat=False
    for j in (a['i']-2,a['i']-1,a['i']+1,a['i']+2):
        for b in byi.get(j,[]):
            if angd(a['ang'],b['ang'])<5:
                t=np.radians(a['ang']); n=np.array([-np.sin(t),np.cos(t)]); 
                if abs(n@np.array([b['cx']-a['cx'],b['cy']-a['cy']]))<6: sat=True
    a['sat']=sat; a['jump']=a['i'] in jump
    if not sat: keep.append(a)
print('after cloud/recurring/satellite filters:',len(keep),' non-jump:',len([a for a in keep if not a['jump']]))
for a in sorted(keep,key=lambda a:(a['jump'],-a['peak'])): print(a['i'],'J' if a['jump'] else ' ',round(a['cx']),round(a['cy']),'len',round(a['len']),'wid',round(a['wid'],1),'ang',round(a['ang']),'pk',round(a['peak'],1),'cf',round(a['cf'],2))
json.dump(keep,open(OUT+'/mf_keep.json','w'))
EOF

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF' 2>/dev/null
import json,glob,os,numpy as np
OUT=os.path.expanduser('~/mnt/outputs/met')
A=[]
for f in sorted(glob.glob(OUT+'/mf_*.json')): A+=json.load(open(f))
jump={i for i in range(116,669) if i%6==2}|{440}
clear=[a for a in A if a['cf']<0.3 and a['i'] not in jump and (a['i']<252 or 384<=a['i']<576)]
print('clear-frame, non-jump candidates (before recurrence/satellite filters):',len(clear))
P=np.array([[a['cx'],a['cy'],a['i']] for a in A])
for a in sorted(clear,key=lambda a:-a['peak']):
    d=np.hypot(P[:,0]-a['cx'],P[:,1]-a['cy']); fr=sorted(set(P[d<20,2].astype(int))-{a['i']})
    print(a['i'],round(a['cx']),round(a['cy']),'len',round(a['len']),'pk',round(a['peak'],1),'ang',round(a['ang']),'others@loc',fr[:8],'(jumps' ,sum(1 for f in fr if f in jump),')')
EOF

# ----
cd "$HOME/mnt/outputs" && D="$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan" && python3 met/crops.py '[[562,364,190],[394,312,300],[396,647,315],[396,506,388],[393,543,408],[397,501,410],[399,491,465],[407,870,378],[408,601,144],[409,855,418],[411,741,315],[414,432,410]]' sheet_m1 90 && python3 met/crops.py '[[415,845,518],[415,793,481],[417,412,517],[419,325,445],[423,317,517],[569,706,169],[571,682,217],[571,915,148],[412,862,363],[416,776,515],[389,672,216],[402,476,522]]' sheet_m2 90 && cp met/sheet_m1.png met/sheet_m2.png "$D/"

# ----
cd "$HOME/mnt/outputs" && D="$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan" && python3 - <<'EOF'
import numpy as np, cv2, json, os
src=open(os.path.expanduser('~/mnt/outputs/met/mf.py')).read().split("if __name__=='__main__' and sys.argv[1]=='inj'")[0]
exec(src)
D_=os.path.expanduser('~/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan')
print('times',[F[i][1] for i in (560,561,562,563,564)])
# full difference images for 560..564
rows=[]
for i in (560,561,562,563,564):
    sess=F[i][0]; nb=[j for j in (i-2,i-1,i+1,i+2) if F[j][0]==sess]
    L,hp=get(i); H_=np.stack([get(j)[1] for j in nb]); Dd=hp-H_.max(0); med=np.median(H_,0)
    s=1.4826*np.median(np.abs((hp-med)-np.median(hp-med)))
    v=np.clip(Dd/(6*s),0,1)*255
    rows.append(v[0:400,180:560])
    if i==562:
        # trace the line: matched filter response along the best line
        R=np.max(np.stack([cv2.filter2D(np.where(Dd>0,Dd,0).astype(np.float32),-1,k)/(s*n) for _,k,n in KER]),0)
        ys,xs=np.where(R>4)
        sel=(xs>300)&(xs<420)&(ys<400)
        xs,ys=xs[sel],ys[sel]; print('line pixels',len(xs),'x range',xs.min(),xs.max(),'y range',ys.min(),ys.max())
        A=np.stack([ys,np.ones_like(ys)],1).astype(float); cx=np.linalg.lstsq(A,xs.astype(float),rcond=None)[0]; print('x = %.4f*y + %.1f'%tuple(cx))
        prof=[]
        for y in range(0,540,10):
            x=cx[0]*y+cx[1]
            if 2<x<957:
                v_=Dd[y:y+10,int(x)-2:int(x)+3].sum()/10; prof.append((y,round(float(v_/s),1)))
        print('profile (y, sum across 5px per row / sigma):',prof)
        np.save(D_+'/f562_D.npy',Dd.astype(np.float32))
cv2.imwrite(D_+'/f562_seq.png',np.hstack(rows).astype(np.uint8))
print('ok')
EOF

# ----
which ffmpeg; python3 -c "import cv2; print(cv2.getBuildInformation().split('Video I/O')[1][:600])"

# ----
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "264|hevc|mpeg4|mjpeg" | head

# ----
cd "$HOME/mnt/outputs" && cat > met/tl.py <<'EOF'
import numpy as np, cv2, json, os, subprocess, sys
exec(open(os.path.expanduser('~/mnt/outputs/met/stack.py')).read().split("cmd=sys.argv[1]")[0])
reg=json.load(open(ST+'/reg.json'))
V40=np.load(ST+'/vig40.npy'); V60=np.load(ST+'/vig60.npy')
def prep(i):
    a=raw(i)-240.0; a=a/(V40 if gain(i)==40 else V60)
    c=cv2.cvtColor(np.clip(a+1000,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2BGR).astype(np.float32)-1000
    return c*(2.0 if gain(i)==40 else 1.0)
ref=prep(480); sky=np.median(ref.reshape(-1,3)[::5],0)
cw=np.array([1.291,0.69,1.019],np.float32)
x=(ref-sky*0.9)*cw; top=float(np.percentile(x.mean(2),99.97)); s_=float(1.4826*np.median(np.abs(x.mean(2)-np.median(x.mean(2)))))
A=6*s_
def stretch(c):
    x=(c-sky*0.9)*cw; L=np.maximum(x.mean(2),0); y=np.arcsinh(L/A)/np.arcsinh(top/A)
    rat=x/np.maximum(x.mean(2,keepdims=True),1e-3); rat=1+0.6*(np.clip(rat,0.5,1.8)-1)
    rgb=np.clip(y[...,None]*rat,0,1)
    return (rgb*255).astype(np.uint8)
OUTW,OUTH=1440,810
D=os.path.expanduser('~/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan')
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OUTW}x{OUTH}','-r','30','-i','-','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',D+'/alhena_timelapse.mp4'],stdin=subprocess.PIPE)
jump={i for i in range(116,669) if i%6==2}|{440}
last=None; times=[]
for i in range(len(F)):
    if 'Alhena_10s' not in F[i][2]: continue
    if i in jump and last is not None: fr=last
    else:
        fr=cv2.resize(stretch(prep(i)),(OUTW,OUTH),interpolation=cv2.INTER_AREA); last=fr
    p.stdin.write(fr.tobytes()); times.append(F[i][1])
p.stdin.close(); p.wait()
json.dump(times,open(D+'/alhena_timelapse_times.json','w')); print('frames',len(times))
EOF
time python3 met/tl.py && ls -la "$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan/"

# ----
cd "$HOME/mnt/outputs" && python3 - <<'EOF'
import re,os
p=os.path.expanduser('~/mnt/outputs/met/tl.py'); s=open(p).read()
s=s.replace("ref=prep(480); sky=np.median(ref.reshape(-1,3)[::5],0)","""SKY={}
for g in (40,60):
    sel=[i for i in C if gain(i)==g][::max(1,len([i for i in C if gain(i)==g])//40)][:40]
    m=np.median(np.stack([cv2.resize(prep(i),(480,270),interpolation=cv2.INTER_AREA) for i in sel]),0)
    m=cv2.GaussianBlur(cv2.medianBlur(m.astype(np.float32),5),(0,0),6)
    SKY[g]=cv2.resize(m,(W,H),interpolation=cv2.INTER_CUBIC)
ref=prep(480)-SKY[60]; sky=np.zeros(3,np.float32)""")
s=s.replace("def stretch(c):\n    x=(c-sky*0.9)*cw","def stretch(c):\n    x=(c+25)*cw")
s=s.replace("x=(ref-sky*0.9)*cw;","x=(ref+25)*cw;")
s=s.replace("else:\n        fr=cv2.resize(stretch(prep(i)),","else:\n        fr=cv2.resize(stretch(prep(i)-SKY[gain(i)]),")
s=s.replace("y=np.arcsinh(L/A)/np.arcsinh(top/A)","y=np.arcsinh(L/A)/np.arcsinh(top/A); y=np.where(y>0.7,0.7+(1-np.exp(-(y-0.7)/0.25))*0.25,y)")
open(p,'w').write(s)
EOF
time python3 met/tl.py

# ----
D="$HOME/mnt/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan"; ls -la "$D/alhena_timelapse.mp4"; cd "$D" && ffmpeg -v error -y -i alhena_timelapse.mp4 -c copy -f segment -segment_time 8 -reset_timestamps 1 tl_part%d.mp4 && ls -la tl_part*

# ----
cd "$HOME/mnt/outputs/met" && ls *.py && tar czf - *.py *.json 2>/dev/null | base64 -w0 | wc -c

