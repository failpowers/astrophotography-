import numpy as np, cv2, json
U='/mnt/user-data/uploads/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan/'
st=np.nan_to_num(np.load(U+'alhena_stack2.npy')); L=st.sum(2); H,W=L.shape
CAT=[('Alhena',99.428,16.399,1.9),('Betelgeuse',88.793,7.407,0.5),('Bellatrix',81.283,6.350,1.6),('Meissa',83.784,9.934,3.4),
('Pollux',116.329,28.026,1.1),('Castor',113.650,31.888,1.6),('Procyon',114.825,5.225,0.4),('Gomeisa',111.788,8.289,2.9),
('Elnath',81.573,28.608,1.65),('Mebsuta',100.983,25.131,3.0),('Mekbuda',106.027,20.570,3.8),('Tejat',95.740,22.514,2.9),
('Propus',93.719,22.507,3.3),('Wasat',110.031,21.982,3.5),('Alzirr',101.322,12.896,3.35),('Tianguan',84.411,21.143,3.0),
('nu Gem',97.241,20.212,4.1),('mu Ori',90.596,9.647,4.1),('lambda Gem',109.523,16.540,3.6),('kappa Gem',116.112,24.398,3.6),
('iota Gem',111.432,27.798,3.8),('theta Gem',103.197,33.961,3.6),('xi Ori',92.985,14.209,4.5),('nu Ori',91.893,14.768,4.4),
('chi1 Ori',88.596,20.276,4.4),('13 Mon',98.226,7.333,4.5),('S Mon',100.244,9.896,4.7),('eps Mon',95.942,4.593,4.4)]
ra0,de0=np.radians(99.428),np.radians(16.399)
def tan(ra,de):
    ra,de=np.radians(ra),np.radians(de); c=np.sin(de)*np.sin(de0)+np.cos(de)*np.cos(de0)*np.cos(ra-ra0)
    xi=np.cos(de)*np.sin(ra-ra0)/c; eta=(np.sin(de)*np.cos(de0)-np.cos(de)*np.sin(de0)*np.cos(ra-ra0))/c; return xi,eta
sc=np.radians(89.3/3600)
cat=np.array([tan(r,d) for _,r,d,_ in CAT])/sc   # tangent-plane pixels (xi east, eta north)
# detect bright stars in image
bg=cv2.medianBlur(cv2.resize(L,(240,135),interpolation=cv2.INTER_AREA).astype(np.float32),5); bg=cv2.resize(bg,(W,H))
hp=cv2.GaussianBlur(L-bg,(0,0),1.5); mx=cv2.dilate(hp,np.ones((15,15),np.uint8)); pk=np.argwhere((hp==mx)&(hp>0))
vals=hp[pk[:,0],pk[:,1]]; o=np.argsort(-vals)[:45]; img=pk[o][:,::-1].astype(float)
best=(0,None)
for par in (1,-1):
    for th in np.arange(0,360,0.5):
        t=np.radians(th); R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        c=cat*np.array([par,1.0]); c=c@R.T; c[:,1]*=-1   # image y down
        dif=(img[:,None,:]-c[None,:,:]).reshape(-1,2)
        Hh,xe,ye=np.histogram2d(dif[:,0],dif[:,1],bins=[np.arange(-1500,3500,6),np.arange(-1500,2600,6)])
        kk=np.unravel_index(Hh.argmax(),Hh.shape)
        if Hh[kk]>best[0]: best=(Hh[kk],(par,th,xe[kk[0]]+3,ye[kk[1]]+3))
print('best votes',best)
par,th,tx,ty=best[1]; t=np.radians(th); R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
def proj(c):
    c=c*np.array([par,1.0]); c=c@R.T; c[:,1]*=-1; return c+np.array([tx,ty])
p=proj(cat)
# refine: match & fit quadratic poly from tangent-plane to image
def feat(c):
    x,y=c[:,0]/1000,c[:,1]/1000; return np.stack([np.ones_like(x),x,y,x*x,x*y,y*y,x**3,x*y*y,x*x*y,y**3],1)
# detect many stars for matching
pk2=np.argwhere((hp==cv2.dilate(hp,np.ones((7,7),np.uint8)))&(hp>np.percentile(hp,99.7))); allimg=pk2[:,::-1].astype(float)
for it in range(3):
    d=np.hypot(p[:,None,0]-allimg[None,:,0],p[:,None,1]-allimg[None,:,1]); j=d.argmin(1); ok=d[np.arange(len(p)),j]<(25 if it==0 else 8)
    inside=(p[:,0]>0)&(p[:,0]<W)&(p[:,1]>0)&(p[:,1]<H)
    m=ok&inside; print('iter',it,'matched',m.sum(),[CAT[i][0] for i in np.where(m)[0]])
    A=feat(cat[m]); deg=6 if m.sum()<12 else 10
    cx=np.linalg.lstsq(A[:,:deg],allimg[j[m],0],rcond=None)[0]; cy=np.linalg.lstsq(A[:,:deg],allimg[j[m],1],rcond=None)[0]
    p=np.stack([feat(cat)[:,:deg]@cx,feat(cat)[:,:deg]@cy],1)
    res=np.hypot(p[m,0]-allimg[j[m],0],p[m,1]-allimg[j[m],1]); print('  rms',res.mean().round(2))
sol=dict(par=int(par),th=float(th),tx=float(tx),ty=float(ty),deg=int(deg),cx=cx.tolist(),cy=cy.tolist())
json.dump(sol,open('alh_wcs.json','w'))
for (n,r,d,mg),q in zip(CAT,p): print(f'{n:12s} {q[0]:8.1f} {q[1]:8.1f}')
