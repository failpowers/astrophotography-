"""Label the Alhena wide field from the plate solution (26 catalogue stars, 0.5 px rms)."""
import numpy as np, cv2, json, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
src,out=sys.argv[1],sys.argv[2]
sol=json.load(open('alh_wcs.json')); y0,x0,h0,w0=np.load('alh_crop.npy')
ra0,de0=np.radians(99.428),np.radians(16.399); sc=np.radians(89.3/3600)
def tan(ra,de):
    ra,de=np.radians(ra),np.radians(de); c=np.sin(de)*np.sin(de0)+np.cos(de)*np.cos(de0)*np.cos(ra-ra0)
    return np.array([np.cos(de)*np.sin(ra-ra0)/c,(np.sin(de)*np.cos(de0)-np.cos(de)*np.sin(de0)*np.cos(ra-ra0))/c])/sc
def feat(c):
    x,y=c[0]/1000,c[1]/1000; return np.array([1,x,y,x*x,x*y,y*y,x**3,x*y*y,x*x*y,y**3])[:sol['deg']]
def pix(ra,de):
    f=feat(tan(ra,de)); return float(f@np.array(sol['cx'])-x0), float(f@np.array(sol['cy'])-y0)
img=Image.open(src).convert('RGBA'); W,H=img.size
L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
def MF(sz,wt='SemiBold'):
    f=ImageFont.truetype('Montserrat-VF.ttf',sz); f.set_variation_by_name(wt); return f
BEB=lambda s:ImageFont.truetype('BebasNeue-Regular.ttf',s)
WHITE=(236,240,246,225); CYAN=(34,211,238,235); DIM=(200,214,232,150)
inside=lambda x,y,m=30:(m<x<W-m and m<y<H-m)
SIG=(0,int(H*0.74),int(W*0.36),H)   # keep labels out of the signature corner
def blocked(x,y): return SIG[0]<=x<=SIG[2] and SIG[1]<=y<=SIG[3]
stars=[('Alhena',99.428,16.399,(16,-30)),('Betelgeuse',88.793,7.407,(16,-30)),('Bellatrix',81.283,6.350,(-112,-30)),
 ('Meissa',83.784,9.934,(14,-30)),('Procyon',114.825,5.225,(16,-34)),('Gomeisa',111.788,8.289,(14,-28)),('Elnath',81.573,28.608,(-84,10)),
 ('Mebsuta',100.983,25.131,(14,-28)),('Tejat',95.740,22.514,(-60,-30)),('Propus',93.719,22.507,(12,-28)),('Wasat',110.031,21.982,(14,-28)),
 ('Alzirr',101.322,12.896,(14,-28))]
f=MF(19,'SemiBold')
for n,ra,de,(dx,dy) in stars:
    x,y=pix(ra,de)
    if not inside(x,y,8) or blocked(x+dx,y+dy): continue
    d.ellipse([x-13,y-13,x+13,y+13],outline=(236,240,246,110),width=2)
    d.text((x+dx,y+dy),n,font=f,fill=WHITE)
objs=[('M35',92.25,24.33,(16,-14)),('NGC 2264 · Christmas Tree',100.24,9.90,(18,-12))]
for n,ra,de,(dx,dy) in objs:
    x,y=pix(ra,de)
    if not inside(x,y): continue
    r=16;
    for a in range(0,360,30): d.arc([x-r,y-r,x+r,y+r],a,a+18,fill=(255,214,140,200),width=2)
    d.text((x+dx+r,y+dy),n,font=MF(17,'Medium'),fill=(255,226,170,220))
cons=[('GEMINI',105.0,24.5),('ORION',85.0,4.6),('CANIS MINOR',113.4,7.6),('TAURUS',84.0,25.8),('MONOCEROS',104.6,4.0)]
for n,ra,de in cons:
    x,y=pix(ra,de)
    if not inside(x,y,60): continue
    fb=BEB(40); tw=sum(d.textlength(ch,font=fb) for ch in n)+6*(len(n)-1); xx=x-tw/2
    for ch in n: d.text((xx,y-20),ch,font=fb,fill=DIM); xx+=d.textlength(ch,font=fb)+6
# Orionid radiant (IMO/EarthSky: in Orion's club, north of Betelgeuse) ~ RA 95 deg, Dec +16 deg
rx,ry=pix(95.0,16.0)
for r,al in ((70,235),):
    for a in range(0,360,20): d.arc([rx-r,ry-r,rx+r,ry+r],a,a+11,fill=CYAN[:3]+(al,),width=3)
d.line([rx-10,ry,rx+10,ry],fill=CYAN,width=2); d.line([rx,ry-10,rx,ry+10],fill=CYAN,width=2)
for ang in range(15,360,45):          # schematic "meteors fly outward" ticks
    t=np.radians(ang); a0,a1=85,120
    d.line([rx+a0*np.cos(t),ry+a0*np.sin(t),rx+a1*np.cos(t),ry+a1*np.sin(t)],fill=CYAN[:3]+(150,),width=2)
fr=BEB(44); lab='ORIONID RADIANT'
tw=sum(d.textlength(ch,font=fr) for ch in lab)+4*(len(lab)-1); xx=rx-tw/2
for ch in lab: d.text((xx,ry+128),ch,font=fr,fill=CYAN); xx+=d.textlength(ch,font=fr)+4
sub='approx. · meteors appear to fly out from here'; fs=MF(17,'Medium'); d.text((rx-d.textlength(sub,font=fs)/2,ry+176),sub,font=fs,fill=(200,240,250,215))
sh=L.split()[3].filter(ImageFilter.GaussianBlur(3)).point(lambda v:int(v*0.8))
S=Image.new('RGBA',(W,H),(0,0,0,0)); S.putalpha(sh)
img.alpha_composite(S); img.alpha_composite(L)
img.convert('RGB').save(out); print('radiant px',round(rx),round(ry))
