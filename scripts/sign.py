import cv2, numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
from wm import embed
src,out=sys.argv[1],sys.argv[2]
STAMP=sys.argv[3] if len(sys.argv)>3 else ''
TARGET=sys.argv[5] if len(sys.argv)>5 else ''
KS=float(sys.argv[4]) if len(sys.argv)>4 else 0.82
img=cv2.imread(src); H,W=img.shape[:2]; k=W/1342*KS          # scale signature to frame
pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).convert('RGBA')
S=3; LW,LH=int(520*k),int(205*k)
layer=Image.new('RGBA',(LW*S,LH*S),(0,0,0,0)); d=ImageDraw.Draw(layer)
f1=ImageFont.truetype('GreatVibes-Regular.ttf',int(58*k*S)); f2=ImageFont.truetype('Allura-Regular.ttf',int(30*k*S))
ink=(238,233,222,200)
d.text((14*k*S,10*k*S),"Oh Hail Nah!!",font=f1,fill=ink)
w=d.textlength("Oh Hail Nah!!",font=f1)/S
d.text((58*k*S,98*k*S),"Astrophotography",font=f2,fill=ink)
sp=lambda t:' '.join(t).replace('   ','  ')
ty=146
if TARGET:
    f4=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',int(15*k*S))
    d.text((60*k*S,ty*k*S),sp(TARGET.upper()),font=f4,fill=(238,233,222,205))
    ty+=24
if STAMP:
    f3=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',int(13*k*S))
    d.text((60*k*S,ty*k*S),sp(STAMP),font=f3,fill=(238,233,222,165))
ts=np.linspace(0,1,200)
for i in range(len(ts)-1):
    t0,t1=ts[i],ts[i+1]
    p=lambda t:((20*k+t*(w+30*k))*S,(92*k+6*k*np.sin(t*np.pi*1.5)-10*k*t)*S)
    wd=max(1,int(S*k*(0.6+2.0*np.sin(np.pi*t0)**0.7)))
    d.line([p(t0),p(t1)],fill=ink,width=wd)
layer=layer.resize((LW,LH),Image.LANCZOS).rotate(3,resample=Image.BICUBIC,expand=True)
from PIL import ImageFilter
import os as _o; pos=(int(30*k),H-layer.height-int(14*k)-int(_o.environ.get("SIGDY","0")))
al=layer.split()[3].filter(ImageFilter.GaussianBlur(max(1.5,2.2*k)))
al=al.point(lambda v:int(min(255,v*1.6)*0.55))
shadow=Image.new('RGBA',layer.size,(0,0,0,0)); shadow.putalpha(al)
pil.alpha_composite(shadow,pos)
pil.alpha_composite(layer,pos)
o=cv2.cvtColor(np.array(pil.convert('RGB')),cv2.COLOR_RGB2BGR)
import os; o=embed(o,float(os.environ.get("WMS","9")))
cv2.imwrite(out,o,[cv2.IMWRITE_JPEG_QUALITY,95])
