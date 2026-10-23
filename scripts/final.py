import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from wm import embed
img=cv2.imread('/mnt/user-data/outputs/moon_enhanced.jpg').astype(np.float32)
# --- light pollution: model sky background with smooth polynomial, subtract ---
lum=img.mean(2)
moon=cv2.dilate((lum>18).astype(np.uint8),np.ones((61,61),np.uint8))>0
ys,xs=np.mgrid[0:img.shape[0],0:img.shape[1]]
xn,yn=xs/img.shape[1],ys/img.shape[0]
A=np.stack([np.ones_like(xn),xn,yn,xn*yn,xn**2,yn**2],-1)
sky=~moon; sel=np.random.default_rng(0).random(sky.sum())<0.1
for c in range(3):
    coef,*_=np.linalg.lstsq(A[sky][sel],img[:,:,c][sky][sel],rcond=None)
    bg=A@coef
    img[:,:,c]=img[:,:,c]-np.maximum(bg-1.0,0)  # leave a hair of floor
# soften residual glow halo just outside the limb
halo=moon & (lum<18)
img[halo]*=0.6
img=np.clip(img,0,255).astype(np.uint8)
# --- signature (drawn at 3x, downsampled for smooth pen strokes) ---
pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).convert('RGBA')
S=3; layer=Image.new('RGBA',(520*S,190*S),(0,0,0,0)); d=ImageDraw.Draw(layer)
f1=ImageFont.truetype('GreatVibes-Regular.ttf',58*S); f2=ImageFont.truetype('Allura-Regular.ttf',30*S)
ink=(238,233,222,205)
d.text((14*S,10*S),"Oh Hail Nah!!",font=f1,fill=ink)
w=d.textlength("Oh Hail Nah!!",font=f1)/S
d.text((58*S,98*S),"Astrophotography",font=f2,fill=ink)
f3=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13*S)
stamp="10.18.26  \u00b7  20:57"
d.text((60*S,146*S),' '.join(stamp).replace('   ','  '),font=f3,fill=(238,233,222,165))
# tapered pen flourish under the name
ts=np.linspace(0,1,160)
for i in range(len(ts)-1):
    t0,t1=ts[i],ts[i+1]
    x0,x1=(20+t0*(w+30))*S,(20+t1*(w+30))*S
    y0=(92+6*np.sin(t0*np.pi*1.5)-10*t0)*S; y1=(92+6*np.sin(t1*np.pi*1.5)-10*t1)*S
    wd=max(1,int(S*(0.6+2.0*np.sin(np.pi*t0)**0.7)))
    d.line([(x0,y0),(x1,y1)],fill=ink,width=wd)
layer=layer.resize((520,190),Image.LANCZOS).rotate(3,resample=Image.BICUBIC,expand=True)
pil.alpha_composite(layer,(30,pil.height-layer.height-14))
out=cv2.cvtColor(np.array(pil.convert('RGB')),cv2.COLOR_RGB2BGR)
out=embed(out)
cv2.imwrite('/mnt/user-data/outputs/moon_OHN_signed.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])
