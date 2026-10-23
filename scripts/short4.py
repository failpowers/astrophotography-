import numpy as np, cv2, subprocess, sys, json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W,H,FPS=1080,1920,30
NAVY=(11,30,63); CYAN=(34,211,238); WHITE=(245,247,250); AMBER=(255,196,92)
def F(size,weight='Bold'):
    f=ImageFont.truetype('Montserrat-VF.ttf',size); f.set_variation_by_name(weight); return f
BEBAS=lambda s:ImageFont.truetype('BebasNeue-Regular.ttf',s)
WIN_Y0,WIN_H=470,960
def ease(t): t=np.clip(t,0,1); return t*t*(3-2*t)
# ---------- sources
U='/mnt/user-data/uploads/DWARF_RAW_WIDE_Alhena_EXP_10_GAIN_60_2026-10-22-04-04-12-073/OHN_meteor_scan/'
times=json.load(open(U+'alhena_timelapse_times.json'))
TLX0=264; TLW=912
cap=cv2.VideoCapture('met/timelapse.mp4'); TL=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    TL.append(fr[:,TLX0:TLX0+TLW].copy())
TL=np.stack(TL); NT=len(TL); print('timelapse frames',NT)
annot=cv2.imread('alh_annot.png'); final=cv2.imread('alh_v6.png')
SAT=int(sys.argv[2]) if len(sys.argv)>2 else 502
SATC=json.load(open('sat_box.json')) if False else None
def tl_win(idx):
    fr=TL[int(np.clip(idx,0,NT-1))]
    return cv2.resize(fr,(W,WIN_H),interpolation=cv2.INTER_LINEAR)   # 912x810 -> 1080x960 (same aspect)
def window(img,c0,h0,c1,h1,t):
    e=ease(t); cx=c0[0]+(c1[0]-c0[0])*e; cy=c0[1]+(c1[1]-c0[1])*e; hh=h0+(h1-h0)*e; s=WIN_H/hh
    M=np.float32([[s,0,W/2-cx*s],[0,s,WIN_H/2-cy*s]])
    return cv2.warpAffine(img,M,(W,WIN_H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(8,6,6))
yy=np.linspace(0,1,H)[:,None,None]
BGc=np.broadcast_to(np.array([24,12,6],np.float32)*(1-yy)+np.array([46,24,10],np.float32)*yy,(H,W,3)).copy()
fm=np.ones((WIN_H,W),np.float32); r=60; ramp=np.linspace(0,1,r); fm[:r]*=ramp[:,None]; fm[-r:]*=ramp[::-1][:,None]; FM=fm[...,None]
# ---------- text layers
def text_c(d,y,txt,font,fill,spacing=0):
    if spacing:
        ws=[d.textlength(ch,font=font) for ch in txt]; tw=sum(ws)+spacing*(len(txt)-1); x=(W-tw)/2
        for ch,wd in zip(txt,ws): d.text((x,y),ch,font=font,fill=fill); x+=wd+spacing
    else:
        tw=d.textlength(txt,font=font); d.text(((W-tw)/2,y),txt,font=font,fill=fill)
def shadowed(layer):
    a=layer.split()[3].filter(ImageFilter.GaussianBlur(6)).point(lambda v:int(v*0.7))
    sh=Image.new('RGBA',layer.size,(0,0,0,0)); sh.putalpha(a)
    out=Image.new('RGBA',layer.size,(0,0,0,0)); out.alpha_composite(sh); out.alpha_composite(layer); return out
BADGE=Image.open('badge.png').convert('RGBA')
def header():
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    bs=150; bdg=BADGE.resize((bs,bs),Image.LANCZOS)
    f=F(62,'Black'); a='OH HAIL NAH'; b='!!'; f2=F(25,'SemiBold'); sub='ASTROPHOTOGRAPHY'; sp=6
    wa=d.textlength(a,font=f); wb=d.textlength(b,font=f); wsub=sum(d.textlength(ch,font=f2) for ch in sub)+sp*(len(sub)-1)
    tw=max(wa+wb,wsub); total=bs+24+tw; x=(W-total)/2; y=135
    L.alpha_composite(bdg,(int(x),y)); tx=x+bs+24
    d.text((tx,y+30),a,font=f,fill=WHITE); d.text((tx+wa,y+30),b,font=f,fill=CYAN)
    xx=tx
    for ch in sub: d.text((xx,y+104),ch,font=f2,fill=CYAN); xx+=d.textlength(ch,font=f2)+sp
    return shadowed(L)
def panel(title,l1,l2,tsize=130,c2=CYAN):
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    text_c(d,318,title,BEBAS(tsize),WHITE,spacing=4)
    text_c(d,1462,l1,F(50,'Bold'),WHITE); text_c(d,1530,l2,F(40,'SemiBold'),c2)
    return shadowed(L)
def clock(txt):
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L); f=F(30,'Bold')
    x,y=40,WIN_Y0+30; tw=d.textlength(txt,font=f)
    d.rounded_rectangle([x,y,x+tw+36,y+50],radius=12,fill=(8,20,44,200),outline=CYAN,width=2); d.text((x+18,y+8),txt,font=f,fill=CYAN)
    return L
def ring(cx,cy,rx,ry,txt):
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    for a in range(0,360,16): d.arc([cx-rx,WIN_Y0+cy-ry,cx+rx,WIN_Y0+cy+ry],a,a+9,fill=AMBER+(255,),width=5)
    f=F(34,'Black'); tw=d.textlength(txt,font=f); tx=int(np.clip(cx-tw/2,30,W-tw-30)); ty=WIN_Y0+cy+ry+16
    d.rounded_rectangle([tx-16,ty-6,tx+tw+16,ty+46],radius=12,fill=(8,20,44,215),outline=AMBER,width=2)
    d.text((tx,ty),txt,font=f,fill=AMBER)
    return shadowed(L)
HDR=header()
P=dict(hook=panel('ORIONID HUNT FROM HOUSTON','Camera aimed at the radiant','Did we catch a meteor?',tsize=104),
       tl=panel('2 HOURS · 10-SECOND FRAMES','Gemini & Orion rising high','03:41 to 05:35 AM, Oct 22',tsize=96),
       sat=panel('WAIT… IS THAT ONE?','Nope: a satellite','It keeps moving across several frames',tsize=120,c2=AMBER),
       clouds=panel('THEN THE CLOUDS ROLLED IN','Meteors caught: 0','Satellites & planes: 4',tsize=96,c2=AMBER),
       stack=panel('389 FRAMES STACKED','Gemini, Orion & the radiant','The Orionids appear to fly out from the circle',tsize=118))
CLK={}
def clock_for(idx):
    t=times[int(np.clip(idx,0,NT-1))]; hh,mm=int(t[9:11]),int(t[11:13]); s=f'{hh:d}:{mm:02d} AM'
    if s not in CLK: CLK[s]=clock(s)
    return CLK[s]
RING=None
def endcard():
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    bs=300; L.alpha_composite(BADGE.resize((bs,bs),Image.LANCZOS),((W-bs)//2,150))
    text_c(d,500,'ORIONID SCOREBOARD',BEBAS(118),WHITE,spacing=4)
    rows=[('Frames shot','669'),('Satellites & planes','4'),('Meteors','0'),('Clouds','plenty')]
    y=660
    for k,v in rows:
        d.rounded_rectangle([150,y,W-150,y+92],radius=18,fill=(8,20,44,200),outline=CYAN,width=2)
        d.text((190,y+22),k,font=F(40,'SemiBold'),fill=WHITE); tw=d.textlength(v,font=F(46,'Black'))
        d.text((W-190-tw,y+18),v,font=F(46,'Black'),fill=AMBER if v in('0','plenty') else CYAN); y+=112
    text_c(d,1150,'Did you see any Orionids this week?',F(44,'Bold'),WHITE)
    text_c(d,1225,'Shot on a DWARF 3 smart telescope',F(34,'Medium'),CYAN)
    text_c(d,1273,'from the Houston suburbs',F(34,'Medium'),CYAN)
    return shadowed(L)
END=endcard()
segs=[('hook',0.0,3.5),('tl',3.5,9.0),('sat',9.0,12.5),('clouds',12.5,15.5),('stack',15.5,25.0),('end',25.0,30.5)]
DUR=30.5; XF=0.5
def tl_index(t):
    if t<9.0: return t/9.0*SAT
    if t<12.5: return SAT
    return SAT+(t-12.5)/3.0*(NT-1-SAT)
def base(name,t):
    c=BGc.copy()
    if name=='end': c[:]=BGc*0.9; return c
    if name in('hook','tl','sat','clouds'): win=tl_win(tl_index(t))
    else:
        u=(t-15.5)/9.5; win=window(annot,(700,520),1000,(1150,600),880,u)
    reg=c[WIN_Y0:WIN_Y0+WIN_H]; c[WIN_Y0:WIN_Y0+WIN_H]=reg*(1-FM)+win.astype(np.float32)*FM
    return c
def texts(name,t):
    if name=='end': return [END]
    o=[P[name]]
    if name in('hook','tl','sat','clouds'): o.append(clock_for(tl_index(t)))
    if name=='sat' and RING is not None and t>9.4: o.append(RING)
    return o
def alpha_over(b,ov,a=1.0):
    o=np.asarray(ov,np.float32); al=o[...,3:4]/255*a; return b*(1-al)+o[...,2::-1]*al
def frame(t):
    i=max(k for k,s_ in enumerate(segs) if s_[1]<=t or k==0); name,s0,s1=segs[i]
    img=base(name,t); a=0.0; nxt=None
    # time-lapse segments flow continuously: no image crossfade between hook/tl/sat/clouds
    cont={'hook','tl','sat','clouds'}
    if i+1<len(segs) and t>s1-XF:
        nxt=segs[i+1][0]; a=ease((t-(s1-XF))/XF)
        if not (name in cont and nxt in cont): img=img*(1-a)+base(nxt,s1)*a
    hdr_a=1.0 if name!='end' else 0.0
    if nxt=='end': hdr_a=1-a
    if hdr_a>0: img=alpha_over(img,HDR,hdr_a)
    fin=1.0 if s0==0 else min(1,(t-s0)/0.25)
    out_a=fin*(1-min(1,2*a)) if nxt else fin
    for ov in texts(name,t): img=alpha_over(img,ov,out_a)
    if nxt and a>0.5:
        for ov in texts(nxt,s1): img=alpha_over(img,ov,(2*a-1))
    if t<0.25: img*=t/0.25
    if t>DUR-0.6: img*=max(0,(DUR-t)/0.6)
    return np.clip(img,0,255).astype(np.uint8)
if __name__=='__main__':
    box=json.load(open('sat_box.json')); RING=ring(box['cx'],box['cy'],box['rx'],box['ry'],'SATELLITE')
    if sys.argv[1]=='stills':
        for t in (1.0,5.0,10.5,14.0,17.0,24.0,28.0): cv2.imwrite(f'met/still4_{t}.jpg',frame(t))
        sys.exit()
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-profile:v','high','-r',str(FPS),'video4_only.mp4'],stdin=subprocess.PIPE)
    n=int(DUR*FPS)
    for k in range(n): p.stdin.write(frame(k/FPS).tobytes())
    p.stdin.close(); p.wait(); print('video frames',n)
