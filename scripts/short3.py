import numpy as np, cv2, subprocess, sys, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W,H,FPS=1080,1920,30
NAVY=(11,30,63); CYAN=(34,211,238); WHITE=(245,247,250)
OUT=sys.argv[1] if len(sys.argv)>1 else 'short_video.mp4'
def F(size,weight='Bold'):
    f=ImageFont.truetype('Montserrat-VF.ttf',size); f.set_variation_by_name(weight); return f
BEBAS=lambda s:ImageFont.truetype('BebasNeue-Regular.ttf',s)
# ---------------- source images (unsigned masters; branding is added by the video) ----------------
sat=cv2.imread('saturn_final.png'); sun=cv2.imread('sun_master2x.png'); sunb=cv2.imread('sun_before2x.png'); sunf=cv2.imread('sun_final.png'); moon=cv2.imread('moon2.png')
dc=cv2.imread('dc.png'); n2264=cv2.imread('n2264b.png'); m41=cv2.imread('m41.png')
MX,MY=1117,656
WIN_Y0,WIN_H=470,960            # image window
def ease(t): return t*t*(3-2*t)
def window(img,c0,h0,c1,h1,t):
    """render 1080xWIN_H window from img, center/height interpolated (subpixel, no jitter)"""
    e=ease(np.clip(t,0,1)); cx=c0[0]+(c1[0]-c0[0])*e; cy=c0[1]+(c1[1]-c0[1])*e; hh=h0+(h1-h0)*e
    s=WIN_H/hh
    M=np.float32([[s,0,W/2-cx*s],[0,s,WIN_H/2-cy*s]])
    return cv2.warpAffine(img,M,(W,WIN_H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
def bg_from(img):
    h,w=img.shape[:2]; s=max(W/w,H/h)*1.15
    b=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
    y=(b.shape[0]-H)//2; x=(b.shape[1]-W)//2; b=b[y:y+H,x:x+W]
    b=cv2.GaussianBlur(b,(0,0),45).astype(np.float32)
    navy=np.array(NAVY[::-1],np.float32)
    return np.clip(b*0.45+navy*0.55,0,255)
BG={k:bg_from(v) for k,v in dict(dc=dc,n2264=n2264,m41=m41).items()}
yy=np.linspace(0,1,H)[:,None,None]
BG['sat']=np.broadcast_to(np.array([22,12,6],np.float32)*(1-yy)+np.array([40,22,10],np.float32)*yy,(H,W,3)).copy()
BG['moon']=BG['sat']
BG['sun']=np.broadcast_to(np.array([6,12,30],np.float32)*(1-yy)+np.array([10,24,56],np.float32)*yy,(H,W,3)).copy()
# window fade mask (soft top/bottom edges into background)
fm=np.ones((WIN_H,W),np.float32); r=60
ramp=np.linspace(0,1,r); fm[:r]*=ramp[:,None]; fm[-r:]*=ramp[::-1][:,None]; FM=fm[...,None]
# ---------------- text overlays ----------------
def text_c(d,y,txt,font,fill,spacing=0):
    if spacing:
        widths=[d.textlength(ch,font=font) for ch in txt]; tw=sum(widths)+spacing*(len(txt)-1); x=(W-tw)/2
        for ch,wd in zip(txt,widths): d.text((x,y),ch,font=font,fill=fill); x+=wd+spacing
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
    wa=d.textlength(a,font=f); wb=d.textlength(b,font=f)
    wsub=sum(d.textlength(ch,font=f2) for ch in sub)+sp*(len(sub)-1)
    tw=max(wa+wb,wsub); total=bs+24+tw; x=(W-total)/2; y=135
    L.alpha_composite(bdg,(int(x),y))
    tx=x+bs+24
    d.text((tx,y+30),a,font=f,fill=WHITE); d.text((tx+wa,y+30),b,font=f,fill=CYAN)
    xx=tx
    for ch in sub: d.text((xx,y+104),ch,font=f2,fill=CYAN); xx+=d.textlength(ch,font=f2)+sp
    return shadowed(L)
def panel(title,l1,l2,tsize=150):
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    text_c(d,318,title,BEBAS(tsize),WHITE,spacing=4)
    text_c(d,1462,l1,F(50,'Bold'),WHITE)
    text_c(d,1530,l2,F(40,'SemiBold'),CYAN)
    return shadowed(L)
def tag(txt,x,y,fill):
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L); f=F(30,'Bold')
    tw=d.textlength(txt,font=f); d.rounded_rectangle([x,y,x+tw+36,y+50],radius=12,fill=(8,20,44,200),outline=fill,width=2)
    d.text((x+18,y+8),txt,font=f,fill=fill); return L
HDR=header()
P=dict(
 hook=panel('THE SUN FROM HOUSTON','1 frame vs. 774 stacked','Shot through a certified solar filter',tsize=112),
 sun=panel('THE SUN','Sunspot groups + bright faculae','Best 774 of 6,450 video frames'),
 moon=panel('THE MOON','Every crater is real data','416 frames stacked'),
 sat=panel('SATURN','Rings through a 35mm lens','Best 1,152 of 9,604 frames'),
 dc=panel('DOUBLE CLUSTER','Two star clusters in Perseus','711 frames · 3 hours',tsize=130),
 n2264=panel('CHRISTMAS TREE','Cluster NGC 2264 in Monoceros','355 frames · 89 minutes',tsize=130))
TB=tag('1 FRAME',40,WIN_Y0+30,(200,200,200)); TA=tag('STACKED',W-40-190,WIN_Y0+30,CYAN)
def endcard():
    L=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    bs=300; L.alpha_composite(BADGE.resize((bs,bs),Image.LANCZOS),((W-bs)//2,130))
    text_c(d,455,'WHICH ONE WINS?',BEBAS(140),WHITE,spacing=4)
    crops=[(sunf,(720,532),880,'1','SUN'),(moon,(727,568),620,'2','MOON'),(sat,(675,580),560,'3','SATURN'),(dc,(1048,610),760,'4','DOUBLE CLUSTER'),(n2264,(1235,650),700,'5','CHRISTMAS TREE')]
    S=236; G=28
    rows=[crops[:3],crops[3:]]; ys=[650,986]
    for row,y in zip(rows,ys):
        x0=(W-len(row)*S-(len(row)-1)*G)//2
        for i,(img,c,sz,num,lab) in enumerate(row):
            x=x0+i*(S+G)
            crop=cv2.getRectSubPix(img,(sz,sz),c); crop=cv2.resize(crop,(S,S),interpolation=cv2.INTER_AREA)
            tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).convert('RGBA')
            m=Image.new('L',(S,S),0); ImageDraw.Draw(m).rounded_rectangle([0,0,S-1,S-1],radius=22,fill=255)
            L.paste(tile,(x,y),m); d.rounded_rectangle([x,y,x+S-1,y+S-1],radius=22,outline=CYAN,width=3)
            # number badge
            r=30; bx,by=x+16,y+16
            d.ellipse([bx,by,bx+2*r,by+2*r],fill=(8,20,44,230),outline=CYAN,width=3)
            f=BEBAS(48); tw=d.textlength(num,font=f); d.text((bx+r-tw/2,by+7),num,font=f,fill=CYAN)
            f2=F(23,'Bold'); tw=d.textlength(lab,font=f2); d.text((x+S/2-tw/2,y+S+12),lab,font=f2,fill=WHITE)
    text_c(d,1300,'Comment 1, 2, 3, 4 or 5',F(48,'Bold'),WHITE)
    text_c(d,1377,'Shot on a DWARF 3 smart telescope',F(34,'Medium'),CYAN)
    text_c(d,1425,'from the Houston suburbs',F(34,'Medium'),CYAN)
    return shadowed(L)
END=endcard()
# ---------------- timeline ----------------
SUN_A=((820,820),1600); SUN_B=((1090,640),760)
segs=[('hook',0.0,3.8),('sun',3.8,8.6),('moon',8.6,13.4),('sat',13.4,18.2),('dc',18.2,23.0),('n2264',23.0,27.8),('end',27.8,33.0)]
DUR=33.0; XF=0.5
def img_layer(name,t):
    if name in('hook','sun'):
        u=t/8.6; win=window(sun,SUN_A[0],SUN_A[1],SUN_B[0],SUN_B[1],u); bg=BG['sun']
        if t<3.8:
            before=window(sunb,SUN_A[0],SUN_A[1],SUN_B[0],SUN_B[1],u)
            wp=ease(np.clip((t-1.4)/1.3,0,1)); xcut=int(W*wp)
            mixed=before.copy(); mixed[:,:xcut]=win[:,:xcut]
            if 0<xcut<W: mixed[:,max(0,xcut-3):xcut+3]=CYAN[::-1]
            win=mixed if t<2.9 else win
        return win,bg
    if name=='moon': u=(t-8.6)/4.8; return window(moon,(720,568),1020,(727,568),820,u),BG['moon']
    if name=='sat': u=(t-13.4)/4.8; return window(sat,(675,580),1150,(675,580),1000,u),BG['sat']
    if name=='dc':  u=(t-18.2)/4.8;  return window(dc,(1030,620),1150,(1050,610),950,u),BG['dc']
    if name=='n2264': u=(t-23.0)/4.8; return window(n2264,(1220,660),1100,(1235,640),900,u),BG['n2264']
    return None,None
def compose(name,t):
    canvas=np.zeros((H,W,3),np.float32)
    if name=='end':
        bg=(BG['dc']*0.5+BG['m41']*0.5)*0.75; canvas[:]=bg
        return canvas,[END]
    win,bg=img_layer(name,t); canvas[:]=bg
    reg=canvas[WIN_Y0:WIN_Y0+WIN_H]; canvas[WIN_Y0:WIN_Y0+WIN_H]=reg*(1-FM)+win.astype(np.float32)*FM
    ovs=[HDR,P[name]]
    if name=='hook' and t<2.9: ovs+= [TB,TA] if t>1.5 else [TB]
    return canvas,ovs
def alpha_over(base,ov,a=1.0):
    o=np.asarray(ov,np.float32); al=o[...,3:4]/255*a
    return base*(1-al)+o[...,2::-1]*al
def base(name,t):
    canvas=np.zeros((H,W,3),np.float32)
    if name=='end':
        canvas[:]=(BG['dc']*0.5+BG['m41']*0.5)*0.75; return canvas
    win,bg=img_layer(name,t); canvas[:]=bg
    reg=canvas[WIN_Y0:WIN_Y0+WIN_H]; canvas[WIN_Y0:WIN_Y0+WIN_H]=reg*(1-FM)+win.astype(np.float32)*FM
    return canvas
def texts(name,t):
    if name=='end': return [END]
    o=[P[name]]
    if name=='hook' and t<2.9: o+= [TB,TA] if t>1.5 else [TB]
    return o
def frame(t):
    i=max(k for k,s_ in enumerate(segs) if s_[1]<=t or k==0); name,s0,s1=segs[i]
    img=base(name,t); a=0.0; nxt=None
    if i+1<len(segs) and t>s1-XF:
        a=ease((t-(s1-XF))/XF); nxt=segs[i+1][0]; img=img*(1-a)+base(nxt,s1)*a
    # header: on for all image segments, fades out into the end card
    hdr_a=1.0 if name!='end' else 0.0
    if nxt=='end': hdr_a=1-a
    if hdr_a>0: img=alpha_over(img,HDR,hdr_a)
    # captions: fade out first, then the next fades in (no overlapping text)
    fin=1.0 if s0==0 else min(1,(t-s0)/0.25+0.0)
    out_a=fin*(1-min(1,2*a)) if nxt else fin
    for ov in texts(name,t): img=alpha_over(img,ov,out_a)
    if nxt and a>0.5:
        for ov in texts(nxt,s1): img=alpha_over(img,ov,(2*a-1))
    if t<0.25: img*=t/0.25
    if t>DUR-0.6: img*=max(0,(DUR-t)/0.6)
    return np.clip(img,0,255).astype(np.uint8)
if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[2]=='stills':
        for t in (0.5,2.2,3.2,6,8.3,11,16,21,26,30): cv2.imwrite(f'still3_{t}.jpg',frame(t))
        sys.exit()
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-profile:v','high','-r',str(FPS),'video3_only.mp4'],stdin=subprocess.PIPE)
    n=int(DUR*FPS)
    for k in range(n): p.stdin.write(frame(k/FPS).tobytes())
    p.stdin.close(); p.wait(); print('video frames',n)
