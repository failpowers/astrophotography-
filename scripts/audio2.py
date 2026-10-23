import numpy as np, subprocess
sr=48000; DUR=30.6
th=np.fromfile('theme.raw',np.float32).reshape(-1,2)
env=np.sqrt(np.convolve(th.mean(1)**2,np.ones(4800)/4800,'same'))
print('theme level last 2s (dB, per 0.25s):',[round(20*np.log10(env[int((13+k*0.25)*sr)]+1e-9),1) for k in range(8)])
n=int(DUR*sr); t=np.arange(n)/sr
out=np.zeros((n,2),np.float32); out[:len(th)]+=th
# ---- original ambient pad in Bb minor (Bb, C, Db, F) ----
P0=11.4; tp=t-P0; on=tp>=0
notes=[116.54,233.08,261.63,277.18,349.23,466.16]   # Bb2 Bb3 C4 Db4 F4 Bb4
gains=[0.55,0.45,0.18,0.35,0.30,0.16]
rng=np.random.default_rng(5)
pad=np.zeros((n,2))
for f,g in zip(notes,gains):
    for ch,det in ((0,-0.18),(1,0.21)):
        ph=rng.uniform(0,2*np.pi)
        w=2*np.pi*(f*(1+det/100))*tp
        tone=np.sin(w+ph)+0.25*np.sin(2*w+ph)+0.08*np.sin(3*w)          # soft, warm partials
        lfo=1+0.12*np.sin(2*np.pi*(0.11+0.03*ch)*tp+ph)
        pad[:,ch]+=g*tone*lfo
# airy shimmer: band-limited noise around 3-6 kHz, very quiet
nz=rng.standard_normal((n,2))
spec=np.fft.rfft(nz,axis=0); fr=np.fft.rfftfreq(n,1/sr); band=np.exp(-((fr-4500)/1800)**2); nzb=np.fft.irfft(spec*band[:,None],n=n,axis=0)
pad+=0.05*nzb/np.abs(nzb).max()
# simple reverb: convolve with decaying noise
ir_len=int(2.2*sr); ir=rng.standard_normal((ir_len,2))*np.exp(-np.arange(ir_len)/sr/0.55)[:,None]; ir[0]+=6
from numpy.fft import rfft,irfft
L=n+ir_len; N=1<<int(np.ceil(np.log2(L)))
wet=np.stack([irfft(rfft(pad[:,c],N)*rfft(ir[:,c],N),N)[:n] for c in range(2)],1)
pad=0.45*pad+0.55*wet/np.abs(wet).max()*np.abs(pad).max()
# envelope: swell in under the theme's ending, gentle fade out
envp=np.clip(tp/2.2,0,1)**1.5*np.clip((DUR-t)/2.2,0,1)
pad*=envp[:,None]*on[:,None]
pad=pad/np.sqrt((pad[t>16]**2).mean())*10**(-22/20)   # pad sits around -22 dB RMS
# duck theme tail into pad
pass
out=out+pad.astype(np.float32)
out.astype(np.float32).tofile('mix2.raw')
subprocess.run(['ffmpeg','-v','error','-y','-f','f32le','-ar',str(sr),'-ac','2','-i','mix2.raw',
  '-ar','48000','-c:a','pcm_s16le','mix2.wav'],check=True)
print('audio done')
