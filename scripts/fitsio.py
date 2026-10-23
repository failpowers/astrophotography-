import numpy as np
def read(path):
    raw=open(path,'rb').read(); hdr={}; off=0
    while True:
        blk=raw[off:off+2880].decode('ascii','replace'); off+=2880; end=False
        for i in range(36):
            c=blk[i*80:(i+1)*80]; k=c[:8].strip()
            if k=='END': end=True;break
            if c[8:10]=='= ':
                v=c[10:].split('/')[0].strip().strip("'").strip(); hdr[k]=v
        if end: break
    bp=int(hdr['BITPIX']); n=int(hdr['NAXIS']); dims=[int(hdr[f'NAXIS{i}']) for i in range(1,n+1)]
    dt={16:'>i2',-32:'>f4',8:'u1',32:'>i4'}[bp]
    a=np.frombuffer(raw,dtype=dt,count=int(np.prod(dims)),offset=off).astype(np.float32)
    a=a*float(hdr.get('BSCALE',1))+float(hdr.get('BZERO',0))
    return a.reshape(dims[::-1]),hdr
