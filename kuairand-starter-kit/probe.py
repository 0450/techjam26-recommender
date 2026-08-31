import numpy as np
import baseline as B
from data import load
splits = load('./KuaiRand-Pure/data')
configs=[('base',dict(k=16,lr=0.001,epochs=40,seed=0)),('lowlr',dict(k=16,lr=0.0005,epochs=40,seed=0)),('hi_lr',dict(k=16,lr=0.002,epochs=40,seed=0)),('k8',dict(k=8,lr=0.001,epochs=40,seed=0)),('k32',dict(k=32,lr=0.001,epochs=40,seed=0))]
for name, cfg in configs:
    res = B.run_fm(splits, **cfg)
    v = res['valid']; t = res['test']
    print(name, 'valid', v['GAUC'], v['nDCG@5'], v['primary'], 'test', t['GAUC'], t['nDCG@5'], t['primary'])
