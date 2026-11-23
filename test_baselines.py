import numpy as np, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from fptm import FPTM
from tmu.tsetlin_machine import TMClassifier
d = load_digits(); X = d.data/16.0
B = np.concatenate([(X > t) for t in (0, .25, .5, .75)], 1).astype(np.uint8)
Xtr, Xte, ytr, yte = train_test_split(B, d.target, test_size=0.3, random_state=0, stratify=d.target)
for kw in [dict(clauses=200,T=20,S=200,L=16,LF=1), dict(clauses=400,T=40,S=100,L=12,LF=1)]:
    tm = FPTM(include_limit=128, seed=1, **kw); best=0; t0=time.time()
    for e in range(60):
        tm.fit_epoch(Xtr, ytr); best=max(best,(tm.predict(Xte)==yte).mean())
    print("FPTM-LF1", kw, f"best {best:.4f} {time.time()-t0:.1f}s")
for C,T,s in [(200,50,5.0),(400,100,5.0)]:
    tm = TMClassifier(C, T, s, platform='CPU', weighted_clauses=False); best=0; t0=time.time()
    for e in range(60):
        tm.fit(Xtr.astype(np.uint32), ytr.astype(np.uint32)); best=max(best,(tm.predict(Xte.astype(np.uint32))==yte).mean())
    print("tmu vanilla", C,T,s, f"best {best:.4f} {time.time()-t0:.1f}s")
