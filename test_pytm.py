import numpy as np, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
d = load_digits(); X = d.data/16.0
B = np.concatenate([(X > t) for t in (0, .25, .5, .75)], 1).astype(np.uint32)
Xtr, Xte, ytr, yte = train_test_split(B, d.target, test_size=0.3, random_state=0, stratify=d.target)
for C,T,s in [(200,50,5.0),(400,100,5.0),(2000,50,10.0)]:
    tm = MultiClassTsetlinMachine(C, T, s); best=0; t0=time.time()
    for e in range(60):
        tm.fit(Xtr, ytr.astype(np.uint32), epochs=1, incremental=True); best=max(best,(tm.predict(Xte)==yte).mean())
    print("pyTM vanilla", C,T,s, f"best {best:.4f} {time.time()-t0:.1f}s")
