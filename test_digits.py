import numpy as np, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from fptm import FPTM, reference_votes
d = load_digits(); X = d.data/16.0
B = np.concatenate([(X > t) for t in (0, .25, .5, .75)], 1).astype(np.uint8)
Xtr, Xte, ytr, yte = train_test_split(B, d.target, test_size=0.3, random_state=0, stratify=d.target)
for name, kw in [("FPTM LF=75", dict(clauses=20,T=20,S=200,L=150,LF=75)),
                 ("FPTM LF=1 ", dict(clauses=20,T=20,S=200,L=150,LF=1))]:
    tm = FPTM(include_limit=200, seed=1, **kw); best=0; t0=time.time()
    for e in range(60):
        tm.fit_epoch(Xtr, ytr); acc=(tm.predict(Xte)==yte).mean(); best=max(best,acc)
    m = tm.export(); V = reference_votes(m, Xte, tm.n_classes, tm.LF)
    print(name, f"last {acc:.4f} best {best:.4f} {time.time()-t0:.1f}s", tm.literal_stats(),
          "ref==numba:", np.array_equal(V, tm.class_votes(Xte)))
