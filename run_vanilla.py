"""Vanilla (strict) Tsetlin Machine baseline: Granmo's reference C implementation
via pyTsetlinMachine (MultiClassTsetlinMachine). Validation-based epoch selection.
Usage: python run_vanilla.py <dataset> <tag> clauses_per_class T s max_epochs [seed]
Same stopping rule as run_fptm.py (patience 10). The seed sets the libc RNG
used by the C library and the order of the training samples.
pyTsetlinMachine defaults are used: boost_true_positive_feedback=1, 8 state
bits, no literal limit.
"""
import json, os, pickle, sys, time
import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from datasets import load

ds, tag = sys.argv[1], sys.argv[2]
C, T = int(sys.argv[3]), int(sys.argv[4])
s, E = float(sys.argv[5]), int(sys.argv[6])
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
PATIENCE = 10
import ctypes
ctypes.CDLL(None).srand(seed)
out = os.path.join("results", ds, f"{tag}_s{seed}")
os.makedirs(out, exist_ok=True)
logf = open(os.path.join(out, "log.jsonl"), "a")
d = load(ds)
perm = np.random.default_rng(seed).permutation(len(d["ytr"]))
Xtr = d["Xtr"][perm]; ytr = d["ytr"][perm].astype(np.uint32)
Xva = d["Xva"].astype(np.uint32); Xte = d["Xte"].astype(np.uint32)
tm = MultiClassTsetlinMachine(C, T, s, indexed=False)
# Memory-lean training (identical algorithm): pyTsetlinMachine.fit re-encodes
# a full uint32 copy of X at every call. Here the machine is created with a
# zero-epoch fit on one sample per class, the training set is bit-packed once
# in chunks with the library encoder, and each epoch calls the C trainer.
from pyTsetlinMachine import tm as _tmmod
_lib = _tmmod._lib
first = np.array([np.flatnonzero(ytr == c)[0] for c in range(int(ytr.max()) + 1)])
tm.fit(Xtr[first], ytr[first], epochs=0, incremental=True)
n, Fh = Xtr.shape[0], tm.number_of_features // 2
enc = np.empty(n * tm.number_of_ta_chunks, dtype=np.uint32)
for i in range(0, n, 2000):
    xc = np.ascontiguousarray(Xtr[i:i + 2000].astype(np.uint32).ravel())
    m = min(2000, n - i)
    _lib.tm_encode(xc, enc[i * tm.number_of_ta_chunks:(i + m) * tm.number_of_ta_chunks], m, Fh, 1, 1, Fh, 1, 1, 0)
Ym = np.ascontiguousarray(ytr).astype(np.uint32)
best = (-1, None, None); t_train = 0.0
for e in range(1, E + 1):
    t0 = time.time(); _lib.mc_tm_fit(tm.mc_tm, enc, Ym, n, 1); dt = time.time() - t0
    t_train += dt
    va = float((tm.predict(Xva) == d["yva"]).mean())
    rec = dict(epoch=e, val=va, train_s=round(dt, 2))
    if va > best[0]:
        best = (va, e, tm.get_state()); rec["best"] = True
    logf.write(json.dumps(rec) + "\n"); logf.flush(); print(json.dumps(rec), flush=True)
    if e - best[1] >= PATIENCE:
        break
tm.set_state(best[2])
te = float((tm.predict(Xte) == d["yte"]).mean())

# Export included literals per clause (strict AND), for hardware mapping.
F = Xtr.shape[1]
model = []
n_cls = int(ytr.max()) + 1
for c in range(n_cls):
    for j in range(C):
        inc = [k for k in range(F) if tm.ta_action(c, j, k) == 1]
        incn = [k for k in range(F) if tm.ta_action(c, j, F + k) == 1]
        model.append(dict(cls=c, polarity=+1 if j % 2 == 0 else -1, clause=j,
                          lits=np.array(inc, dtype=np.int64),
                          lits_neg=np.array(incn, dtype=np.int64)))
from fptm import strict_votes
pred_ref = np.argmax(strict_votes(model, d["Xte"], n_cls), 1)
agree = float((pred_ref == tm.predict(Xte)).mean())
print("export/predict agreement", agree, flush=True)
lens = np.array([len(m["lits"]) + len(m["lits_neg"]) for m in model])
summary = dict(dataset=ds, tag=tag, seed=seed, epochs_run=e, impl="pyTsetlinMachine", clauses=C, T=T, s=s,
               epochs=E, best_epoch=best[1], val_acc=best[0], test_acc=te,
               train_time_s=round(t_train, 1), export_agreement=agree, n_features=F, n_classes=n_cls,
               total=int(lens.sum()), per_clause_mean=float(lens.mean()),
               per_clause_max=int(lens.max()), empty_clauses=int((lens == 0).sum()))
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
pickle.dump(dict(model=model, LF=None, strict=True, n_classes=n_cls, n_features=F),
            open(os.path.join(out, "model.pkl"), "wb"))
print("SUMMARY", json.dumps(summary), flush=True)
