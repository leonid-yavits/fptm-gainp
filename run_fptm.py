"""Train FPTM (or strict TM via LF=1) with validation-based epoch selection.
Saves the best-on-validation model (exported clause literal lists) and a log.
Usage: python run_fptm.py <dataset> <tag> clauses T S L LF include_limit max_epochs [seed]
Training stops after max_epochs or after PATIENCE epochs without a new best
validation accuracy, whichever comes first (env PATIENCE, default 10; 0 = no early stop).
Output: $RESULTS_ROOT/<dataset>/<tag>_s<seed>/ (default results/)
"""
import json, os, pickle, sys, time
import numpy as np
from fptm import FPTM, reference_votes, _seed
from datasets import load

ds, tag = sys.argv[1], sys.argv[2]
C, T, S, L, LF, IL, E = map(int, sys.argv[3:10])
seed = int(sys.argv[10]) if len(sys.argv) > 10 else 0
PATIENCE = int(os.environ.get("PATIENCE", "10"))   # 0 disables early stopping
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "0"))  # early stopping applies only after this epoch
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "0"))  # >0: checkpoint every N epochs, resume if present
# With checkpointing, the in-kernel RNG is reseeded every epoch so a resumed run equals an uninterrupted one.
out = os.path.join(os.environ.get("RESULTS_ROOT", "results"), ds, f"{tag}_s{seed}")
os.makedirs(out, exist_ok=True)
logf = open(os.path.join(out, "log.jsonl"), "a")

d = load(ds)
tm = FPTM(C, T, S, L=L, LF=LF, include_limit=IL, seed=seed)
best = (-1, None, None)
t_train = 0.0
e0 = 0
ckpt = os.path.join(out, "ckpt.pkl")
if CKPT_EVERY and os.path.exists(ckpt):
    st = pickle.load(open(ckpt, "rb"))
    tm, best, e0, t_train = st["tm"], st["best"], st["epoch"], st["t_train"]
    logf.close()
    keep = [l for l in open(os.path.join(out, "log.jsonl")) if json.loads(l)["epoch"] <= e0]
    open(os.path.join(out, "log.jsonl"), "w").writelines(keep)
    logf = open(os.path.join(out, "log.jsonl"), "a")
    print(f"RESUMED at epoch {e0}", flush=True)
elif CKPT_EVERY:   # fresh start: drop log lines of a run that died before its first checkpoint
    logf.close(); logf = open(os.path.join(out, "log.jsonl"), "w")
e = e0
for e in range(e0 + 1, E + 1):
    if CKPT_EVERY and tm._init:
        _seed(seed * 1000003 + e)
    t0 = time.time(); tm.fit_epoch(d["Xtr"], d["ytr"]); dt = time.time() - t0
    t_train += dt
    va = float((tm.predict(d["Xva"]) == d["yva"]).mean())
    rec = dict(epoch=e, val=va, train_s=round(dt, 2), **tm.literal_stats())
    if va > best[0]:
        best = (va, e, tm.snapshot())
        rec["best"] = True
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    print(json.dumps(rec), flush=True)
    if CKPT_EVERY and e % CKPT_EVERY == 0:
        pickle.dump(dict(tm=tm, best=best, epoch=e, t_train=t_train), open(ckpt + ".tmp", "wb"))
        os.replace(ckpt + ".tmp", ckpt)
    if PATIENCE and e >= MIN_EPOCHS and e - best[1] >= PATIENCE:
        break

tm.restore(best[2])
te_pred = tm.predict(d["Xte"])
te = float((te_pred == d["yte"]).mean())
model = tm.export()
# independent NumPy cross-check of the exported model on the test set
V = reference_votes(model, d["Xte"], tm.n_classes, LF)
assert np.array_equal(np.argmax(V, 1), te_pred)
summary = dict(dataset=ds, tag=tag, epochs_run=e, clauses=C, T=T, S=S, L=L, LF=LF,
               include_limit=IL, epochs=E, seed=seed, patience=PATIENCE, min_epochs=MIN_EPOCHS, best_epoch=best[1],
               val_acc=best[0], test_acc=te, train_time_s=round(t_train, 1),
               n_features=int(d["Xtr"].shape[1]), n_classes=int(tm.n_classes),
               **tm.literal_stats())
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
pickle.dump(dict(model=model, LF=LF, n_classes=tm.n_classes,
                 n_features=int(d["Xtr"].shape[1])),
            open(os.path.join(out, "model.pkl"), "wb"))
print("SUMMARY", json.dumps(summary), flush=True)
