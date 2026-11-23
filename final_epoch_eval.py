"""Score the model state after epoch 300 (Artem's protocol: no best-epoch selection).
Reads ckpt.pkl of the long runs (saved at epoch 300) and writes summary.json/model.pkl
in the same format as run_fptm.py, so hw_eval.py and check_gainp.py run unchanged.
Usage: python3 final_epoch_eval.py <src_root> <dst_root>"""
import json, os, pickle, sys, glob
import numpy as np
from fptm import reference_votes
from datasets import load
src, dst = sys.argv[1], sys.argv[2]
cache = {}
for ck in sorted(glob.glob(f"{src}/*/*/ckpt.pkl")):
    st = pickle.load(open(ck, "rb"))
    old = json.load(open(os.path.join(os.path.dirname(ck), "summary.json")))
    if st["epoch"] != 300:
        print("SKIP (checkpoint at epoch %d)" % st["epoch"], ck); continue
    ds = old["dataset"]; d = cache.setdefault(ds, load(ds)); tm = st["tm"]
    va = float((tm.predict(d["Xva"]) == d["yva"]).mean())
    pred = tm.predict(d["Xte"]); te = float((pred == d["yte"]).mean())
    model = tm.export()
    assert np.array_equal(np.argmax(reference_votes(model, d["Xte"], tm.n_classes, old["LF"]), 1), pred)
    s = dict(old); s.update(epochs_run=300, best_epoch=300, selection="final epoch 300", val_acc=va, test_acc=te, **tm.literal_stats())
    out = os.path.join(dst, ds, os.path.basename(os.path.dirname(ck))); os.makedirs(out, exist_ok=True)
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1)
    pickle.dump(dict(model=model, LF=old["LF"], n_classes=tm.n_classes, n_features=int(d["Xtr"].shape[1])), open(os.path.join(out, "model.pkl"), "wb"))
    np.save(os.path.join(out, "test_pred.npy"), pred)
    print(ds, s["tag"], "val %.2f test %.2f" % (100 * va, 100 * te))
