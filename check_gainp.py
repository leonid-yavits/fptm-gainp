"""Bit-exact checks. Both mappings execute their generated COMPARE/WRITE/READ
programs on real test samples and are compared with an independent NumPy
reference of the trained model (fptm.reference_votes / strict_votes).
Usage: python check_gainp.py <dataset> <run_dir_name> <nS> <nC> [results_root]"""
import pickle, sys, numpy as np
from datasets import load
from gainp_sim import run_sample_stationary
from gainp_clause import run_clause_stationary
from fptm import reference_votes, strict_votes
ds, tag, nS, nC = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
root = sys.argv[5] if len(sys.argv) > 5 else "results"
M = pickle.load(open(f"{root}/{ds}/{tag}/model.pkl", "rb"))
d = load(ds)
strict = bool(M.get("strict", False))
LF = M["LF"] or 1
def ref(X):
    V = strict_votes(M["model"], X, M["n_classes"]) if strict else reference_votes(M["model"], X, M["n_classes"], LF)
    return np.argmax(V, 1)
X = d["Xte"][:nS]
p, hw = run_sample_stationary(M["model"], M["n_classes"], LF, strict, M["n_features"], X)
print(f"{ds} {tag} mapping S match: {float((ref(X) == hw).mean())} n={nS} cycles {p.cycles} rows {p.n_rows}")
# Mapping C: simulate in chunks so that the replicated array fits in memory
Xc = d["Xte"][:nC]
pc0, _, info = run_clause_stationary(M["model"], M["n_classes"], LF, strict, None)
chunk = int(max(1, min(nC, 1.5e8 // max(1, info["F_used"] * info["columns"]))))
hc = np.concatenate([run_clause_stationary(M["model"], M["n_classes"], LF, strict, Xc[i:i + chunk])[1]
                     for i in range(0, nC, chunk)])
print(f"{ds} {tag} mapping C match: {float((ref(Xc) == hc).mean())} n={nC} cycles {pc0.cycles} {info}")
