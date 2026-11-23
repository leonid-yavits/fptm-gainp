"""Dataset loader. All datasets return Boolean train / validation / test splits.
Model selection (epoch, hyperparameters) uses validation only; test is reported once."""
import json, os
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
TREETM = os.path.join(ROOT, "inputs", "treetm", "TreeTM_datasets")


def load(name):
    if name in ("alerce", "nslkdd"):
        P = np.load(os.path.join(TREETM, name, "prepared.npz"))
        meta = json.load(open(os.path.join(TREETM, name, "meta.json")))
        return dict(Xtr=P["Btr"], ytr=P["ytr"], Xva=P["Bva"], yva=P["yva"],
                    Xte=P["Bte"], yte=P["yte"], classes=meta["leaves"])
    if name == "wos":
        P = np.load(os.path.join(ROOT, "data", "wos_WOS46985_YL1_k10000.npz"))
        # 10% of the training split as validation, stratified, seed 0
        Xtr, Xva, ytr, yva = train_test_split(P["Xtr"], P["ytr"], test_size=0.1,
                                              random_state=0, stratify=P["ytr"])
        return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=P["Xte"], yte=P["yte"],
                    classes=["CS", "ECE", "Psychology", "MAE", "Civil",
                             "Medical", "Biochemistry"])
    raise ValueError(name)
