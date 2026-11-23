"""
Booleanization of the evaluation datasets for FPTM / vanilla TM.

WOS (Kowsari et al., 2017, Mendeley Data):
  expects the extracted WebOfScience folder, e.g. WOS46985/{X.txt,YL1.txt,YL2.txt,Y.txt}.
  Binary bag of n-grams (CountVectorizer, binary=True), then chi2 SelectKBest,
  following the FPTM IMDb preparation script (prepare_dataset.py).

ALeRCE light curve classifier (Sanchez-Saez et al., 2021, Zenodo 4279622):
  expects the labeled set CSV (oid, classALeRCE) and the features parquet
  (index or column 'oid'). Each numeric feature is thermometer-encoded with
  thresholds at training-set quantiles. A missing value encodes as all zeros
  plus one dedicated "missing" bit.

All splits are stratified, with a fixed seed. Fitting (vocabulary, chi2,
quantiles) uses the training split only.
"""
import argparse, os
import numpy as np


def prep_wos(root, subset="WOS46985", level="YL1", ngram=2, k=10000,
             test_size=0.2, seed=0, out="data"):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    d = os.path.join(root, subset)
    with open(os.path.join(d, "X.txt"), encoding="utf-8", errors="ignore") as f:
        docs = [l.strip() for l in f]
    y = np.loadtxt(os.path.join(d, f"{level}.txt"), dtype=np.int64)
    assert len(docs) == len(y), (len(docs), len(y))
    _, y = np.unique(y, return_inverse=True)
    dtr, dte, ytr, yte = train_test_split(docs, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    vec = CountVectorizer(ngram_range=(1, ngram), binary=True, lowercase=True,
                          min_df=2, stop_words="english")
    Xtr = vec.fit_transform(dtr)
    Xte = vec.transform(dte)
    skb = SelectKBest(chi2, k=min(k, Xtr.shape[1])).fit(Xtr, ytr)
    Xtr = skb.transform(Xtr).toarray().astype(np.uint8)
    Xte = skb.transform(Xte).toarray().astype(np.uint8)
    name = f"wos_{subset}_{level}_k{Xtr.shape[1]}"
    _save(out, name, Xtr, ytr, Xte, yte)


def prep_alerce(labels_csv, features_parquet, n_thr=8, test_size=0.2, seed=0,
                label_col="classALeRCE", hierarchy=False, out="data"):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    lab = pd.read_csv(labels_csv)
    feat = pd.read_parquet(features_parquet)
    if "oid" in feat.columns:
        feat = feat.set_index("oid")
    lab = lab.drop_duplicates("oid").set_index("oid")
    df = feat.join(lab[[label_col]], how="inner")
    yname = df.pop(label_col).astype(str).values
    if hierarchy:  # top level of the ALeRCE taxonomy (Sanchez-Saez et al., 2021)
        top = {"SNIa": "Transient", "SNIbc": "Transient", "SNII": "Transient", "SLSN": "Transient",
               "QSO": "Stochastic", "AGN": "Stochastic", "Blazar": "Stochastic",
               "YSO": "Stochastic", "CV/Nova": "Stochastic",
               "LPV": "Periodic", "E": "Periodic", "DSCT": "Periodic",
               "RRL": "Periodic", "CEP": "Periodic", "Periodic-Other": "Periodic"}
        unknown = sorted(set(yname) - set(top))
        assert not unknown, f"unmapped classes: {unknown}"
        yname = np.array([top[c] for c in yname])
    classes, y = np.unique(yname, return_inverse=True)
    num = df.select_dtypes(include=[np.number]).astype(np.float64)
    num = num.replace([np.inf, -np.inf], np.nan)
    Ftr, Fte, ytr, yte = train_test_split(num.values, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    qs = np.linspace(0, 1, n_thr + 2)[1:-1]
    thr = np.nanquantile(Ftr, qs, axis=0)  # (n_thr, n_feat)

    def enc(F):
        miss = np.isnan(F)
        bits = [(np.nan_to_num(F, nan=-np.inf) > thr[t]) for t in range(n_thr)]
        return np.concatenate(bits + [miss], axis=1).astype(np.uint8)
    Xtr, Xte = enc(Ftr), enc(Fte)
    keep = Xtr.std(0) > 0  # drop constant bits (fit on train only)
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    name = f"alerce_{'top' if hierarchy else 'leaf'}_t{n_thr}"
    _save(out, name, Xtr, ytr, Xte, yte, classes=classes)


def _save(out, name, Xtr, ytr, Xte, yte, classes=None):
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, name + ".npz"), Xtr=Xtr, ytr=ytr,
                        Xte=Xte, yte=yte,
                        classes=np.array([] if classes is None else classes))
    print(f"{name}: train {Xtr.shape}, test {Xte.shape}, classes {len(np.unique(ytr))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="ds", required=True)
    w = sub.add_parser("wos"); w.add_argument("root"); w.add_argument("--subset", default="WOS46985")
    w.add_argument("--level", default="YL1"); w.add_argument("--k", type=int, default=10000)
    w.add_argument("--ngram", type=int, default=2)
    a = sub.add_parser("alerce"); a.add_argument("labels"); a.add_argument("features")
    a.add_argument("--thr", type=int, default=8); a.add_argument("--top", action="store_true")
    args = ap.parse_args()
    if args.ds == "wos":
        prep_wos(args.root, args.subset, args.level, args.ngram, args.k)
    else:
        prep_alerce(args.labels, args.features, args.thr, hierarchy=args.top)
