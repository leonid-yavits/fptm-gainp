# TreeTM datasets (prepared)

The exact train, validation and test splits used in the TreeTM paper, for two public datasets.
Each folder holds `prepared.npz` (the arrays) and `meta.json` (taxonomy and metadata).

| Folder | Domain | Bits | Raw features | Train / Val / Test | Leaves | Depth |
|---|---|---|---|---|---|---|
| `nslkdd` | Network intrusion (NSL-KDD) | 178 | 122 (38 numeric + one-hot categoricals) | 113,375 / 12,598 / 18,794 | 23 | 3 |
| `alerce` | Astronomical alerts (ALeRCE, ZTF) | 1,172 | 152 | 88,916 / 9,880 / 24,700 | 15 | 2 |

## Arrays in `prepared.npz`

| Key | Shape | Type | Content |
|---|---|---|---|
| `Btr`, `Bva`, `Bte` | (n, bits) | uint8 | Booleanized inputs for Tsetlin Machines (thermometer code, 7 quantile levels, plus missing-value bits; one-hot categoricals for NSL-KDD, where a threshold at the training minimum becomes the strict test x > min) |
| `Ftr`, `Fva`, `Fte` | (n, features) | float32 | Raw features for non-TM baselines (random forest, gradient boosting, MLP) |
| `ytr`, `yva`, `yte` | (n,) | int64 | Leaf-class index into `meta["leaves"]` |
| `Bnov`, `Fnov` | (3,750, ...) | | NSL-KDD only: test records of 17 attack types absent from training (no leaf label) |

## Metadata in `meta.json`

- `taxonomy`: nested dictionary of the class tree, root key `"root"`, leaves are the entries of `leaves`
- `leaves`: leaf-class names, in label-index order
- `bit_names`: the condition each Boolean input encodes (for clause interpretation)
- `class_counts_train`, `class_counts_test`, `n_train`, `n_val`, `n_test`, `source`, `split`

## Loading

```python
import json, numpy as np
from taxonomy import Taxonomy          # included here

P = np.load('nslkdd/prepared.npz')
meta = json.load(open('nslkdd/meta.json'))
tax = Taxonomy(meta['taxonomy'])        # children, parent, leaves, paths, depth, node_labels(u, y), ...
Xtr, ytr = P['Btr'], P['ytr']
```

`taxonomy.py` builds per-node training labels (`tax.node_labels(node, y)`) and level labels (`tax.level_label(y, depth)`).

## Sources

- NSL-KDD: Canadian Institute for Cybersecurity; attack categories from KDD Cup 1999 (UCI, DOI 10.24432/C51C7N)
- ALeRCE light-curve classifier labeled set and features: Zenodo, DOI 10.5281/zenodo.4279623

Code: https://github.com/leonid-yavits/TreeTM
