"""
Fuzzy-Pattern Tsetlin Machine (FPTM): Python/Numba port.

Line-by-line port of the multiclass reference implementation
src/FuzzyPatternTM.jl from https://github.com/BooBSD/FuzzyPatternTM
(A. Hnilov, "Fuzzy-Pattern Tsetlin Machine", arXiv:2508.08350, 2025).

Port notes
- Julia is unavailable in this environment, so the algorithm is re-expressed
  in Numba. Clause evaluation, vote, Feedback 1 / Feedback 2, the
  include/exclude thresholds and the state bounds follow the Julia code.
- The Julia trainer is Hogwild-parallel over samples. This port trains
  sequentially (deterministic for a given seed). Accuracy may therefore
  differ slightly from multithreaded Julia runs.
- LF = 1 reproduces strict (all-or-nothing) clause evaluation, as stated in
  the FPTM README ("LF = 1 corresponds to the same internal logic used in the
  Tsetlin.jl library").
"""
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------
# Clause evaluation (check_clause in FuzzyPatternTM.jl)
# ----------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _clause_score(x, inc, n, inci, ni, LF):
    length = n + ni
    c = length if (0 < length < LF) else LF
    for k in range(n):
        if c <= 0:
            return 0
        c -= 1 - x[inc[k]]
    for k in range(ni):
        if c <= 0:
            return 0
        c -= x[inci[k]]
    return c if c > 0 else 0


@njit(cache=True)
def _class_vote(x, cls, inc, ninc, inci, ninci, LF):
    H = inc.shape[2]
    pos = 0
    neg = 0
    for j in range(H):
        pos += _clause_score(x, inc[cls, 0, j], ninc[cls, 0, j],
                             inci[cls, 0, j], ninci[cls, 0, j], LF)
        neg += _clause_score(x, inc[cls, 1, j], ninc[cls, 1, j],
                             inci[cls, 1, j], ninci[cls, 1, j], LF)
    return pos - neg


@njit(cache=True)
def _rebuild(state, out, include_limit):
    n = 0
    for i in range(state.shape[0]):
        if state[i] >= include_limit:
            out[n] = i
            n += 1
    return n


# ----------------------------------------------------------------------------
# Feedback (feedback! in FuzzyPatternTM.jl)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _feedback(x, cls, positive, S, Si, inc, ninc, inci, ninci,
              T, s, L, LF, include_limit, state_max):
    F = x.shape[0]
    H = S.shape[2]
    v = _class_vote(x, cls, inc, ninc, inci, ninci, LF)
    if v > T:
        v = T
    elif v < -T:
        v = -T
    if positive:
        update = (T - v) / (2.0 * T)
        s1 = 0
    else:
        update = (T + v) / (2.0 * T)
        s1 = 1
    s2 = 1 - s1

    # Feedback 1
    for j in range(H):
        if np.random.random() < update:
            c = S[cls, s1, j]
            ci = Si[cls, s1, j]
            # The include lists are rebuilt only when a state crosses the
            # include threshold. This changes speed only, not the result.
            ch = False
            chi = False
            if _clause_score(x, inc[cls, s1, j], ninc[cls, s1, j],
                             inci[cls, s1, j], ninci[cls, s1, j], LF) > 0:
                if ninc[cls, s1, j] + ninci[cls, s1, j] <= L:
                    for i in range(F):
                        if x[i] == 1:
                            if c[i] < state_max:
                                c[i] += 1
                                if c[i] == include_limit:
                                    ch = True
                        else:
                            if ci[i] < state_max:
                                ci[i] += 1
                                if ci[i] == include_limit:
                                    chi = True
                for i in range(F):
                    if x[i] == 0:
                        if c[i] < include_limit and c[i] > 0:
                            c[i] -= 1
                    else:
                        if ci[i] < include_limit and ci[i] > 0:
                            ci[i] -= 1
            else:
                for _ in range(s):
                    i = np.random.randint(0, F)
                    if c[i] > 0:
                        if c[i] == include_limit:
                            ch = True
                        c[i] -= 1
                    i = np.random.randint(0, F)
                    if ci[i] > 0:
                        if ci[i] == include_limit:
                            chi = True
                        ci[i] -= 1
            if ch:
                ninc[cls, s1, j] = _rebuild(c, inc[cls, s1, j], include_limit)
            if chi:
                ninci[cls, s1, j] = _rebuild(ci, inci[cls, s1, j], include_limit)

    # Feedback 2
    for j in range(H):
        if np.random.random() < update:
            if _clause_score(x, inc[cls, s2, j], ninc[cls, s2, j],
                             inci[cls, s2, j], ninci[cls, s2, j], LF) > 0:
                c = S[cls, s2, j]
                ci = Si[cls, s2, j]
                ch = False
                chi = False
                for i in range(F):
                    if x[i] == 0:
                        if c[i] < include_limit:
                            c[i] += 1
                            if c[i] == include_limit:
                                ch = True
                    else:
                        if ci[i] < include_limit:
                            ci[i] += 1
                            if ci[i] == include_limit:
                                chi = True
                if ch:
                    ninc[cls, s2, j] = _rebuild(c, inc[cls, s2, j], include_limit)
                if chi:
                    ninci[cls, s2, j] = _rebuild(ci, inci[cls, s2, j], include_limit)


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _train_epoch(X, Y, perm, n_classes, S, Si, inc, ninc, inci, ninci,
                 T, s, L, LF, include_limit, state_max):
    for idx in range(perm.shape[0]):
        k = perm[idx]
        x = X[k]
        y = Y[k]
        _feedback(x, y, True, S, Si, inc, ninc, inci, ninci,
                  T, s, L, LF, include_limit, state_max)
        for cls in range(n_classes):
            if cls != y:
                _feedback(x, cls, False, S, Si, inc, ninc, inci, ninci,
                          T, s, L, LF, include_limit, state_max)


@njit(cache=True)
def _votes(X, n_classes, inc, ninc, inci, ninci, LF):
    N = X.shape[0]
    V = np.zeros((N, n_classes), dtype=np.int64)
    for k in range(N):
        for cls in range(n_classes):
            V[k, cls] = _class_vote(X[k], cls, inc, ninc, inci, ninci, LF)
    return V


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
class FPTM:
    """Multiclass FPTM. Hyperparameter names follow FuzzyPatternTM.jl.

    clauses: clauses per class (half positive, half negative polarity).
    T: vote clamp. S: specificity divisor, s = round(F / S).
    L: literal budget for Feedback 1 growth. LF: literal-miss budget.
    """

    def __init__(self, clauses, T, S, L=16, LF=4, states_num=256,
                 include_limit=128, seed=0):
        assert clauses >= 2 and clauses % 2 == 0, \
            "clauses must be even (the Julia code uses floor(clauses/2) per polarity)"
        self.clauses, self.T, self.S, self.L, self.LF = clauses, T, S, L, LF
        self.state_max = states_num - 1
        self.include_limit = include_limit
        self.seed = seed
        self._init = False

    def _initialize(self, F, n_classes):
        H = self.clauses // 2
        self.F, self.n_classes, self.H = F, n_classes, H
        self.s = int(round(F / self.S))
        fill = self.include_limit - 1
        self.Sm = np.full((n_classes, 2, H, F), fill, dtype=np.uint8)
        self.Si = np.full((n_classes, 2, H, F), fill, dtype=np.uint8)
        idt = np.uint16 if F < 65536 else np.uint32
        self.inc = np.zeros((n_classes, 2, H, F), dtype=idt)
        self.inci = np.zeros((n_classes, 2, H, F), dtype=idt)
        self.ninc = np.zeros((n_classes, 2, H), dtype=np.int64)
        self.ninci = np.zeros((n_classes, 2, H), dtype=np.int64)
        self._rng = np.random.default_rng(self.seed)
        _seed(self.seed)
        self._init = True

    def fit_epoch(self, X, Y, shuffle=True):
        X = np.ascontiguousarray(X, dtype=np.uint8)
        Y = np.ascontiguousarray(Y, dtype=np.int64)
        if not self._init:
            self._initialize(X.shape[1], int(Y.max()) + 1)
        perm = self._rng.permutation(len(Y)) if shuffle else np.arange(len(Y))
        _train_epoch(X, Y, perm.astype(np.int64), self.n_classes,
                     self.Sm, self.Si, self.inc, self.ninc, self.inci, self.ninci,
                     self.T, self.s, self.L, self.LF,
                     self.include_limit, self.state_max)

    def class_votes(self, X):
        X = np.ascontiguousarray(X, dtype=np.uint8)
        return _votes(X, self.n_classes, self.inc, self.ninc,
                      self.inci, self.ninci, self.LF)

    def predict(self, X):
        return np.argmax(self.class_votes(X), axis=1)

    # ---- compiled-model export (for hardware mapping) ----------------------
    def export(self):
        """Return the compiled model: for each (class, polarity, clause), the
        included plain and negated literal indices. This is what inference
        hardware must store."""
        model = []
        for c in range(self.n_classes):
            for p in range(2):
                for j in range(self.H):
                    model.append(dict(
                        cls=c, polarity=+1 if p == 0 else -1, clause=j,
                        lits=self.inc[c, p, j, :self.ninc[c, p, j]].astype(np.int64).copy(),
                        lits_neg=self.inci[c, p, j, :self.ninci[c, p, j]].astype(np.int64).copy()))
        return model

    def literal_stats(self):
        n = self.ninc + self.ninci
        return dict(total=int(n.sum()), per_clause_mean=float(n.mean()),
                    per_clause_max=int(n.max()), per_clause_min=int(n.min()),
                    empty_clauses=int((n == 0).sum()))

    def snapshot(self):
        return {k: getattr(self, k).copy() for k in
                ("Sm", "Si", "inc", "inci", "ninc", "ninci")}

    def restore(self, snap):
        for k, v in snap.items():
            setattr(self, k, v.copy())


def reference_votes(model, X, n_classes, LF):
    """Plain-NumPy reference of FPTM inference from the exported model.
    Used to cross-check the Numba kernel and to count hardware operations."""
    X = np.asarray(X, dtype=np.int64)
    V = np.zeros((X.shape[0], n_classes), dtype=np.int64)
    for cl in model:
        n = len(cl["lits"]) + len(cl["lits_neg"])
        budget = n if 0 < n < LF else LF
        miss = (1 - X[:, cl["lits"]]).sum(1) + X[:, cl["lits_neg"]].sum(1)
        score = np.maximum(budget - miss, 0)
        V[:, cl["cls"]] += cl["polarity"] * score
    return V


def strict_votes(model, X, n_classes):
    """Strict (vanilla TM) inference from an exported model: a clause outputs 1
    only if all included literals are satisfied; empty clauses output 0 at
    inference, as in Granmo's reference implementation."""
    X = np.asarray(X, dtype=np.int64)
    V = np.zeros((X.shape[0], n_classes), dtype=np.int64)
    for cl in model:
        n = len(cl["lits"]) + len(cl["lits_neg"])
        if n == 0:
            continue
        miss = (1 - X[:, cl["lits"]]).sum(1) + X[:, cl["lits_neg"]].sum(1)
        V[:, cl["cls"]] += cl["polarity"] * (miss == 0)
    return V
