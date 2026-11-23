"""
CPU inference baseline for every trained model (FPTM, FPTM with LF=1, vanilla TM).

Three single-threaded Numba kernels are run on every model. Each kernel is
verified and timed in two pilot passes over the test set; every kernel within
1.5x of the fastest pilot time is then timed in full, and the fastest is
reported (the per-model best CPU baseline):
  dense  - bit-packed sample; per clause, failing literals = popcount(pos & ~x)
           + popcount(neg & x) over 64-bit words;
  sparse - literal lists per clause with early exit once the clause score is 0
           (as in the reference check_clause);
  sliced - 64 samples bit-sliced per 64-bit word. Strict and vanilla models: a
           clause is the AND of its literal words, with early exit. FPTM models:
           a vertical binary miss counter per clause with a sticky overflow word.
The score follows Eq. (1) for FPTM/LF=1 models and the strict rule (empty
clause outputs 0) for the exported vanilla TM.
The predictions are checked against fptm.reference_votes / strict_votes on the full
test set. Timing covers input bit-packing plus inference of the full test set,
repeated until at least MIN_TIME seconds have elapsed; the median of 5 such
measurements is reported as time per sample.

Energy is not measured: the virtual machine exposes no power counters (no RAPL).
"""
import glob, json, os, pickle, sys, time
import numpy as np
import numba as nb
from datasets import load
from fptm import reference_votes, strict_votes

nb.set_num_threads(1)
MIN_TIME = 0.5


@nb.njit(cache=True, inline="always")
def _popcount(v):
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((v * np.uint64(0x0101010101010101)) >> np.uint64(56))


@nb.njit(cache=True)
def _predict(Xp, pos, neg, budget, strict_empty0, nlit, cls, pol, n_classes):
    N, W = Xp.shape
    C = pos.shape[0]
    out = np.empty(N, np.int64)
    v = np.zeros(n_classes, np.int64)
    for i in range(N):
        v[:] = 0
        for j in range(C):
            if strict_empty0 and nlit[j] == 0:
                continue
            m = 0
            for w in range(W):
                x = Xp[i, w]
                m += _popcount(pos[j, w] & ~x) + _popcount(neg[j, w] & x)
            s = budget[j] - m
            if s > 0:
                v[cls[j]] += pol[j] * s
        best = 0
        for c in range(1, n_classes):
            if v[c] > v[best]:
                best = c
        out[i] = best
    return out


@nb.njit(cache=True)
def _predict_sparse(X, idx, neg, start, budget, strict_empty0, cls, pol, n_classes):
    N = X.shape[0]
    C = budget.shape[0]
    out = np.empty(N, np.int64)
    v = np.zeros(n_classes, np.int64)
    for i in range(N):
        v[:] = 0
        for j in range(C):
            a, b = start[j], start[j + 1]
            if strict_empty0 and a == b:
                continue
            bj = budget[j]
            m = 0
            for t in range(a, b):
                if X[i, idx[t]] == neg[t]:      # plain literal fails on 0, negated on 1
                    m += 1
                    if m >= bj:
                        break
            s = bj - m
            if s > 0:
                v[cls[j]] += pol[j] * s
        best = 0
        for c in range(1, n_classes):
            if v[c] > v[best]:
                best = c
        out[i] = best
    return out


@nb.njit(cache=True)
def _predict_sliced(XT, N, idx, neg, start, strict_empty0, cls, pol, n_classes):
    F, B = XT.shape
    C = cls.shape[0]
    out = np.empty(N, np.int64)
    V = np.zeros((64, n_classes), np.int64)
    full = ~np.uint64(0)
    for blk in range(B):
        V[:, :] = 0
        for j in range(C):
            a, b = start[j], start[j + 1]
            if strict_empty0 and a == b:
                continue
            m = full
            for t in range(a, b):
                w = XT[idx[t], blk]
                if neg[t] == 1:
                    w = ~w
                m &= w
                if m == 0:
                    break
            if m == full:
                for k in range(64):
                    V[k, cls[j]] += pol[j]
            else:
                while m != 0:
                    low = m & (~m + np.uint64(1))
                    V[_popcount(low - np.uint64(1)), cls[j]] += pol[j]
                    m &= m - np.uint64(1)
        for s in range(64):
            i = blk * 64 + s
            if i >= N:
                break
            best = 0
            for c in range(1, n_classes):
                if V[s, c] > V[s, best]:
                    best = c
            out[i] = best
    return out


@nb.njit(cache=True)
def _predict_sliced_fuzzy(XT, N, idx, neg, start, budget, cls, pol, n_classes):
    """Bit-sliced FPTM: per 64-sample block and clause, a vertical binary miss
    counter of bitlen(b_j) planes with a sticky overflow word."""
    F, B = XT.shape
    C = cls.shape[0]
    out = np.empty(N, np.int64)
    V = np.zeros((64, n_classes), np.int64)
    P = np.zeros(8, np.uint64)
    for blk in range(B):
        V[:, :] = 0
        for j in range(C):
            a, b = start[j], start[j + 1]
            bj = budget[j]
            w = 0
            while (1 << w) <= bj:
                w += 1
            for q in range(w):
                P[q] = 0
            ovf = np.uint64(0)
            for t in range(a, b):
                x = XT[idx[t], blk]
                f = x if neg[t] == 1 else ~x          # miss: plain literal on 0, negated on 1
                for q in range(w):
                    c = P[q] & f
                    P[q] ^= f
                    f = c
                    if f == 0:
                        break
                ovf |= f
                if ovf == ~np.uint64(0):
                    break
            if ovf == ~np.uint64(0):
                continue
            for k in range(64):
                if (ovf >> np.uint64(k)) & np.uint64(1):
                    continue
                m = 0
                for q in range(w):
                    m |= int((P[q] >> np.uint64(k)) & np.uint64(1)) << q
                sc = bj - m
                if sc > 0:
                    V[k, cls[j]] += pol[j] * sc
        for s in range(64):
            i = blk * 64 + s
            if i >= N:
                break
            best = 0
            for c in range(1, n_classes):
                if V[s, c] > V[s, best]:
                    best = c
            out[i] = best
    return out


def sparse_model(M, strict):
    model, LF = M["model"], (M["LF"] or 1)
    idx, neg, start, budget, cls, pol = [], [], [0], [], [], []
    for cl in model:
        for f in cl["lits"]:
            idx.append(f); neg.append(0)
        for f in cl["lits_neg"]:
            idx.append(f); neg.append(1)
        start.append(len(idx))
        n = len(cl["lits"]) + len(cl["lits_neg"])
        budget.append(1 if strict else (n if 0 < n < LF else LF))
        cls.append(cl["cls"]); pol.append(cl["polarity"])
    A = lambda x, t: np.array(x, dtype=t)
    return (A(idx, np.int64), A(neg, np.uint8), A(start, np.int64), A(budget, np.int64),
            A(cls, np.int64), A(pol, np.int64))


def slice_inputs(X):
    """Transpose to (F, ceil(N/64)) words, bit s of word b = sample 64*b + s."""
    N, F = X.shape
    B = (N + 63) // 64
    P = np.zeros((B * 64, F), np.uint8)
    P[:N] = X
    P = P.reshape(B, 64, F).transpose(2, 0, 1)            # F, B, 64
    return np.ascontiguousarray(np.packbits(P, axis=2, bitorder="little").view(np.uint64)[:, :, 0])


def pack(X):
    """Pack Boolean rows into little-endian 64-bit words."""
    N, F = X.shape
    W = (F + 63) // 64
    P = np.zeros((N, W * 64), np.uint8)
    P[:, :F] = X
    b = np.packbits(P, axis=1, bitorder="little")
    return np.ascontiguousarray(b.view(np.uint64))


def compile_model(M, strict):
    model, LF, F = M["model"], (M["LF"] or 1), M["n_features"]
    W = (F + 63) // 64
    C = len(model)
    pos = np.zeros((C, W * 64), np.uint8); neg = np.zeros((C, W * 64), np.uint8)
    budget = np.zeros(C, np.int64); nlit = np.zeros(C, np.int64)
    cls = np.zeros(C, np.int64); pol = np.zeros(C, np.int64)
    for j, cl in enumerate(model):
        pos[j, cl["lits"]] = 1; neg[j, cl["lits_neg"]] = 1
        n = len(cl["lits"]) + len(cl["lits_neg"])
        nlit[j] = n
        budget[j] = 1 if strict else (n if 0 < n < LF else LF)
        cls[j] = cl["cls"]; pol[j] = cl["polarity"]
    pk = lambda A: np.ascontiguousarray(np.packbits(A, axis=1, bitorder="little").view(np.uint64))
    return pk(pos), pk(neg), budget, nlit, cls, pol


def _time(fn):
    times = []
    for _ in range(5):
        reps, t0 = 0, time.perf_counter()
        while True:
            fn()
            reps += 1
            if time.perf_counter() - t0 >= MIN_TIME:
                break
        times.append((time.perf_counter() - t0) / reps)
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def bench(path, data):
    M = pickle.load(open(path, "rb"))
    S = json.load(open(os.path.join(os.path.dirname(path), "summary.json")))
    strict = bool(M.get("strict", False))
    unit = strict or (M["LF"] or 1) == 1          # every live clause is a strict conjunction
    X = np.ascontiguousarray(data["Xte"].astype(np.uint8))
    N, K = X.shape[0], M["n_classes"]
    ref = np.argmax(strict_votes(M["model"], X, K) if strict
                    else reference_votes(M["model"], X, K, M["LF"] or 1), 1)
    pos, neg, budget, nlit, cls, pol = compile_model(M, strict)
    sidx, sneg, sstart, sbud, scls, spol = sparse_model(M, strict)
    runs = {
        "dense": lambda: _predict(pack(X), pos, neg, budget, strict, nlit, cls, pol, K),
        "sparse": lambda: _predict_sparse(X, sidx, sneg, sstart, sbud, strict, scls, spol, K),
    }
    if not unit:
        runs["sliced"] = lambda: _predict_sliced_fuzzy(slice_inputs(X), N, sidx, sneg, sstart, sbud, scls, spol, K)
    if unit:
        # an empty LF=1 clause keeps the all-ones mask and scores 1; vanilla empty clauses are skipped
        runs["sliced"] = lambda: _predict_sliced(slice_inputs(X), N, sidx, sneg, sstart, strict, scls, spol, K)
    res = {}
    for name, fn in runs.items():                  # pilot pass: correctness and a first time
        t0 = time.perf_counter()
        match = float((fn() == ref).mean())
        res[name] = dict(match=match, us_pilot=1e6 * (time.perf_counter() - t0) / N)
    for name, fn in runs.items():                  # second pilot pass after JIT warm-up
        t0 = time.perf_counter(); fn()
        res[name]["us_pilot"] = 1e6 * (time.perf_counter() - t0) / N
    best = min(res, key=lambda k: res[k]["us_pilot"])
    for name in runs:                              # full timing for the fastest kernel and any within 1.5x of it
        if res[name]["us_pilot"] <= 1.5 * res[best]["us_pilot"]:
            med, lo, hi = _time(runs[name])
            res[name].update(us=1e6 * med / N, us_min=1e6 * lo / N, us_max=1e6 * hi / N)
    best = min((k for k in res if "us" in res[k]), key=lambda k: res[k]["us"])
    return dict(dataset=S["dataset"], tag=S["tag"], seed=int(S.get("seed", 0)),
                clauses_total=len(M["model"]), n_test=int(N), kernels=res, best_kernel=best,
                cpu_match=min(r["match"] for r in res.values()),
                cpu_us_per_sample=res[best]["us"], cpu_us_min=res[best]["us_min"], cpu_us_max=res[best]["us_max"])


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = sys.argv[2] if len(sys.argv) > 2 else f"{root}/cpu_bench.json"
    rows, cache = [], {}
    for path in sorted(glob.glob(f"{root}/*/*/model.pkl")):
        ds = path.split("/")[-3]
        if ds not in cache:
            cache.clear(); cache[ds] = load(ds)
        r = bench(path, cache[ds])
        rows.append(r)
        print(r["dataset"], r["tag"], r["seed"], "match", r["cpu_match"], "best", r["best_kernel"],
              " ".join("%s %.2f" % (k, v.get("us", v["us_pilot"])) for k, v in r["kernels"].items()), flush=True)
    json.dump(dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
                   threads=1, numba=nb.__version__, numpy=np.__version__, rows=rows),
              open(out, "w"), indent=1)
