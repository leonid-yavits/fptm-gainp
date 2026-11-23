"""
Mapping C (clause-stationary) on GainP: program generator and functional
simulator.

Each GainP column holds one live clause (after normalize_clause) as a ternary
pattern over the used input features (TCAM storage):
  required value 1  -> store 1
  required value 0  -> store 0
  feature not used  -> store "don't care" (both gain cells = 1), never mismatches
Conflicting literal pairs are removed at compile time and lower the clause
budget (normalize_clause), so no extra rows are needed.
Each column also stores its budget b_col in bitlen(LF) binary rows.

The sample drives the searchlines. No input data is written into the array.

Strict model (every live clause has b = 1): one COMPARE over all used feature
rows, one WRITE of the match bit, one READ of that row.
FPTM model: per used feature, one COMPARE on that row and one WRITE of the
match bit M (M = 0 means miss), then a saturating thermometer update
T_r <- T_r | (T_{r-1} & ~M), r = min(LF, t) .. 1. After the last feature, the
thermometer is converted to a binary miss count m, the column computes
s = max(0, b_col - m) with a borrow chain, and the score rows are READ.

The program is executed by the simulator below with the same COMPARE / WRITE /
READ semantics as gainp_sim.Prog, plus ternary storage. Cycle and active-row
counts are taken from the executed program.

The cross-column vote reduction (signed sum of clause scores per class) is not
a GainP primitive. It is done outside the array, here in NumPy, and is
reported as an operation count only.
"""
import numpy as np
from gainp_sim import bitlen, normalize_clause, _norm


class CProg:
    """COMPARE / WRITE / READ on a ternary array. Input rows are compared
    against the per-column query value (the sample bit of that column's
    sample block)."""

    def __init__(self, n_inputs, stored, dc, query_fn):
        self.n_in = n_inputs
        self.cycles = self.n_cmp = self.n_wr = self.n_rd = 0
        self.act = 0
        self.exec = stored is not None
        self.next = n_inputs
        self.rows = {}
        if self.exec:
            ncol = stored.shape[1]
            self.M = np.zeros((n_inputs + 128, ncol), dtype=bool)
            self.DC = np.zeros((n_inputs + 128, ncol), dtype=bool)
            self.M[:n_inputs] = stored
            self.DC[:n_inputs] = dc
            self.q = query_fn
            self.tag = np.zeros(ncol, dtype=bool)

    def row(self, name):
        if name not in self.rows:
            self.rows[name] = self.next
            self.next += 1
        return self.rows[name]

    @property
    def n_rows(self):
        return self.next

    def compare(self, cube):
        """cube: list of (row, value); value 'x' = searchline driven by the sample."""
        self.cycles += 1; self.n_cmp += 1; self.act += len(cube)
        if self.exec:
            m = np.ones(self.tag.shape, dtype=bool)
            for r, v in cube:
                qv = self.q(r) if v == "x" else bool(v)
                m &= (self.M[r] == qv) | self.DC[r]
            self.tag |= m

    def write(self, r):
        self.cycles += 1; self.n_wr += 1; self.act += 1
        if self.exec:
            self.M[r] = self.tag
            self.DC[r] = False
            self.tag[:] = False

    def read(self, r):
        self.cycles += 1; self.n_rd += 1; self.act += 1
        return self.M[r].copy() if self.exec else None

    def func(self, cubes, r):
        cs = _norm(cubes)
        if cs == [[]]:
            self.compare([]); self.write(r)
            return
        for c in cs:
            self.compare(c)
        self.write(r)


def _compile(model, n_classes, LF, strict_semantics):
    norm = [(m["cls"], m["polarity"]) + normalize_clause(m, LF, strict_semantics) for m in model]
    live = [t for t in norm if t[2] is not None]
    used = sorted({i for t in live for i, v in t[2]})
    const = np.zeros(n_classes, dtype=np.int64)
    for cls, pol, req, b, k in norm:
        const[cls] += pol * k
    strict_prog = all(t[3] == 1 for t in live)
    return live, used, const, strict_prog


def run_clause_stationary(model, n_classes, LF, strict_semantics, X=None):
    """Generate and (if X is given) execute the Mapping C program.
    Returns (prog, pred, info). pred is None when X is None."""
    live, used, const, strict_prog = _compile(model, n_classes, LF, strict_semantics)
    Fu, nl = len(used), len(live)
    kb = bitlen(LF)
    stored = dc = qfn = None
    B = 0
    if X is not None and nl > 0:
        B = X.shape[0]
        T = np.zeros((Fu, nl), dtype=bool)
        D = np.ones((Fu, nl), dtype=bool)
        pos = {f: r for r, f in enumerate(used)}
        for j, (cls, pol, req, b, k) in enumerate(live):
            for i, v in req:
                T[pos[i], j] = bool(v); D[pos[i], j] = False
        stored = np.tile(T, (1, B)); dc = np.tile(D, (1, B))
        Xu = X[:, used].astype(bool)                       # (B, Fu)
        qfn = lambda r: np.repeat(Xu[:, r], nl)            # column block = one sample
    p = CProg(Fu, stored, dc, qfn)
    if strict_prog:
        Q = p.row("Q")
        p.compare([(r, "x") for r in range(Fu)])
        p.write(Q)
        rows_s = [Q]
    else:
        brow = [p.row(f"b{j}") for j in range(kb)]
        if p.exec:                                         # budgets stored offline
            bc = np.array([t[3] for t in live], dtype=np.int64)
            for j, r in enumerate(brow):
                p.M[r] = np.tile(((bc >> j) & 1).astype(bool), B)
        Mrow = p.row("M")
        Tr = [None] + [p.row(f"T{r}") for r in range(1, LF + 1)]
        for r in range(1, LF + 1):
            p.func([], Tr[r])
        for t, i in enumerate(range(Fu), start=1):
            p.compare([(i, "x")]); p.write(Mrow)
            for r in range(min(LF, t), 1, -1):
                p.func([[(Tr[r], 1)], [(Tr[r - 1], 1), (Mrow, 0)]], Tr[r])
            p.func([[(Tr[1], 1)], [(Mrow, 0)]], Tr[1])
        mrow = [p.row(f"m{j}") for j in range(kb)]
        for j in range(kb):
            cubes = []
            for m in range(1, LF + 1):
                if (m >> j) & 1:
                    c = [(Tr[m], 1)]
                    if m + 1 <= LF:
                        c.append((Tr[m + 1], 0))
                    cubes.append(c)
            p.func(cubes, mrow[j])
        # D = b_col - m, neg = borrow out; s = D & ~neg
        Drow = [p.row(f"D{j}") for j in range(kb)]
        cr = [p.row("c0"), p.row("c1")]
        neg = p.row("neg")
        br = "K0"
        for j in range(kb):
            x, y = brow[j], mrow[j]
            bo = neg if j == kb - 1 else cr[j % 2]
            p.func([[(x, 0), (y, 1)], [(x, 0), (br, 1)], [(y, 1), (br, 1)]], bo)
            p.func([[(x, 1), (y, 0), (br, 0)], [(x, 0), (y, 1), (br, 0)],
                    [(x, 0), (y, 0), (br, 1)], [(x, 1), (y, 1), (br, 1)]], Drow[j])
            br = bo
        rows_s = [p.row(f"s{j}") for j in range(kb)]
        for j in range(kb):
            p.func([[(Drow[j], 1), (neg, 0)]], rows_s[j])
    # read out the score rows
    S = np.zeros(nl * B, dtype=np.int64) if p.exec else None
    for j, r in enumerate(rows_s):
        v = p.read(r)
        if p.exec:
            S |= v.astype(np.int64) << j
    pred = None
    if X is not None:
        W = np.zeros((nl, n_classes), dtype=np.int64)
        for j, (cls, pol, req, b, k) in enumerate(live):
            W[j, cls] = pol
        if nl > 0:
            votes = S.reshape(B, nl) @ W + const[None, :]   # reduction, outside GainP
        else:
            votes = np.tile(const, (X.shape[0], 1))
        pred = np.argmax(votes, axis=1)
    info = dict(F_used=Fu, columns=nl, rows=p.n_rows, score_bits=len(rows_s),
                reduction_adds=nl, strict_program=strict_prog)
    return p, pred, info
