"""
Functional and cycle-level model of FPTM / strict-TM inference on GainP.

GainP primitives (Levi, Harel, Teman, Yavits, "GainP: A Gain Cell Embedded
DRAM-Based Associative In-Memory Processor"):
  COMPARE  one cycle. A query pattern is applied on the searchlines of the
           selected rows. All other rows are masked (both searchlines high).
           Each column senses match/mismatch. The result is OR-accumulated
           in the per-column tag register/accumulator.
  WRITE    one cycle. The tag register is written into one CAM row of every
           column (nominal value to the top gain cell, inverted value to the
           bottom gain cell). The tag register is reset after a write.
  A Boolean function of a few rows costs one COMPARE per cube of its
  sum-of-products form plus one WRITE. Full-adder carry is 3 cubes + write
  (4 cycles) and full-adder sum is 4 cubes + write (5 cycles), i.e. 9 cycles
  per bit, matching the GainP vector-addition cost n x (5 + 4).

Mapping S (sample-stationary): each logical GainP column holds one input
sample (F Boolean input rows) plus working rows. The trained model is
compiled into a data-independent COMPARE/WRITE program that is broadcast to
all columns. The program is executed on real test samples, so the in-array
classification can be checked bit-exactly against the software model.
"""
import numpy as np

K0, K1 = "K0", "K1"   # constant operands


def _norm(cubes):
    """Normalise cubes with constant operands. Returns a list of cubes over rows
    only, dropping infeasible cubes. A cube with no literals matches all columns."""
    out = []
    for c in cubes:
        lits, ok = [], True
        for opnd, v in c:
            if opnd == K0 or opnd == K1:
                if (opnd == K1) != bool(v):
                    ok = False
                    break
            else:
                lits.append((opnd, v))
        if ok:
            out.append(lits)
    # a cube with no literals is always true: the function is constant 1
    if any(len(c) == 0 for c in out):
        return [[]]
    return out


class Prog:
    """Records (and optionally executes) a GainP COMPARE/WRITE program."""

    def __init__(self, n_inputs, X=None):
        self.n_in = n_inputs
        self.rows = {}
        self.next = n_inputs
        self.cycles = self.n_cmp = self.n_wr = self.n_rd = 0
        self.act = 0              # sum over cycles of active rows (per column)
        self.exec = X is not None
        if self.exec:
            self.M = np.zeros((n_inputs + 256, X.shape[0]), dtype=bool)
            self.M[:n_inputs] = X.T.astype(bool)
            self.tag = np.zeros(X.shape[0], dtype=bool)

    def row(self, name):
        if name not in self.rows:
            self.rows[name] = self.next
            self.next += 1
            if self.exec and self.next > self.M.shape[0]:
                self.M = np.vstack([self.M, np.zeros((256, self.M.shape[1]), bool)])
        return self.rows[name]

    @property
    def n_rows(self):
        return self.next

    # ---- primitives ------------------------------------------------------
    def compare(self, cube):
        self.cycles += 1; self.n_cmp += 1; self.act += len(cube)
        if self.exec:
            m = np.ones(self.tag.shape, dtype=bool)
            for r, v in cube:
                m &= (self.M[r] == bool(v))
            self.tag |= m

    def write(self, r):
        self.cycles += 1; self.n_wr += 1; self.act += 1
        if self.exec:
            self.M[r] = self.tag
            self.tag[:] = False

    def read(self, r):
        self.cycles += 1; self.n_rd += 1; self.act += 1
        return self.M[r].copy() if self.exec else None

    # ---- composite ---------------------------------------------------------
    def func(self, cubes, r):
        """Row r := OR of cubes (cubes may contain constant operands)."""
        cs = _norm(cubes)
        if cs == [[]]:
            self.compare([]); self.write(r)          # constant 1
            return
        for c in cs:
            self.compare(c)
        self.write(r)                                 # zero cubes -> writes 0


# ----------------------------------------------------------------------------
# Arithmetic on row vectors (LSB first). Missing high bits are constant 0.
# ----------------------------------------------------------------------------
def _bit(V, j):
    return V[j] if j < len(V) else K0


def add_into(p, A, S, carry, width):
    """A[0:width] += S (unsigned). The carry out of bit width-1 is dropped,
    so width must be large enough for the result."""
    c = K0
    for j in range(width):
        a, s = A[j], _bit(S, j)
        if s == K0 and c == K0:
            break                                     # nothing to propagate
        last = (j == width - 1)
        co = carry[j % 2]
        if not last:
            p.func([[(a, 1), (s, 1)], [(a, 1), (c, 1)], [(s, 1), (c, 1)]], co)
        p.func([[(a, 1), (s, 0), (c, 0)], [(a, 0), (s, 1), (c, 0)],
                [(a, 0), (s, 0), (c, 1)], [(a, 1), (s, 1), (c, 1)]], a)
        c = K0 if last else co


def sub_to(p, X, Y, D, borrow, width):
    """D[0:width] = X - Y in two's complement (X, Y unsigned, zero-extended)."""
    b = K0
    for j in range(width):
        x, y = _bit(X, j), _bit(Y, j)
        last = (j == width - 1)
        bo = borrow[j % 2]
        if not last:
            p.func([[(x, 0), (y, 1)], [(x, 0), (b, 1)], [(y, 1), (b, 1)]], bo)
        p.func([[(x, 1), (y, 0), (b, 0)], [(x, 0), (y, 1), (b, 0)],
                [(x, 0), (y, 0), (b, 1)], [(x, 1), (y, 1), (b, 1)]], D[j])
        b = K0 if last else bo


def signed_gt(p, V, B, out, borrow, width):
    """out := (V > B), signed two's complement, width bits.
    Unsigned compare on sign-flipped operands: V > B  <=>  borrow of (B' - V')."""
    b = K0
    for j in range(width):
        flip = (j == width - 1)
        vb, bb = (0, 1) if flip else (1, 0)          # literal polarity for V'=1 / B'=0
        v1 = (V[j], vb)          # V'_j = 1
        b0 = (B[j], bb)          # B'_j = 0
        dst = out if j == width - 1 else borrow[j % 2]
        # borrow_out of (B' - V') = (~B' & V') | (~B' & b) | (V' & b)
        p.func([[b0, v1], [b0, (b, 1)], [v1, (b, 1)]], dst)
        b = dst


def bitlen(x):
    return max(1, int(x).bit_length())


# ----------------------------------------------------------------------------
# Mapping S compiler
# ----------------------------------------------------------------------------
def normalize_clause(m, LF, strict_semantics):
    """Exact compile-time simplification of one clause.
    Returns (req, b, const): the clause score equals const + S, where
    S = max(0, b - misses over req) (b >= 1), or S = 0 when req is None.
    req is a list of (feature_row, required_value).
      * A feature included both plainly and negated misses exactly once for
        every input. FPTM: it lowers the budget by one. Strict TM: the clause
        never fires.
      * FPTM with b > n' (budget above the number of remaining literals):
        the score never clamps, so b - n' is a constant.
      * FPTM empty clause: constant LF (FuzzyPatternTM.jl check_clause).
    """
    pos, neg = set(int(i) for i in m["lits"]), set(int(i) for i in m["lits_neg"])
    conf = pos & neg
    n = len(pos) + len(neg)
    req = [(i, 1) for i in sorted(pos - conf)] + [(i, 0) for i in sorted(neg - conf)]
    if strict_semantics:
        if n == 0 or conf:
            return None, 0, 0
        return req, 1, 0
    if n == 0:
        return None, 0, LF
    b = min(n, LF) - len(conf)
    if b <= 0:
        return None, 0, 0
    const = 0
    if b > len(req):
        const, b = b - len(req), len(req)
    if b == 0:
        return None, 0, const
    return req, b, const


def compile_sample_stationary(p, model, n_classes, LF, strict_semantics):
    """Emit the inference program for an exported model.
    model: list of dicts(cls, polarity, lits, lits_neg).
    strict_semantics: True for a vanilla TM (clause = AND of literals, empty
    clause outputs 0 at inference). False for FPTM (score max(0, min(n, LF) -
    misses), empty clause scores LF).
    Returns the rows holding the predicted class index (LSB first)."""
    norm = [[(m["polarity"],) + normalize_clause(m, LF, strict_semantics)
             for m in model if m["cls"] == c] for c in range(n_classes)]
    pos_max, neg_max = [], []
    for cl in norm:
        pos_max.append(sum(b + k for pol, r, b, k in cl if pol > 0))
        neg_max.append(sum(b + k for pol, r, b, k in cl if pol < 0))
    wP = bitlen(max(pos_max)); wN = bitlen(max(neg_max))
    w = max(wP, wN) + 1
    ci = bitlen(n_classes - 1)
    Prow = [p.row(f"P{j}") for j in range(wP)]
    Nrow = [p.row(f"N{j}") for j in range(wN)]
    Vrow = [p.row(f"V{j}") for j in range(w)]
    Brow = [p.row(f"B{j}") for j in range(w)]
    Irow = [p.row(f"I{j}") for j in range(ci)]
    carry = [p.row("c0"), p.row("c1")]
    gt = p.row("gt")
    Q = p.row("Q")
    for c in range(n_classes):
        kP = sum(k for pol, r, b, k in norm[c] if pol > 0)
        kN = sum(k for pol, r, b, k in norm[c] if pol < 0)
        for j, r in enumerate(Prow):
            p.func([[]] if (kP >> j) & 1 else [], r)
        for j, r in enumerate(Nrow):
            p.func([[]] if (kN >> j) & 1 else [], r)
        runP, runN = kP, kN
        for pol, req, b, k in norm[c]:
            if req is None:
                continue
            if b == 1:
                p.compare(req); p.write(Q)          # strict AND in one COMPARE
                S = [Q]
            else:
                S = fuzzy_clause_score(p, req, b)
            if pol > 0:
                runP += b; add_into(p, Prow, S, carry, bitlen(runP))
            else:
                runN += b; add_into(p, Nrow, S, carry, bitlen(runN))
        sub_to(p, Prow, Nrow, Vrow, carry, w)
        if c == 0:
            for j in range(w):
                p.func([[(Vrow[j], 1)]], Brow[j])
            for j in range(ci):
                p.func([], Irow[j])
        else:
            signed_gt(p, Vrow, Brow, gt, carry, w)
            for j in range(w):
                p.func([[(gt, 1), (Vrow[j], 1)], [(gt, 0), (Brow[j], 1)]], Brow[j])
            for j in range(ci):
                cubes = [[(gt, 0), (Irow[j], 1)]]
                if (c >> j) & 1:
                    cubes.append([(gt, 1)])
                p.func(cubes, Irow[j])
    return Irow


def fuzzy_clause_score(p, req, b):
    """FPTM clause score max(0, b - misses) over literals `req`
    (list of (input_row, required_value)), b = min(n, LF) >= 2.
    Two exact implementations; the cheaper one (known statically) is emitted."""
    ct = Prog(0); _score_thermometer(ct, [(0, 1)] * len(req), b)
    cb = Prog(0); _score_binary(cb, [(0, 1)] * len(req), b)
    if ct.cycles <= cb.cycles:
        return _score_thermometer(p, req, b)
    return _score_binary(p, req, b)


def _therm_cost(n, b):
    kb = bitlen(b)
    conv = sum(sum(1 for m in range(b + 1) if ((b - m) >> j) & 1) + 1 for j in range(kb))
    return b + sum(3 * min(b, t) for t in range(1, n + 1)) + conv


def _bin_cost(n, b):
    wm = bitlen(n)
    inc = sum(5 * bitlen(t) - 2 for t in range(1, n + 1))
    return wm + inc + 4 * (wm - 1) + 3 * wm + 2 * bitlen(b) + 2


def _score_thermometer(p, req, b):
    """Saturating thermometer miss counter T_r = [misses >= r], r = 1..b."""
    T = [None] + [p.row(f"T{r}") for r in range(1, b + 1)]
    for r in range(1, b + 1):
        p.func([], T[r])                                        # clear
    for t, (i, v) in enumerate(req, start=1):
        u = 1 - v                                               # miss value
        for r in range(min(b, t), 1, -1):
            p.func([[(T[r], 1)], [(T[r - 1], 1), (i, u)]], T[r])
        p.func([[(T[1], 1)], [(i, u)]], T[1])
    # score s = b - m', m' = max{r : T_r = 1}; binary encode
    kb = bitlen(b)
    S = [p.row(f"s{j}") for j in range(kb)]
    for j in range(kb):
        cubes = []
        for m in range(b + 1):
            if ((b - m) >> j) & 1:
                cube = []
                if m >= 1:
                    cube.append((T[m], 1))
                if m + 1 <= b:
                    cube.append((T[m + 1], 0))
                cubes.append(cube)
        p.func(cubes, S[j])
    return S


def _score_binary(p, req, b):
    """Binary miss counter, then score = max(0, b - m) with clamp."""
    n = len(req)
    wm = bitlen(n)
    Mr = [p.row(f"m{j}") for j in range(wm)]
    cr = [p.row("bc0"), p.row("bc1")]
    for r in Mr:
        p.func([], r)
    for t, (i, v) in enumerate(req, start=1):
        u = 1 - v
        a = bitlen(t)
        c = (i, u)                                              # carry-in literal = miss
        for j in range(a):
            last = (j == a - 1)
            if not last:
                p.func([[(Mr[j], 1), c]], cr[j % 2])
            p.func([[(Mr[j], 1), (c[0], 1 - c[1])], [(Mr[j], 0), c]], Mr[j])
            c = (cr[j % 2], 1)
    # D = b - m (two's complement over wm+1 bits), neg = borrow out
    D = [p.row(f"d{j}") for j in range(wm)]
    bconst = [K1 if (b >> j) & 1 else K0 for j in range(wm)]
    br = K0
    neg = p.row("neg")
    for j in range(wm):
        x, y = bconst[j], Mr[j]
        bo = neg if j == wm - 1 else cr[j % 2]
        p.func([[(x, 0), (y, 1)], [(x, 0), (br, 1)], [(y, 1), (br, 1)]], bo)
        p.func([[(x, 1), (y, 0), (br, 0)], [(x, 0), (y, 1), (br, 0)],
                [(x, 0), (y, 0), (br, 1)], [(x, 1), (y, 1), (br, 1)]], D[j])
        br = bo
    kb = bitlen(b)
    S = [p.row(f"s{j}") for j in range(kb)]
    for j in range(kb):
        p.func([[(D[j], 1), (neg, 0)]], S[j])
    return S


def run_sample_stationary(model, n_classes, LF, strict, n_features, X=None):
    p = Prog(n_features, X)
    I = compile_sample_stationary(p, model, n_classes, LF, strict)
    pred = None
    if p.exec:
        pred = np.zeros(X.shape[0], dtype=np.int64)
        for j, r in enumerate(I):
            pred |= p.read(r).astype(np.int64) << j
    else:
        for r in I:
            p.read(r)
    return p, pred
