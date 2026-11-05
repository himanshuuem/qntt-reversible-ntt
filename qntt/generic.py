"""
qntt.generic  --  fully parametric generator  (word width n, prime q, transform size N)
=====================================================================================
GBuilder(n, q, v1=False) builds the library-composed NTT multiplier for ANY
    word width n,   modulus q with 2^(n-1) < q < 2^n,   N a power of two with N | (q-1)
using only X / CNOT / Toffoli.  v1=False is the faithful reuse-only design (V0: every multiplication goes
through the generic Multiplier block);  v1=True is the constant-aware design (V1).
For (n=4, q=13) it reproduces the original builders gate-for-gate (21,240 / 6,545 gates) -- see tests.

Preconditions (derived, and re-checked by the exhaustive macro tests):
  * 2^n < 2q  =>  c = 2^n mod q = 2^n - q,  so MODADD's corrected sum V = S + c < 2^n  and MODSUB's
    correction (2^n - q) is a single subtraction  (the block-level design's 'q > 8' condition is q > 2^(n-1)).
  * r_fold (number of MODMUL fold rounds) is found by exhaustive search over the functional model.
"""
import random
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def find_rfold(n, q, limit=40):
    c, mask = (1 << n) % q, (1 << n) - 1
    for r in range(1, limit):
        ok = True
        for x in range(q):
            for y in range(q):
                p = x * y; hi, lo = p >> n, p & mask
                for _ in range(r):
                    pc = hi * c; s = lo + (pc & mask)
                    lo, hi = s & mask, (pc >> n) + (s >> n)
                if lo % q != (x * y) % q: ok = False; break
            if not ok: break
        if ok: return r
    raise ValueError("no r_fold found")


def root_of_unity(q, N):
    """smallest primitive N-th root of unity mod q (N a power of two >= 2): w^(N/2) = -1."""
    for w in range(2, q):
        if pow(w, N // 2, q) == q - 1: return w
    raise ValueError("no primitive root")


class GBuilder:
    def __init__(self, n, q, v1=False, named=False, flags=None, rfold=None):
        assert (1 << (n - 1)) < q < (1 << n), "need 2^(n-1) < q < 2^n"
        self.n, self.q, self.v1 = n, q, v1
        self.c, self.negq = (1 << n) % q, (1 << n) - q
        self.rfold = rfold or find_rfold(n, q)
        self.f = dict(identity=True, lookup_twiddle=True, cnot_selectors=True, lookup_fold=True)
        if flags: self.f.update(flags)
        self.qc = QuantumCircuit(name=f"qntt_n{n}_q{q}")
        self.nq = 0; self.named = named; self._uid = 0
        self.const_init, self.consts = [], {}
        self.inst, self.gates, self._stack = Counter(), {}, []
        self.zeros = self.new(n, "zero")

    # ------------------------------------------------------------ allocation / ledger
    def new(self, k, label="anc"):
        if self.named:
            from qiskit.circuit import QuantumRegister
            self._uid += 1; reg = QuantumRegister(k, f"{label}{self._uid}")
            self.qc.add_register(reg); qs = list(reg)
        else:
            qs = [Qubit() for _ in range(k)]; self.qc.add_bits(qs)
        self.nq += k
        return qs

    def const(self, value, label=None):
        if value in self.consts: return self.consts[value]
        reg = self.new(self.n, label or f"const{value}")
        for k in range(self.n):
            if (value >> k) & 1: self.const_init.append(self.qc.find_bit(reg[k]).index)
        self.consts[value] = reg
        return reg

    def zext(self, bit): return [bit] + self.zeros[: self.n - 1]

    def _begin(self, name): self._stack.append((name, len(self.qc.data)))

    def _end(self):
        name, start = self._stack.pop()
        g = self.gates.setdefault(name, Counter())
        for inst in self.qc.data[start:]: g[inst.operation.name] += 1
        self.inst[name] += 1

    # ------------------------------------------------------------ Cuccaro primitives
    def _ops(self, a, b, c0, cout=None):
        n, ops = len(a), []
        def MAJ(c, bb, aa): ops.extend([("cx", aa, bb), ("cx", aa, c), ("ccx", c, bb, aa)])
        def UMA(c, bb, aa): ops.extend([("ccx", c, bb, aa), ("cx", aa, c), ("cx", c, bb)])
        MAJ(c0, b[0], a[0])
        for i in range(1, n): MAJ(a[i - 1], b[i], a[i])
        if cout is not None: ops.append(("cx", a[n - 1], cout))
        for i in range(n - 1, 0, -1): UMA(a[i - 1], b[i], a[i])
        UMA(c0, b[0], a[0])
        return ops

    def _emit(self, ops):
        for g, *qs in ops: getattr(self.qc, g)(*qs)

    def _add(self, a, b, c0, cout=None): self._emit(self._ops(a, b, c0, cout))
    def _sub(self, a, b, c0): self._emit(list(reversed(self._ops(a, b, c0, None))))

    # ------------------------------------------------------------ the four blocks
    def adder(self, A, B):
        n = self.n; self._begin("Adder")
        S, Cout, c0 = self.new(n, "S"), self.new(1, "Cout")[0], self.new(1, "c0")[0]
        for k in range(n): self.qc.cx(B[k], S[k])
        self._add(A, S, c0, Cout); self._end()
        return S, Cout

    def subtractor(self, A, B):
        n = self.n; self._begin("Subtractor")
        D, c0, pad = self.new(n + 1, "D"), self.new(1, "c0")[0], self.new(1, "zpad")[0]
        for k in range(n): self.qc.cx(A[k], D[k])
        self._sub(list(B) + [pad], D, c0); self._end()
        return D[:n], D[n]

    def multiplier(self, A, B):
        n = self.n; self._begin("Multiplier")
        P, c0, pp = self.new(2 * n, "P"), self.new(1, "c0")[0], self.new(n * (n - 1), "pp")
        for j in range(n): self.qc.ccx(A[0], B[j], P[j])
        for i in range(1, n):
            row = pp[n * (i - 1): n * i]
            for j in range(n): self.qc.ccx(A[i], B[j], row[j])
            self._add(row, P[i:i + n], c0, P[i + n])
        self._end()
        return P

    def modulo(self, V, Qr):
        n = self.n; self._begin("Modulo")
        zs, c0 = self.new(n, "zs"), self.new(1, "c0")[0]
        qext, r_prev = list(Qr) + [zs[0], zs[1]], zs
        for i in range(n - 1, -1, -1):
            rn = [V[i]] + list(r_prev)
            t = self.new(n + 2, "t")
            for k in range(n + 1): self.qc.cx(rn[k], t[k])
            self._sub(qext, t, c0)
            out = self.new(n, "r")
            for k in range(n):
                self.qc.cx(t[k], out[k]); self.qc.cx(rn[k], t[k]); self.qc.ccx(t[n + 1], t[k], out[k])
            r_prev = out
        self._end()
        return r_prev

    # ------------------------------------------------------------ constant-aware primitives (V1)
    @staticmethod
    def _anf(f, n):
        size = 1 << n
        dcs = [i for i, v in enumerate(f) if v is None]
        def cost_terms(t):
            a = t[:]
            for i in range(n):
                bit = 1 << i
                for m in range(size):
                    if m & bit: a[m] ^= a[m ^ bit]
            terms = [m for m in range(size) if a[m]]
            return (sum(bin(m).count("1") >= 2 for m in terms), len(terms)), terms
        base = [0 if v is None else v for v in f]
        if len(dcs) <= 12:
            best = None
            for assign in range(1 << len(dcs)):
                t = base[:]
                for k, i in enumerate(dcs): t[i] = (assign >> k) & 1
                cst, terms = cost_terms(t)
                if best is None or cst < best[0]: best = (cst, terms)
            return best[1]
        rng = random.Random(1234)                                   # hill-climb for many don't-cares
        best = None
        for restart in range(6):
            t = base[:]
            for i in dcs: t[i] = rng.randrange(2) if restart else 0
            cst, terms = cost_terms(t)
            improved = True
            while improved:
                improved = False
                for i in dcs:
                    t[i] ^= 1
                    c2, tm2 = cost_terms(t)
                    if c2 < cst: cst, terms, improved = c2, tm2, True
                    else: t[i] ^= 1
            if best is None or cst < best[0]: best = (cst, terms)
        return best[1]

    _ANF_CACHE = {}

    def lookup(self, y, func, nout, name="ConstMul", key=None):
        n = self.n; self._begin(name)
        ck = (n, nout, key) if key is not None else None
        if ck in GBuilder._ANF_CACHE: terms = GBuilder._ANF_CACHE[ck]
        else:
            terms = [self._anf([None if func(v) is None else (func(v) >> k) & 1 for v in range(1 << n)], n) for k in range(nout)]
            if ck is not None: GBuilder._ANF_CACHE[ck] = terms
        out, mono = self.new(nout, "lk"), {}
        def get(m):
            if bin(m).count("1") == 1: return y[m.bit_length() - 1]
            if m in mono: return mono[m]
            low = m & -m; a = get(m ^ low); q_ = self.new(1, "mono")[0]
            self.qc.ccx(a, y[low.bit_length() - 1], q_); mono[m] = q_
            return q_
        for k, ts in enumerate(terms):
            for m in ts:
                if m == 0: self.qc.x(out[k])
                else: self.qc.cx(get(m), out[k])
        self._end()
        return out

    def sel_const(self, bit, cval):
        self._begin("CondConst"); out, zi = [], 0
        for k in range(self.n):
            if (cval >> k) & 1:
                q_ = self.new(1, "sel")[0]; self.qc.cx(bit, q_); out.append(q_)
            else: out.append(self.zeros[zi]); zi += 1
        self._end()
        return out

    def mul_hi_c(self, hi):
        nout = (((1 << self.n) - 1) * self.c).bit_length()
        return self.lookup(hi, lambda v: v * self.c, nout, key=("hc", self.q)) + self.new(2 * self.n - nout, "z")

    # ------------------------------------------------------------ macros
    def modadd(self, x, t):
        S, cout = self.adder(x, t)
        corr = (self.sel_const(cout, self.c) if (self.v1 and self.f["cnot_selectors"])
                else self.multiplier(self.zext(cout), self.const(self.c, "c"))[: self.n])
        V, _ = self.adder(S, corr)
        return self.modulo(V, self.const(self.q, "q"))

    def modsub(self, x, t):
        D, bw = self.subtractor(x, t)
        corr = (self.sel_const(bw, self.negq) if (self.v1 and self.f["cnot_selectors"])
                else self.multiplier(self.zext(bw), self.const(self.negq, "2^n-q"))[: self.n])
        R, _ = self.subtractor(D, corr)
        return R

    def modmul(self, x, y):
        n = self.n
        P = self.multiplier(x, y); lo, hi = P[:n], P[n:]
        for _ in range(self.rfold):
            P2 = self.mul_hi_c(hi) if (self.v1 and self.f["lookup_fold"]) else self.multiplier(hi, self.const(self.c, "c"))
            clo, chi = P2[:n], P2[n:]
            lo2, carry = self.adder(lo, clo)
            hi2, _ = self.adder(chi, self.zext(carry))
            lo, hi = lo2, hi2
        return self.modulo(lo, self.const(self.q, "q"))

    def modmul_const(self, y, w, side="wy"):
        """(w*y) mod q, w a compile-time constant."""
        if self.v1 and w == 1 and self.f["identity"]: return y
        if self.v1 and self.f["lookup_twiddle"]:
            return self.lookup(y, lambda v: (w * v) % self.q if v < self.q else None, self.n, key=("mc", self.q, w))
        return self.modmul(self.const(w, f"w{w}"), y) if side == "wy" else self.modmul(y, self.const(w, f"w{w}"))

    def bu(self, x, y, w):
        t = self.modmul_const(y, w, "wy")
        return self.modadd(x, t), self.modsub(x, t)

    # ------------------------------------------------------------ transforms / pipeline
    @staticmethod
    def _bitrev(i, bits): return int(format(i, f"0{bits}b")[::-1], 2) if bits else 0

    def ntt(self, vec, omega):
        N = len(vec); L = N.bit_length() - 1
        a = [vec[self._bitrev(i, L)] for i in range(N)]
        blk = 2
        while blk <= N:
            half, step = blk // 2, N // blk
            for start in range(0, N, blk):
                for j in range(half):
                    u, v = self.bu(a[start + j], a[start + j + half], pow(omega, j * step, self.q))
                    a[start + j], a[start + j + half] = u, v
            blk *= 2
        return a

    def inputs(self, N, count=2):
        return [[self.new(self.n, f"in{r}_{i}") for i in range(N)] for r in range(count)]

    def pipeline(self, N):
        """c(x) = a(x)*b(x) mod (x^N - 1), coefficients mod q."""
        (a, b), omega = self.inputs(N), root_of_unity(self.q, N)
        winv, ninv = pow(omega, -1, self.q), pow(N, -1, self.q)
        Na, Nb = self.ntt(a, omega), self.ntt(b, omega)
        P = [self.modmul(Na[k], Nb[k]) for k in range(N)]
        I = self.ntt(P, winv)
        c = [self.modmul_const(I[k], ninv, "yw") for k in range(N)]
        return a, b, c

    def ntt_only(self, N):
        (a,), omega = self.inputs(N, 1), root_of_unity(self.q, N)
        return a, self.ntt(a, omega)

    # ------------------------------------------------------------ reporting
    def summary(self):
        tot = Counter()
        for g in self.gates.values(): tot.update(g)
        return {"n": self.n, "q": self.q, "v1": self.v1, "rfold": self.rfold, "qubits": self.nq,
                "gates": len(self.qc.data), "ccx": tot["ccx"], "cx": tot["cx"], "x": tot["x"],
                "t_count": 7 * tot["ccx"], "instances": dict(self.inst),
                "block_gates": {k: sum(v.values()) for k, v in self.gates.items()}}
