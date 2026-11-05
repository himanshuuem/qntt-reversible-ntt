"""
qntt.const_aware  --  constant-aware realisation (variant V1)
=============================================================
The reuse-only design (V0, builder.py) pushes EVERY multiplication through the generic 4x4 Multiplier
block. But in a 4-point NTT multiplier 16 of the 20 MODMUL calls (12 butterfly twiddles + 4 N^-1 scalings)
multiply by a COMPILE-TIME CONSTANT, and 9 of those twiddles are 1 (identity). V1 specialises every
multiplication that has a constant operand; variable x variable products (the 4 pointwise MODMULs) still use
the generic blocks. Techniques used (all standard in isolation - see design document Section VI-E for prior art):
  * identity twiddle (w = 1): t = y, no gates                                   (classical NTT 'trivial twiddle')
  * constant modular multiplier t = (w*y) mod q on a 4-bit residue: compile-time algebraic-normal-form
    (ANF) lookup network, Toffoli/CNOT only, exact, with don't-cares for y in [13,15]
  * constant selector bit*c (replaces the Multiplier in MODADD/MODSUB corrections): CNOT fan-out
  * hi*c in the MODMUL fold: ANF lookup instead of the generic Multiplier
V1 deliberately DROPS the 'no gate outside the four blocks' rule; V0 is the faithful reuse-only design.
"""
from .builder import Builder, Q, W, WINV, NINV, C_FOLD, NEGQ, R_FOLD


class BuilderCA(Builder):
    """Per-technique switches allow a cumulative ablation; all False reproduces V0 exactly, all True is V1."""

    def __init__(self, label="qntt", named=False, identity=True, lookup_twiddle=True,
                 cnot_selectors=True, lookup_fold=True):
        super().__init__(label, named)
        self.f = dict(identity=identity, lookup_twiddle=lookup_twiddle,
                      cnot_selectors=cnot_selectors, lookup_fold=lookup_fold)

    # ---------------------------------------------------------------- ANF lookup
    @staticmethod
    def _anf(f):
        """f: list of 16 values in {0,1,None}. Return the ANF term list (monomial masks) with the
        don't-care assignment that minimises (#nonlinear terms, #terms)."""
        dcs = [i for i, v in enumerate(f) if v is None]
        best = None
        for assign in range(1 << len(dcs)):
            t = [0 if v is None else v for v in f]
            for k, i in enumerate(dcs): t[i] = (assign >> k) & 1
            a = t[:]
            for i in range(4):
                for m in range(16):
                    if m >> i & 1: a[m] ^= a[m ^ (1 << i)]
            terms = [m for m in range(16) if a[m]]
            cost = (sum(bin(m).count("1") >= 2 for m in terms), len(terms))
            if best is None or cost < best[0]: best = (cost, terms)
        return best[1]

    def lookup(self, y, func, nout, name="ConstMul"):
        """out = func(y) for a 4-bit register y, func(v) in [0,2^nout) or None (don't care)."""
        self._begin(name)
        terms = [self._anf([None if func(v) is None else (func(v) >> k) & 1 for v in range(16)])
                 for k in range(nout)]
        out = self.new(nout, "lk")
        mono = {}

        def get(m):
            if bin(m).count("1") == 1: return y[m.bit_length() - 1]
            if m in mono: return mono[m]
            low = m & -m
            a = get(m ^ low)
            q = self.new(1, "mono")[0]
            self.qc.ccx(a, y[low.bit_length() - 1], q)
            mono[m] = q
            return q

        for k, ts in enumerate(terms):
            for m in ts:
                if m == 0: self.qc.x(out[k])
                else: self.qc.cx(get(m), out[k])
        self._end()
        return out

    def modmul_const(self, y, w):
        """t = (w*y) mod q for residue y<q (identity wiring when w == 1)."""
        if w == 1 and self.f["identity"]: return y
        if self.f["lookup_twiddle"]:
            return self.lookup(y, lambda v: (w * v) % Q if v < Q else None, 4)
        return Builder.modmul(self, self.const(w, f"w{w}"), y)         # generic reuse-only path

    def mul_const8(self, hi, c):
        """8-bit product hi*c (hi a nibble, c a small constant): ANF lookup + two idle |0> high bits."""
        out = self.lookup(hi, lambda v: v * c, 6)
        return out + self.new(2, "z")

    def sel_const(self, bit, c):
        """corr = bit * c as a 4-qubit operand: CNOT fan-out of `bit` onto fresh qubits, shared |0> elsewhere."""
        self._begin("CondConst")
        out, zi = [], 0
        for k in range(4):
            if (c >> k) & 1:
                q = self.new(1, "sel")[0]; self.qc.cx(bit, q); out.append(q)
            else:
                out.append(self.zeros[zi]); zi += 1
        self._end()
        return out

    # -------------------------------------------------------------------- macros
    def modadd(self, x, t):
        if not self.f["cnot_selectors"]: return Builder.modadd(self, x, t)
        S, cout = self.adder(x, t)
        V, _ = self.adder(S, self.sel_const(cout, C_FOLD))
        return self.modulo(V, self.const(Q, "q"))

    def modsub(self, x, t):
        if not self.f["cnot_selectors"]: return Builder.modsub(self, x, t)
        D, bw = self.subtractor(x, t)
        R, _ = self.subtractor(D, self.sel_const(bw, NEGQ))
        return R

    def modmul(self, x, y):
        """variable x variable modular multiplication (generic Multiplier, constant-aware fold)."""
        if not self.f["lookup_fold"]: return Builder.modmul(self, x, y)
        P = self.multiplier(x, y)
        lo, hi = P[:4], P[4:]
        for _ in range(R_FOLD):
            P2 = self.mul_const8(hi, C_FOLD)
            clo, chi = P2[:4], P2[4:]
            lo2, carry = self.adder(lo, clo)
            hi2, _ = self.adder(chi, self.zext(carry))
            lo, hi = lo2, hi2
        return self.modulo(lo, self.const(Q, "q"))

    def bu(self, x, y, w):
        t = self.modmul_const(y, w)
        return self.modadd(x, t), self.modsub(x, t)

    # ------------------------------------------------------------------ pipeline
    def ntt4(self, vec, w):
        v = self._bitrev4(vec)
        u0, v0 = self.bu(v[0], v[1], 1)
        u1, v1 = self.bu(v[2], v[3], 1)
        s = [u0, v0, u1, v1]
        X0, X2 = self.bu(s[0], s[2], 1)
        X1, X3 = self.bu(s[1], s[3], w)
        return [X0, X1, X2, X3]

    def pipeline(self):
        a = [self.new(4, f"a{i}") for i in range(4)]
        b = [self.new(4, f"b{i}") for i in range(4)]
        Na, Nb = self.ntt4(a, W), self.ntt4(b, W)
        Pk = [self.modmul(Na[k], Nb[k]) for k in range(4)]
        Iv = self.ntt4(Pk, WINV)
        c = [self.modmul_const(Iv[k], NINV) for k in range(4)]
        return a, b, c
