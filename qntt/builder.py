"""
qntt.builder
============
Gate-level Qiskit realisation of the 4-point NTT-based reversible polynomial
multiplier (q=13, w=5) described in the block-level design.

Architecture (mirrors the block-level design, Sections II-IV)
-------------------------------------------------
  four black-box blocks  : Adder, Subtractor, Multiplier, Modulo
  three macros           : MODADD, MODSUB, MODMUL      (built ONLY from blocks)
  one Butterfly Unit     : BU = MODMUL + MODADD + MODSUB
  pipeline               : NTT(a), NTT(b) -> pointwise MODMUL -> INTT -> N^-1 scaling

IMPORTANT HONESTY NOTE
----------------------
The block-level design's four blocks are given SVG netlists (22/30/82/248 gates) that are not
available here.  The blocks below are *functional equivalents* with the same
black-box interface (4-bit operands, Cout/Borrow flags, 8-bit product, 4-bit
remainder), built only from X / CNOT / Toffoli (CCX).  Their gate counts differ
from Table I of the block-level design (see `block_costs()`), so absolute totals differ; the
block-INSTANCE structure above the blocks is exactly the block-level design's, except for one
step the block-level design's accounting leaves out (see `fold_step` in MODMUL).

Everything below block level uses ONLY block instances: no gate is placed outside
a block.  Blocks never modify their inputs (inputs are restored), outputs are on
fresh qubits, and no garbage is uncomputed (same convention as the block-level design).
"""
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

Q, W, WINV, NINV = 13, 5, 8, 10          # modulus, root, inverse root, N^-1 mod q
C_FOLD = 16 % Q                            # fold constant c = 16 mod q   (=3)
NEGQ = 16 - Q                              # MODSUB correction constant   (=3)
R_FOLD = 3                                 # fixed number of MODMUL fold rounds
NBITS = 4


class Builder:
    """Owns one QuantumCircuit and a ledger of block instances / gates."""

    def __init__(self, label="qntt", named=False):
        self.named = named          # True -> QuantumRegisters with readable names (small circuits / drawings)
        self._uid = 0
        self.qc = QuantumCircuit(name=label)
        self.n = 0
        self.const_init = []                     # qubit indices that must start in |1>
        self.consts = {}                         # value -> list of qubits
        self.inst = Counter()                    # block-instance counts
        self.gates = {}                          # block name -> Counter of gate names
        self.names = {}                          # qubit index -> label (for debugging)
        self.zeros = None
        self._depth = []                         # stack of current block names
        self.zeros = self.new(4, "zero")         # shared read-only |0> qubits (zero-extend)

    # ---------------------------------------------------------------- allocation
    def new(self, k, label="anc"):
        if self.named:
            from qiskit.circuit import QuantumRegister
            self._uid += 1
            reg = QuantumRegister(k, f"{label}{self._uid}")
            self.qc.add_register(reg)
            qs = list(reg)
        else:
            qs = [Qubit() for _ in range(k)]
            self.qc.add_bits(qs)
        out = list(range(self.n, self.n + k))
        for i, idx in enumerate(out):
            self.names[idx] = f"{label}[{i}]"
        self.n += k
        return qs

    def const(self, value, label=None):
        """Constant-initialised ancilla register (design document: 'constant bits').
        Prepared by X gates in the HARNESS (input preparation), not counted as circuit gates."""
        if value in self.consts:
            return self.consts[value]
        reg = self.new(NBITS, label or f"const{value}")
        for k in range(NBITS):
            if (value >> k) & 1:
                self.const_init.append(self.qc.find_bit(reg[k]).index)
        self.consts[value] = reg
        return reg

    def zext(self, bit):
        """zero_extend(bit) -> 4-qubit operand [bit, 0, 0, 0] (shared zero qubits, read-only)."""
        return [bit] + self.zeros[:3]

    # -------------------------------------------------------------------- ledger
    def _begin(self, name):
        self._depth.append((name, len(self.qc.data)))

    def _end(self):
        name, start = self._depth.pop()
        c = self.gates.setdefault(name, Counter())
        for inst in self.qc.data[start:]:
            c[inst.operation.name] += 1
        self.inst[name] += 1

    # ---------------------------------------------------------- Cuccaro primitives
    def _maj(self, c, b, a):
        self.qc.cx(a, b); self.qc.cx(a, c); self.qc.ccx(c, b, a)

    def _uma(self, c, b, a):
        self.qc.ccx(c, b, a); self.qc.cx(a, c); self.qc.cx(c, b)

    def _cuccaro_ops(self, a, b, c0, cout=None):
        """Return the gate list of  b := a + b  (a preserved, c0 returned to 0)."""
        n = len(a)
        ops = []
        def MAJ(c, bb, aa): ops.extend([("cx", aa, bb), ("cx", aa, c), ("ccx", c, bb, aa)])
        def UMA(c, bb, aa): ops.extend([("ccx", c, bb, aa), ("cx", aa, c), ("cx", c, bb)])
        MAJ(c0, b[0], a[0])
        for i in range(1, n): MAJ(a[i - 1], b[i], a[i])
        if cout is not None: ops.append(("cx", a[n - 1], cout))
        for i in range(n - 1, 0, -1): UMA(a[i - 1], b[i], a[i])
        UMA(c0, b[0], a[0])
        return ops

    def _emit(self, ops):
        for g, *qs in ops:
            getattr(self.qc, g)(*qs)

    def _cuccaro_add(self, a, b, c0, cout=None):
        self._emit(self._cuccaro_ops(a, b, c0, cout))

    def _cuccaro_sub(self, a, b, c0):
        """b := b - a  (exact gate-reversal of the adder; every gate is self-inverse)."""
        self._emit(list(reversed(self._cuccaro_ops(a, b, c0, None))))

    # ------------------------------------------------------- the four given blocks
    def adder(self, A, B):
        """S,Cout = A+B.  Inputs preserved."""
        self._begin("Adder")
        S = self.new(4, "S"); Cout = self.new(1, "Cout")[0]; c0 = self.new(1, "c0")[0]
        for k in range(4): self.qc.cx(B[k], S[k])
        self._cuccaro_add(A, S, c0, Cout)
        self._end()
        return S, Cout

    def subtractor(self, A, B):
        """D,Borrow = A-B (D=(A-B) mod 16, Borrow=1 iff A<B).  Inputs preserved."""
        self._begin("Subtractor")
        D5 = self.new(5, "D"); c0 = self.new(1, "c0")[0]; pad = self.new(1, "zpad")[0]
        for k in range(4): self.qc.cx(A[k], D5[k])
        self._cuccaro_sub(list(B) + [pad], D5, c0)
        self._end()
        return D5[:4], D5[4]

    def multiplier(self, A, B):
        """P = A*B (8 bits) via partial products + three Cuccaro rows.  Inputs preserved."""
        self._begin("Multiplier")
        P = self.new(8, "P"); c0 = self.new(1, "c0")[0]; pp = self.new(12, "pp")
        for j in range(4): self.qc.ccx(A[0], B[j], P[j])
        for i in range(1, 4):
            row = pp[4 * (i - 1): 4 * i]
            for j in range(4): self.qc.ccx(A[i], B[j], row[j])
            self._cuccaro_add(row, P[i:i + 4], c0, P[i + 4])
        self._end()
        return P

    def modulo(self, V, Qr):
        """R = V mod Qr for 4-bit V and 4-bit divisor Qr>=1: 4-round restoring division."""
        self._begin("Modulo")
        zs = self.new(4, "zs"); c0 = self.new(1, "c0")[0]
        qext = list(Qr) + [zs[0], zs[1]]
        r_prev = zs
        for rd, i in enumerate([3, 2, 1, 0]):
            rn = [V[i]] + list(r_prev)                      # (r<<1)|v_i : wiring only
            t = self.new(6, f"t{rd}")
            for k in range(5): self.qc.cx(rn[k], t[k])
            self._cuccaro_sub(qext, t, c0)                  # t = rn - q  (6-bit, bit5 = borrow)
            out = self.new(4, f"r{rd}")
            for k in range(4):                              # out = borrow ? rn : t
                self.qc.cx(t[k], out[k])
                self.qc.cx(rn[k], t[k])
                self.qc.ccx(t[5], t[k], out[k])
            r_prev = out
        self._end()
        return r_prev

    # -------------------------------------------------------------------- macros
    def modadd(self, x, t):
        S, cout = self.adder(x, t)
        P = self.multiplier(self.zext(cout), self.const(C_FOLD, "c"))
        V, _ = self.adder(S, P[:4])
        return self.modulo(V, self.const(Q, "q"))

    def modsub(self, x, t):
        D, bw = self.subtractor(x, t)
        P = self.multiplier(self.zext(bw), self.const(NEGQ, "16-q"))
        R, _ = self.subtractor(D, P[:4])
        return R

    def modmul(self, x, y):
        P = self.multiplier(x, y)
        lo, hi = P[:4], P[4:]
        for _ in range(R_FOLD):
            P2 = self.multiplier(hi, self.const(C_FOLD, "c"))
            clo, chi = P2[:4], P2[4:]
            lo2, carry = self.adder(lo, clo)
            # hi <- chi + carry : a genuine 4-bit addition (see design document-gap finding F1).
            hi2, _ = self.adder(chi, self.zext(carry))
            lo, hi = lo2, hi2
        return self.modulo(lo, self.const(Q, "q"))

    # ----------------------------------------------------------------- butterfly
    def bu(self, x, y, w_reg):
        t = self.modmul(w_reg, y)
        return self.modadd(x, t), self.modsub(x, t)

    # ------------------------------------------------------------------ pipeline
    @staticmethod
    def _bitrev4(v): return [v[0], v[2], v[1], v[3]]

    def ntt4(self, vec, w_reg, one):
        v = self._bitrev4(vec)
        u0, v0 = self.bu(v[0], v[1], one)
        u1, v1 = self.bu(v[2], v[3], one)
        s = [u0, v0, u1, v1]
        X0, X2 = self.bu(s[0], s[2], one)
        X1, X3 = self.bu(s[1], s[3], w_reg)
        return [X0, X1, X2, X3]

    def pipeline(self):
        """Full multiplier c(x) = a(x)*b(x) mod (x^4-1), coefficients mod 13."""
        a = [self.new(4, f"a{i}") for i in range(4)]
        b = [self.new(4, f"b{i}") for i in range(4)]
        one, w, winv = self.const(1, "1"), self.const(W, "w"), self.const(WINV, "w^-1")
        ninv = self.const(NINV, "N^-1")
        Na = self.ntt4(a, w, one)
        Nb = self.ntt4(b, w, one)
        Pk = [self.modmul(Na[k], Nb[k]) for k in range(4)]
        # inverse core: ntt4() bit-reverses P internally (=(P0,P2,P1,P3)), twiddle w^-1
        Iv = self.ntt4(Pk, winv, one)
        c = [self.modmul(Iv[k], ninv) for k in range(4)]
        return a, b, c


# ------------------------------------------------------------------ reporting
def block_costs():
    """Gate cost of each block (one instance) in THIS implementation."""
    out = {}
    for name in ("Adder", "Subtractor", "Multiplier", "Modulo"):
        b = Builder()
        A, B = b.new(4, "A"), b.new(4, "B")
        {"Adder": lambda: b.adder(A, B), "Subtractor": lambda: b.subtractor(A, B),
         "Multiplier": lambda: b.multiplier(A, B), "Modulo": lambda: b.modulo(A, B)}[name]()
        g = b.gates[name]
        out[name] = {"total": sum(g.values()), "ccx": g["ccx"], "cx": g["cx"], "x": g["x"],
                     "qubits_added": b.n - 4 - 8}
    return out

THESIS_TABLE_I = {"Adder": 22, "Subtractor": 30, "Multiplier": 82, "Modulo": 248}
