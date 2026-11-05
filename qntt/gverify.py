"""Verification helpers for GBuilder circuits (exact classical simulation of the circuit's own gate list)."""
import numpy as np
from .batch import Packed
from .sim import compile_ops, run_classical, pack, idx_of
from .generic import GBuilder, root_of_unity


def conv_ref(a, b, q):
    N = len(a)
    return [sum(a[i] * b[(r - i) % N] for i in range(N)) % q for r in range(N)]


def ntt_ref(a, q, w):
    N = len(a)
    return [sum(a[i] * pow(w, i * k, q) for i in range(N)) % q for k in range(N)]


def verify_components(n, q, v1=False, consts=None):
    """Exhaustive: all four blocks (all n-bit operand pairs), MODADD/MODSUB/MODMUL (all residue pairs),
    and every constant multiplier (all constants, all residues). Returns dict of failure counts."""
    out = {}
    def harness(build):
        b = GBuilder(n, q, v1=v1); x, t = b.new(n, "x"), b.new(n, "t"); res = build(b, x, t)
        ops, idx = compile_ops(b.qc)
        return b, x, t, res, ops, idx
    def run(b, ops, idx, regs_in, vals):
        init = list(b.const_init)
        for reg, v in zip(regs_in, vals): init += [idx[reg[k]] for k in range(n) if v >> k & 1]
        return run_classical(ops, b.nq, init)
    spec = {
        "Adder": (lambda b, x, t: b.adder(x, t), lambda a, c: [(a + c) % (1 << n), (a + c) >> n], range(1 << n)),
        "Subtractor": (lambda b, x, t: b.subtractor(x, t), lambda a, c: [(a - c) % (1 << n), int(a < c)], range(1 << n)),
        "Multiplier": (lambda b, x, t: (b.multiplier(x, t),), lambda a, c: [a * c], range(1 << n)),
        "Modulo": (lambda b, x, t: (b.modulo(x, t),), lambda a, c: [a % c], range(1, 1 << n)),
    }
    for name, (build, ref, cr) in spec.items():
        b, x, t, res, ops, idx = harness(build)
        regs = [list(r) if isinstance(r, (list, tuple)) else [r] for r in res]
        bad = 0
        for a in range(1 << n):
            for c in cr:
                bits = run(b, ops, idx, (x, t), (a, c))
                got = [pack(bits, [idx[z] for z in r]) for r in regs]
                bad += not (got == ref(a, c) and pack(bits, [idx[z] for z in x]) == a and pack(bits, [idx[z] for z in t]) == c)
        out[name] = bad
    for name, fn, ref in [("MODADD", lambda b, x, t: b.modadd(x, t), lambda a, c: (a + c) % q),
                          ("MODSUB", lambda b, x, t: b.modsub(x, t), lambda a, c: (a - c) % q),
                          ("MODMUL", lambda b, x, t: b.modmul(x, t), lambda a, c: (a * c) % q)]:
        b, x, t, res, ops, idx = harness(fn)
        ir = [idx[z] for z in res]; bad = 0
        for a in range(q):
            for c in range(q):
                bits = run(b, ops, idx, (x, t), (a, c)); bad += pack(bits, ir) != ref(a, c)
        out[name] = bad
    if v1:
        bad = 0
        for w in (consts if consts is not None else range(1, q)):
            b = GBuilder(n, q, v1=True); y = b.new(n, "y"); tt = b.modmul_const(y, w)
            ops, idx = compile_ops(b.qc); iy, it = [idx[z] for z in y], [idx[z] for z in tt]
            for v in range(q):
                bits = run_classical(ops, b.nq, list(b.const_init) + [iy[k] for k in range(n) if v >> k & 1])
                bad += pack(bits, it) != (w * v) % q or pack(bits, iy) != v
        out["ConstMul(all w)"] = bad
    return out


def verify_pipeline(b, a_regs, b_regs, c_regs, N, q, tests=1 << 17, seed=1, exhaustive=False, chunk=None):
    """Random (or exhaustive) verification of the full multiplier with the bit-packed simulator."""
    sim = Packed(b.qc, b.const_init)
    ia = [idx_of(sim.idx, r) for r in a_regs]; ib = [idx_of(sim.idx, r) for r in b_regs]; ic = [idx_of(sim.idx, r) for r in c_regs]
    n = b.n; rng = np.random.default_rng(seed); checked = fails = 0
    chunk = chunk or int(max(1 << 12, min(1 << 18, 4e8 / (8 * len(b.qc.qubits)) * 64)))
    total = q ** (2 * N) if exhaustive else tests
    done = 0
    while done < total:
        B = min(chunk, total - done)
        if exhaustive:
            t = np.arange(done, done + B, dtype=np.int64); digs = []
            for _ in range(2 * N): digs.append(t % q); t = t // q
            av, bv = digs[:N], digs[N:]
        else:
            av = [rng.integers(0, q, B, dtype=np.int64) for _ in range(N)]
            bv = [rng.integers(0, q, B, dtype=np.int64) for _ in range(N)]
            if done == 0:                                          # edge cases first
                for i in range(N):
                    av[i][0] = 0; bv[i][0] = 0; av[i][1] = q - 1; bv[i][1] = q - 1
                    av[i][2] = 1 if i == 0 else 0; bv[i][2] = q - 1
        inputs = {}
        for i in range(N):
            for k in range(n):
                inputs[ia[i][k]] = ((av[i] >> k) & 1).astype(np.uint8); inputs[ib[i][k]] = ((bv[i] >> k) & 1).astype(np.uint8)
        st = sim.run(inputs, B)
        exp = [sum(av[i] * bv[(r - i) % N] for i in range(N)) % q for r in range(N)]
        for r in range(N): fails += int(np.count_nonzero(Packed.read(st, ic[r], B) != exp[r]))
        done += B; checked += B
    return {"checked": checked, "failures": fails, "exhaustive": exhaustive}


def verify_ntt(b, a_regs, X_regs, N, q, exhaustive=True, tests=1 << 16, seed=2):
    sim = Packed(b.qc, b.const_init); w = root_of_unity(q, N)
    ia = [idx_of(sim.idx, r) for r in a_regs]; ix = [idx_of(sim.idx, r) for r in X_regs]
    n = b.n; rng = np.random.default_rng(seed)
    if exhaustive:
        B = q ** N; t = np.arange(B, dtype=np.int64); av = []
        for _ in range(N): av.append(t % q); t = t // q
    else:
        B = tests; av = [rng.integers(0, q, B, dtype=np.int64) for _ in range(N)]
    inputs = {ia[i][k]: ((av[i] >> k) & 1).astype(np.uint8) for i in range(N) for k in range(n)}
    st = sim.run(inputs, B); fails = 0
    for k in range(N):
        exp = sum(av[i] * pow(w, i * k, q) for i in range(N)) % q
        fails += int(np.count_nonzero(Packed.read(st, ix[k], B) != exp))
    return {"checked": B, "failures": fails, "exhaustive": exhaustive}
