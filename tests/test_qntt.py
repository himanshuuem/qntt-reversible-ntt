"""pytest-compatible tests (also runnable as a script).  python3 -m pytest -q tests  |  python3 tests/test_qntt.py"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qntt import Builder, Q
from qntt.sim import compile_ops, run_classical, pack, idx_of

def _blocks(kind):
    b = Builder(); A, B = b.new(4), b.new(4)
    outs = {"adder": lambda: b.adder(A, B), "sub": lambda: b.subtractor(A, B),
            "mul": lambda: (b.multiplier(A, B),), "mod": lambda: (b.modulo(A, B),)}[kind]()
    regs = [list(o) if isinstance(o, (list, tuple)) else [o] for o in outs]
    ops, idx = compile_ops(b.qc)
    return b, A, B, regs, ops, idx

def _check(kind, ref, bs=range(16)):
    b, A, B, regs, ops, idx = _blocks(kind)
    ia, ib = idx_of(idx, A), idx_of(idx, B)
    for a in range(16):
        for c in bs:
            init = [ia[k] for k in range(4) if a >> k & 1] + [ib[k] for k in range(4) if c >> k & 1]
            bits = run_classical(ops, b.n, init)
            assert [pack(bits, idx_of(idx, r)) for r in regs] == ref(a, c)
            assert pack(bits, ia) == a and pack(bits, ib) == c          # inputs restored

def test_adder():      _check("adder", lambda a, c: [(a + c) % 16, (a + c) >> 4])
def test_subtractor(): _check("sub", lambda a, c: [(a - c) % 16, int(a < c)])
def test_multiplier(): _check("mul", lambda a, c: [a * c])
def test_modulo():     _check("mod", lambda a, c: [a % c], bs=range(1, 16))

def _macro(name, ref):
    b = Builder(); x, t = b.new(4), b.new(4)
    out = getattr(b, name)(x, t)
    ops, idx = compile_ops(b.qc); ix, it, io = idx_of(idx, x), idx_of(idx, t), idx_of(idx, out)
    for a in range(Q):
        for c in range(Q):
            init = list(b.const_init) + [ix[k] for k in range(4) if a >> k & 1] + [it[k] for k in range(4) if c >> k & 1]
            assert pack(run_classical(ops, b.n, init), io) == ref(a, c)

def test_modadd(): _macro("modadd", lambda a, c: (a + c) % Q)
def test_modsub(): _macro("modsub", lambda a, c: (a - c) % Q)
def test_modmul(): _macro("modmul", lambda a, c: (a * c) % Q)

def test_pipeline_thesis_example_and_random():
    b = Builder(); a_, b_, c_ = b.pipeline()
    ops, idx = compile_ops(b.qc)
    ia = [idx_of(idx, r) for r in a_]; ib = [idx_of(idx, r) for r in b_]; ic = [idx_of(idx, r) for r in c_]
    conv = lambda av, bv: [sum(av[i] * bv[(r - i) % 4] for i in range(4)) % Q for r in range(4)]
    def run(av, bv):
        init = list(b.const_init)
        for i in range(4):
            init += [ia[i][k] for k in range(4) if av[i] >> k & 1] + [ib[i][k] for k in range(4) if bv[i] >> k & 1]
        bits = run_classical(ops, b.n, init)
        return [pack(bits, r) for r in ic]
    assert run([3, 7, 2, 9], [5, 0, 11, 4]) == [0, 12, 1, 4]            # thesis worked example
    random.seed(0)
    for _ in range(200):
        av = [random.randrange(Q) for _ in range(4)]; bv = [random.randrange(Q) for _ in range(4)]
        assert run(av, bv) == conv(av, bv)

if __name__ == "__main__" and False:  # (runner moved to bottom)
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

# ---------------- constant-aware variant V1 ----------------
def test_v1_constant_multipliers_all_constants():
    from qntt import BuilderCA
    for w in range(1, Q):
        b = BuilderCA(); y = b.new(4); t = b.modmul_const(y, w)
        ops, idx = compile_ops(b.qc); iy, it = idx_of(idx, y), idx_of(idx, t)
        for v in range(Q):
            bits = run_classical(ops, b.n, [iy[k] for k in range(4) if v >> k & 1])
            assert pack(bits, it) == (w * v) % Q and pack(bits, iy) == v

def test_v1_pipeline_and_ablation_regression():
    from qntt import BuilderCA
    import random
    conv = lambda av, bv: [sum(av[i] * bv[(r - i) % 4] for i in range(4)) % Q for r in range(4)]
    for flags in [dict(), dict(identity=False, lookup_twiddle=False, cnot_selectors=False, lookup_fold=False)]:
        b = BuilderCA(**flags); a_, b_, c_ = b.pipeline()
        ops, idx = compile_ops(b.qc)
        ia = [idx_of(idx, r) for r in a_]; ib = [idx_of(idx, r) for r in b_]; ic = [idx_of(idx, r) for r in c_]
        random.seed(5)
        for _ in range(150):
            av = [random.randrange(Q) for _ in range(4)]; bv = [random.randrange(Q) for _ in range(4)]
            init = list(b.const_init)
            for i in range(4):
                init += [ia[i][k] for k in range(4) if av[i] >> k & 1] + [ib[i][k] for k in range(4) if bv[i] >> k & 1]
            bits = run_classical(ops, b.n, init)
            assert [pack(bits, r) for r in ic] == conv(av, bv)
        if not flags: assert len(b.qc.data) == 6545
        else: assert len(b.qc.data) == 21240            # all switches off == reuse-only V0

# ---------------- parametric generator ----------------
def test_generic_regression_n4_q13():
    from qntt.generic import GBuilder
    for v1, gates in [(False, 21240), (True, 6545)]:
        b = GBuilder(4, 13, v1=v1); b.pipeline(4)
        assert len(b.qc.data) == gates

def test_generic_q5_exhaustive_components_and_pipeline():
    from qntt.generic import GBuilder
    from qntt.gverify import verify_components, verify_pipeline
    for v1 in (False, True):
        comp = verify_components(3, 5, v1=v1)
        assert all(v == 0 for v in comp.values()), comp
        b = GBuilder(3, 5, v1=v1); a, bb, c = b.pipeline(4)
        r = verify_pipeline(b, a, bb, c, 4, 5, exhaustive=True)       # all 5^8 inputs
        assert r["failures"] == 0 and r["checked"] == 5 ** 8

def test_generic_lu_benchmark_ntt4_q5():
    from qntt.generic import GBuilder
    from qntt.gverify import verify_ntt
    for v1, gates in [(False, 2352), (True, 932)]:
        b = GBuilder(3, 5, v1=v1); a, X = b.ntt_only(4)
        assert len(b.qc.data) == gates
        assert verify_ntt(b, a, X, 4, 5, exhaustive=True)["failures"] == 0   # all 5^4 inputs

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
