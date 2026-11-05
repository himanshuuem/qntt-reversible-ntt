"""Run every verification for the Qiskit realisation and write results.json + figures/.
python3 run_all.py            (the exhaustive 13^8 run is separate: verify_exhaustive.py)"""
import json, os, random, time
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qntt import Builder, block_costs, THESIS_TABLE_I, Q
from qntt.sim import compile_ops, run_classical, pack, idx_of, run_aer_mps

R = {}
random.seed(7)

# ------------------------------------------------------------------ 1. blocks
def build_block(kind, named=False):
    b = Builder(named=named)
    A, B = b.new(4, "A"), b.new(4, "B")
    outs = {"Adder": lambda: b.adder(A, B), "Subtractor": lambda: b.subtractor(A, B),
            "Multiplier": lambda: (b.multiplier(A, B),), "Modulo": lambda: (b.modulo(A, B),)}[kind]()
    regs = [list(o) if isinstance(o, (list, tuple)) else [o] for o in outs]
    return b, A, B, regs

def ref_block(kind, a, c):
    return {"Adder": lambda: [(a + c) % 16, (a + c) >> 4], "Subtractor": lambda: [(a - c) % 16, int(a < c)],
            "Multiplier": lambda: [a * c], "Modulo": lambda: [a % c]}[kind]()

R["blocks"] = {}
for kind in ["Adder", "Subtractor", "Multiplier", "Modulo"]:
    b, A, B, regs = build_block(kind)
    ops, idx = compile_ops(b.qc)
    ia, ib = idx_of(idx, A), idx_of(idx, B)
    fails = n = 0
    for a in range(16):
        for c in (range(1, 16) if kind == "Modulo" else range(16)):
            init = [ia[k] for k in range(4) if a >> k & 1] + [ib[k] for k in range(4) if c >> k & 1]
            bits = run_classical(ops, b.n, init)
            got = [pack(bits, idx_of(idx, r)) for r in regs]
            ok = got == ref_block(kind, a, c) and pack(bits, ia) == a and pack(bits, ib) == c
            fails += (not ok); n += 1
    # Aer-MPS cross-check on random cases
    aer_f = 0
    for _ in range(12):
        a, c = random.randrange(16), random.randrange(1 if kind == "Modulo" else 0, 16)
        init = [ia[k] for k in range(4) if a >> k & 1] + [ib[k] for k in range(4) if c >> k & 1]
        vals = run_aer_mps(b.qc, init, regs)
        aer_f += (vals != ref_block(kind, a, c))
    R["blocks"][kind] = {"exhaustive_cases": n, "failures": fails, "aer_mps_cases": 12, "aer_failures": int(aer_f)}
    print(kind, R["blocks"][kind])

# ------------------------------------------------------------------ 2. macros
R["macros"] = {}
for kind in ["MODADD", "MODSUB", "MODMUL"]:
    b = Builder(); x, t = b.new(4, "x"), b.new(4, "t")
    out = {"MODADD": b.modadd, "MODSUB": b.modsub, "MODMUL": b.modmul}[kind](x, t)
    ops, idx = compile_ops(b.qc)
    ix, it, io = idx_of(idx, x), idx_of(idx, t), idx_of(idx, out)
    f = 0
    for a in range(Q):
        for c in range(Q):
            init = list(b.const_init) + [ix[k] for k in range(4) if a >> k & 1] + [it[k] for k in range(4) if c >> k & 1]
            bits = run_classical(ops, b.n, init)
            exp = {"MODADD": (a + c) % Q, "MODSUB": (a - c) % Q, "MODMUL": (a * c) % Q}[kind]
            f += not (pack(bits, io) == exp and pack(bits, ix) == a and pack(bits, it) == c)
    aer_f = 0
    for _ in range(6):
        a, c = random.randrange(Q), random.randrange(Q)
        init = list(b.const_init) + [ix[k] for k in range(4) if a >> k & 1] + [it[k] for k in range(4) if c >> k & 1]
        v = run_aer_mps(b.qc, init, [out])[0]
        aer_f += v != {"MODADD": (a + c) % Q, "MODSUB": (a - c) % Q, "MODMUL": (a * c) % Q}[kind]
    gates = sum(sum(g.values()) for g in b.gates.values())
    R["macros"][kind] = {"exhaustive_cases": Q * Q, "failures": f, "aer_cases": 6, "aer_failures": int(aer_f),
                         "instances": dict(b.inst), "gates": gates, "qubits": b.n,
                         "ccx": sum(g["ccx"] for g in b.gates.values())}
    print(kind, R["macros"][kind])

# ------------------------------------------------------------------ 3. pipeline
t0 = time.time()
b = Builder(); a_, b_, c_ = b.pipeline()
build_s = time.time() - t0
ops, idx = compile_ops(b.qc)
ia = [idx_of(idx, r) for r in a_]; ib = [idx_of(idx, r) for r in b_]; ic = [idx_of(idx, r) for r in c_]
def init_for(av, bv):
    init = list(b.const_init)
    for i in range(4):
        init += [ia[i][k] for k in range(4) if av[i] >> k & 1] + [ib[i][k] for k in range(4) if bv[i] >> k & 1]
    return init
conv = lambda av, bv: [sum(av[i] * bv[(r - i) % 4] for i in range(4)) % Q for r in range(4)]
tests = [([3, 7, 2, 9], [5, 0, 11, 4]), ([0] * 4, [0] * 4), ([12] * 4, [12] * 4), ([1, 0, 0, 0], [5, 0, 11, 4])]
tests += [([random.randrange(Q) for _ in range(4)], [random.randrange(Q) for _ in range(4)]) for _ in range(2000)]
fails = 0
for av, bv in tests:
    bits = run_classical(ops, b.n, init_for(av, bv))
    fails += [pack(bits, r) for r in ic] != conv(av, bv)
aer = []
for av, bv in tests[:3]:
    ta = time.time()
    got = run_aer_mps(b.qc, init_for(av, bv), c_)
    # run_aer_mps returns one int per register passed -> c_ is 4 regs
    aer.append({"a": av, "b": bv, "got": got, "expected": conv(av, bv), "ok": got == conv(av, bv),
                "seconds": round(time.time() - ta, 1)})
    print("aer pipeline", aer[-1])
cnt = {"ccx": 0, "cx": 0, "x": 0}
for g in b.gates.values():
    for k in cnt: cnt[k] += g[k]
R["pipeline"] = {"qubits": b.n, "gates": len(b.qc.data), "ccx": cnt["ccx"], "cx": cnt["cx"],
                 "instances": dict(b.inst), "depth": b.qc.depth(), "classical_tests": len(tests),
                 "classical_failures": int(fails), "aer_mps": aer, "build_seconds": round(build_s, 2),
                 "t_count_7_per_toffoli": 7 * cnt["ccx"], "constant_qubits_set_to_1": len(b.const_init)}
print({k: v for k, v in R["pipeline"].items() if k != "aer_mps"})

# ------------------------------------------------------------------ 4. superposition (true quantum) test
R["superposition"] = {}
for kind in ["Adder", "Subtractor"]:
    bb = Builder(); A, B = bb.new(4, "A"), bb.new(4, "B")
    bb.qc.h(A + B)                                   # uniform superposition over all 256 (a,b)
    outs = bb.adder(A, B) if kind == "Adder" else bb.subtractor(A, B)
    sv = Statevector(bb.qc)
    ix = {q: i for i, q in enumerate(bb.qc.qubits)}
    expect = np.zeros(2 ** bb.n)
    for a in range(16):
        for c in range(16):
            s, flag = ((a + c) % 16, (a + c) >> 4) if kind == "Adder" else ((a - c) % 16, int(a < c))
            idxv = 0
            for k in range(4):
                idxv |= (a >> k & 1) << ix[A[k]]; idxv |= (c >> k & 1) << ix[B[k]]
                idxv |= (s >> k & 1) << ix[outs[0][k]]
            idxv |= flag << ix[outs[1]]
            expect[idxv] = 1 / 16
    fid = abs(np.vdot(expect, sv.data)) ** 2
    R["superposition"][kind] = {"qubits": bb.n, "fidelity_with_expected_state": round(float(fid), 12)}
    print(kind, "superposition fidelity", fid)

# ------------------------------------------------------------------ 5. Clifford+T transpile
R["clifford_t"] = {}
for kind in ["Adder", "Subtractor", "Multiplier", "Modulo"]:
    b2, A, B, regs = build_block(kind)
    tc = transpile(b2.qc, basis_gates=["h", "t", "tdg", "cx", "x"], optimization_level=0)
    ops_ = tc.count_ops()
    ccx = block_costs()[kind]["ccx"]
    R["clifford_t"][kind] = {"toffoli": ccx, "t_plus_tdg": ops_.get("t", 0) + ops_.get("tdg", 0),
                             "cx_after": ops_.get("cx", 0), "h": ops_.get("h", 0), "depth": tc.depth()}
    print(kind, R["clifford_t"][kind])

# ------------------------------------------------------------------ 6. cost tables
costs = block_costs()
R["block_costs"] = {k: dict(v, thesis_table_I=THESIS_TABLE_I[k]) for k, v in costs.items()}
inst = R["pipeline"]["instances"]
R["totals"] = {
    "qiskit_realisation_gates": sum(inst[k] * costs[k]["total"] for k in inst),
    "thesis_accounting_gates": 19032,
    "thesis_accounting_instances": {"Adder": 84, "Subtractor": 24, "Multiplier": 104, "Modulo": 32},
    "qiskit_instances": inst,
    "extra_adders_vs_thesis": inst["Adder"] - 84,
    "thesis_costs_with_faithful_instances": sum(inst[k] * THESIS_TABLE_I[k] for k in inst),
}
print(R["totals"])

# ------------------------------------------------------------------ 7. figures
os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
for kind in ["Adder", "Subtractor", "Multiplier"]:
    b3, *_ = build_block(kind, named=True)
    fig = b3.qc.draw("mpl", fold=-1, scale=0.55)
    fig.savefig(f"figures/{kind.lower()}_block.png", dpi=170, bbox_inches="tight"); print("saved", kind)
json.dump(R, open("results.json", "w"), indent=1)
print("DONE")
