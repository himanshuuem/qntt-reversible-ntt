"""Variant V1 (constant-aware) results + cumulative ablation vs the reuse-only design V0 -> results_v1.json"""
import json, random, time
from collections import Counter
from qntt import Builder, BuilderCA, Q
from qntt.sim import compile_ops, run_classical, pack, idx_of, run_aer_mps

conv = lambda av, bv: [sum(av[i] * bv[(r - i) % 4] for i in range(4)) % Q for r in range(4)]
def evaluate(b, n_random=400, aer=0):
    a, bb, c = b.pipeline()
    ops, idx = compile_ops(b.qc)
    ia = [idx_of(idx, r) for r in a]; ib = [idx_of(idx, r) for r in bb]; ic = [idx_of(idx, r) for r in c]
    def init_for(av, bv):
        init = list(b.const_init)
        for i in range(4):
            init += [ia[i][k] for k in range(4) if av[i] >> k & 1] + [ib[i][k] for k in range(4) if bv[i] >> k & 1]
        return init
    tests = [([3, 7, 2, 9], [5, 0, 11, 4])] + [([random.randrange(Q) for _ in range(4)], [random.randrange(Q) for _ in range(4)]) for _ in range(n_random)]
    fails = sum([pack(run_classical(ops, b.n, init_for(*t)), r) for r in ic] != conv(*t) for t in tests) if False else 0
    for av, bv in tests:
        bits = run_classical(ops, b.n, init_for(av, bv))
        fails += [pack(bits, r) for r in ic] != conv(av, bv)
    aer_res = []
    for av, bv in tests[:aer]:
        t0 = time.time(); got = run_aer_mps(b.qc, init_for(av, bv), c)
        aer_res.append({"a": av, "b": bv, "got": got, "ok": got == conv(av, bv), "seconds": round(time.time() - t0, 1)})
    tot = Counter()
    for g in b.gates.values(): tot.update(g)
    return {"gates": len(b.qc.data), "ccx": tot["ccx"], "cx": tot["cx"], "x": tot["x"], "t_count": 7 * tot["ccx"],
            "qubits": b.n, "depth": b.qc.depth(), "tests": len(tests), "failures": int(fails),
            "block_gates": {k: sum(v.values()) for k, v in b.gates.items()}, "instances": dict(b.inst), "aer_mps": aer_res}

random.seed(11)
R = {}
cfgs = [("V0 reuse-only (reference)", None),
        ("+ identity twiddles (w=1 free)", dict(identity=True, lookup_twiddle=False, cnot_selectors=False, lookup_fold=False)),
        ("+ lookup constant multipliers (w=5,8; N^-1)", dict(identity=True, lookup_twiddle=True, cnot_selectors=False, lookup_fold=False)),
        ("+ CNOT constant selectors in MODADD/MODSUB", dict(identity=True, lookup_twiddle=True, cnot_selectors=True, lookup_fold=False)),
        ("+ lookup fold multiplier hi*c  (= V1)", dict(identity=True, lookup_twiddle=True, cnot_selectors=True, lookup_fold=True)),
        ("regression: all switches off", dict(identity=False, lookup_twiddle=False, cnot_selectors=False, lookup_fold=False))]
R["ablation"] = []
for name, flags in cfgs:
    b = Builder() if flags is None else BuilderCA(**flags)
    last = flags is not None and all(flags.values())
    r = evaluate(b, aer=3 if last else 0)
    r["config"] = name; R["ablation"].append(r)
    print(f'{name:48s} gates={r["gates"]:6d} ccx={r["ccx"]:5d} T={r["t_count"]:6d} qubits={r["qubits"]:5d} fails={r["failures"]}')
    for a in r["aer_mps"]: print("   aer", a)
R["v0_regression_matches"] = R["ablation"][0]["gates"] == R["ablation"][-1]["gates"]
v0, v1 = R["ablation"][0], R["ablation"][4]
R["savings_v1_vs_v0"] = {k: round(100 * (1 - v1[k] / v0[k]), 1) for k in ["gates", "ccx", "t_count", "qubits", "depth"]}
print("regression ok:", R["v0_regression_matches"], "savings %:", R["savings_v1_vs_v0"])
json.dump(R, open("results_v1.json", "w"), indent=1)
