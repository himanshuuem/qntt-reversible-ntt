"""Negacyclic vs cyclic: cost of constant-oblivious reuse when there are no identity twiddles.
python3 revision/run_negacyclic.py   -> revision/negacyclic_results.json"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from qntt.generic import GBuilder, root_of_unity
from qntt.gverify import verify_components, verify_pipeline
from negacyclic import NBuilder, psi_2n, brv, verify

OFF = dict(identity=False, lookup_twiddle=False, cnot_selectors=False, lookup_fold=False)
R = {}

def stats(b):
    s = b.summary()
    qc = b.qc
    s["toffoli_depth"] = qc.depth(filter_function=lambda ins: ins.operation.name == "ccx")
    s["depth"] = qc.depth()
    return {k: s[k] for k in ("n", "q", "qubits", "gates", "ccx", "cx", "t_count", "depth", "toffoli_depth")}

def nega(N, n, q, v1, flags=None, tests=1 << 16):
    b = NBuilder(n, q, v1=v1, flags=flags); a, bb, c = b.nega_pipeline(N)
    s = stats(b); s.update(N=N, ring="negacyclic", psi=psi_2n(q, N))
    s["pipeline"] = verify(b, a, bb, c, N, q, tests=tests)
    return s

def cyc(N, n, q, v1, flags=None, tests=1 << 16):
    b = GBuilder(n, q, v1=v1, flags=flags); a, bb, c = b.pipeline(N)
    s = stats(b); s.update(N=N, ring="cyclic")
    s["pipeline"] = verify_pipeline(b, a, bb, c, N, q, tests=tests)
    return s

def log(k, s):
    R[k] = s; json.dump(R, open(os.path.join(HERE, "negacyclic_results.json"), "w"), indent=1)
    print(k, {x: s.get(x) for x in ("gates", "ccx", "qubits", "toffoli_depth")}, s.get("pipeline"), flush=True)

t0 = time.time()
# components (exhaustive) for every constant the negacyclic designs use
for n, q, Ns in ((5, 17, (4, 8)), (7, 97, (4, 8))):
    consts = set()
    for N in Ns:
        p = psi_2n(q, N); pi = pow(p, -1, q); L = N.bit_length() - 1
        consts |= {pow(p, brv(k, L), q) for k in range(1, N)} | {pow(pi, brv(k, L), q) for k in range(1, N)} | {pow(N, -1, q)}
    R[f"components_n{n}_q{q}"] = verify_components(n, q, v1=True, consts=sorted(consts))
    print("components", n, q, R[f"components_n{n}_q{q}"], flush=True)

# N=4, q=17: cyclic V0, V0' (identity wired), V1  vs  negacyclic V0, cumulative ablation, V1
log("cyc_N4_q17_V0", cyc(4, 5, 17, False))
log("cyc_N4_q17_V0prime", cyc(4, 5, 17, True, flags=dict(OFF, identity=True)))
log("cyc_N4_q17_V1", cyc(4, 5, 17, True))
log("nega_N4_q17_V0", nega(4, 5, 17, False))
log("nega_N4_q17_lookup", nega(4, 5, 17, True, flags=dict(OFF, lookup_twiddle=True)))
log("nega_N4_q17_lookup_sel", nega(4, 5, 17, True, flags=dict(OFF, lookup_twiddle=True, cnot_selectors=True)))
log("nega_N4_q17_V1", nega(4, 5, 17, True))
# scaling
log("cyc_N8_q17_V0prime", cyc(8, 5, 17, True, flags=dict(OFF, identity=True)))
log("nega_N8_q17_V0", nega(8, 5, 17, False))
log("nega_N8_q17_V1", nega(8, 5, 17, True))
log("cyc_N4_q97_V0prime", cyc(4, 7, 97, True, flags=dict(OFF, identity=True), tests=1 << 15))
log("nega_N4_q97_V0", nega(4, 7, 97, False, tests=1 << 15))
log("nega_N4_q97_V1", nega(4, 7, 97, True, tests=1 << 15))
log("cyc_N8_q97_V0prime", cyc(8, 7, 97, True, flags=dict(OFF, identity=True), tests=1 << 15))
log("nega_N8_q97_V0", nega(8, 7, 97, False, tests=1 << 15))
log("nega_N8_q97_V1", nega(8, 7, 97, True, tests=1 << 15))
print("done in", round(time.time() - t0), "s")
