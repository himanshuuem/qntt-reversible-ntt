"""Scaling + like-for-like study.   python3 run_scaling.py <group>      groups: lu | n5 | n7
Appends to results_scaling.json."""
import json, sys, time, os
from qntt.generic import GBuilder, root_of_unity
from qntt.gverify import verify_components, verify_pipeline, verify_ntt

group = sys.argv[1]
OUT = "results_scaling.json"
R = json.load(open(OUT)) if os.path.exists(OUT) else {}

def full(N, n, q, v1, exhaustive=False, tests=1 << 16, comps=True, consts=None):
    t0 = time.time()
    b = GBuilder(n, q, v1=v1); a, bb, c = b.pipeline(N); s = b.summary(); s["N"] = N; s["omega"] = root_of_unity(q, N)
    s["build_s"] = round(time.time() - t0, 1)
    s["pipeline"] = verify_pipeline(b, a, bb, c, N, q, tests=tests, exhaustive=exhaustive)
    if comps:
        ws = sorted(set([root_of_unity(q, N), pow(root_of_unity(q, N), -1, q), pow(N, -1, q)] + [pow(root_of_unity(q, N), k, q) for k in range(N)])) if consts == "used" else consts
        s["components"] = verify_components(n, q, v1=v1, consts=ws if v1 else None)
    s["verify_s"] = round(time.time() - t0, 1)
    return s

def log(key, s):
    R[key] = s; json.dump(R, open(OUT, "w"), indent=1)
    print(key, {k: s[k] for k in ("gates", "ccx", "t_count", "qubits")}, s.get("pipeline"), s.get("components"), flush=True)

if group == "lu":                      # Lu et al. benchmark: 4-point NTT, q = 5, root 2 (n = 3 bits suffice)
    for v1 in (False, True):
        b = GBuilder(3, 5, v1=v1); a, X = b.ntt_only(4); s = b.summary()
        s["ntt_exhaustive"] = verify_ntt(b, [a], X, 4, 5, exhaustive=True) if False else verify_ntt(b, a, X, 4, 5, exhaustive=True)
        log(f"lu_ntt4_q5_{'V1' if v1 else 'V0'}", s)
    for v1 in (False, True):
        log(f"mult_N4_q5_{'V1' if v1 else 'V0'}", full(4, 3, 5, v1, exhaustive=True))
if group == "n5":
    for N in (4, 8, 16):
        for v1 in (True, False):
            if N == 16 and not v1 and len(sys.argv) < 3: continue
            log(f"mult_N{N}_q17_{'V1' if v1 else 'V0'}", full(N, 5, 17, v1, tests=1 << 16, consts="used"))
if group == "n7":
    for N in (4, 8):
        for v1 in (True, False):
            log(f"mult_N{N}_q97_{'V1' if v1 else 'V0'}", full(N, 7, 97, v1, tests=1 << 15, consts="used"))
