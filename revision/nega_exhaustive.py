"""Exhaustive check (all q^(2N) inputs) of the negacyclic multiplier: outputs and input restoration.
python3 revision/nega_exhaustive.py V1 4 5 17 revision/nega_exhaustive_V1_N4_q17.json"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from negacyclic import NBuilder, verify
var, N, n, q, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
b = NBuilder(n, q, v1=(var == "V1")); a, bb, c = b.nega_pipeline(N)
t0 = time.time()
r = verify(b, a, bb, c, N, q, exhaustive=True, chunk=1 << 20, log=f"{var} N={N} q={q}")
r.update(variant=var, N=N, q=q, ring="negacyclic", gates=len(b.qc.data), qubits=b.nq, seconds=round(time.time() - t0, 1))
json.dump(r, open(out, "w"), indent=1); print(r)
