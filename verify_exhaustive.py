"""Exhaustive verification of the FULL Qiskit multiplier over all 13^8 input pairs.
Usage: python3 verify_exhaustive.py [chunk_log2=20] [max_chunks=all]"""
import sys, time, json
import numpy as np
from qntt import Builder, BuilderCA
from qntt.batch import Packed, digits, conv_expected, Q
from qntt.sim import idx_of

import os
LOG = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else None      # chunks to run in this invocation (resumable)
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "v0"      # v0 = reuse-only, v1 = constant-aware
STATE = "exhaustive_state.json" if VARIANT == "v0" else f"exhaustive_state_{VARIANT}.json"
state = json.load(open(STATE)) if os.path.exists(STATE) else {"next_chunk": 0, "checked": 0, "fails": 0, "seconds": 0.0}
MAXC = None
TOTAL = Q ** 8
B = 1 << LOG

b = (Builder() if VARIANT == 'v0' else BuilderCA()); a, bb, c = b.pipeline()
sim = Packed(b.qc, b.const_init)
ia = [idx_of(sim.idx, r) for r in a]; ib = [idx_of(sim.idx, r) for r in bb]; ic = [idx_of(sim.idx, r) for r in c]
nchunks = (TOTAL + B - 1) // B
t0 = time.time(); checked = state["checked"]; fails = state["fails"]
end = nchunks if BATCH is None else min(nchunks, state["next_chunk"] + BATCH)
for ch in range(state["next_chunk"], end):
    lo = ch * B; hi = min(TOTAL, lo + B); n = hi - lo
    t = np.arange(lo, hi, dtype=np.int64)
    av, bv = digits(t)
    inputs = {}
    for i in range(4):
        for k in range(4):
            inputs[ia[i][k]] = ((av[i] >> k) & 1).astype(np.uint8)
            inputs[ib[i][k]] = ((bv[i] >> k) & 1).astype(np.uint8)
    st = sim.run(inputs, n)
    exp = conv_expected(av, bv)
    for r in range(4):
        got = Packed.read(st, ic[r], n)
        fails += int(np.count_nonzero(got != exp[r]))
    checked += n
    if ch % 25 == 0 or ch == end - 1:
        print(f"chunk {ch+1}/{nchunks} checked={checked:,} fails={fails}", flush=True)
state = {"next_chunk": end, "checked": checked, "fails": fails, "seconds": state["seconds"] + time.time() - t0}
json.dump(state, open(STATE, "w"))
res = dict(state, total_space=TOTAL, complete=checked == TOTAL)
json.dump(dict(res, variant=VARIANT, qubits=b.n, gates=len(b.qc.data)), open("exhaustive_result.json" if VARIANT == "v0" else f"exhaustive_result_{VARIANT}.json", "w"), indent=1)
print(res)
