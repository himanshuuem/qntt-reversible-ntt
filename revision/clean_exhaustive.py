"""Exhaustive check (all 13^8 inputs) of the Bennett-cleaned V0 or V1 multiplier:
outputs correct, inputs restored, constant qubits back to |1>, every other qubit back to |0>."""
import sys, json, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from qiskit import QuantumCircuit
from qntt import Builder, BuilderCA
from qntt.batch import Packed, digits, conv_expected, Q
from qntt.sim import idx_of

variant, outpath = sys.argv[2], sys.argv[3]
LOG = 20
b = (Builder() if variant == "V0" else BuilderCA()); a, bb, c = b.pipeline(); qc = b.qc
n0 = qc.num_qubits
cl = QuantumCircuit(n0 + 16); cl.compose(qc, qubits=range(n0), inplace=True)
sim0 = Packed(qc, b.const_init)
ia = [idx_of(sim0.idx, r) for r in a]; ib = [idx_of(sim0.idx, r) for r in bb]; ic = [idx_of(sim0.idx, r) for r in c]
copy = [[n0 + 4 * i + k for k in range(4)] for i in range(4)]
for i in range(4):
    for k in range(4): cl.cx(ic[i][k], copy[i][k])
cl.compose(qc.inverse(), qubits=range(n0), inplace=True)
sim = Packed(cl, b.const_init)
inq = set(q for r in ia + ib for q in r); outq = set(q for r in copy for q in r); ones = set(b.const_init)
zero_q = np.array([q for q in range(cl.num_qubits) if q not in inq | outq | ones])
one_q = np.array(sorted(ones))
TOTAL = Q ** 8; B = 1 << LOG; nch = (TOTAL + B - 1) // B
fails = dict(output=0, input=0, constant=0, ancilla=0); checked = 0; t0 = time.time()
for ch in range(nch):
    lo = ch * B; hi = min(TOTAL, lo + B); n = hi - lo
    av, bv = digits(np.arange(lo, hi, dtype=np.int64))
    inputs = {}
    for i in range(4):
        for k in range(4):
            inputs[ia[i][k]] = ((av[i] >> k) & 1).astype(np.uint8)
            inputs[ib[i][k]] = ((bv[i] >> k) & 1).astype(np.uint8)
    st = sim.run(inputs, n)
    words = (n + 63) // 64
    # mask out padding bits in the last word
    mask = np.full(words, np.uint64(0xFFFFFFFFFFFFFFFF))
    if n % 64: mask[-1] = np.uint64((1 << (n % 64)) - 1)
    exp = conv_expected(av, bv)
    for r in range(4):
        fails["output"] += int(np.count_nonzero(Packed.read(st, copy[r], n) != exp[r]))
        fails["input"] += int(np.count_nonzero(Packed.read(st, ia[r], n) != av[r]))
        fails["input"] += int(np.count_nonzero(Packed.read(st, ib[r], n) != bv[r]))
    fails["ancilla"] += int(np.count_nonzero(st[zero_q] & mask))
    fails["constant"] += int(np.count_nonzero((~st[one_q]) & mask))
    checked += n
    if ch % 100 == 0 or ch == nch - 1:
        print(f"{variant} chunk {ch+1}/{nch} checked={checked:,} fails={fails} t={time.time()-t0:.0f}s", flush=True)
res = dict(variant=variant, checked=checked, total=TOTAL, complete=checked == TOTAL, failures=fails,
           seconds=round(time.time() - t0, 1), qubits=cl.num_qubits, gates=sum(cl.count_ops().values()))
json.dump(res, open(outpath, "w"), indent=1); print(res)
