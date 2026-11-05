"""Measure Toffoli-depth, depth, and a Bennett-cleaned (compute-copy-uncompute) variant of V0 and V1,
and check, on random inputs, (a) outputs correct, (b) inputs and constant registers restored,
(c) for the cleaned variant: every non-output qubit returned to its initial value."""
import sys, json, time
sys.path.insert(0, sys.argv[1])          # path to qntt_qiskit
import numpy as np
from qiskit import QuantumCircuit
from qntt import Builder, BuilderCA
from qntt.batch import Packed, digits, conv_expected, Q
from qntt.sim import idx_of

NT = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 16
rng = np.random.default_rng(2026)
out = {}

def check(qc, const_init, ia, ib, ic, extra_out=None, clean=False):
    sim = Packed(qc, const_init)
    t = rng.integers(0, Q ** 8, NT, dtype=np.int64)
    av, bv = digits(t)
    inputs = {}
    for i in range(4):
        for k in range(4):
            inputs[ia[i][k]] = ((av[i] >> k) & 1).astype(np.uint8)
            inputs[ib[i][k]] = ((bv[i] >> k) & 1).astype(np.uint8)
    st = sim.run(inputs, NT)
    exp = conv_expected(av, bv)
    res = {"tests": NT}
    regs = ic if extra_out is None else extra_out
    res["output_failures"] = int(sum(np.count_nonzero(Packed.read(st, regs[r], NT) != exp[r]) for r in range(4)))
    # inputs restored
    f = 0
    for i in range(4):
        f += int(np.count_nonzero(Packed.read(st, ia[i], NT) != av[i]))
        f += int(np.count_nonzero(Packed.read(st, ib[i], NT) != bv[i]))
    res["input_restore_failures"] = f
    ones = set(const_init)
    f = sum(int(np.count_nonzero(Packed.read(st, [q], NT) != 1)) for q in ones)
    res["constant_restore_failures"] = f
    if clean:
        keep = set(q for r in ia + ib for q in r) | ones | set(q for r in regs for q in r)
        dirty = 0
        for q in range(qc.num_qubits):
            if q in keep: continue
            dirty += int(np.count_nonzero(Packed.read(st, [q], NT) != 0))
        res["ancilla_not_clean"] = dirty
    return res

def tof_depth(qc):
    return qc.depth(filter_function=lambda ins: ins.operation.name == "ccx")

for name, B in (("V0", Builder), ("V1", BuilderCA)):
    t0 = time.time()
    b = B(); a, bb, c = b.pipeline(); qc = b.qc
    sim = Packed(qc, b.const_init)
    ia = [idx_of(sim.idx, r) for r in a]; ib = [idx_of(sim.idx, r) for r in bb]; ic = [idx_of(sim.idx, r) for r in c]
    ops = qc.count_ops()
    r = {"qubits": qc.num_qubits, "gates": sum(ops.values()), "toffoli": ops.get("ccx", 0),
         "depth": qc.depth(), "toffoli_depth": tof_depth(qc)}
    r["check_garbage_kept"] = check(qc, b.const_init, ia, ib, ic)
    # Bennett: compute, copy 16 output bits to a fresh register, uncompute
    n0 = qc.num_qubits
    cl = QuantumCircuit(n0 + 16)
    cl.compose(qc, qubits=range(n0), inplace=True)
    copy = [[n0 + 4 * i + k for k in range(4)] for i in range(4)]
    for i in range(4):
        for k in range(4):
            cl.cx(ic[i][k], copy[i][k])
    cl.compose(qc.inverse(), qubits=range(n0), inplace=True)
    o2 = cl.count_ops()
    r["clean"] = {"qubits": cl.num_qubits, "gates": sum(o2.values()), "toffoli": o2.get("ccx", 0),
                  "depth": cl.depth(), "toffoli_depth": tof_depth(cl)}
    r["clean"]["check"] = check(cl, b.const_init, ia, ib, ic, extra_out=copy, clean=True)
    r["seconds"] = round(time.time() - t0, 1)
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
