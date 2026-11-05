"""Simulation helpers: (1) fast exact classical simulator for X/CX/CCX circuits,
(2) Qiskit-Aer MPS runner (bond dimension 1 for classical reversible circuits), and
(3) Statevector helper for small blocks (superposition tests)."""
from qiskit import QuantumCircuit


def compile_ops(qc):
    idx = {q: i for i, q in enumerate(qc.qubits)}
    ops = []
    for inst in qc.data:
        ops.append((inst.operation.name, tuple(idx[q] for q in inst.qubits)))
    return ops, idx


def run_classical(ops, nqubits, init_ones):
    """Apply the gate list to a basis state given as the set of qubit indices that are |1>."""
    bits = bytearray(nqubits)
    for i in init_ones: bits[i] = 1
    for name, q in ops:
        if name == "ccx":
            if bits[q[0]] and bits[q[1]]: bits[q[2]] ^= 1
        elif name == "cx":
            if bits[q[0]]: bits[q[1]] ^= 1
        elif name == "x":
            bits[q[0]] ^= 1
        else:
            raise ValueError(f"non-classical gate {name}")
    return bits


def pack(bits, reg_idx):
    return sum(bits[i] << k for k, i in enumerate(reg_idx))


def idx_of(idx, reg):
    return [idx[q] for q in reg]


def run_aer_mps(qc, init_ones, regs_to_read, shots=1):
    """Run on qiskit-aer's matrix-product-state simulator (exact for classical circuits)."""
    from qiskit_aer import AerSimulator
    full = QuantumCircuit(qc.qubits, name="run")
    from qiskit.circuit import ClassicalRegister
    cr = ClassicalRegister(sum(len(r) for r in regs_to_read), "out")
    full.add_register(cr)
    for i in init_ones: full.x(qc.qubits[i])
    full.compose(qc, qubits=qc.qubits, inplace=True)
    k = 0
    for reg in regs_to_read:
        for q in reg:
            full.measure(q, cr[k]); k += 1
    sim = AerSimulator(method="matrix_product_state")
    res = sim.run(full, shots=shots).result()
    counts = res.get_counts()
    bitstr = max(counts, key=counts.get)[::-1]            # little-endian
    vals, k = [], 0
    for reg in regs_to_read:
        v = sum(int(bitstr[k + j]) << j for j in range(len(reg)))
        vals.append(v); k += len(reg)
    return vals
