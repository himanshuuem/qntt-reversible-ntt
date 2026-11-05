"""Bit-packed batch simulator for X/CX/CCX circuits: 64 test vectors per uint64 word.
Used for EXHAUSTIVE verification (all 13^8 = 815,730,721 input pairs of the full multiplier)."""
import numpy as np
from .sim import compile_ops, idx_of

Q = 13

class Packed:
    def __init__(self, qc, const_init):
        self.ops, self.idx = compile_ops(qc)
        self.n = len(qc.qubits)
        self.const_init = list(const_init)

    def run(self, inputs, B):
        """inputs: {qubit_index: bool array length B}. Returns packed state [n, words]."""
        words = (B + 63) // 64
        st = np.zeros((self.n, words), dtype=np.uint64)
        ones = np.full(words, np.uint64(0xFFFFFFFFFFFFFFFF))
        for q in self.const_init: st[q] = ones
        for q, arr in inputs.items():
            pad = np.zeros(words * 64, dtype=np.uint8); pad[:B] = arr
            st[q] = np.packbits(pad, bitorder="little").view(np.uint64)
        for name, q in self.ops:
            if name == "ccx": st[q[2]] ^= st[q[0]] & st[q[1]]
            elif name == "cx": st[q[1]] ^= st[q[0]]
            else: st[q[0]] ^= ones
        return st

    @staticmethod
    def read(st, reg_idx, B):
        val = np.zeros(B, dtype=np.int64)
        for k, i in enumerate(reg_idx):
            bits = np.unpackbits(st[i].view(np.uint8), bitorder="little")[:B]
            val |= bits.astype(np.int64) << k
        return val


def digits(t):
    """t in [0,13^8) -> (a0..a3, b0..b3) base-13 digits (int64 arrays)."""
    d = []
    for _ in range(8):
        d.append(t % Q); t = t // Q
    return d[:4], d[4:]


def conv_expected(a, b):
    return [sum(a[i] * b[(r - i) % 4] for i in range(4)) % Q for r in range(4)]
