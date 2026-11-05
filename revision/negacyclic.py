"""Negacyclic NTT multiplier: c(x) = a(x) b(x) mod (x^N + 1), coefficients mod q.

Merged form used by ML-KEM-style implementations (no separate pre-/post-twist):
  forward : Cooley-Tukey, natural-order input, bit-reversed output, twiddles psi^brv(k)
  inverse : Gentleman-Sande, bit-reversed input, natural-order output, twiddles psi^-brv(k),
            followed by scaling with N^-1
where psi is a primitive 2N-th root of unity (psi^N = -1).  No butterfly twiddle equals 1,
so the identity-twiddle saving of the cyclic design does not exist here.
Built on GBuilder, so V0 (reuse-only) and V1 (constant-aware) use exactly the same blocks
and switches as the cyclic designs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qntt.generic import GBuilder
from qntt.batch import Packed
from qntt.sim import idx_of


def psi_2n(q, N):
    """smallest primitive 2N-th root of unity mod q."""
    for p in range(2, q):
        if pow(p, N, q) == q - 1: return p
    raise ValueError("2N does not divide q-1")


def brv(i, bits): return int(format(i, f"0{bits}b")[::-1], 2) if bits else 0


class NBuilder(GBuilder):
    def gs_bu(self, x, y, s):
        """Gentleman-Sande butterfly: (x + y, (x - y) * s) mod q."""
        return self.modadd(x, y), self.modmul_const(self.modsub(x, y), s, "yw")

    def nega_ntt(self, vec, psi):
        N = len(vec); L = N.bit_length() - 1; q = self.q
        a = list(vec); t, m = N, 1
        while m < N:
            t //= 2
            for i in range(m):
                S = pow(psi, brv(m + i, L), q)
                for j in range(2 * i * t, 2 * i * t + t):
                    a[j], a[j + t] = self.bu(a[j], a[j + t], S)
            m *= 2
        return a

    def nega_intt(self, vec, psi):
        N = len(vec); L = N.bit_length() - 1; q = self.q
        pinv = pow(psi, -1, q); a = list(vec); t, m = 1, N
        while m > 1:
            h, j1 = m // 2, 0
            for i in range(h):
                S = pow(pinv, brv(h + i, L), q)
                for j in range(j1, j1 + t):
                    a[j], a[j + t] = self.gs_bu(a[j], a[j + t], S)
                j1 += 2 * t
            t *= 2; m = h
        ninv = pow(N, -1, q)
        return [self.modmul_const(a[k], ninv, "yw") for k in range(N)]

    def nega_pipeline(self, N):
        (a, b), psi = self.inputs(N), psi_2n(self.q, N)
        Na, Nb = self.nega_ntt(a, psi), self.nega_ntt(b, psi)
        P = [self.modmul(Na[k], Nb[k]) for k in range(N)]
        return a, b, self.nega_intt(P, psi)


def nega_ref(av, bv, q):
    N = len(av)
    out = []
    for r in range(N):
        s = 0
        for i in range(N):
            j = r - i
            s = s + av[i] * bv[j] if j >= 0 else s - av[i] * bv[j + N]
        out.append(s % q)
    return out


def verify(b, a, bb, c, N, q, tests=1 << 16, exhaustive=False, seed=7, chunk=1 << 18, log=None):
    """Bit-packed check of outputs; also checks that inputs are restored."""
    sim = Packed(b.qc, b.const_init)
    ia = [idx_of(sim.idx, r) for r in a]; ib = [idx_of(sim.idx, r) for r in bb]; ic = [idx_of(sim.idx, r) for r in c]
    rng = np.random.default_rng(seed); n = b.n
    total = q ** (2 * N) if exhaustive else tests
    done = fails = restore = 0
    while done < total:
        B = min(chunk, total - done)
        if exhaustive:
            t = np.arange(done, done + B, dtype=np.int64); d = []
            for _ in range(2 * N): d.append(t % q); t = t // q
            av, bv = d[:N], d[N:]
        else:
            av = [rng.integers(0, q, B, dtype=np.int64) for _ in range(N)]
            bv = [rng.integers(0, q, B, dtype=np.int64) for _ in range(N)]
            if done == 0:
                for i in range(N):
                    av[i][0] = bv[i][0] = 0; av[i][1] = bv[i][1] = q - 1
                    av[i][2] = 1 if i == 0 else 0; bv[i][2] = q - 1
        inp = {}
        for i in range(N):
            for k in range(n):
                inp[ia[i][k]] = ((av[i] >> k) & 1).astype(np.uint8); inp[ib[i][k]] = ((bv[i] >> k) & 1).astype(np.uint8)
        st = sim.run(inp, B)
        exp = nega_ref(av, bv, q)
        for r in range(N):
            fails += int(np.count_nonzero(Packed.read(st, ic[r], B) != exp[r]))
            restore += int(np.count_nonzero(Packed.read(st, ia[r], B) != av[r]))
            restore += int(np.count_nonzero(Packed.read(st, ib[r], B) != bv[r]))
        done += B
        if log and (done // B) % 200 == 0: print(f"{log} {done:,}/{total:,} fails={fails}", flush=True)
    return {"checked": done, "failures": fails, "input_restore_failures": restore, "exhaustive": exhaustive}
