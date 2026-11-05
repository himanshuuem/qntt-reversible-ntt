"""Finding F1: the fold step  hi <- chi + carry  is a genuine addition.
The block-level design counts only 3 Adders in MODMUL (the lo+clo adds).  Here we show that realising the
hi-update by plain wiring (drop / OR / XOR) gives wrong answers, i.e. a further adder is required."""
import json
Q, C = 13, 3
def modmul(x, y, rfold, mode):
    p = x * y; hi, lo = p >> 4, p & 15
    for _ in range(rfold):
        pc = hi * C; chi, clo = pc >> 4, pc & 15
        s = lo + clo; lo, carry = s & 15, s >> 4
        hi = {"add": chi + carry, "drop": chi, "or": chi | carry, "xor": chi ^ carry}[mode]
    return lo % Q
out = {}
for mode in ["add", "drop", "or", "xor"]:
    row = {}
    for rf in range(1, 7):
        bad = sum(modmul(x, y, rf, mode) != (x * y) % Q for x in range(Q) for y in range(Q))
        row[rf] = bad
    out[mode] = row
    print(f"{mode:5s} failures out of 169 for r_fold=1..6:", [row[r] for r in range(1, 7)])
# when is the carry nonzero AND chi nonzero simultaneously (needs a real adder)?
both = sum(1 for x in range(Q) for y in range(Q)
           if (lambda p: (p >> 4) * C >> 4)(x * y) and
           ((x * y & 15) + (((x * y >> 4) * C) & 15)) >> 4)
print("input pairs where round-1 chi>0 and carry=1 simultaneously:", both)
out["pairs_chi_and_carry"] = both
json.dump(out, open("finding_f1.json", "w"), indent=1)
