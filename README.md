# Qiskit realisation of the reversible 4-point NTT multiplier (q = 13, w = 5)

Gate-level (X / CNOT / Toffoli only) Qiskit implementation of the internship report's design:
**four blocks -> three macros -> Butterfly Unit -> full NTT pipeline**, with verification.

    pip install -r requirements.txt
    python3 tests/test_qntt.py        # fast unit tests (~10 s)
    python3 run_all.py                # blocks/macros exhaustive + Aer-MPS + Statevector + Clifford+T + figures (~6 min, Aer MPS dominates)
    python3 verify_exhaustive.py 20 200   # EXHAUSTIVE: all 13^8 = 815,730,721 inputs; resumable, run repeatedly (~8 min total)
    python3 run_v1.py                 # constant-aware variant V1 + cumulative ablation -> results_v1.json
    python3 verify_exhaustive.py 20 800 v1   # exhaustive 13^8 check of V1 (~4 min)
    python3 finding_f1.py             # why the report's MODMUL accounting needs one more adder

## Use it
```python
from qntt import Builder
b = Builder()                      # Builder(named=True) gives readable registers for drawing small circuits
a, bb, c = b.pipeline()            # c(x) = a(x)*b(x) mod (x^4-1), coefficients mod 13
qc = b.qc                          # a normal qiskit.QuantumCircuit (4,716 qubits, 21,240 gates)
# constants must be prepared as |1> on qubits listed in b.const_init (input preparation, like the report's "constant bits")
```

## Layout
| file | role |
|---|---|
| `qntt/builder.py` | Adder, Subtractor, Multiplier, Modulo; MODADD/MODSUB/MODMUL; BU; pipeline; ledger of block instances & gates |
| `qntt/sim.py` | exact classical simulator for X/CX/CCX circuits; Aer-MPS runner |
| `qntt/batch.py` | bit-packed batch simulator (64 vectors per word) for exhaustive checks |
| `run_all.py`, `verify_exhaustive.py`, `finding_f1.py` | verification + results (`results.json`, `exhaustive_result.json`, `finding_f1.json`) |
| `figures/` | Qiskit-drawn gate-level diagrams of the Adder, Subtractor, Multiplier blocks |

## What is and is not faithful to the report  (read this before citing numbers)
* The report's four blocks are SVG netlists (22/30/82/248 gates) that were **not available** here. The blocks in
  this package are **functional equivalents** with the same black-box interface (4-bit operands; Cout/Borrow;
  8-bit product; 4-bit remainder; inputs restored; garbage never uncomputed). Their gate counts are
  29 / 34 / 91 / 212, not 22 / 30 / 82 / 248. Everything above block level follows the report.
* **Finding F1.** The report counts 3 Adders per MODMUL, but the fold step `hi <- chi + carry` is a real addition.
  Realising it by wiring (drop / OR / XOR of the carry) gives wrong results (`finding_f1.py`); a faithful
  reuse-only realisation needs 3 more Adder instances per MODMUL (144 Adders in total, not 84; +1,320 gates at
  Table I costs: 19,032 -> 20,352).
* Constants (c=3, 16-q=3, q=13, twiddles, N^-1) are prepared with X gates by the harness and are not counted as
  circuit gates (the report counts them as constant ancilla inputs).
* Classical-simulation results are exact for these circuits (they contain only X/CNOT/Toffoli); they are
  cross-checked against qiskit-aer's MPS simulator and, for the Adder/Subtractor, a 18/19-qubit Statevector
  run on a uniform superposition of all 256 inputs.

## Variant V1 (constant-aware) and the price of the reuse-only rule
`qntt/const_aware.py` (`BuilderCA`) specialises every multiplication with a compile-time constant operand (identity twiddles,
ANF-lookup constant multipliers, CNOT constant selectors, lookup fold multiplier) and keeps the generic blocks only for the four
variable x variable pointwise products. It drops the report's "no gate outside the blocks" rule on purpose.
`BuilderCA(identity=..., lookup_twiddle=..., cnot_selectors=..., lookup_fold=...)` exposes each technique as a switch;
all four off reproduces V0 exactly (21,240 gates) - a built-in regression check.

| config | gates | Toffoli | T-count | qubits |
|---|---|---|---|---|
| V0 reuse-only | 21,240 | 7,600 | 53,200 | 4,716 |
| + identity twiddles | 14,490 | 5,152 | 36,064 | 3,227 |
| + lookup constant multipliers | 9,437 | 3,295 | 23,065 | 2,135 |
| + CNOT constant selectors | 7,301 | 2,335 | 16,345 | 1,679 |
| + lookup fold (= V1) | 6,545 | 1,951 | 13,657 | 1,615 |

V1 verified on all 13^8 inputs (0 failures) and by Aer-MPS. None of the individual techniques is new (see report Section VI-E);
the lookup constant multipliers rely on 4-bit operands and do not scale to Kyber-size words.

## Parametric generator, scaling and like-for-like comparison  (`qntt/generic.py`, `qntt/gverify.py`, `run_scaling.py`)
`GBuilder(n, q, v1=False)` builds the design for ANY word width n, prime q with 2^(n-1) < q < 2^n, and power-of-two N | q-1
(`.pipeline(N)` = full multiplier, `.ntt_only(N)` = forward transform). For (n=4, q=13) it reproduces V0/V1 gate-for-gate (21,240 / 6,545).
`python3 run_scaling.py lu | n5 | n7` regenerates `results_scaling.json` (Tables XVII-XIX of the report).

| N | q (n) | gates V0 -> V1 | saved | T-count V0 -> V1 | verification |
|---|---|---|---|---|---|
| 4 | 5 (3) | 9,192 -> 3,759 | 59.1% | 22,176 -> 7,777 | all 5^8 inputs |
| 4 | 13 (4) | 21,240 -> 6,545 | 69.2% | 53,200 -> 13,657 | all 13^8 inputs |
| 4 | 17 (5) | 36,164 -> 11,246 | 68.9% | 92,596 -> 22,057 | random + exhaustive components |
| 8 | 17 (5) | 97,708 -> 28,878 | 70.4% | 249,788 -> 57,113 | random + exhaustive components |
| 16 | 17 (5) | 246,176 -> 70,173 | 71.5% | 628,768 -> 139,874 | random + exhaustive components |
| 4 | 97 (7) | 67,180 -> 26,943 | 59.9% | 175,364 -> 48,300 | random + exhaustive components |
| 8 | 97 (7) | 181,412 -> 68,208 | 62.4% | 472,780 -> 124,936 | random + exhaustive components |

Like-for-like at Lu et al.'s benchmark (4-point NTT, q=5, root 2; all 625 inputs verified): V0 = 2,352 gates, V1 = 932 gates
vs. 1,980 (optimised) / 3,548 (unoptimised) reported in the paper. Indicative only: block designs and register widths differ.
For N >= 8 correctness = exhaustively verified components + randomly tested composition (full-input exhaustion is infeasible).
Lookup constant multipliers cost ~2x per extra bit of word width (7 gates at n=3 -> 1,136 at n=8); break-even with a generic MODMUL
is extrapolated (not measured) near n = 9-10.

## Paper and revision measurements

This code accompanies the paper *Reversible NTT Polynomial Multiplication from Fixed Arithmetic Blocks:
Exhaustive Gate-Level Verification and the Cost of Constant-Oblivious Reuse*
(J. S. Khatun, Himanshu, S. Ghosh; IIEST Shibpur).

`revision/` adds the garbage-free (compute-copy-uncompute) variants, depth and Toffoli-depth, and an
exhaustive check over all 13^8 inputs that outputs are correct and every input, constant and ancilla
qubit is restored. See `revision/README.md`.

Note on naming: identifiers such as `THESIS_TABLE_I`, `thesis_accounting_gates` and
`test_pipeline_thesis_example_and_random` refer to the documented block-level design
(Table II of the paper). They are kept unchanged so that the stored result files stay reproducible.
