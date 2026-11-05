# Revision measurements (garbage-free variants and depth)

Run from the `qntt_qiskit` folder:

    python3 revision/clean_depth.py . 65536 revision/clean_depth.json      # depth, Toffoli-depth, garbage-free variants (random check)
    python3 revision/clean_exhaustive.py . V1 revision/clean_V1.json       # all 13^8 inputs, ~3 min
    python3 revision/clean_exhaustive.py . V0 revision/clean_V0.json       # all 13^8 inputs, ~8 min

The garbage-free circuit is compute -> CNOT-copy the 16 output bits -> inverse.
The exhaustive check confirms, on every input, correct outputs, restored inputs,
constant qubits back to |1>, and every other qubit back to |0>.
Results: 0 failures for V0 (4,732 qubits, 42,496 gates) and V1 (1,631 qubits, 13,106 gates).

## Negacyclic transform (ring of ML-KEM-type schemes)

    python3 revision/run_negacyclic.py                                   # cyclic vs negacyclic, V0 / ablation / V1 -> negacyclic_results.json (~1 min)
    python3 revision/nega_exhaustive.py V1 4 5 17 revision/nega_exhaustive_V1_N4_q17.json   # all 17^8 inputs (~1 h)

`negacyclic.py` builds c(x) = a(x) b(x) mod (x^N + 1) in the merged form used by ML-KEM implementations
(Cooley-Tukey forward with psi^brv(k) twiddles, Gentleman-Sande inverse, N^-1 scaling) from the same blocks.
No butterfly twiddle equals 1, so V0 is already the rule-compatible baseline. Result: the cost of a library
without constant-operand blocks is higher than in the cyclic ring at every size measured
(N=4, q=17: 3.07x gates / 3.99x Toffoli vs 2.14x / 2.77x cyclic).
