# Revision measurements (garbage-free variants and depth)

Run from the `qntt_qiskit` folder:

    python3 revision/clean_depth.py . 65536 revision/clean_depth.json      # depth, Toffoli-depth, garbage-free variants (random check)
    python3 revision/clean_exhaustive.py . V1 revision/clean_V1.json       # all 13^8 inputs, ~3 min
    python3 revision/clean_exhaustive.py . V0 revision/clean_V0.json       # all 13^8 inputs, ~8 min

The garbage-free circuit is compute -> CNOT-copy the 16 output bits -> inverse.
The exhaustive check confirms, on every input, correct outputs, restored inputs,
constant qubits back to |1>, and every other qubit back to |0>.
Results: 0 failures for V0 (4,732 qubits, 42,496 gates) and V1 (1,631 qubits, 13,106 gates).
