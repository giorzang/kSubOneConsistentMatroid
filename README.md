# README — One-Consistent Algorithm for k-submodular Maximization under Matroid

## 1. Overview

This repository contains the C++ implementation used in our paper:

***One-Consistent Algorithm for k-submodular Maximization under Matroid***

The code focuses on maximizing **$k$-submodular objectives** under **matroid constraints** (with an emphasis on **partition matroid**) and optional **one-consistent constraints** on label set sizes. The repository provides:

* Objective oracles (**$k$-IC**).
* Data preprocessing utilities (text edge list → binary graph; random partition generation).

---

## 2. Repository structure

```
src/
  algs/
    stream_algos.h       # Algorithms: StreamGreedy, StreamRandom, Algorithm1, Algorithm2
    result.h             # Output struct: objective, errors, queries, time, memory
    runtime_seed.h       # Random seed + timing utilities
  matroid/
    partition_matroid.h  # Partition matroid definition and checks
  objectvalue/
    kic.h                # k-IC oracle (Monte Carlo + OpenMP)
    objectvalue_common.h # Common types for oracles
  mygraph.h              # tinyGraph: binary I/O + adjacency indices + node/edge weights
  kfunctions.h           # k-submodular oracle interface: evaluate/marginal
  kfunctions_impl.h      # Objective binding via compile-time macro (KFUNC_*)
  matroid.h              # Shared matroid utilities
  matroid_impl.h         # Matroid implementations (if separated)
  main.cpp               # Main entry: load graph, run algorithm, export CSV

data/
  *.txt                  # Example edge lists / partitions
  gen_partition.py       # Generate random partition file
  preprocess.cpp         # Preprocess edge list → tinyGraph (.bin)
  preprocess_kic.cpp     # Preprocess for k-IC (if separate pipeline)

Makefile
runSTREAM.sh      	 # Example run scripts (datasets/configs)
```

---

## 3. Requirements

* **C++17** compiler (tested with `g++`).
* **OpenMP** is required for the **$k$-IC** objective (`-fopenmp`).
* Linux is recommended for reproducibility and HPC runs.

---

## 4. Build

Use the provided `Makefile`.

```bash
make
```

Common targets:

```bash
make preprockic
make kic
```

> For `kic`, the build must include OpenMP flags (`-fopenmp`), as enforced by `objectvalue/kic.h`.

---

## 5. Input formats

### 5.1 Edge list (text)

A text file containing one edge per line:

```
u v
```

where `u, v` are integer node ids.

### 5.2 Partition file (for partition matroid)

A text file with `n` lines:

```
node_id part_id
```

used to define the ground set partition for a partition matroid constraint.

To generate a random partition:

```bash
python3 data/gen_partition.py <edge_list.txt> <p> <output-part.txt>
```

### 5.3 Binary graph (`.bin`)

Experiments are typically executed on a binary `tinyGraph` format for faster I/O and evaluation.

Example preprocessing:

```bash
./preproc input_edges.txt output.bin K undirected randomize_node seed
```

---

## 6. Running experiments

`src/main.cpp` is the main entry point. Executables are produced per objective (depending on the build macro). Typical usage:

* Load `.bin` graph,
* Specify algorithm variant and constraints (partition capacities, fairness bounds, `eps`, `seed`, etc.),
* Export results to CSV.

Example environment for k-IC:

```bash
export OMP_NUM_THREADS=32
export KIC_MC=1000
export KIC_SEED=42
./kic graph.bin <args> out.csv
```

Example scripts:

```bash
bash runSTREAM.sh
```

---

## 7. Output metrics

The output CSV (and/or printed logs) typically includes:

* `f_value`: objective value,
* `queries`: number of oracle queries (evaluate/marginal),
* `time_sec`: wall-clock time,
* `matroid_error`: matroid constraint violation,
* `total_error`: aggregated violation.

Exact fields are defined in `src/algs/result.h`.

---

## 8. Reproducibility notes

To reproduce results:

1. Fix random seeds in preprocessing and algorithm runs.
2. For k-IC: fix `KIC_SEED` and `KIC_MC`; report `OMP_NUM_THREADS`.
3. Record compiler version and flags (`-O2`, `-fopenmp`), and machine configuration.
