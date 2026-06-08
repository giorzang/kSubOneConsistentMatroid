#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sensor_k3.py

Input raw sensor log (whitespace-delimited, 8 fields):
  date time epoch moteid temperature humidity light voltage

We build K=3 channels: temperature, humidity, light (ignore voltage).
Steps:
  - optional outlier filtering (treat as missing)
  - discretize: bin = floor(value / delta_i)
  - empirical entropy H = - sum_b p_b log(p_b)   (base e or 2)

Outputs (for strict preprocess_sensor.cpp pipeline):
  - entropy.txt : moteid  H_temp  H_hum  H_light
  - edges.txt   : moteid moteid 1.0        (dummy self-loop; enough to declare nodes)
  - parts.txt   : moteid part_id           (balanced random partition into p parts)

Usage:
  python3 prepare_sensor_k3.py data.txt --p 10 --seed 42 --out_dir out

Then:
  ./preprocess_sensor out/edges.txt out/parts.txt out/entropy.txt sensor.bin 3 1 0 42
"""

import os
import math
import argparse
import random
from collections import defaultdict, Counter

def entropy_from_counts(cnt: Counter, base: str) -> float:
    total = sum(cnt.values())
    if total <= 0:
        return 0.0
    if base == "2":
        logf = lambda x: math.log(x, 2)
    else:
        logf = math.log
    H = 0.0
    for c in cnt.values():
        p = c / total
        H -= p * logf(p)
    return H

def discretize(v: float, delta: float) -> int:
    return math.floor(v / delta)

def is_finite(x: float) -> bool:
    return math.isfinite(x)

def sanitize_entropy(h: float) -> float:
    if not math.isfinite(h):
        return 0.0
    return h if h >= 0.0 else 0.0

def balanced_partition(nodes, p: int, seed: int):
    # Deterministic across runs: sort then shuffle by seed
    nodes = sorted(nodes)
    rng = random.Random(seed)
    rng.shuffle(nodes)

    n = len(nodes)
    base = n // p
    rem = n % p
    parts = {}
    idx = 0
    for g in range(p):
        size_g = base + (1 if g < rem else 0)
        for _ in range(size_g):
            parts[nodes[idx]] = g
            idx += 1
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", help="raw sensor log txt")
    ap.add_argument("--out_dir", default=".", help="output directory")

    # K=3 deltas
    ap.add_argument("--delta_temp", type=float, default=0.5)
    ap.add_argument("--delta_hum", type=float, default=1.0)
    ap.add_argument("--delta_light", type=float, default=5.0)

    # entropy log base
    ap.add_argument("--log_base", choices=["e", "2"], default="2")

    # parts
    ap.add_argument("--p", type=int, default=1, help="number of parts (>=1)")
    ap.add_argument("--seed", type=int, default=42)

    # outlier filters (treat outside as missing)
    ap.add_argument("--temp_min", type=float, default=-40.0)
    ap.add_argument("--temp_max", type=float, default=125.0)
    ap.add_argument("--hum_min", type=float, default=0.0)
    ap.add_argument("--hum_max", type=float, default=100.0)
    ap.add_argument("--light_min", type=float, default=0.0)
    ap.add_argument("--light_max", type=float, default=1e18)

    args = ap.parse_args()

    if args.p <= 0:
        raise SystemExit("Error: --p must be >= 1")
    if args.delta_temp <= 0 or args.delta_hum <= 0 or args.delta_light <= 0:
        raise SystemExit("Error: all deltas must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)
    out_entropy = os.path.join(args.out_dir, "entropy.txt")
    out_edges   = os.path.join(args.out_dir, "edges.txt")
    out_parts   = os.path.join(args.out_dir, "parts.txt")

    # bins per moteid
    bins_temp = defaultdict(Counter)
    bins_hum  = defaultdict(Counter)
    bins_light= defaultdict(Counter)

    moteids = set()

    with open(args.input_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            # expect >= 8 columns
            if len(cols) < 8:
                continue

            try:
                moteid = int(cols[3])
                temp   = float(cols[4])
                hum    = float(cols[5])
                light  = float(cols[6])
            except Exception:
                continue

            moteids.add(moteid)

            # temperature
            if is_finite(temp) and (args.temp_min <= temp <= args.temp_max):
                b = discretize(temp, args.delta_temp)
                bins_temp[moteid][b] += 1

            # humidity
            if is_finite(hum) and (args.hum_min <= hum <= args.hum_max):
                b = discretize(hum, args.delta_hum)
                bins_hum[moteid][b] += 1

            # light
            if is_finite(light) and (args.light_min <= light <= args.light_max):
                b = discretize(light, args.delta_light)
                bins_light[moteid][b] += 1

    if not moteids:
        raise SystemExit("Error: no moteid parsed from input.")

    moteids_sorted = sorted(moteids)

    # 1) entropy.txt (K=3)
    with open(out_entropy, "w", encoding="utf-8") as out:
        for mid in moteids_sorted:
            Ht = sanitize_entropy(entropy_from_counts(bins_temp[mid], args.log_base))
            Hh = sanitize_entropy(entropy_from_counts(bins_hum[mid], args.log_base))
            Hl = sanitize_entropy(entropy_from_counts(bins_light[mid], args.log_base))
            out.write(f"{mid} {Ht:.10f} {Hh:.10f} {Hl:.10f}\n")

    # 2) edges.txt dummy self-loops
    with open(out_edges, "w", encoding="utf-8") as out:
        for mid in moteids_sorted:
            out.write(f"{mid} {mid} 1.0\n")

    # 3) parts.txt balanced random
    part_map = balanced_partition(moteids_sorted, args.p, args.seed)
    with open(out_parts, "w", encoding="utf-8") as out:
        for mid in moteids_sorted:
            out.write(f"{mid} {part_map[mid]}\n")

    print("OK.")
    print(f"  |V| (moteids) = {len(moteids_sorted)}")
    print(f"  K             = 3  (temp, hum, light)")
    print(f"  p             = {args.p}, seed = {args.seed}")
    print("  Outputs:")
    print(f"    {out_entropy}")
    print(f"    {out_edges}")
    print(f"    {out_parts}")
    print("Next step:")
    print(f"  ./preprocess_sensor {out_edges} {out_parts} {out_entropy} sensor.bin 3 1 0 {args.seed}")

if __name__ == "__main__":
    main()
