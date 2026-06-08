#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input file with lines: t a v (tab/space separated)")
    ap.add_argument("output", help="Output file with lines: t_id a_id v_norm")
    ap.add_argument("--save-maps-prefix", default="",
                    help="If set, save mapping files: <prefix>.t.map and <prefix>.a.map")
    ap.add_argument("--float-format", default="{:.10f}",
                    help="Format for v_norm (default '{:.10f}')")
    args = ap.parse_args()

    # -------- pass 1: compute vmax --------
    vmax = 0.0
    total = 0
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Line {line_no}: expected >=3 columns, got {len(parts)}")
            try:
                v = float(parts[2])
            except Exception as e:
                raise ValueError(f"Line {line_no}: cannot parse v={parts[2]!r} as float") from e
            if v > vmax:
                vmax = v
            total += 1

    # avoid division by zero
    def norm(v: float) -> float:
        if vmax <= 0.0:
            return 0.0
        x = v / vmax
        # numerical safety
        if x < 0.0: x = 0.0
        if x > 1.0: x = 1.0
        return x

    # -------- pass 2: re-index t and a, keep all rows --------
    t_map = {}  # old_t -> new_t
    a_map = {}  # old_a -> new_a

    t_order = []  # to save maps in id order
    a_order = []

    fmt = args.float_format

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Line {line_no}: expected >=3 columns, got {len(parts)}")

            t_old, a_old = parts[0], parts[1]
            try:
                v = float(parts[2])
            except Exception as e:
                raise ValueError(f"Line {line_no}: cannot parse v={parts[2]!r} as float") from e

            if t_old not in t_map:
                t_map[t_old] = len(t_map)
                t_order.append(t_old)

            if a_old not in a_map:
                a_map[a_old] = len(a_map)
                a_order.append(a_old)

            t_id = t_map[t_old]
            a_id = a_map[a_old]
            v_norm = norm(v)

            fout.write(f"{t_id}\t{a_id}\t{fmt.format(v_norm)}\n")

    # -------- optional: save mapping files --------
    if args.save_maps_prefix:
        with open(args.save_maps_prefix + ".t.map", "w", encoding="utf-8") as f:
            for new_id, old in enumerate(t_order):
                f.write(f"{old}\t{new_id}\n")
        with open(args.save_maps_prefix + ".a.map", "w", encoding="utf-8") as f:
            for new_id, old in enumerate(a_order):
                f.write(f"{old}\t{new_id}\n")

    print("[OK]")
    print(f"  lines processed : {total}")
    print(f"  unique t        : {len(t_map)}")
    print(f"  unique a        : {len(a_map)}")
    print(f"  vmax            : {vmax}")
    print(f"  output          : {args.output}")

if __name__ == "__main__":
    main()
