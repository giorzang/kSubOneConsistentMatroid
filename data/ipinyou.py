#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract (t,a,v) = (BidID, AdvertiserID, BiddingPrice) from iPinYou bid logs (season 2/3).

- Input: directory containing many bid files (*.txt, *.txt.bz2, *.bz2)
- Output: TSV file with columns: t  a  v

Default behavior:
- Deduplicate by (t,a) using max(v) via SQLite (disk-based, RAM-friendly).

Usage examples:
  python extract_tav_from_bids.py --input_dir training3rd --output tav.tsv
  python extract_tav_from_bids.py --input_dir data --output tav.tsv --recursive
  python extract_tav_from_bids.py --input_dir data --output tav.tsv --no_dedup
"""

import argparse
import bz2
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple


def open_maybe_bz2(path: Path):
    """
    Open a file that may be plain text or bz2-compressed, returning a text-mode file object.
    """
    # Common iPinYou: *.txt.bz2 or *.bz2
    if path.suffix == ".bz2" or path.name.endswith(".txt.bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_bid_files(input_dir: Path, recursive: bool, name_contains: str) -> Iterable[Path]:
    """
    Yield candidate bid files. We filter by filename containing `name_contains` (default "bid.").
    """
    if recursive:
        it = input_dir.rglob("*")
    else:
        it = input_dir.glob("*")

    for p in it:
        if not p.is_file():
            continue
        # basic filter: only consider files with "bid." in the name (customizable)
        if name_contains and (name_contains not in p.name):
            continue
        # allow .txt, .bz2, .txt.bz2, or already uncompressed
        yield p


def is_int_str(s: str) -> bool:
    # advertiserID and bidding price are non-negative integers in iPinYou logs
    return s.isdigit()


def parse_tav_from_bid_line(line: str) -> Optional[Tuple[str, int, int]]:
    """
    Parse one bid-log TSV line and extract:
      t = BidID  (field 0)
      a = AdvertiserID (heuristic: last integer before final column UserTags)
      v = BiddingPrice (integer immediately before advertiserID)

    Returns None if parsing fails.
    """
    line = line.rstrip("\n")
    if not line:
        return None

    fields = line.split("\t")
    if len(fields) < 6:
        return None

    t = fields[0]
    if not t:
        return None

    # Heuristic robust across season 2/3:
    # - last field is usually UserTags (may be "null" or comma-separated list)
    # - advertiserID is typically the last *integer* before the last field
    # - bidding price is the previous integer before advertiserID
    last = len(fields) - 1

    # find advertiser index: nearest integer when scanning backward from last-1
    adv_idx = None
    for i in range(last - 1, -1, -1):
        if is_int_str(fields[i]):
            adv_idx = i
            break
    if adv_idx is None:
        return None

    # find bidding price index: previous integer before advertiserID
    bidp_idx = None
    for i in range(adv_idx - 1, -1, -1):
        if is_int_str(fields[i]):
            bidp_idx = i
            break
    if bidp_idx is None:
        return None

    a = int(fields[adv_idx])
    v = int(fields[bidp_idx])

    # sanity checks (optional but useful)
    if a <= 0:
        return None
    if v < 0:
        return None

    return (t, a, v)


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tav (
            t TEXT NOT NULL,
            a INTEGER NOT NULL,
            v INTEGER NOT NULL,
            PRIMARY KEY (t, a)
        );
    """)
    return conn


def upsert_batch(conn: sqlite3.Connection, rows):
    # keep max(v) for each (t,a)
    conn.executemany("""
        INSERT INTO tav (t, a, v) VALUES (?, ?, ?)
        ON CONFLICT(t, a) DO UPDATE SET v = CASE WHEN excluded.v > tav.v THEN excluded.v ELSE tav.v END;
    """, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing bid files (season 2/3).")
    ap.add_argument("--output", required=True, help="Output TSV path (t\\ta\\tv).")
    ap.add_argument("--recursive", action="store_true", help="Search files recursively.")
    ap.add_argument("--name_contains", default="bid.", help='Only process files whose name contains this substring (default "bid.").')
    ap.add_argument("--no_dedup", action="store_true", help="Do not deduplicate; write (t,a,v) rows as they appear (streaming).")
    ap.add_argument("--with_header", action="store_true", help="Write header line: t\\ta\\tv")
    ap.add_argument("--sqlite", default=None, help="SQLite DB path for dedup (default: output + .sqlite).")

    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.output)

    files = list(iter_bid_files(input_dir, args.recursive, args.name_contains))
    files.sort()

    if not files:
        raise SystemExit(f"No files found in {input_dir} matching name_contains={args.name_contains!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_dedup:
        # streaming write: fastest, but may contain duplicates
        total_lines = 0
        kept = 0
        with open(out_path, "wt", encoding="utf-8") as out:
            if args.with_header:
                out.write("t\ta\tv\n")

            for fp in files:
                with open_maybe_bz2(fp) as f:
                    for line in f:
                        total_lines += 1
                        tav = parse_tav_from_bid_line(line)
                        if tav is None:
                            continue
                        t, a, v = tav
                        out.write(f"{t}\t{a}\t{v}\n")
                        kept += 1

        print(f"[OK] Processed {len(files)} files, scanned {total_lines} lines, wrote {kept} rows -> {out_path}")
        return

    # dedup mode: SQLite-based (RAM-friendly)
    db_path = Path(args.sqlite) if args.sqlite else Path(str(out_path) + ".sqlite")
    if db_path.exists():
        db_path.unlink()  # recreate to avoid mixing runs

    conn = init_db(db_path)

    total_lines = 0
    parsed = 0
    batch = []
    BATCH_SIZE = 20000

    try:
        for fp in files:
            with open_maybe_bz2(fp) as f:
                for line in f:
                    total_lines += 1
                    tav = parse_tav_from_bid_line(line)
                    if tav is None:
                        continue
                    batch.append(tav)
                    parsed += 1
                    if len(batch) >= BATCH_SIZE:
                        upsert_batch(conn, batch)
                        conn.commit()
                        batch.clear()

        if batch:
            upsert_batch(conn, batch)
            conn.commit()
            batch.clear()

        # export to output TSV
        with open(out_path, "wt", encoding="utf-8") as out:
            if args.with_header:
                out.write("t\ta\tv\n")
            for t, a, v in conn.execute("SELECT t, a, v FROM tav ORDER BY a, t;"):
                out.write(f"{t}\t{a}\t{v}\n")

        # stats
        cur = conn.execute("SELECT COUNT(*) FROM tav;")
        distinct_rows = cur.fetchone()[0]

        print(f"[OK] Processed {len(files)} files")
        print(f"     scanned lines   : {total_lines}")
        print(f"     parsed (t,a,v)  : {parsed}")
        print(f"     distinct (t,a)  : {distinct_rows}  (kept max v)")
        print(f"     output          : {out_path}")
        print(f"     sqlite db       : {db_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
