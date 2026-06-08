#!/usr/bin/env python3
# preIntelLab.py  (pairwise-cov + SPD projection, output only .bin)
#
# Build a tinyGraph (.bin) encoding per-topic covariance matrices Sigma^(t)
# for Intel Lab dataset (temperature/humidity/light).
#
# Output binary format MUST match mygraph::tinyGraph::write_binary:
# [uint64 n][uint64 m][uint64 K][uint8 undirected]
# nodes:  K doubles weights  (we store diag Sigma[u,u])
#         K doubles alpha    (unused = 1.0)
# edges:  [uint32 u][uint32 v] then K doubles weights (we store Sigma[u,v], u<v)
#
# Graph is undirected COMPLETE: m = n*(n-1)/2, edges ordered by (u=0..n-1, v=u+1..n-1).
#
import argparse
import struct
import numpy as np
import pandas as pd

COLS = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]

def _read_intellab(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="python")

    # Keep rows with all 3 modalities present (per-row).
    # Note: we DO NOT require all motes to be present at the same epoch.
    df = df.dropna(subset=["epoch", "moteid", "temperature", "humidity", "light"]).copy()

    df["moteid"] = df["moteid"].astype(int)

    # epoch sometimes appears as float in some dumps; cast safely to int64
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(np.int64)
    return df

def _pivot_matrix(df: pd.DataFrame, mote2idx: dict[int, int], col: str) -> pd.DataFrame:
    tmp = df[["epoch", "moteid", col]].copy()
    tmp["idx"] = tmp["moteid"].map(mote2idx)
    # mean if multiple records at same epoch for a mote
    piv = tmp.pivot_table(index="epoch", columns="idx", values=col, aggfunc="mean")
    # ensure columns are 0..n-1 (some may be missing -> reindex)
    return piv

def _nearest_spd_by_eig(S: np.ndarray, eps: float) -> np.ndarray:
    """Project symmetric matrix to SPD by eigenvalue clipping."""
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def _cov_spd_from_pivot_pairwise(piv: pd.DataFrame,
                                ridge: float,
                                center_only: bool,
                                min_periods: int,
                                spd_eps: float) -> np.ndarray:
    """
    Compute covariance matrix (n x n) from a pivot table (epochs x n),
    allowing missing values via pairwise deletion.
    """
    if piv.shape[1] < 2:
        raise ValueError("Pivot has <2 sensors columns; cannot build covariance.")

    # Center (and optionally standardize) per sensor column; NaNs remain
    if center_only:
        X = piv - piv.mean(axis=0)
    else:
        X = (piv - piv.mean(axis=0)) / (piv.std(axis=0, ddof=1) + 1e-12)

    # Pairwise covariance using overlapping epochs; yields NaN if overlap < min_periods
    Sdf = X.cov(min_periods=min_periods)
    S = Sdf.to_numpy(dtype=float)
    n = S.shape[0]

    # Fill NaNs: off-diagonal -> 0, diagonal -> small positive
    if np.isnan(S).any():
        for i in range(n):
            if np.isnan(S[i, i]) or S[i, i] <= 0.0:
                S[i, i] = 1e-12
        nan_mask = np.isnan(S)
        S[nan_mask] = 0.0

    # Symmetrize + ridge
    S = 0.5 * (S + S.T)
    S = S + ridge * np.eye(n)

    # Ensure SPD (Cholesky-safe)
    S = _nearest_spd_by_eig(S, eps=spd_eps)
    return S

def write_tinygraph_complete_cov(out_bin: str, Sigmas: list[np.ndarray]) -> None:
    K = len(Sigmas)
    n = Sigmas[0].shape[0]
    assert K > 0
    assert all(S.shape == (n, n) for S in Sigmas)

    m = n * (n - 1) // 2  # complete graph u<v

    with open(out_bin, "wb") as f:
        # header: [uint64 n][uint64 m][uint64 K][uint8 undirected]
        f.write(struct.pack("<QQQB", n, m, K, 1))  # undirected=1

        # nodes: weights (diag) then alpha (unused=1)
        for u in range(n):
            for t in range(K):
                diag = float(Sigmas[t][u, u])
                if not np.isfinite(diag) or diag <= 0.0:
                    diag = 1e-12
                f.write(struct.pack("<d", diag))
            for t in range(K):
                f.write(struct.pack("<d", 1.0))

        # edges: store covariance off-diagonal, in required complete-edge order
        for u in range(n):
            for v in range(u + 1, n):
                f.write(struct.pack("<II", u, v))
                for t in range(K):
                    val = float(Sigmas[t][u, v])
                    if not np.isfinite(val):
                        val = 0.0
                    f.write(struct.pack("<d", val))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to Intel Lab data (e.g., Intellab.txt)")
    ap.add_argument("--out", dest="out_bin", required=True, help="Output .bin for tinyGraph")
    ap.add_argument("--min-samples", type=int, default=200,
                    help="Filter out motes with fewer than this many records (per-row).")
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="Add ridge*I to covariance for numerical stability.")
    ap.add_argument("--center-only", action="store_true",
                    help="If set: mean-center only (true covariance). Otherwise standardize -> correlation-like.")
    ap.add_argument("--min-periods", type=int, default=30,
                    help="Min overlapping epochs for a pairwise covariance entry; else set to 0 (off-diag) / tiny (diag).")
    ap.add_argument("--spd-eps", type=float, default=1e-10,
                    help="Eigenvalue floor when projecting to SPD (Cholesky-safe).")
    args = ap.parse_args()

    df = _read_intellab(args.in_path)

    # Filter motes with enough rows
    cnt = df.groupby("moteid").size()
    keep = cnt[cnt >= args.min_samples].index
    df = df[df["moteid"].isin(keep)].copy()

    moteids = np.array(sorted(df["moteid"].unique()), dtype=int)
    n = len(moteids)
    if n < 2:
        raise ValueError("Need at least 2 motes after filtering. Try lowering --min-samples.")

    mote2idx = {m: i for i, m in enumerate(moteids)}

    # Build pivots (epochs x n), NaNs allowed
    P_temp = _pivot_matrix(df, mote2idx, "temperature")
    P_hum  = _pivot_matrix(df, mote2idx, "humidity")
    P_lgt  = _pivot_matrix(df, mote2idx, "light")

    # Ensure columns are exactly 0..n-1 (some may be absent in pivot due to missing)
    P_temp = P_temp.reindex(columns=range(n))
    P_hum  = P_hum.reindex(columns=range(n))
    P_lgt  = P_lgt.reindex(columns=range(n))

    Sig_temp = _cov_spd_from_pivot_pairwise(P_temp, ridge=args.ridge, center_only=args.center_only,
                                            min_periods=args.min_periods, spd_eps=args.spd_eps)
    Sig_hum  = _cov_spd_from_pivot_pairwise(P_hum,  ridge=args.ridge, center_only=args.center_only,
                                            min_periods=args.min_periods, spd_eps=args.spd_eps)
    Sig_lgt  = _cov_spd_from_pivot_pairwise(P_lgt,  ridge=args.ridge, center_only=args.center_only,
                                            min_periods=args.min_periods, spd_eps=args.spd_eps)

    write_tinygraph_complete_cov(args.out_bin, [Sig_temp, Sig_hum, Sig_lgt])

    # Diagnostics (optional but useful)
    m = n * (n - 1) // 2
    print(f"[OK] wrote {args.out_bin} with n={n}, K=3, m={m}")
    print(f"[INFO] min_samples={args.min_samples}, ridge={args.ridge}, center_only={args.center_only}, "
          f"min_periods={args.min_periods}, spd_eps={args.spd_eps}")
    print("[NOTE] internal node_id is 0..n-1 after filtering; original moteids were:", moteids.tolist())

if __name__ == "__main__":
    main()
