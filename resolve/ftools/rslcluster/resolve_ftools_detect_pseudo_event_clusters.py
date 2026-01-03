#!/usr/bin/env python 

"""
resolve_ftools_detect_pseudo_event_clusters.py

XRISM/Resolve event clustering tool (per-pixel), with debug-friendly annotations.

Key improvements:
- True cluster ID: all events in the same cluster share the same cluster_id (= start row index in that pixel stream).
- Member index: 1..N within a cluster.
- CL_MODE: 0 none, 1 large, 2 small (origin of clustering decision).
- CL_REASON: bitmask that records which conditions were true and whether the row was used as START/CONTINUE.
"""

import os
import argparse
import datetime
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time

# -------------------------
# Plot settings
# -------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 8})



# -------------------------
# CL_REASON bit definitions
# -------------------------
# Start-condition related
BIT_IS_LP = 0               # ITYPE == 3
BIT_IS_LS = 1               # ITYPE == 4
BIT_LO_GT_LARGE = 2         # LO_RES_PH > threshold_large
BIT_LO_LE_LARGE = 3         # LO_RES_PH <= threshold_large
BIT_NT_LT_LIMIT = 4         # NEXT_INTERVAL < interval_limit
BIT_NT_EQ_SECONDTHRES = 5         # NEXT_INTERVAL == SECOND_THRES_USE_LEN (75) or == unknown clusters (105)
BIT_RT_LT_MIN = 6           # RISE_TIME < rt_min
BIT_RT_GT_MAX = 7           # RISE_TIME > rt_max

# Continue-condition related
BIT_ITYPE_IN_34 = 8         # ITYPE in (3,4)
BIT_PR_LT_LIMIT = 9         # PREV_INTERVAL < interval_limit
BIT_PR_EQ_SECONDTHRES = 10  # PREV_INTERVAL == SECOND_THRES_USE_LEN (75) or == unknown clusters (105)

# Outcome flags
BIT_START_OK = 14           # this row is used as a cluster start
BIT_CONT_OK = 15            # this row is used as a cluster continuation

# CL_MODE codes
CLMODE_NONE = 0
CLMODE_LARGE = 1
CLMODE_SMALL = 2

UNKNOWNS_SECOND_THRES_USE_LEN = 105

def _setbit(x: int, bit: int) -> int:
    return x | (1 << bit)

@dataclass(frozen=True)
class Params:
    threshold_large: int = 12235
    threshold_small: int = 3000  # currently unused in logic (kept for compatibility / future tuning)
    interval_limit: int = 40
    SECOND_THRES_USE_LEN: int = 75
    SECOND_THRES_UNKNOWN: int = 105
    rt_min: int = 32
    rt_max: int = 58
    mjdref: int = 58484

def compute_datetime_array(mjdref: int, times: np.ndarray):
    """Convert TIME values (sec) to datetime using MJD reference."""
    ref_time = Time(mjdref, format="mjd")
    return [ref_time.datetime + datetime.timedelta(seconds=float(t)) for t in times]


def ensure_directory(path: str, verbose: bool = False) -> None:
    """Create output directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            print(f"Created directory: {path}")
    elif verbose:
        print(f"Directory already exists: {path}")


def identify_clusters(
    events,
    mode: str,
    params: Params,
    debug: bool = False,
):
    """
    Identify clusters in a single-pixel event stream.

    Returns per-row arrays (length = len(events)):
      - cluster_ids: 0 if not clustered, otherwise cluster_id (= start row index in this pixel stream)
      - member_ids: 0 if not clustered, otherwise 1..N within the cluster
      - cl_mode: 0 none, 1 large, 2 small (mode code applied to this clustered row; also stored for noncluster rows)
      - cl_reason: bitmask describing which conditions were true and whether START/CONT was used
      - prev_lo_res_ph: previous row LO_RES_PH (0 for first row)
      - prev_itype: previous row ITYPE (0 for first row)
    """
    n = len(events)

    cluster_ids = np.zeros(n, dtype=np.int64)
    member_ids = np.zeros(n, dtype=np.int64)
    cl_mode = np.zeros(n, dtype=np.int16)
    cl_reason = np.zeros(n, dtype=np.int32)

    prev_lo_res_ph = np.zeros(n, dtype=np.int64)
    prev_itype = np.zeros(n, dtype=np.int64)

    if mode == "large":
        mode_code = CLMODE_LARGE
    elif mode == "small":
        mode_code = CLMODE_SMALL
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")

    i = 0
    while i < n:
        # Build diagnostic bits for the current row (even if it won't become a cluster)
        reason = 0
        itype = int(events["ITYPE"][i])
        lo = int(events["LO_RES_PH"][i])
        nt = int(events["NEXT_INTERVAL"][i])
        rt = int(events["RISE_TIME"][i])

        if itype == 3:
            reason = _setbit(reason, BIT_IS_LP)
        if itype == 4:
            reason = _setbit(reason, BIT_IS_LS)

        if lo > params.threshold_large:
            reason = _setbit(reason, BIT_LO_GT_LARGE)
        if lo <= params.threshold_large:
            reason = _setbit(reason, BIT_LO_LE_LARGE)

        if nt < params.interval_limit:
            reason = _setbit(reason, BIT_NT_LT_LIMIT)
        if nt == params.SECOND_THRES_USE_LEN:
            reason = _setbit(reason, BIT_NT_EQ_SECONDTHRES)

        if rt < params.rt_min:
            reason = _setbit(reason, BIT_RT_LT_MIN)
        if rt > params.rt_max:
            reason = _setbit(reason, BIT_RT_GT_MAX)

        # Decide start condition
        if mode == "large":
            is_cluster_start = (itype == 3 and lo > params.threshold_large)
        else:
            # small
            is_cluster_start = (
                (itype == 3)
                and (lo <= params.threshold_large)
                and ((nt < params.interval_limit) or (nt == params.SECOND_THRES_USE_LEN) or (nt == params.SECOND_THRES_UNKNOWN))
                and ((rt < params.rt_min) or (rt > params.rt_max))
            )

        # Always store diagnostic bits for noncluster rows too
        cl_mode[i] = mode_code
        cl_reason[i] = reason

        if not is_cluster_start:
            i += 1
            continue

        # Start cluster
        cluster_id = i  # stable ID: start row index within this pixel stream
        reason_start = _setbit(reason, BIT_START_OK)

        if debug:
            print(
                f"[{mode}] Cluster start @ {i} | ITYPE={itype} LO={lo} NT={nt} RT={rt}"
            )

        cluster_ids[i] = cluster_id
        member_ids[i] = 1
        cl_mode[i] = mode_code
        cl_reason[i] = reason_start

        member_count = 1
        i += 1

        # Continue cluster while condition holds
        while i < n:
            itype_j = int(events["ITYPE"][i])
            pr = int(events["PREV_INTERVAL"][i])

            cont_reason = 0
            if itype_j in (3, 4):
                cont_reason = _setbit(cont_reason, BIT_ITYPE_IN_34)
            if pr < params.interval_limit:
                cont_reason = _setbit(cont_reason, BIT_PR_LT_LIMIT)
            if pr == params.SECOND_THRES_USE_LEN:
                cont_reason = _setbit(cont_reason, BIT_PR_EQ_SECONDTHRES)

            cont_ok = (itype_j in (3, 4)) and (
                (pr < params.interval_limit) or (pr == params.SECOND_THRES_USE_LEN) or (pr == params.SECOND_THRES_UNKNOWN) 
            )
            if not cont_ok:
                # still keep start-style diag bits for this row already stored above (will be overwritten below only if cont_ok)
                break

            cont_reason = _setbit(cont_reason, BIT_CONT_OK)

            if debug:
                lo_j = int(events["LO_RES_PH"][i])
                print(
                    f"[{mode}] Cluster cont  @ {i} | ITYPE={itype_j} LO={lo_j} PR={pr}"
                )

            member_count += 1
            cluster_ids[i] = cluster_id
            member_ids[i] = member_count
            cl_mode[i] = mode_code
            cl_reason[i] = cont_reason
            i += 1

    # previous-row diagnostics
    for k in range(1, n):
        prev_lo_res_ph[k] = int(events["LO_RES_PH"][k - 1])
        prev_itype[k] = int(events["ITYPE"][k - 1])

    return cluster_ids, member_ids, cl_mode, cl_reason, prev_lo_res_ph, prev_itype


def plot_cluster(
    events,
    cluster_ids: np.ndarray,
    pixel: int,
    output_dir: str,
    mjdref: int,
    show: bool = False,
    outname: str = "cluster",
    input_fits: str = "input.fits"
):
    """
    Diagnostic plot per pixel:
      (1) TIME vs LO_RES_PH
      (2) NEXT_INTERVAL vs LO_RES_PH
    Highlight clustered events (cluster_ids > 0).
    """
    # Separate Lp and Ls
    Lp_mask = events["ITYPE"] == 3
    Ls_mask = events["ITYPE"] == 4
    events_Lp = events[Lp_mask]
    events_Ls = events[Ls_mask]

    times_Lp = compute_datetime_array(mjdref, events_Lp["TIME"])
    times_Ls = compute_datetime_array(mjdref, events_Ls["TIME"])

    clustered_mask = np.asarray(cluster_ids) > 0
    clustered = events[clustered_mask]
    cl_Lp = clustered[clustered["ITYPE"] == 3]
    cl_Ls = clustered[clustered["ITYPE"] == 4]
    times_cl_Lp = compute_datetime_array(mjdref, cl_Lp["TIME"])
    times_cl_Ls = compute_datetime_array(mjdref, cl_Ls["TIME"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # (1) TIME vs LO_RES_PH
    axes[0].plot(times_Lp, events_Lp["LO_RES_PH"], ".", alpha=0.6, ms=1, label="Lp")
    axes[0].plot(times_Ls, events_Ls["LO_RES_PH"], ".", alpha=0.6, ms=1, label="Ls")
    axes[0].scatter(times_cl_Lp, cl_Lp["LO_RES_PH"], s=10, label="Clustered Lp")
    axes[0].scatter(times_cl_Ls, cl_Ls["LO_RES_PH"], s=10, label="Clustered Ls")
    axes[0].set_xlabel("TIME")
    axes[0].set_ylabel("LO_RES_PH")
    axes[0].set_title(f"{outname} : Pixel {pixel} - TIME")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # (2) NEXT_INTERVAL vs LO_RES_PH
    axes[1].plot(events_Lp["NEXT_INTERVAL"], events_Lp["LO_RES_PH"], ".", alpha=0.6, ms=2)
    axes[1].plot(events_Ls["NEXT_INTERVAL"], events_Ls["LO_RES_PH"], ".", alpha=0.6, ms=2)
    axes[1].scatter(cl_Lp["NEXT_INTERVAL"], cl_Lp["LO_RES_PH"], s=10)
    axes[1].scatter(cl_Ls["NEXT_INTERVAL"], cl_Ls["LO_RES_PH"], s=10)
    axes[1].set_xlabel("NEXT_INTERVAL")
    axes[1].set_xlim(0, 255)
    axes[1].set_title("NEXT_INTERVAL")
    axes[1].legend(["Lp", "Ls", "Clustered Lp", "Clustered Ls"])
    axes[1].grid(True, alpha=0.2)

    # Annotation
    stats_text = (
        f"Lp: {len(events_Lp)}\n"
        f"Ls: {len(events_Ls)}\n"
        f"Clustered Lp: {len(cl_Lp)}\n"
        f"Clustered Ls: {len(cl_Ls)}"
    )
    axes[1].text(
        0.99,
        0.95,
        stats_text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"cluster_summary_{outname}_pixel{pixel}.png")
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def process_pixel_data(
    data,
    pixel: int,
    mode: str,
    params: Params,
    output_dir: str | None = None,
    show: bool = False,
    debug: bool = False,
    outname: str = "cluster",
    input_fits: str = "input.fits"
):
    """
    Run clustering for a specific pixel and return global-length arrays (aligned to 'data' rows).
    """
    pixel_mask = (data["PIXEL"] == pixel)
    pixel_events = data[pixel_mask]

    (cluster_ids,
     member_ids,
     cl_mode,
     cl_reason,
     prev_lo_ph,
     prev_itypes) = identify_clusters(pixel_events, mode=mode, params=params, debug=debug)

    # Prepare output arrays matching global data length
    n_rows = len(data)
    cluster_array = np.zeros(n_rows, dtype=np.int64)
    member_array = np.zeros(n_rows, dtype=np.int64)
    mode_array = np.zeros(n_rows, dtype=np.int16)
    reason_array = np.zeros(n_rows, dtype=np.int32)
    prev_lo_array = np.zeros(n_rows, dtype=np.int64)
    prev_type_array = np.zeros(n_rows, dtype=np.int64)

    cluster_array[pixel_mask] = cluster_ids
    member_array[pixel_mask] = member_ids
    mode_array[pixel_mask] = cl_mode
    reason_array[pixel_mask] = cl_reason
    prev_lo_array[pixel_mask] = prev_lo_ph
    prev_type_array[pixel_mask] = prev_itypes

    if output_dir:
        plot_cluster(
            events=pixel_events,
            cluster_ids=cluster_ids,
            pixel=pixel,
            output_dir=output_dir,
            show=show,
            outname=outname,
            mjdref=params.mjdref,
            input_fits=input_fits
        )

        # cluster stats + CL_REASON visualization
        plot_cluster_stats_for_pixel(
            events=pixel_events,
            pixel=pixel,
            cluster_ids=cluster_ids,
            member_ids=member_ids,
            cl_mode=cl_mode,
            cl_reason=cl_reason,
            outdir=output_dir,           # same figdir
            outprefix=f"{outname}",      # keep prefix consistent
            top_n_clusters=40,
            input_fits=input_fits
        )

    if debug:
        n_clustered = int(np.sum(cluster_ids > 0))
        print(f"..... Finished pixel {pixel}: clustered_rows={n_clustered} / events={len(pixel_events)}")



    return cluster_array, member_array, mode_array, reason_array, prev_lo_array, prev_type_array


def parse_args():
    p = argparse.ArgumentParser(description="Cluster pseudo events in a Resolve FITS event file (per pixel).")
    p.add_argument("input_fits", help="Path to input FITS file (EVENTS in HDU=1)")

    p.add_argument("--usepixels", "-p", type=str, default=",".join(map(str, range(36))),
                   help="Comma-separated list of pixel numbers to process (default: 0–35)")

    p.add_argument("--mode", "-m", choices=["large", "small"], default="large",
                   help="Clustering mode: 'large' or 'small'")

    # Output columns
    p.add_argument("--col_cluster", type=str, default="ICLUSTER",
                   help="Column name to store cluster ID (0=none, >0=cluster_id)")
    p.add_argument("--col_member", type=str, default="IMEMBER",
                   help="Column name to store member index (0=none, 1..N within cluster)")
    p.add_argument("--col_mode", type=str, default="CL_MODE",
                   help="Column name to store mode code (0 none, 1 large, 2 small)")
    p.add_argument("--col_reason", type=str, default="CL_REASON",
                   help="Column name to store diagnostic reason bitmask")

    # Diagnostics
    p.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    p.add_argument("--show", "-s", action="store_true", help="Display plots interactively")
    p.add_argument("--outname", "-o", type=str, default="add_cluster_", help="Output file prefix")
    p.add_argument("--figdir", "-f", type=str, default="fig_cluster", help="Directory to save plots")

    # Parameters
    p.add_argument("--threshold_large", type=int, default=12235, help="Threshold for large pseudo events (LO_RES_PH)")
    p.add_argument("--threshold_small", type=int, default=3000, help="Threshold for small pseudo events (reserved)")
    p.add_argument("--interval_limit", type=int, default=40, help="Time interval limit to continue a cluster")
    p.add_argument("--SECOND_THRES_USE_LEN", type=int, default=75, help="PSP-defined special value (usually 75)")
    p.add_argument("--rt_min", type=int, default=32, help="Minimum RISE_TIME threshold")
    p.add_argument("--rt_max", type=int, default=58, help="Maximum RISE_TIME threshold")
    p.add_argument("--mjdref", type=int, default=58484, help="Reference MJD for time conversion")

    return p.parse_args()


#### added for visualize col_reason info 

# -------------------------
# CL_REASON utilities / visualization
# -------------------------

BIT_NAMES = {
    BIT_IS_LP: "IS_LP(ITYPE=3)",
    BIT_IS_LS: "IS_LS(ITYPE=4)",
    BIT_LO_GT_LARGE: "LO>TH_LARGE",
    BIT_LO_LE_LARGE: "LO<=TH_LARGE",
    BIT_NT_LT_LIMIT: "NEXT<INT_LIMIT",
    BIT_NT_EQ_SECONDTHRES: "NEXT==SECOND_THRES",
    BIT_RT_LT_MIN: "RISE<RT_MIN",
    BIT_RT_GT_MAX: "RISE>RT_MAX",
    BIT_ITYPE_IN_34: "ITYPE in (3,4)",
    BIT_PR_LT_LIMIT: "PREV<INT_LIMIT",
    BIT_PR_EQ_SECONDTHRES: "PREV==SECOND_THRES",
    BIT_START_OK: "START_OK",
    BIT_CONT_OK: "CONT_OK",
}

BITS_TO_PLOT = sorted(BIT_NAMES.keys())


def _hasbit(x: np.ndarray, bit: int) -> np.ndarray:
    """Vectorized check for a bit in int array."""
    return ((x.astype(np.int64) >> bit) & 1).astype(np.int8)


def summarize_clusters_for_pixel(
    events,
    cluster_ids: np.ndarray,
    member_ids: np.ndarray,
    cl_mode: np.ndarray,
    cl_reason: np.ndarray,
):
    """
    Build per-cluster summary arrays for a single pixel stream.

    Returns:
      - cluster_unique: cluster IDs (>0)
      - cluster_sizes: number of rows in each cluster
      - n_lp, n_ls: ITYPE=3/4 counts per cluster
      - n_start, n_cont: START_OK/CONT_OK counts per cluster (based on cl_reason bits)
      - reason_bit_counts: (n_clusters, n_bits) matrix: counts of each bit within each cluster
    """
    clustered_mask = cluster_ids > 0
    if not np.any(clustered_mask):
        return None

    cid = cluster_ids[clustered_mask].astype(np.int64)
    itype = events["ITYPE"][clustered_mask].astype(np.int64)
    reason = cl_reason[clustered_mask].astype(np.int64)

    cluster_unique, inv = np.unique(cid, return_inverse=True)
    n_clusters = len(cluster_unique)

    # cluster sizes
    cluster_sizes = np.bincount(inv, minlength=n_clusters)

    # counts by ITYPE
    n_lp = np.bincount(inv, weights=(itype == 3).astype(np.int64), minlength=n_clusters).astype(np.int64)
    n_ls = np.bincount(inv, weights=(itype == 4).astype(np.int64), minlength=n_clusters).astype(np.int64)

    # start/cont (from bits)
    start_ok = _hasbit(reason, BIT_START_OK)
    cont_ok = _hasbit(reason, BIT_CONT_OK)
    n_start = np.bincount(inv, weights=start_ok, minlength=n_clusters).astype(np.int64)
    n_cont = np.bincount(inv, weights=cont_ok, minlength=n_clusters).astype(np.int64)

    # reason bit matrix per cluster
    reason_bit_counts = np.zeros((n_clusters, len(BITS_TO_PLOT)), dtype=np.int64)
    for j, b in enumerate(BITS_TO_PLOT):
        bj = _hasbit(reason, b)
        reason_bit_counts[:, j] = np.bincount(inv, weights=bj, minlength=n_clusters).astype(np.int64)

    return cluster_unique, cluster_sizes, n_lp, n_ls, n_start, n_cont, reason_bit_counts


def plot_cluster_stats_for_pixel(
    *,
    events,
    pixel: int,
    cluster_ids: np.ndarray,
    member_ids: np.ndarray,
    cl_mode: np.ndarray,
    cl_reason: np.ndarray,
    outdir: str,
    outprefix: str,
    top_n_clusters: int = 40,
    input_fits: str = "input.fits"    
):
    """
    Save per-pixel cluster statistics & CL_REASON visualization.

    Outputs:
      - PNG: size histogram, bit frequency bar, heatmap(cluster x bit), start/cont summary
      - CSV: per-cluster summary table (one row per cluster ID)
    """
    ensure_directory(outdir, verbose=False)

    summary = summarize_clusters_for_pixel(
        events=events,
        cluster_ids=cluster_ids,
        member_ids=member_ids,
        cl_mode=cl_mode,
        cl_reason=cl_reason,
    )
    if summary is None:
        # nothing clustered for this pixel
        return

    (cluster_unique,
     cluster_sizes,
     n_lp,
     n_ls,
     n_start,
     n_cont,
     reason_bit_counts) = summary

    # ------------------------------------------------------------
    # Save per-cluster table
    # ------------------------------------------------------------
    csv_path = os.path.join(outdir, f"{outprefix}cluster_stats_pixel{pixel}.csv")
    header_cols = [
        "pixel", "cluster_id", "cluster_size", "n_lp", "n_ls", "n_start_ok", "n_cont_ok"
    ] + [f"bit{b}_{BIT_NAMES[b]}" for b in BITS_TO_PLOT]

    table = np.column_stack([
        np.full_like(cluster_unique, pixel, dtype=np.int64),
        cluster_unique,
        cluster_sizes,
        n_lp,
        n_ls,
        n_start,
        n_cont,
        reason_bit_counts,
    ])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header_cols) + "\n")
        for row in table:
            f.write(",".join(map(str, row.tolist())) + "\n")

    # ------------------------------------------------------------
    # Aggregate bit frequencies across clustered rows (for this pixel)
    # ------------------------------------------------------------
    # Total counts per bit across all clusters
    bit_total = reason_bit_counts.sum(axis=0)
    bit_labels = [f"b{b}:{BIT_NAMES[b]}" for b in BITS_TO_PLOT]

    # ------------------------------------------------------------
    # Choose top-N clusters by size for heatmap
    # ------------------------------------------------------------
    order = np.argsort(cluster_sizes)[::-1]
    top = order[: min(top_n_clusters, len(order))]
    heat = reason_bit_counts[top, :]  # (topN, n_bits)

    # Normalize heatmap rows by cluster size to show "fraction of rows in cluster with bit"
    denom = cluster_sizes[top].reshape(-1, 1)
    heat_frac = np.divide(heat, denom, out=np.zeros_like(heat, dtype=float), where=(denom > 0))

    # ------------------------------------------------------------
    # Figure 1: Cluster size distribution
    # ------------------------------------------------------------
    fig1 = plt.figure(figsize=(8, 4.5))
    ax = fig1.add_subplot(1, 1, 1)

    max_size = int(np.max(cluster_sizes))
    # bins: 0.5, 1.5, 2.5, ... , max_size+0.5
    bins = np.arange(0.5, max_size + 1.5, 1.0)

    ax.hist(cluster_sizes, bins=bins, alpha=0.8, rwidth=0.9)

    # x ticks at integer centers
    ax.set_xticks(np.arange(1, max_size + 1, 1))

    ax.set_xlabel("Cluster size (#rows in cluster)")
    ax.set_ylabel("Count (#clusters)")
    ax.set_title(f"Pixel {pixel} prefix={outprefix}: Cluster size distribution (Nclusters={len(cluster_sizes)})")
    ax.grid(True, alpha=0.2)

    # ------------------------------------------------------------
    # Debug / provenance annotation (bottom-left, small & gray)
    # ------------------------------------------------------------
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta_text = (
        f"input_fits: {os.path.basename(input_fits)}\n"
        f"pixel: {pixel}\n"
        f"Nclusters: {len(cluster_sizes)}\n"
        f"generated: {timestamp}"
    )

    fig1.text(
        0.01, 0.01,
        meta_text,
        ha="left",
        va="bottom",
        fontsize=8,
        color="gray",
        family="monospace",
    )

    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, f"{outprefix}cluster_size_hist_pixel{pixel}.png"))
    plt.close(fig1)

    # ------------------------------------------------------------
    # Figure 2: Bit frequency bar (total counts)
    # ------------------------------------------------------------
    fig2 = plt.figure(figsize=(12, 5))
    ax = fig2.add_subplot(1, 1, 1)
    x = np.arange(len(BITS_TO_PLOT))
    ax.bar(x, bit_total)
    ax.set_xticks(x)
    ax.set_xticklabels(bit_labels, rotation=45, ha="right")
    ax.set_ylabel("Count (clustered rows)")
    ax.set_title(f"Pixel {pixel} prefix={outprefix}: CL_REASON bit frequencies (clustered rows only)")
    ax.grid(True, axis="y", alpha=0.2)

    fig2.text(
        0.01, 0.01,
        meta_text,
        ha="left",
        va="bottom",
        fontsize=7,
        color="gray",
        family="monospace",
    )

    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, f"{outprefix}cl_reason_bitfreq_pixel{pixel}.png"))
    plt.close(fig2)

    # ------------------------------------------------------------
    # Figure 3: Heatmap (top clusters x bits) as fraction
    # ------------------------------------------------------------
    fig3 = plt.figure(figsize=(12, 6))
    ax = fig3.add_subplot(1, 1, 1)
    im = ax.imshow(heat_frac, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(BITS_TO_PLOT)))
    ax.set_xticklabels([f"b{b}" for b in BITS_TO_PLOT], rotation=0)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"cid={int(cluster_unique[i])}\nsize={int(cluster_sizes[i])}" for i in top])
    ax.set_title(f"Pixel {pixel} prefix={outprefix}: Top-{len(top)} clusters × CL_REASON bits (fraction per cluster)")
    ax.set_xlabel("Bits")
    ax.set_ylabel("Clusters (sorted by size)")
    fig3.colorbar(im, ax=ax, label="Fraction of rows in cluster with bit")

    fig3.text(
        0.01, 0.01,
        meta_text,
        ha="left",
        va="bottom",
        fontsize=7,
        color="gray",
        family="monospace",
    )

    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, f"{outprefix}cl_reason_heatmap_pixel{pixel}.png"))
    plt.close(fig3)

    # ------------------------------------------------------------
    # Figure 4: Consistency check (Lp/Ls vs START/CONT)
    # Expectation: START_OK == Lp, CONT_OK == Ls for clustered rows
    # ------------------------------------------------------------
    total_lp = int(n_lp.sum())
    total_ls = int(n_ls.sum())
    total_start = int(n_start.sum())
    total_cont = int(n_cont.sum())

    # --- Consistency checks (global totals) ---
    ok_start = (total_lp == total_start)
    ok_cont  = (total_ls == total_cont)
    ok_all = ok_start and ok_cont

    # --- Find irregular rows (within clustered rows only) ---
    clustered_mask = (cluster_ids > 0)
    itype_c = events["ITYPE"][clustered_mask].astype(np.int64)
    reason_c = cl_reason[clustered_mask].astype(np.int64)

    is_lp = (itype_c == 3)
    is_ls = (itype_c == 4)

    is_start = (((reason_c >> BIT_START_OK) & 1) == 1)
    is_cont  = (((reason_c >> BIT_CONT_OK) & 1) == 1)

    # Expected mapping:
    #   START_OK -> Lp, CONT_OK -> Ls
    # Irregular patterns:
    #   (A) start on Ls
    #   (B) cont  on Lp
    #   (C) clustered row but neither start nor cont is set
    #   (D) both start and cont set (should not happen)
    n_start_on_ls = int(np.sum(is_start & is_ls))
    n_cont_on_lp  = int(np.sum(is_cont  & is_lp))
    n_neither     = int(np.sum(~is_start & ~is_cont))
    n_both        = int(np.sum(is_start & is_cont))

    n_irregular = n_start_on_ls + n_cont_on_lp + n_neither + n_both

    fig4 = plt.figure(figsize=(11, 5.0))
    ax1 = fig4.add_subplot(1, 2, 1)
    ax1.bar(["Lp", "Ls"], [total_lp, total_ls])
    ax1.set_title(f"Pixel {pixel} prefix={outprefix}: ITYPE composition (clustered rows)")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y", alpha=0.2)

    ax2 = fig4.add_subplot(1, 2, 2)
    ax2.bar(["START_OK", "CONT_OK"], [total_start, total_cont])
    ax2.set_title(f"Pixel {pixel} prefix={outprefix}: START/CONT usage (clustered rows)")
    ax2.set_ylabel("Count")
    ax2.grid(True, axis="y", alpha=0.2)

    # --- Add a big OK/NG banner + detailed diagnostics ---
    if ok_all and (n_irregular == 0):
        banner = "CONSISTENCY: OK"
        detail = (
            f"Lp ({total_lp}) == START_OK ({total_start})\n"
            f"Ls ({total_ls}) == CONT_OK ({total_cont})\n"
            f"Irregular rows: 0"
        )
    else:
        banner = "CONSISTENCY: NG (CHECK!)"
        detail = (
            f"Lp ({total_lp}) vs START_OK ({total_start}) => {'OK' if ok_start else 'NG'}\n"
            f"Ls ({total_ls}) vs CONT_OK ({total_cont}) => {'OK' if ok_cont else 'NG'}\n"
            f"Irregular rows breakdown:\n"
            f"  start on Ls : {n_start_on_ls}\n"
            f"  cont  on Lp : {n_cont_on_lp}\n"
            f"  neither bit : {n_neither}\n"
            f"  both bits   : {n_both}\n"
            f"  TOTAL irregular: {n_irregular}"
        )

    # Put banner at the top of the figure
    fig4.suptitle(banner, fontsize=14)

    # Put detail box on the right subplot (top-left corner inside axes)
    ax2.text(
        0.02, 0.98, detail,
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=2),
    )

    # If NG, emphasize the whole figure border by thickening spines
    if not (ok_all and n_irregular == 0):
        for ax in (ax1, ax2):
            for sp in ax.spines.values():
                sp.set_linewidth(3)

    fig4.text(
        0.01, 0.01,
        meta_text,
        ha="left",
        va="bottom",
        fontsize=7,
        color="gray",
        family="monospace",
    )
    
    fig4.tight_layout(rect=[0, 0, 1, 0.92])
    fig4.savefig(os.path.join(outdir, f"{outprefix}cluster_composition_pixel{pixel}.png"))
    plt.close(fig4)


def main():
    args = parse_args()

    params = Params(
        threshold_large=args.threshold_large,
        threshold_small=args.threshold_small,
        interval_limit=args.interval_limit,
        SECOND_THRES_USE_LEN=args.SECOND_THRES_USE_LEN,
        rt_min=args.rt_min,
        rt_max=args.rt_max,
        mjdref=args.mjdref,
    )

    ensure_directory(args.figdir, verbose=args.debug)

    output_fits = args.outname + os.path.basename(args.input_fits)

    with fits.open(args.input_fits) as hdul:
        data = hdul[1].data
        n_rows = len(data)

        # totals (global arrays aligned to HDU=1 rows)
        cluster_total = np.zeros(n_rows, dtype=np.int64)
        member_total = np.zeros(n_rows, dtype=np.int64)
        mode_total = np.zeros(n_rows, dtype=np.int16)
        reason_total = np.zeros(n_rows, dtype=np.int32)
        prev_lo_total = np.zeros(n_rows, dtype=np.int64)
        prev_type_total = np.zeros(n_rows, dtype=np.int64)

        pixel_list = list(map(int, args.usepixels.split(",")))
        for pixel in pixel_list:
            print(f"Processing pixel {pixel} ...")
            (cluster, member, clmode, clreason, prev_lo, prev_type) = process_pixel_data(
                data=data,
                pixel=pixel,
                mode=args.mode,
                params=params,
                output_dir=args.figdir,
                show=args.show,
                debug=args.debug,
                outname=args.outname,
                input_fits = args.input_fits
            )

            # Overwrite by pixel rows (safe and future-proof)
            m = (data["PIXEL"] == pixel)
            cluster_total[m] = cluster[m]
            member_total[m] = member[m]
            mode_total[m] = clmode[m]
            reason_total[m] = clreason[m]
            prev_lo_total[m] = prev_lo[m]
            prev_type_total[m] = prev_type[m]

        # Avoid overwrite errors: remove pre-existing columns with the same name
        col_names_to_replace = [
            "PREV_LO_RES_PH",
            "PREV_ITYPE",
            args.col_cluster,
            args.col_member,
            args.col_mode,
            args.col_reason,
        ]
        orig_cols = hdul[1].columns
        filtered_cols = fits.ColDefs([c for c in orig_cols if c.name not in col_names_to_replace])

        new_cols = filtered_cols + fits.ColDefs([
            fits.Column(name=args.col_cluster, format="J", array=cluster_total),
            fits.Column(name=args.col_member, format="J", array=member_total),
            fits.Column(name=args.col_mode, format="I", array=mode_total),
            fits.Column(name=args.col_reason, format="J", array=reason_total),
            fits.Column(name="PREV_LO_RES_PH", format="J", array=prev_lo_total),
            fits.Column(name="PREV_ITYPE", format="J", array=prev_type_total),
        ])

        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=hdul[1].header)
        hdul[1] = new_hdu
        hdul.writeto(output_fits, overwrite=True)

    print(f"Saved output FITS to: {output_fits}")


if __name__ == "__main__":
    main()
