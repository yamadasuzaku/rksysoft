#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time


def read_header_and_data(path: str):
    """
    Robust reader for a whitespace-delimited text like:
    #MJD GSC_ch1 GSC_ch1_e ... BAT_ch4 BAT_ch4_e
    <numbers...>
    ...
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # find first non-empty line as header
    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "":
            continue
        header_idx = i
        header_line = s
        break
    if header_idx is None:
        raise ValueError("File is empty.")

    if not header_line.startswith("#"):
        raise ValueError("Header line must start with '#', e.g. '#MJD ...'")

    colnames = header_line.lstrip("#").strip().split()
    if len(colnames) < 2:
        raise ValueError(f"Header parsing failed. Got column names: {colnames}")

    ncol = len(colnames)

    # parse data lines: keep only rows with exactly ncol numeric fields
    rows = []
    bad = 0
    for j in range(header_idx + 1, len(lines)):
        s = lines[j].strip()
        if s == "" or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != ncol:
            bad += 1
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            bad += 1
            continue

    if bad > 0:
        print(f"[Info] Skipped {bad} malformed line(s) (column count mismatch or non-numeric).")

    if len(rows) == 0:
        # Return empty structured array
        data = np.zeros(0, dtype=[(c, "f8") for c in colnames])
        return colnames, data

    arr = np.asarray(rows, dtype=float)  # (N, ncol)
    data = np.zeros(arr.shape[0], dtype=[(c, "f8") for c in colnames])
    for k, c in enumerate(colnames):
        data[c] = arr[:, k]

    return colnames, data


def valid_mask(rate, err=None):
    m = np.isfinite(rate) & (rate > 0.0)
    if err is not None:
        m &= np.isfinite(err) & (err >= 0.0)
    return m


def plot_series(ax, x_mjd, rate, err, label):
    m = valid_mask(rate, err)
    if np.count_nonzero(m) == 0:
        return False
    ax.errorbar(
        x_mjd[m],
        rate[m],
        yerr=err[m] if err is not None else None,
        fmt="o",
        ms=3,
        lw=1,
        capsize=2,
        label=label,
    )
    return True


def add_top_date_axis(ax_bottom):
    ax_top = ax_bottom.twiny()

    xmin, xmax = ax_bottom.get_xlim()
    dt_min = Time(xmin, format="mjd", scale="utc").to_datetime()
    dt_max = Time(xmax, format="mjd", scale="utc").to_datetime()
    ax_top.set_xlim(mdates.date2num(dt_min), mdates.date2num(dt_max))

    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax_top.xaxis.set_major_locator(locator)
    ax_top.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_top.set_xlabel("Date (UTC)")
    return ax_top


def main():
    parser = argparse.ArgumentParser(
        description="Plot MAXI/GSC and/or Swift/BAT light curves with MJD + date axes."
    )
    parser.add_argument("input", help="Input text file (whitespace-delimited with '#MJD ...' header).")
    parser.add_argument(
        "--which",
        choices=["maxi", "bat", "both"],
        default="both",
        help="Select which data to plot: maxi (MAXI/GSC), bat (Swift/BAT), or both.",
    )
    parser.add_argument("--out", default=None, help="Output image file (e.g., lc.png).")
    parser.add_argument("--title", default=None, help="Plot title.")
    args = parser.parse_args()

    colnames, data = read_header_and_data(args.input)

    if data.size == 0:
        print("[Error] No valid numeric rows were read. Check file format / delimiter / header.")
        sys.exit(1)

    # MJD column name
    mjd_col = "MJD" if "MJD" in colnames else colnames[0]
    mjd = data[mjd_col].astype(float)

    # Convert MJD to datetime for potential debugging (top axis uses conversion on limits)
    # dt = Time(mjd, format="mjd", scale="utc").to_datetime()

    # Define expected channels (edit here if your file uses different names)
    gsc_ch = [("GSC_ch1", "GSC_ch1_e"),
              ("GSC_ch2", "GSC_ch2_e"),
              ("GSC_ch3", "GSC_ch3_e")]
    bat_ch = [("BAT_ch1", "BAT_ch1_e"),
              ("BAT_ch2", "BAT_ch2_e"),
              ("BAT_ch3", "BAT_ch3_e"),
              ("BAT_ch4", "BAT_ch4_e")]

    gsc_avail = [(r, e) for (r, e) in gsc_ch if (r in colnames and e in colnames)]
    bat_avail = [(r, e) for (r, e) in bat_ch if (r in colnames and e in colnames)]

    if args.which in ("maxi", "both") and len(gsc_avail) == 0:
        print("Warning: No MAXI/GSC columns found in the file.")
    if args.which in ("bat", "both") and len(bat_avail) == 0:
        print("Warning: No Swift/BAT columns found in the file.")

    # If user asked for something not available, exit cleanly
    if args.which == "maxi" and len(gsc_avail) == 0:
        print("[Error] '--which maxi' was requested but GSC columns are missing.")
        sys.exit(1)
    if args.which == "bat" and len(bat_avail) == 0:
        print("[Error] '--which bat' was requested but BAT columns are missing.")
        sys.exit(1)
    if args.which == "both" and (len(gsc_avail) == 0 and len(bat_avail) == 0):
        print("[Error] Neither GSC nor BAT columns exist in the file.")
        sys.exit(1)

    # Layout
    if args.which == "both":
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_gsc, ax_bat = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        ax_gsc = ax_bat = ax

    fig.suptitle(args.title if args.title else args.input)

    # Plot
    if args.which in ("maxi", "both") and len(gsc_avail) > 0:
        any_plotted = False
        for r, e in gsc_avail:
            any_plotted |= plot_series(ax_gsc, mjd, data[r], data[e], label=r)
        ax_gsc.set_ylabel("MAXI/GSC rate")
        ax_gsc.grid(True, alpha=0.3)
        if any_plotted:
            ax_gsc.legend(loc="best", fontsize=9)
        else:
            ax_gsc.text(0.5, 0.5, "No valid (finite & >0) MAXI points",
                        transform=ax_gsc.transAxes, ha="center", va="center")

    if args.which in ("bat", "both") and len(bat_avail) > 0:
        any_plotted = False
        for r, e in bat_avail:
            any_plotted |= plot_series(ax_bat, mjd, data[r], data[e], label=r)
        ax_bat.set_ylabel("Swift/BAT rate")
        ax_bat.grid(True, alpha=0.3)
        if any_plotted:
            ax_bat.legend(loc="best", fontsize=9)
        else:
            ax_bat.text(0.5, 0.5, "No valid (finite & >0) BAT points",
                        transform=ax_bat.transAxes, ha="center", va="center")

    # Bottom axis MJD + top axis Date
    ax_bottom = ax_bat if args.which == "both" else ax_gsc

    # Safe xlim: use finite mjd only
    mjd_finite = mjd[np.isfinite(mjd)]
    if mjd_finite.size == 0:
        print("[Error] MJD column has no finite values.")
        sys.exit(1)

    ax_bottom.set_xlim(mjd_finite.min(), mjd_finite.max())
    ax_bottom.set_xlabel("MJD")
    add_top_date_axis(ax_bottom)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.out:
        fig.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
