#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime

# --- Plot settings ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8})


def compute_datetime_array(mjdref, times):
    """Convert TIME values from seconds to datetime using MJD reference."""
    ref_time = Time(mjdref, format='mjd')
    return [ref_time.datetime + datetime.timedelta(seconds=float(t)) for t in times]


def ensure_directory(path, verbose=False):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            print(f"Created directory: {path}")
    elif verbose:
        print(f"Directory already exists: {path}")


def identify_clusters(events, mode, threshold_large, threshold_small,
                      interval_limit, rt_min, rt_max, SECOND_THRES_USE_LEN, debug=False):
    """Identify clusters of events using specified clustering criteria."""
    cluster_indices, member_indices = [], []
    prev_lo_res_ph, prev_itype = [], []

    i, n_events = 0, len(events)
    while i < n_events:
        is_cluster_start = False
        member_count = 0

        if mode == "large":
            is_cluster_start = (events["ITYPE"][i] == 3 and events["LO_RES_PH"][i] > threshold_large)
        elif mode == "small":
            is_cluster_start = (
                events["ITYPE"][i] == 3 and
                events["LO_RES_PH"][i] <= threshold_large and
                (events["NEXT_INTERVAL"][i] < interval_limit or events["NEXT_INTERVAL"][i] == SECOND_THRES_USE_LEN) and
                (events["RISE_TIME"][i] < rt_min or events["RISE_TIME"][i] > rt_max)
            )
        else:
            raise ValueError(f"Unknown clustering mode: {mode}")

        if is_cluster_start:
            if debug:
                print(f"[{mode}] Cluster start @ {i} | ITYPE={events['ITYPE'][i]} LO={events['LO_RES_PH'][i]} NT={events['NEXT_INTERVAL'][i]}")
            cluster_indices.append(i)
            member_count += 1
            member_indices.append(member_count)
            i += 1
            while i < n_events and (events["ITYPE"][i] in (3, 4)) and \
                  (events["PREV_INTERVAL"][i] < interval_limit or events["PREV_INTERVAL"][i] == SECOND_THRES_USE_LEN):

                if debug:
                    print(f"[{mode}] Cluster continue @ {i} -> ITYPE={events['ITYPE'][i]} LO={events['LO_RES_PH'][i]} PR={events['PREV_INTERVAL'][i]}")
                cluster_indices.append(i)
                member_count += 1
                member_indices.append(member_count)
                i += 1
        else:
            cluster_indices.append(0)
            member_indices.append(0)
            i += 1

    for k in range(n_events):
        prev_lo_res_ph.append(int(events["LO_RES_PH"][k-1]) if k > 0 else 0)
        prev_itype.append(int(events["ITYPE"][k-1]) if k > 0 else 0)

    return cluster_indices, member_indices, prev_lo_res_ph, prev_itype


def plot_cluster(events, cluster_indices, pixel, output_dir, mjdref, show=False, outname="test"):
    """Plot clusters in time vs LO_RES_PH and NEXT_INTERVAL vs LO_RES_PH."""
    color_lp, color_ls = "blue", "green"
    color_cluster_lp, color_cluster_ls = "red", "orange"

    # Separate Lp and Ls events
    Lp_mask = events['ITYPE'] == 3
    Ls_mask = events['ITYPE'] == 4
    events_Lp, events_Ls = events[Lp_mask], events[Ls_mask]
    times_Lp = compute_datetime_array(mjdref, events_Lp['TIME'])
    times_Ls = compute_datetime_array(mjdref, events_Ls['TIME'])

    clustered = events[cluster_indices]
    cl_Lp = clustered[clustered['ITYPE'] == 3]
    cl_Ls = clustered[clustered['ITYPE'] == 4]
    times_cl_Lp = compute_datetime_array(mjdref, cl_Lp['TIME'])
    times_cl_Ls = compute_datetime_array(mjdref, cl_Ls['TIME'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Time vs LO_RES_PH
    axes[0].plot(times_Lp, events_Lp['LO_RES_PH'], '.', color=color_lp, label="Lp", alpha=0.6, ms=1)
    axes[0].plot(times_Ls, events_Ls['LO_RES_PH'], '.', color=color_ls, label="Ls", alpha=0.6, ms=1)
    axes[0].scatter(times_cl_Lp, cl_Lp['LO_RES_PH'], color=color_cluster_lp, label="Clustered Lp", s=10)
    axes[0].scatter(times_cl_Ls, cl_Ls['LO_RES_PH'], color=color_cluster_ls, label="Clustered Ls", s=10)
    axes[0].set_xlabel("TIME")
    axes[0].set_ylabel("LO_RES_PH")
    axes[0].set_title(f"{outname} : Pixel {pixel} - TIME")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # NEXT_INTERVAL vs LO_RES_PH
    axes[1].plot(events_Lp['NEXT_INTERVAL'], events_Lp['LO_RES_PH'], '.', color=color_lp, alpha=0.6, ms=2)
    axes[1].plot(events_Ls['NEXT_INTERVAL'], events_Ls['LO_RES_PH'], '.', color=color_ls, alpha=0.6, ms=2)
    axes[1].scatter(cl_Lp['NEXT_INTERVAL'], cl_Lp['LO_RES_PH'], color=color_cluster_lp, s=10)
    axes[1].scatter(cl_Ls['NEXT_INTERVAL'], cl_Ls['LO_RES_PH'], color=color_cluster_ls, s=10)
    axes[1].set_xlabel("NEXT_INTERVAL")
    axes[1].set_xlim(0, 255)
    axes[1].set_title("NEXT_INTERVAL")
    axes[1].legend(["Lp", "Ls", "Clustered Lp", "Clustered Ls"])
    axes[1].grid(True, alpha=0.2)

    # Annotate counts
    stats_text = (f"Lp: {len(events_Lp)}\nLs: {len(events_Ls)}\n"
                  f"Clustered Lp: {len(cl_Lp)}\nClustered Ls: {len(cl_Ls)}")
    axes[1].text(0.99, 0.95, stats_text, transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"cluster_summary_{outname}_pixel{pixel}.png")
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()


def process_pixel_data(data, pixel, mode, params, output_dir=None, show=False, debug=False, outname="cluster"):
    """
    Process clustering for a specific pixel.
    Returns cluster-related arrays: cluster ID, member ID, previous LO_RES_PH, previous ITYPE.
    """
    pixel_mask = data['PIXEL'] == pixel
    pixel_events = data[pixel_mask]

    cluster_ids, member_ids, prev_lo_ph, prev_itypes = identify_clusters(
        pixel_events, mode=mode,
        threshold_large=params.threshold_large,
        threshold_small=params.threshold_small,
        interval_limit=params.interval_limit,
        rt_min=params.rt_min,
        rt_max=params.rt_max,
        SECOND_THRES_USE_LEN=params.SECOND_THRES_USE_LEN,
        debug=debug
    )

    ncluster = np.sum(np.array(cluster_ids) > 0)
    if debug:
        print(f"..... Finished pixel {pixel}: {ncluster} clusters / {len(pixel_events)} events")

    # Prepare output arrays matching original data length
    n_rows = len(data)
    cluster_array = np.zeros(n_rows, dtype=np.int64)
    member_array = np.zeros(n_rows, dtype=np.int64)
    prev_lo_array = np.zeros(n_rows, dtype=np.int64)
    prev_type_array = np.zeros(n_rows, dtype=np.int64)

    cluster_array[pixel_mask] = cluster_ids
    member_array[pixel_mask] = member_ids
    prev_lo_array[pixel_mask] = prev_lo_ph
    prev_type_array[pixel_mask] = prev_itypes

    if output_dir:
        plot_cluster(
            events=pixel_events,
            cluster_indices=cluster_ids,
            pixel=pixel,
            output_dir=output_dir,
            show=show,
            outname=outname,
            mjdref=params.mjdref
        )

    return cluster_array, member_array, prev_lo_array, prev_type_array


def main():
    parser = argparse.ArgumentParser(description="Cluster pseudo events (Ls type) from a FITS file.")
    parser.add_argument("input_fits", help="Path to input FITS file")
    parser.add_argument("--usepixels", "-p", type=str, default=",".join(map(str, range(36))),
                        help="Comma-separated list of pixel numbers to process (default: 0–35)")
    parser.add_argument("--mode", "-m", choices=["large", "small"], default="large",
                        help="Clustering mode: 'large' or 'small'")
    parser.add_argument("--col_cluster", type=str, default="ICLUSTER",
                        help="Column name to store cluster index")
    parser.add_argument("--col_member", type=str, default="IMEMBER",
                        help="Column name to store member index")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    parser.add_argument("--show", "-s", action="store_true", help="Display plots")
    parser.add_argument("--outname", "-o", type=str, default="add_cluster_", help="Output file prefix")
    parser.add_argument("--figdir", "-f", type=str, default="fig_cluster", help="Directory to save plots")

    # Parameters that can override clustering behavior
    parser.add_argument("--threshold_large", type=int, default=12235, help="Threshold for large pseudo events")
    parser.add_argument("--threshold_small", type=int, default=3000, help="Threshold for small pseudo events")
    parser.add_argument("--interval_limit", type=int, default=40, help="Time interval to continue a cluster")
    parser.add_argument("--SECOND_THRES_USE_LEN", type=int, default=75, help="SECOND_THRES_USE_LEN (75 defined in PSP)")
    parser.add_argument("--rt_min", type=int, default=32, help="Minimum RISE_TIME threshold")
    parser.add_argument("--rt_max", type=int, default=58, help="Maximum RISE_TIME threshold")
    parser.add_argument("--mjdref", type=int, default=58484, help="Reference MJD for time conversion")

    args = parser.parse_args()

    # Create a parameter holder class from parsed args
    class Params:
        threshold_large = args.threshold_large
        threshold_small = args.threshold_small
        interval_limit = args.interval_limit
        SECOND_THRES_USE_LEN = args.SECOND_THRES_USE_LEN
        rt_min = args.rt_min
        rt_max = args.rt_max
        mjdref = args.mjdref

    params = Params()

    ensure_directory(args.figdir, verbose=args.debug)

    # Output FITS file name
    output_fits = args.outname + os.path.basename(args.input_fits)

    with fits.open(args.input_fits) as hdul:
        data = hdul[1].data
        n_rows = len(data)

        cluster_total = np.zeros(n_rows, dtype=np.int64)
        member_total = np.zeros(n_rows, dtype=np.int64)
        prev_lo_total = np.zeros(n_rows, dtype=np.int64)
        prev_type_total = np.zeros(n_rows, dtype=np.int64)

        pixel_list = list(map(int, args.usepixels.split(",")))
        for pixel in pixel_list:
            print(f"Processing pixel {pixel} ...")
            cluster, member, prev_lo, prev_type = process_pixel_data(
                data=data,
                pixel=pixel,
                mode=args.mode,
                params=params,
                output_dir=args.figdir,
                show=args.show,
                debug=args.debug,
                outname=args.outname
            )
            cluster_total += cluster
            member_total += member
            prev_lo_total += prev_lo
            prev_type_total += prev_type

        # Filter out pre-existing columns to avoid overwrite errors
        col_names_to_replace = ["PREV_LO_RES_PH", "PREV_ITYPE", args.col_cluster, args.col_member]
        orig_cols = hdul[1].columns
        filtered_cols = fits.ColDefs([col for col in orig_cols if col.name not in col_names_to_replace])

        new_cols = filtered_cols + fits.ColDefs([
            fits.Column(name=args.col_cluster, format="J", array=cluster_total),
            fits.Column(name=args.col_member, format="J", array=member_total),
            fits.Column(name="PREV_LO_RES_PH", format="J", array=prev_lo_total),
            fits.Column(name="PREV_ITYPE", format="J", array=prev_type_total)
        ])

        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=hdul[1].header)
        hdul[1] = new_hdu
        hdul.writeto(output_fits, overwrite=True)
        print(f"Saved output FITS to: {output_fits}")

if __name__ == "__main__":
    main()