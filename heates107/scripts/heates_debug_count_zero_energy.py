#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug utility to inspect how many zero values are stored in each channel's
'energy' dataset in an HDF5 file.

Concept:
    If calibration (cal) has succeeded, the 'energy' dataset is expected to
    contain at least some non-zero values.
    Therefore, channels that are NOT all-zero are likely to be pixels for which
    calibration succeeded.

Main outputs:
    1. Summary report at the beginning:
       - number of pixels that contain at least one non-zero energy
       - list of such pixel numbers
       - number of pixels that are completely zero
       - list of such pixel numbers
    2. Per-channel detailed statistics
    3. CSV output saved by default as:
       debug_count_zero_energy_<input_basename_without_ext>.csv

Example:
    python debug_count_zero_energy.py 20260323_run0013_mass_trans11.hdf5
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count zero entries in each channel's energy dataset in an HDF5 file."
    )
    parser.add_argument(
        "input_hdf5",
        help="Input HDF5 file path"
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output file path. "
            "If omitted, the default is "
            "debug_count_zero_energy_<input_basename_without_ext>.csv"
        )
    )
    parser.add_argument(
        "--dataset",
        default="energy",
        help="Dataset name to inspect (default: energy)"
    )
    return parser.parse_args()


def is_channel_name(name):
    return name.startswith("chan")


def extract_channel_number(name):
    """
    Convert 'chan12' -> 12 for sorting.
    If conversion fails, return a large number so it goes to the end.
    """
    try:
        return int(name.replace("chan", ""))
    except ValueError:
        return 10**9


def make_default_output_name(input_hdf5):
    """
    Example:
        20260323_run0013_mass_trans11.hdf5
        -> debug_count_zero_energy_20260323_run0013_mass_trans11.csv
    """
    base = os.path.basename(input_hdf5)
    stem, _ = os.path.splitext(base)
    return "debug_count_zero_energy_{}.csv".format(stem)


def inspect_dataset(dset):
    """
    Read the dataset and return basic statistics.
    """
    arr = dset[()]

    total = int(arr.size)
    zero_count = int((arr == 0).sum())

    if np.issubdtype(arr.dtype, np.floating):
        nan_count = int(np.isnan(arr).sum())
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            finite_vals = arr[finite_mask]
            finite_mean = float(np.mean(finite_vals))
            finite_std = float(np.std(finite_vals))
            finite_min = float(np.min(finite_vals))
            finite_max = float(np.max(finite_vals))
        else:
            finite_mean = float("nan")
            finite_std = float("nan")
            finite_min = float("nan")
            finite_max = float("nan")
    else:
        nan_count = 0
        if total > 0:
            finite_mean = float(np.mean(arr))
            finite_std = float(np.std(arr))
            finite_min = float(np.min(arr))
            finite_max = float(np.max(arr))
        else:
            finite_mean = float("nan")
            finite_std = float("nan")
            finite_min = float("nan")
            finite_max = float("nan")

    zero_fraction = zero_count / total if total > 0 else float("nan")
    nonzero_count = total - zero_count
    is_all_zero = (total > 0 and zero_count == total)
    has_any_nonzero = (nonzero_count > 0)

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "total": total,
        "zero_count": zero_count,
        "nonzero_count": nonzero_count,
        "nan_count": nan_count,
        "zero_fraction": zero_fraction,
        "mean": finite_mean,
        "std": finite_std,
        "min": finite_min,
        "max": finite_max,
        "is_all_zero": is_all_zero,
        "has_any_nonzero": has_any_nonzero,
    }


def format_channel_list(ch_list):
    """
    Convert ['chan1', 'chan2', 'chan10'] -> '1, 2, 10'
    """
    nums = [str(extract_channel_number(ch)) for ch in ch_list]
    return ", ".join(nums) if nums else "(none)"


def main():
    args = parse_args()

    if not os.path.exists(args.input_hdf5):
        print("ERROR: file not found: {}".format(args.input_hdf5), file=sys.stderr)
        sys.exit(1)

    output_csv = args.output if args.output is not None else make_default_output_name(args.input_hdf5)

    results = []

    with h5py.File(args.input_hdf5, "r") as f:
        channel_names = sorted(
            [key for key in f.keys() if is_channel_name(key)],
            key=extract_channel_number
        )

        if not channel_names:
            print("No channel groups like 'chan1', 'chan2', ... were found.")
            sys.exit(1)

        for ch_name in channel_names:
            grp = f[ch_name]

            if args.dataset not in grp:
                results.append({
                    "channel": ch_name,
                    "channel_num": extract_channel_number(ch_name),
                    "total": "",
                    "zero_count": "",
                    "nonzero_count": "",
                    "zero_fraction": "",
                    "nan_count": "",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "dtype": "",
                    "shape": "",
                    "is_all_zero": "",
                    "has_any_nonzero": "",
                    "status": "missing_dataset",
                })
                continue

            dset = grp[args.dataset]

            if not isinstance(dset, h5py.Dataset):
                results.append({
                    "channel": ch_name,
                    "channel_num": extract_channel_number(ch_name),
                    "total": "",
                    "zero_count": "",
                    "nonzero_count": "",
                    "zero_fraction": "",
                    "nan_count": "",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "dtype": "",
                    "shape": "",
                    "is_all_zero": "",
                    "has_any_nonzero": "",
                    "status": "not_dataset",
                })
                continue

            stat = inspect_dataset(dset)

            results.append({
                "channel": ch_name,
                "channel_num": extract_channel_number(ch_name),
                "total": stat["total"],
                "zero_count": stat["zero_count"],
                "nonzero_count": stat["nonzero_count"],
                "zero_fraction": stat["zero_fraction"],
                "nan_count": stat["nan_count"],
                "mean": stat["mean"],
                "std": stat["std"],
                "min": stat["min"],
                "max": stat["max"],
                "dtype": stat["dtype"],
                "shape": str(stat["shape"]),
                "is_all_zero": stat["is_all_zero"],
                "has_any_nonzero": stat["has_any_nonzero"],
                "status": "ok",
            })

    # ---------------------------------
    # Summary report
    # ---------------------------------
    ok_results = [r for r in results if r["status"] == "ok"]

    nonzero_pixels = [r["channel"] for r in ok_results if r["has_any_nonzero"]]
    all_zero_pixels = [r["channel"] for r in ok_results if r["is_all_zero"]]

    print("=" * 120)
    print("SUMMARY REPORT")
    print("=" * 120)
    print("Input file   : {}".format(args.input_hdf5))
    print("Dataset name : {}".format(args.dataset))
    print("Output CSV   : {}".format(output_csv))
    print("-" * 120)
    print("Interpretation:")
    print("  If calibration (cal) has succeeded, energy is expected to contain non-zero values.")
    print("  Therefore, pixels that are NOT all-zero are likely to be candidates for successful calibration.")
    print("-" * 120)
    print("Pixels with at least one non-zero energy value : {}".format(len(nonzero_pixels)))
    print("Pixel numbers                                 : {}".format(format_channel_list(nonzero_pixels)))
    print("-" * 120)
    print("Pixels with all energy values equal to zero   : {}".format(len(all_zero_pixels)))
    print("Pixel numbers                                 : {}".format(format_channel_list(all_zero_pixels)))
    print("=" * 120)
    print()

    # ---------------------------------
    # Per-channel report
    # ---------------------------------
    print("=" * 120)
    print("PER-CHANNEL STATISTICS")
    print("=" * 120)
    print(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}".format(
            "channel", "total", "zeros", "nonzero", "zero_frac", "NaN",
            "min", "max", "mean", "std"
        )
    )
    print("-" * 120)

    for r in results:
        ch_name = r["channel"]

        if r["status"] == "missing_dataset":
            print("{:>8}  {:>10}".format(ch_name, "MISSING"))
            continue

        if r["status"] == "not_dataset":
            print("{:>8}  {:>10}".format(ch_name, "NOT_DATASET"))
            continue

        print(
            "{:>8}  {:10d}  {:10d}  {:10d}  {:10.4f}  {:10d}  {:14.6g}  {:14.6g}  {:14.6g}  {:14.6g}".format(
                ch_name,
                r["total"],
                r["zero_count"],
                r["nonzero_count"],
                r["zero_fraction"],
                r["nan_count"],
                r["min"],
                r["max"],
                r["mean"],
                r["std"],
            )
        )

    print("-" * 120)
    print("Valid channels inspected : {}".format(len(ok_results)))
    print("=" * 120)

    # ---------------------------------
    # Save CSV
    # ---------------------------------
    with open(output_csv, "w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "channel",
                "channel_num",
                "total",
                "zero_count",
                "nonzero_count",
                "zero_fraction",
                "nan_count",
                "min",
                "max",
                "mean",
                "std",
                "dtype",
                "shape",
                "is_all_zero",
                "has_any_nonzero",
                "status",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("CSV summary written to: {}".format(output_csv))


if __name__ == "__main__":
    main()