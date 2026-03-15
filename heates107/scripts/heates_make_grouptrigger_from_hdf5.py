#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot spectra from HDF5 files and generate group-trigger JSON.

Main features
-------------
1. Read per-channel event data from HDF5 files.
2. Apply good-event selection.
3. Optionally apply beam selection:
   - --beam-on   -> use only beam == 1
   - --beam-off  -> use only beam == 0
   - default     -> do not apply any beam cut
4. Optionally apply energy selection:
   - if both --emin and --emax are given, apply energy cut
   - default -> do not apply any energy cut
5. Generate:
   - merged spectrum
   - per-channel spectra
   - per-channel count histogram
   - group-trigger JSON using only alive pixels

Definition of alive pixel
-------------------------
A channel is considered alive if it has at least one selected event
after all enabled cuts are applied.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot HDF5 spectra and generate group-trigger JSON."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Input HDF5 files."
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=None,
        help="Minimum energy for the optional energy cut."
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=None,
        help="Maximum energy for the optional energy cut."
    )
    parser.add_argument(
        "--bin",
        type=float,
        default=5.0,
        help="Histogram bin width in energy units."
    )
    parser.add_argument(
        "--beam-on",
        action="store_true",
        help="Apply beam == 1 selection."
    )
    parser.add_argument(
        "--beam-off",
        action="store_true",
        help="Apply beam == 0 selection."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output_energy_hist",
        help="Output directory for figures and JSON files."
    )
    parser.add_argument(
        "--max-channels-in-overlay",
        type=int,
        default=100,
        help="Maximum number of channels to overlay in the per-channel spectrum figure."
    )

    args = parser.parse_args()

    if args.beam_on and args.beam_off:
        parser.error("Use only one of --beam-on or --beam-off.")

    if (args.emin is None) ^ (args.emax is None):
        parser.error("Specify both --emin and --emax together, or neither.")

    return args


def extract_channel_id(group_name):
    """
    Extract integer channel ID from group names such as 'chan0' or 'chan12'.

    Parameters
    ----------
    group_name : str
        HDF5 group name.

    Returns
    -------
    int or None
        Channel ID if parsing succeeds, otherwise None.
    """
    if not group_name.startswith("chan"):
        return None

    suffix = group_name[4:]
    if not suffix.isdigit():
        return None

    return int(suffix)


def load_channel_data(h5_filename):
    """
    Load event data for each channel from an HDF5 file.

    Parameters
    ----------
    h5_filename : str or Path
        Input HDF5 file.

    Returns
    -------
    dict
        Dictionary keyed by channel ID. Each value is another dictionary:
        {
            ch_id: {
                "energy": np.ndarray,
                "good": np.ndarray,
                "beam": np.ndarray or None
            }
        }
    """
    channel_data = {}

    with h5py.File(h5_filename, "r") as h5_file:
        for key in h5_file.keys():
            ch_id = extract_channel_id(key)
            if ch_id is None:
                continue

            group = h5_file[key]

            if "energy" not in group or "good" not in group:
                continue

            energy_array = group["energy"][:]
            good_array = group["good"][:]
            beam_array = group["beam"][:] if "beam" in group else None

            channel_data[ch_id] = {
                "energy": energy_array,
                "good": good_array,
                "beam": beam_array,
            }

    return channel_data


def select_channel_energies(channel_data, beam_mode=None, emin=None, emax=None):
    """
    Apply event selections and return selected energies for each channel.

    Parameters
    ----------
    channel_data : dict
        Per-channel raw event data.
    beam_mode : str or None
        None      -> no beam cut
        "on"      -> beam == 1
        "off"     -> beam == 0
    emin, emax : float or None
        Energy cut range. If both are None, no energy cut is applied.

    Returns
    -------
    selected_energies : dict
        Dictionary keyed by channel ID with selected energy arrays.
    beam_summary : dict
        Summary of beam counts before applying beam selection:
        {
            "beam_on": int,
            "beam_off": int,
            "beam_missing": int
        }
    """
    selected_energies = {}

    beam_on_count = 0
    beam_off_count = 0
    beam_missing_count = 0

    for ch_id, values in channel_data.items():
        energy_array = values["energy"]
        good_array = values["good"]
        beam_array = values["beam"]

        mask = (good_array == True)

        good_indices = np.where(mask)[0]

        if beam_array is not None:
            beam_on_count += int(np.sum(beam_array[good_indices] == 1))
            beam_off_count += int(np.sum(beam_array[good_indices] == 0))
        else:
            beam_missing_count += int(len(good_indices))

        if beam_mode is not None:
            if beam_array is None:
                # If beam selection is requested but the dataset does not exist,
                # no event can pass the beam cut for this channel.
                mask &= False
            elif beam_mode == "on":
                mask &= (beam_array == 1)
            elif beam_mode == "off":
                mask &= (beam_array == 0)

        if emin is not None and emax is not None:
            mask &= (energy_array >= emin) & (energy_array <= emax)

        selected_energies[ch_id] = energy_array[mask]

    beam_summary = {
        "beam_on": beam_on_count,
        "beam_off": beam_off_count,
        "beam_missing": beam_missing_count,
    }

    return selected_energies, beam_summary


def get_alive_channels(channel_energies):
    """
    Return a sorted list of alive channels.

    A channel is alive if it has at least one selected event.
    """
    alive_channels = [
        ch_id for ch_id, energies in channel_energies.items() if len(energies) > 0
    ]
    return sorted(alive_channels)


def merge_energies(channel_energies):
    """
    Concatenate all selected energies into a single 1D array.
    """
    arrays = [arr for arr in channel_energies.values() if len(arr) > 0]
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)


def make_energy_bins(energies, bin_width):
    """
    Create histogram bin edges from data.

    Parameters
    ----------
    energies : np.ndarray
        Input energy array.
    bin_width : float
        Histogram bin width.

    Returns
    -------
    np.ndarray
        Histogram bin edges.
    """
    if len(energies) == 0:
        return np.array([0.0, bin_width], dtype=float)

    data_min = np.min(energies)
    data_max = np.max(energies)

    left_edge = np.floor(data_min / bin_width) * bin_width
    right_edge = np.ceil(data_max / bin_width) * bin_width + bin_width

    return np.arange(left_edge, right_edge, bin_width)


def plot_merged_spectrum(channel_energies, bin_width, title, output_png):
    """
    Plot the merged spectrum from all alive channels.
    """
    energies = merge_energies(channel_energies)
    bins = make_energy_bins(energies, bin_width)

    counts, edges = np.histogram(energies, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 6))
    plt.step(centers, counts, where="mid", label="Merged alive channels")
    plt.xlabel("Energy")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def plot_per_channel_spectra(
    channel_energies,
    bin_width,
    title,
    output_png,
    max_channels_in_overlay=100
):
    """
    Plot per-channel spectra in one overlay figure.
    """
    alive_channels = get_alive_channels(channel_energies)
    channels_to_plot = alive_channels[:max_channels_in_overlay]

    all_energies = merge_energies({ch: channel_energies[ch] for ch in channels_to_plot})
    bins = make_energy_bins(all_energies, bin_width)

    plt.figure(figsize=(12, 7))

    for ch_id in channels_to_plot:
        energies = channel_energies[ch_id]
        counts, edges = np.histogram(energies, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.step(
            centers,
            counts,
            where="mid",
            label="ch{0} (N={1})".format(ch_id, len(energies))
        )

    plt.xlabel("Energy")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title(title)

    if channels_to_plot:
        plt.legend(fontsize=6, ncol=3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def plot_channel_count_histogram(channel_energies, title, output_png):
    """
    Plot the selected event counts for each alive channel.
    """
    alive_channels = get_alive_channels(channel_energies)
    counts = [len(channel_energies[ch_id]) for ch_id in alive_channels]

    plt.figure(figsize=(12, 6))
    plt.bar(alive_channels, counts)
    plt.xlabel("Channel")
    plt.ylabel("Selected event counts")
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def build_grouptrigger_map(alive_channels):
    """
    Build group-trigger mapping using only alive pixels.

    Rules
    -----
    - Interior pixel:
        use immediate alive neighbors on both sides
        e.g. 2 -> [1, 3]
    - Edge pixel:
        round inward using the nearest two alive channels
        e.g. 0 -> [1, 2]
             4 -> [2, 3]
    """
    result = {}

    if len(alive_channels) == 0:
        return result

    if len(alive_channels) == 1:
        ch = alive_channels[0]
        result[str(ch)] = [ch, ch]
        return result

    if len(alive_channels) == 2:
        ch0, ch1 = alive_channels
        result[str(ch0)] = [ch1, ch1]
        result[str(ch1)] = [ch0, ch0]
        return result

    for index, ch in enumerate(alive_channels):
        if index == 0:
            neighbors = [alive_channels[1], alive_channels[2]]
        elif index == len(alive_channels) - 1:
            neighbors = [alive_channels[-3], alive_channels[-2]]
        else:
            neighbors = [alive_channels[index - 1], alive_channels[index + 1]]

        result[str(ch)] = neighbors

    return result


def save_grouptrigger_json(grouptrigger_map, output_json):
    """
    Save the group-trigger mapping to a JSON file.
    """
    with open(output_json, "w", encoding="utf-8") as output_file:
        json.dump(grouptrigger_map, output_file, indent=2, ensure_ascii=False)


def print_summary(h5_filename, channel_energies, beam_summary, beam_mode, emin, emax):
    """
    Print a summary of the current file and selection.
    """
    alive_channels = get_alive_channels(channel_energies)
    total_events = int(sum(len(channel_energies[ch_id]) for ch_id in alive_channels))

    print("File               : {0}".format(Path(h5_filename).name))
    print("Beam mode          : {0}".format(
        "none" if beam_mode is None else beam_mode
    ))
    print("Energy cut         : {0}".format(
        "none" if (emin is None or emax is None) else "[{0}, {1}]".format(emin, emax)
    ))
    print("Beam ON events     : {0}".format(beam_summary["beam_on"]))
    print("Beam OFF events    : {0}".format(beam_summary["beam_off"]))
    print("Beam missing       : {0}".format(beam_summary["beam_missing"]))
    print("Alive channels     : {0}".format(len(alive_channels)))
    print("Alive channel list : {0}".format(alive_channels))
    print("Selected events    : {0}".format(total_events))
    print("")


def build_mode_label(beam_mode, emin, emax):
    """
    Build a short label for file names and figure titles.
    """
    labels = []

    if beam_mode is None:
        labels.append("beam_all")
    elif beam_mode == "on":
        labels.append("beam_on")
    elif beam_mode == "off":
        labels.append("beam_off")

    if emin is not None and emax is not None:
        labels.append("E_{0:g}_{1:g}".format(emin, emax))
    else:
        labels.append("E_all")

    return "_".join(labels)


def main():
    args = parse_args()

    if args.beam_on:
        beam_mode = "on"
    elif args.beam_off:
        beam_mode = "off"
    else:
        beam_mode = None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for h5_filename in args.files:
        stem = Path(h5_filename).stem
        mode_label = build_mode_label(beam_mode, args.emin, args.emax)

        raw_channel_data = load_channel_data(h5_filename)

        channel_energies, beam_summary = select_channel_energies(
            raw_channel_data,
            beam_mode=beam_mode,
            emin=args.emin,
            emax=args.emax
        )

        print_summary(
            h5_filename=h5_filename,
            channel_energies=channel_energies,
            beam_summary=beam_summary,
            beam_mode=beam_mode,
            emin=args.emin,
            emax=args.emax
        )

        merged_png = outdir / "{0}_{1}_merged_spectrum.png".format(stem, mode_label)
        per_channel_png = outdir / "{0}_{1}_per_channel_spectra.png".format(stem, mode_label)
        count_hist_png = outdir / "{0}_{1}_channel_count_histogram.png".format(stem, mode_label)
        json_path = outdir / "{0}_{1}_grouptrigger.json".format(stem, mode_label)

        plot_merged_spectrum(
            channel_energies=channel_energies,
            bin_width=args.bin,
            title="{0} : merged spectrum ({1})".format(stem, mode_label),
            output_png=merged_png
        )

        plot_per_channel_spectra(
            channel_energies=channel_energies,
            bin_width=args.bin,
            title="{0} : per-channel spectra ({1})".format(stem, mode_label),
            output_png=per_channel_png,
            max_channels_in_overlay=args.max_channels_in_overlay
        )

        plot_channel_count_histogram(
            channel_energies=channel_energies,
            title="{0} : selected counts per channel ({1})".format(stem, mode_label),
            output_png=count_hist_png
        )

        alive_channels = get_alive_channels(channel_energies)
        grouptrigger_map = build_grouptrigger_map(alive_channels)
        save_grouptrigger_json(grouptrigger_map, json_path)

        print("Saved:")
        print("  {0}".format(merged_png))
        print("  {0}".format(per_channel_png))
        print("  {0}".format(count_hist_png))
        print("  {0}".format(json_path))
        print("")


if __name__ == "__main__":
    main()