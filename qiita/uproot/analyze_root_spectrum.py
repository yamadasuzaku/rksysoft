#!/usr/bin/env python3
"""
ROOT TTree spectrum analysis using uproot.

Main features
-------------
- Read a ROOT file and a TTree with uproot
- Build 1D histogram of energy (eV)
- Optionally perform Gaussian fit
- If line energy is not specified, auto-detect the strongest peak
- Optionally analyze by channel (channum)
- Skip channels with zero events
- Debug mode to inspect all available branches
- Designed to be extendable for additional cut conditions

Compatibility
-------------
- Written in a Python 3.9-friendly style
- Plotting uses matplotlib only
"""

import argparse
import math
import sys

import numpy as np
import uproot
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def str2bool(value):
    """
    Robust boolean parser for argparse.
    """
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "t", "yes", "y", "1"):
        return True
    if value in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def has_branch(array_container, branch_name):
    """
    Check whether a branch exists in the object returned by uproot.

    This supports both:
    - dict-like containers
    - NumPy structured arrays
    """
    if isinstance(array_container, dict):
        return branch_name in array_container

    dtype = getattr(array_container, "dtype", None)
    if dtype is not None and dtype.names is not None:
        return branch_name in dtype.names

    return False


def gaussian(x, amplitude, mean, sigma, offset):
    """
    Gaussian + constant background model.
    """
    return offset + amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def compute_fwhm_from_sigma(sigma):
    """
    Convert Gaussian sigma to FWHM.
    """
    return 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma


def safe_minmax(arr):
    """
    Return finite min/max from an array.
    """
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Array has no finite values.")
    return np.min(arr[finite]), np.max(arr[finite])


def build_histogram(energy_array, nbins=400, xmin=None, xmax=None):
    """
    Build histogram for energy.
    """
    energy_array = np.asarray(energy_array)
    finite = np.isfinite(energy_array)
    energy_array = energy_array[finite]

    if energy_array.size == 0:
        raise ValueError("No valid energy events remain after filtering.")

    if xmin is None or xmax is None:
        data_min, data_max = safe_minmax(energy_array)
        if xmin is None:
            xmin = data_min
        if xmax is None:
            xmax = data_max

    counts, edges = np.histogram(energy_array, bins=nbins, range=(xmin, xmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers


def auto_find_peak_energy(centers, counts):
    """
    Find the energy of the strongest histogram bin.
    """
    if counts.size == 0 or np.max(counts) <= 0:
        raise ValueError("Cannot auto-detect peak because histogram is empty.")
    peak_index = np.argmax(counts)
    return centers[peak_index]


def fit_peak_from_histogram(centers, counts, target_energy=None, fit_window=80.0):
    """
    Fit a Gaussian peak in a histogram.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for fitting. Please install scipy or run with --fit false."
        )

    if target_energy is None:
        target_energy = auto_find_peak_energy(centers, counts)

    fit_mask = (
        np.isfinite(centers) &
        np.isfinite(counts) &
        (centers >= target_energy - fit_window) &
        (centers <= target_energy + fit_window)
    )

    x_fit = centers[fit_mask]
    y_fit = counts[fit_mask]

    if x_fit.size < 5:
        raise RuntimeError(
            "Too few histogram bins in the fit window. Increase --fit-window."
        )

    y_max = float(np.max(y_fit))
    y_min = float(np.min(y_fit))

    amplitude0 = max(y_max - y_min, 1.0)
    mean0 = float(x_fit[np.argmax(y_fit)])
    sigma0 = max(fit_window / 6.0, 1.0)
    offset0 = max(y_min, 0.0)

    p0 = [amplitude0, mean0, sigma0, offset0]

    lower_bounds = [0.0, target_energy - fit_window, 0.1, 0.0]
    upper_bounds = [np.inf, target_energy + fit_window, fit_window * 2.0, np.inf]

    popt, pcov = curve_fit(
        gaussian,
        x_fit,
        y_fit,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000
    )

    perr = np.sqrt(np.diag(pcov))
    model = gaussian(x_fit, *popt)
    residuals = y_fit - model

    result = {
        "target_energy": target_energy,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "model": model,
        "residuals": residuals,
        "params": {
            "amplitude": popt[0],
            "mean": popt[1],
            "sigma": popt[2],
            "offset": popt[3],
        },
        "errors": {
            "amplitude": perr[0],
            "mean": perr[1],
            "sigma": perr[2],
            "offset": perr[3],
        },
        "fwhm": compute_fwhm_from_sigma(popt[2]),
        "fwhm_err": compute_fwhm_from_sigma(perr[2]),
    }
    return result


def parse_cut_expression(cut_expr):
    """
    Parse a generic cut expression of the form:
        branch:min:max
    """
    parts = cut_expr.split(":")
    if len(parts) != 3:
        raise ValueError(
            "Invalid cut expression '{}'. Use branch:min:max".format(cut_expr)
        )

    branch_name = parts[0].strip()
    min_str = parts[1].strip()
    max_str = parts[2].strip()

    min_value = None if min_str == "" else float(min_str)
    max_value = None if max_str == "" else float(max_str)

    return branch_name, min_value, max_value


def apply_cuts(data_dict, require_good=False, cut_expressions=None):
    """
    Apply cuts to arrays stored in a dictionary or structured array.
    """
    n_events = len(data_dict["energy"])
    mask = np.ones(n_events, dtype=bool)

    mask &= np.isfinite(data_dict["energy"])

    if require_good:
        if not has_branch(data_dict, "good"):
            print("WARNING: require_good=True but branch 'good' is not available.")
            print("WARNING: Skipping good-event selection.")
        else:
            mask &= data_dict["good"].astype(bool)

    if cut_expressions:
        for cut_expr in cut_expressions:
            branch_name, min_value, max_value = parse_cut_expression(cut_expr)
            if not has_branch(data_dict, branch_name):
                raise KeyError(
                    "Branch '{}' not found for cut '{}'".format(branch_name, cut_expr)
                )

            arr = data_dict[branch_name]
            this_mask = np.isfinite(arr)

            if min_value is not None:
                this_mask &= (arr >= min_value)
            if max_value is not None:
                this_mask &= (arr <= max_value)

            mask &= this_mask

    return mask


def summarize_channels(channum_array):
    """
    Summarize active channels and their event counts.
    """
    unique_ch, counts = np.unique(channum_array, return_counts=True)
    active_mask = counts > 0
    unique_ch = unique_ch[active_mask]
    counts = counts[active_mask]

    order = np.argsort(unique_ch)
    unique_ch = unique_ch[order]
    counts = counts[order]

    return unique_ch, counts


# ----------------------------------------------------------------------
# Debug / inspection helpers
# ----------------------------------------------------------------------
def inspect_tree_branches(tree):
    """
    Inspect all branches in the TTree and return summary information.
    """
    summary_list = []

    for branch_name, branch_obj in tree.items():
        info = {
            "name": branch_name,
            "typename": getattr(branch_obj, "typename", "unknown"),
            "interpretation": str(getattr(branch_obj, "interpretation", "unknown")),
            "num_entries": getattr(branch_obj, "num_entries", None),
        }
        summary_list.append(info)

    return summary_list


def print_branch_summary(summary_list):
    """
    Print branch summary in a readable format.
    """
    print("")
    print("=== Branch summary ===")
    print("Number of branches:", len(summary_list))
    for i, info in enumerate(summary_list):
        print(
            "[{0:3d}] {1:20s} | entries={2} | typename={3} | interpretation={4}".format(
                i,
                str(info["name"]),
                str(info["num_entries"]),
                str(info["typename"]),
                str(info["interpretation"]),
            )
        )


def plot_branch_entries(summary_list, output_path=None, show=False):
    """
    Plot number of entries for all branches.
    """
    if not summary_list:
        print("WARNING: No branch information to plot.")
        return

    names = [info["name"] for info in summary_list]
    entries = [
        -1 if info["num_entries"] is None else info["num_entries"]
        for info in summary_list
    ]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(names)), 6))
    ax.bar(x, entries)
    ax.set_xlabel("Branch index")
    ax.set_ylabel("Number of entries")
    ax.set_title("TTree branch entry counts")
    ax.grid(True, axis="y", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print("Saved:", output_path)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_spectrum_only(centers, counts, title, output_path=None, show=False):
    """
    Plot only the spectrum.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers, counts, where="mid")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Counts / bin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print("Saved:", output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_spectrum_with_fit(
    centers,
    counts,
    fit_result,
    title,
    output_path=None,
    show=False
):
    """
    Plot data, Gaussian model, and residuals.
    """
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.step(centers, counts, where="mid", label="Data")
    ax_top.plot(fit_result["x_fit"], fit_result["model"], label="Gaussian fit")
    ax_top.set_ylabel("Counts / bin")
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    text = (
        "Mean = {:.3f} ± {:.3f} eV\n"
        "Sigma = {:.3f} ± {:.3f} eV\n"
        "FWHM = {:.3f} ± {:.3f} eV"
    ).format(
        fit_result["params"]["mean"],
        fit_result["errors"]["mean"],
        fit_result["params"]["sigma"],
        fit_result["errors"]["sigma"],
        fit_result["fwhm"],
        fit_result["fwhm_err"],
    )
    ax_top.text(
        0.98, 0.97, text,
        transform=ax_top.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax_bot.axhline(0.0, linewidth=1.0)
    ax_bot.plot(
        fit_result["x_fit"],
        fit_result["residuals"],
        marker="o",
        linestyle="-"
    )
    ax_top.set_yscale("log")    
    ax_bot.set_xlabel("Energy (eV)")
    ax_bot.set_ylabel("Residual")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print("Saved:", output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_overlay_by_channel(
    energy_array,
    channum_array,
    active_channels,
    nbins,
    xmin,
    xmax,
    title,
    output_path=None,
    show=False,
    max_channels_to_plot=None
):
    """
    Overlay spectra by channel.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    channels_to_plot = active_channels
    if max_channels_to_plot is not None:
        channels_to_plot = active_channels[:max_channels_to_plot]

    for ch in channels_to_plot:
        mask = (channum_array == ch)
        ch_energy = energy_array[mask]
        if ch_energy.size == 0:
            continue

        counts, edges = np.histogram(ch_energy, bins=nbins, range=(xmin, xmax))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, counts, where="mid", alpha=0.6, linewidth=1.0)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Counts / bin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print("Saved:", output_path)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze energy spectrum in a ROOT TTree using uproot."
    )

    parser.add_argument("root_file", help="Input ROOT file path")
    parser.add_argument(
        "--tree-name",
        default="tree",
        help="TTree name (default: tree)"
    )

    parser.add_argument(
        "--fit",
        type=str2bool,
        default=False,
        help="Whether to perform Gaussian fit (true/false)"
    )
    parser.add_argument(
        "--line-energy",
        type=float,
        default=None,
        help="Expected line energy in eV. If omitted, the strongest peak is used."
    )
    parser.add_argument(
        "--fit-window",
        type=float,
        default=80.0,
        help="Half-width of fit region in eV (default: 80)"
    )

    parser.add_argument(
        "--use-all-chan",
        type=str2bool,
        default=True,
        help="If true, ignore channum and analyze all events together."
    )
    parser.add_argument(
        "--require-good",
        type=str2bool,
        default=False,
        help="If true, require good == True"
    )
    parser.add_argument(
        "--cut",
        action="append",
        default=[],
        help=(
            "Additional cut in the form branch:min:max "
            "(can be specified multiple times)"
        )
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=2000,
        help="Number of histogram bins"
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Minimum energy for histogram"
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Maximum energy for histogram"
    )

    parser.add_argument(
        "--show",
        type=str2bool,
        default=True,
        help="Show figures interactively"
    )
    parser.add_argument(
        "--output-prefix",
        default="spectrum",
        help="Prefix for output figure files"
    )
    parser.add_argument(
        "--max-channels-to-plot",
        type=int,
        default=None,
        help="Optional maximum number of channels to overlay"
    )

    parser.add_argument(
        "--debug-branches",
        type=str2bool,
        default=False,
        help="If true, print and plot all branch information"
    )
    parser.add_argument(
        "--list-branches",
        type=str2bool,
        default=False,
        help="If true, list all branches and exit"
    )

    args = parser.parse_args()

    print("Opening ROOT file:", args.root_file)
    root_obj = uproot.open(args.root_file)
    tree = root_obj[args.tree_name]

    # ------------------------------------
    # Debug / inspection mode
    # ------------------------------------
    branch_summary = inspect_tree_branches(tree)

    if args.debug_branches or args.list_branches:
        print_branch_summary(branch_summary)
        plot_branch_entries(
            branch_summary,
            output_path=args.output_prefix + "_branch_entries.png",
            show=args.show if args.debug_branches else False
        )

        branch_names = [info["name"] for info in branch_summary]
        print("")
        print("=== Quick branch existence check ===")
        for key_name in ["energy", "channum", "good", "dt", "dt_us"]:
            print(
                "{0:12s} : {1}".format(
                    key_name,
                    "FOUND" if key_name in branch_names else "MISSING"
                )
            )

    if args.list_branches:
        print("")
        print("Exiting after listing branches because --list-branches true was specified.")
        return

    # Base branches for main analysis
    branches = [
        "energy",
        "channum",
        "good",
        "pretrig_rms",
        "rise_time",
        "promptness",
        "pulse_rms",
        "filt_phase",
    ]

    available_branches = set(tree.keys())
    read_branches = [b for b in branches if b in available_branches]

    for cut_expr in args.cut:
        branch_name, _, _ = parse_cut_expression(cut_expr)
        if branch_name not in read_branches and branch_name in available_branches:
            read_branches.append(branch_name)

    print("")
    print("=== Read branch summary ===")
    print("Requested branches:", branches)
    print("Available branches:", sorted(list(available_branches)))
    print("Branches to read   :", read_branches)

    if "energy" not in read_branches:
        print("ERROR: Required branch 'energy' is missing.", file=sys.stderr)
        sys.exit(1)

    arrays = tree.arrays(read_branches, library="np")

    mask = apply_cuts(
        arrays,
        require_good=args.require_good,
        cut_expressions=args.cut
    )

    energy = arrays["energy"][mask]
    channum = arrays["channum"][mask] if has_branch(arrays, "channum") else None

    print("")
    print("=== Event summary after cuts ===")
    print("Total events after cuts:", energy.size)

    if energy.size == 0:
        print("No events remain after cuts.", file=sys.stderr)
        sys.exit(1)

    if args.xmin is None or args.xmax is None:
        e_min, e_max = safe_minmax(energy)
        xmin = e_min if args.xmin is None else args.xmin
        xmax = e_max if args.xmax is None else args.xmax
    else:
        xmin = args.xmin
        xmax = args.xmax

    if args.use_all_chan:
        counts, edges, centers = build_histogram(
            energy,
            nbins=args.nbins,
            xmin=xmin,
            xmax=xmax
        )

        print("")
        print("=== All-event spectrum summary ===")
        print("Number of events used:", energy.size)

        if args.fit:
            fit_result = fit_peak_from_histogram(
                centers,
                counts,
                target_energy=args.line_energy,
                fit_window=args.fit_window
            )

            print("Peak target energy   : {:.3f} eV".format(fit_result["target_energy"]))
            print("Peak centroid        : {:.3f} +/- {:.3f} eV".format(
                fit_result["params"]["mean"],
                fit_result["errors"]["mean"]
            ))
            print("Peak sigma           : {:.3f} +/- {:.3f} eV".format(
                fit_result["params"]["sigma"],
                fit_result["errors"]["sigma"]
            ))
            print("Peak FWHM            : {:.3f} +/- {:.3f} eV".format(
                fit_result["fwhm"],
                fit_result["fwhm_err"]
            ))

            plot_spectrum_with_fit(
                centers,
                counts,
                fit_result,
                title="Energy spectrum (all events)",
                output_path=args.output_prefix + "_all_fit.png",
                show=args.show
            )
        else:
            plot_spectrum_only(
                centers,
                counts,
                title="Energy spectrum (all events)",
                output_path=args.output_prefix + "_all.png",
                show=args.show
            )

    else:
        if channum is None:
            print("WARNING: Branch 'channum' is missing.")
            print("WARNING: Falling back to all-event analysis.")

            counts, edges, centers = build_histogram(
                energy,
                nbins=args.nbins,
                xmin=xmin,
                xmax=xmax
            )

            if args.fit:
                fit_result = fit_peak_from_histogram(
                    centers,
                    counts,
                    target_energy=args.line_energy,
                    fit_window=args.fit_window
                )

                plot_spectrum_with_fit(
                    centers,
                    counts,
                    fit_result,
                    title="Energy spectrum (fallback all events)",
                    output_path=args.output_prefix + "_fallback_all_fit.png",
                    show=args.show
                )
            else:
                plot_spectrum_only(
                    centers,
                    counts,
                    title="Energy spectrum (fallback all events)",
                    output_path=args.output_prefix + "_fallback_all.png",
                    show=args.show
                )
            return

        active_channels, per_channel_counts = summarize_channels(channum)

        print("")
        print("=== Active channel summary ===")
        print("Number of active channels:", len(active_channels))
        for ch, n_evt in zip(active_channels, per_channel_counts):
            print("Channel {:4d} : {:8d} events".format(int(ch), int(n_evt)))

        active_mask = np.isin(channum, active_channels)
        energy_active = energy[active_mask]
        channum_active = channum[active_mask]

        counts_all, edges_all, centers_all = build_histogram(
            energy_active,
            nbins=args.nbins,
            xmin=xmin,
            xmax=xmax
        )

        plot_overlay_by_channel(
            energy_active,
            channum_active,
            active_channels,
            nbins=args.nbins,
            xmin=xmin,
            xmax=xmax,
            title="Energy spectra by active channel",
            output_path=args.output_prefix + "_by_channel_overlay.png",
            show=args.show,
            max_channels_to_plot=args.max_channels_to_plot
        )

        if args.fit:
            fit_result_all = fit_peak_from_histogram(
                centers_all,
                counts_all,
                target_energy=args.line_energy,
                fit_window=args.fit_window
            )

            print("")
            print("=== Combined active-channel fit summary ===")
            print("Events in active channels: {}".format(energy_active.size))
            print("Peak target energy   : {:.3f} eV".format(fit_result_all["target_energy"]))
            print("Peak centroid        : {:.3f} +/- {:.3f} eV".format(
                fit_result_all["params"]["mean"],
                fit_result_all["errors"]["mean"]
            ))
            print("Peak sigma           : {:.3f} +/- {:.3f} eV".format(
                fit_result_all["params"]["sigma"],
                fit_result_all["errors"]["sigma"]
            ))
            print("Peak FWHM            : {:.3f} +/- {:.3f} eV".format(
                fit_result_all["fwhm"],
                fit_result_all["fwhm_err"]
            ))

            plot_spectrum_with_fit(
                centers_all,
                counts_all,
                fit_result_all,
                title="Combined spectrum (active channels only)",
                output_path=args.output_prefix + "_active_channels_fit.png",
                show=args.show
            )
        else:
            plot_spectrum_only(
                centers_all,
                counts_all,
                title="Combined spectrum (active channels only)",
                output_path=args.output_prefix + "_active_channels.png",
                show=args.show
            )


if __name__ == "__main__":
    main()