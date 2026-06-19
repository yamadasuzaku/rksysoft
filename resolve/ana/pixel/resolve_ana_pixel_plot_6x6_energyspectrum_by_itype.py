#!/usr/bin/env python

import argparse
import os

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

params = {"xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 8}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(params)

ITYPE_NAMES = ["Hp", "Mp", "Ms", "Lp", "Ls"]
g_types = [rf"$\rm {name}$" for name in ITYPE_NAMES]

# ITYPE ごとの色を明示的に固定する。
# matplotlib の color cycle に依存すると、バージョンや rcParams の違いで
# 色が変わったり、複数系列が見分けにくくなったりするため、ここでは
# 色覚多様性にも比較的強い Okabe--Ito 系の配色を使う。
ITYPE_COLORS = {
    0: "#0072B2",  # Hp: blue
    1: "#E69F00",  # Mp: orange
    2: "#009E73",  # Ms: green
    3: "#D55E00",  # Lp: vermillion
    4: "#CC79A7",  # Ls: reddish purple
}


def get_itype_color(itype):
    """Return a stable plotting color for a given ITYPE."""
    return ITYPE_COLORS.get(itype, f"C{itype % 10}")


def format_compact_count(count):
    """Format a count value so that it fits in a small panel title."""
    count = int(count)
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 10_000:
        return f"{count / 1_000:.1f}k"
    return f"{count:,d}"


def format_full_count(count):
    """Format a count value for terminal output."""
    return f"{int(count):,d}"


def format_exposure(exposure):
    """Format exposure in seconds. Return 'unknown' when unavailable."""
    if exposure is None or not np.isfinite(exposure):
        return "unknown"
    if exposure >= 1_000.0:
        return f"{exposure / 1_000.0:.2f} ks"
    return f"{exposure:.1f} s"


def get_header_value(hdus, keys, default=None):
    """
    Read the first available FITS header value from EVENTS first, then PRIMARY.

    Resolve cleaned event files normally carry OBJECT and EXPOSURE in the EVENTS
    extension, but this function is intentionally defensive so that the script
    also works with intermediate files whose keywords are slightly different.
    """
    headers = []
    if len(hdus) > 1:
        headers.append(hdus[1].header)
    if len(hdus) > 0:
        headers.append(hdus[0].header)

    for header in headers:
        for key in keys:
            if key in header:
                return header[key]
    return default


def plot_spectrum_points(ax, x, y, yerr, color, alpha=0.90, show_errorbar=True):
    """
    Plot one spectrum with a faint connecting line and compact error bars.

    The old script called errorbar() twice for each ITYPE -- once for a line and
    once for points -- and both calls used color="k".  With five ITYPEs this
    easily makes each panel almost black.  Here we draw the line separately and
    call errorbar() only once, using the same ITYPE color for all artists.
    """
    ax.plot(x, y, "-", color=color, linewidth=0.55, alpha=0.35)

    if show_errorbar:
        handle = ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=".",
            color=color,
            ecolor=color,
            markersize=2.0,
            elinewidth=0.40,
            capsize=0,
            alpha=alpha,
        )
        # エラーバーだけ少し薄くして、点と線を見やすくする。
        for bar in handle[2]:
            bar.set_alpha(0.30)
    else:
        ax.plot(x, y, ".", color=color, markersize=2.0, alpha=alpha)


def filter_strings_by_indices(strings, indices):
    """Return strings selected by integer indices."""
    return [strings[i] for i in indices]


def propagate_division_error(y1, yerr1, y2, yerr2):
    """Calculate y1/y2 and its propagated statistical error."""
    result = np.zeros_like(y1, dtype=float)
    propagated_error = np.zeros_like(y1, dtype=float)

    valid_indices = (y1 > 0) & (y2 > 0)
    result[valid_indices] = y1[valid_indices] / y2[valid_indices]
    propagated_error[valid_indices] = result[valid_indices] * np.sqrt(
        (yerr1[valid_indices] / y1[valid_indices]) ** 2
        + (yerr2[valid_indices] / y2[valid_indices]) ** 2
    )

    return result, propagated_error


def get_filename_without_extension(filepath):
    """Return basename without extension."""
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]


def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.25


def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return 2 * ev - 0.5


def gen_energy_hist(epi2, bin_width):
    bins = np.arange(0, 40e3, bin_width)
    ncount, bin_edges = np.histogram(epi2, bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_half_width = bin_width / 2
    ncount_sqrt = np.sqrt(ncount)
    energy = bin_centers
    return energy, ncount, bin_half_width, ncount_sqrt


def count_events_in_band(epi2, itype_list, pix_num_list, itype, pixel, ene_min, ene_max):
    """Count events in the plotted energy band for one pixel and one ITYPE."""
    mask = (
        (pix_num_list == pixel)
        & (itype_list == itype)
        & (epi2 >= ene_min)
        & (epi2 <= ene_max)
    )
    return int(np.count_nonzero(mask))


def make_suptitle(ifile, objectname, exposure, ene_min, ene_max, outtag, ratioflag):
    mode = "ratio" if ratioflag else "spectrum"
    basename = os.path.basename(ifile)
    return (
        f"{basename}: {objectname}, exp={format_exposure(exposure)}, "
        f"{mode}, energy {ene_min:g}-{ene_max:g} eV, itype {outtag}"
    )


def print_run_summary(
    ifile,
    objectname,
    exposure,
    ene_min,
    ene_max,
    bin_width,
    itypenames,
    ratioflag,
    ylogflag,
    commonymax,
    g_ymax,
    outfile,
    total_counts_by_itype,
    hp_counts_by_pixel,
    verbose=False,
):
    """Print only compact, useful information to stdout."""
    print("\n=== Resolve 6x6 spectrum plot ===")
    print(f"Input        : {ifile}")
    print(f"Object       : {objectname}")
    print(f"Exposure     : {format_exposure(exposure)}")
    print(f"Energy range : {ene_min:g}--{ene_max:g} eV")
    print(f"Bin width    : {bin_width:g} eV")
    print("ITYPE        : " + ", ".join(f"{i}:{ITYPE_NAMES[i]}" for i in itypenames))
    print(f"Mode         : {'ratio to central 4 pixels' if ratioflag else 'counts spectrum'}")
    print(f"Y scale      : {'log' if ylogflag else 'linear'}")

    if commonymax:
        print(f"Common ymax  : {g_ymax:g}")
    else:
        print("Common ymax  : disabled")

    print("Counts in selected band:")
    for itype in itypenames:
        print(f"  {ITYPE_NAMES[itype]:>2s} : {format_full_count(total_counts_by_itype.get(itype, 0))}")

    if hp_counts_by_pixel:
        hp_values = np.array(list(hp_counts_by_pixel.values()), dtype=float)
        min_pixel = min(hp_counts_by_pixel, key=hp_counts_by_pixel.get)
        max_pixel = max(hp_counts_by_pixel, key=hp_counts_by_pixel.get)
        print(
            "Hp per pixel : "
            f"min PIXEL={min_pixel} ({format_full_count(hp_counts_by_pixel[min_pixel])}), "
            f"median {np.median(hp_values):.0f}, "
            f"max PIXEL={max_pixel} ({format_full_count(hp_counts_by_pixel[max_pixel])})"
        )

    if verbose:
        print("\nPer-pixel Hp counts in selected band:")
        for pixel in sorted(hp_counts_by_pixel):
            print(f"  PIXEL={pixel:2d}  Hp={format_full_count(hp_counts_by_pixel[pixel])}")

    print(f"Output       : {outfile}")
    print("==================================\n")


def plot_spec_6x6(
    ifile,
    bin_width,
    ene_min,
    ene_max,
    itypenames=[0],
    commonymax=True,
    ratioflag=False,
    outtag="auto",
    ylogflag=False,
    plotflag=False,
    commonymaxvalue=None,
    show_errorbar=True,
    verbose=False,
):
    pixel_map = np.array(
        [
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],  # DETY
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],  # DETX
            [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18, 25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27, 29, 30],  # PIXEL
        ]
    )

    with fits.open(ifile) as hdus:
        objectname = get_header_value(hdus, ["OBJECT"], default="UNKNOWN")
        exposure = get_header_value(hdus, ["EXPOSURE", "ONTIME", "LIVETIME"], default=np.nan)
        try:
            exposure = float(exposure)
        except (TypeError, ValueError):
            exposure = np.nan

        itype_list = np.asarray(hdus[1].data["ITYPE"]).copy()
        pix_num_list = np.asarray(hdus[1].data["PIXEL"]).copy()
        epi2 = np.asarray(hdus[1].data["EPI2"]).copy()

    fig, ax = plt.subplots(6, 6, figsize=(21.3, 12), sharex=True)
    fig.suptitle(make_suptitle(ifile, objectname, exposure, ene_min, ene_max, outtag, ratioflag), fontsize=11)

    g_ymax = 0
    hp_counts_by_pixel = {}
    total_counts_by_itype = {
        itype: int(np.count_nonzero((itype_list == itype) & (epi2 >= ene_min) & (epi2 <= ene_max)))
        for itype in itypenames
    }

    if ratioflag:
        stored_hist_dic = {}
        for itype in itypenames:
            cutid = np.where(
                (itype_list == itype)
                & ((pix_num_list == 0) | (pix_num_list == 17) | (pix_num_list == 18) | (pix_num_list == 35))
            )[0]
            epi2_itype_allpixel = epi2[cutid]
            energy, ncount, xerr, yerr = gen_energy_hist(epi2_itype_allpixel, bin_width)
            stored_hist_dic[itype] = (energy, ncount, xerr, yerr)

    for i in range(36):
        dety = 6 - pixel_map.T[i][0]
        detx = pixel_map.T[i][1] - 1
        pixel = pixel_map.T[i][2]
        axp = ax[dety, detx]

        hp_count = count_events_in_band(epi2, itype_list, pix_num_list, 0, pixel, ene_min, ene_max)
        hp_counts_by_pixel[pixel] = hp_count

        # タイトル左側に PIXEL、右側に小さく Hp の総カウントを出す。
        # Hp カウントは「描画しているエネルギー範囲内」のイベント数である。
        axp.set_title(f"PIXEL={pixel}", loc="left", fontsize=8.0, pad=2)
        axp.set_title(f"Hp={format_compact_count(hp_count)}", loc="right", fontsize=6.5, pad=2, color=get_itype_color(0))

        mask_pix = pix_num_list == pixel
        epi2_pix = epi2[mask_pix]
        itype_pix = itype_list[mask_pix]

        y_min, y_max = np.inf, -np.inf

        for itype in itypenames:
            color = get_itype_color(itype)
            mask_itype = itype_pix == itype
            epi2_pix_itype = epi2_pix[mask_itype]

            energy, ncount, xerr, yerr = gen_energy_hist(epi2_pix_itype, bin_width)
            mask = (ene_min < energy) & (energy < ene_max)

            energy, ncount, yerr = energy[mask], ncount[mask], yerr[mask]

            if mask.any():
                current_ymin = np.amin(ncount)
                current_ymax = np.amax(ncount)
                if g_ymax < current_ymax:
                    g_ymax = current_ymax

                y_min = min(y_min, current_ymin)
                y_max = max(y_max, current_ymax)

                if ratioflag:
                    ave_energy, ave_ncount, ave_xerr, ave_yerr = stored_hist_dic[itype]
                    ave_energy, ave_ncount, ave_yerr = ave_energy[mask], ave_ncount[mask], ave_yerr[mask]
                    ratio, ratio_err = propagate_division_error(ncount, yerr, ave_ncount, ave_yerr)
                    plot_spectrum_points(
                        axp,
                        energy,
                        ratio,
                        ratio_err,
                        color=color,
                        show_errorbar=show_errorbar,
                    )
                else:
                    plot_spectrum_points(
                        axp,
                        energy,
                        ncount,
                        yerr,
                        color=color,
                        show_errorbar=show_errorbar,
                    )

                axp.set_xlim(ene_min, ene_max)

                if ratioflag:
                    if ylogflag:
                        axp.semilogy()
                else:
                    if y_min > 0 and y_max > 0:
                        axp.set_ylim(y_min, y_max + 5)
                    if np.any(ncount > 0) and ylogflag:
                        axp.semilogy()

        if dety == 5 and detx == 0:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=get_itype_color(itype),
                    marker=".",
                    linestyle="-",
                    linewidth=0.8,
                    markersize=4,
                    label=g_types[itype],
                )
                for itype in itypenames
            ]
            axp.legend(handles=legend_handles, loc="lower left")

    for i in range(6):
        if ratioflag:
            ax[i, 0].set_ylabel("ratio to cen4")
            ax[5, i].set_xlabel(rf"$\rm Energy\ (eV)$")
        else:
            ax[i, 0].set_ylabel(rf"$\rm Counts/{bin_width:g}eV$")
            ax[5, i].set_xlabel(rf"$\rm Energy\ (eV)$")

    if commonymax:
        if commonymaxvalue is not None:
            g_ymax = commonymaxvalue
        for i in range(36):
            dety = 6 - pixel_map.T[i][0]
            detx = pixel_map.T[i][1] - 1
            if ratioflag:
                ax[dety, detx].set_ylim(0.01, 10)
                if ylogflag:
                    ax[dety, detx].semilogy()
            else:
                ax[dety, detx].set_ylim(0.1, g_ymax)
                if ylogflag:
                    ax[dety, detx].semilogy()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ftag = get_filename_without_extension(ifile)
    if ratioflag:
        outfile = f"resolve_spec_plot6x6_{ftag}_EneRan{ene_min}-{ene_max}eV_itype{outtag}_bin{bin_width}_ratio.png"
    else:
        outfile = f"resolve_spec_plot6x6_{ftag}_EneRan{ene_min}-{ene_max}eV_itype{outtag}_bin{bin_width}.png"
    plt.savefig(outfile, dpi=150)

    if ratioflag:
        fig_check, ax_check = plt.subplots(figsize=(8, 10), sharex=True)
        fig_check.suptitle(
            f"Check all spec, {os.path.basename(ifile)}: Energy range {ene_min:g}-{ene_max:g} eV"
        )

        for itype in itypenames:
            ave_energy, ave_ncount, ave_xerr, ave_yerr = stored_hist_dic[itype]
            color = get_itype_color(itype)
            ax_check.errorbar(
                ave_energy,
                ave_ncount,
                yerr=ave_yerr,
                fmt="o-",
                color=color,
                ecolor=color,
                markersize=2.5,
                linewidth=0.7,
                elinewidth=0.4,
                capsize=0,
                label=ITYPE_NAMES[itype],
            )
        ax_check.legend()
        fig_check.tight_layout()
        fig_check.savefig("checkall_" + outfile, dpi=150)

    print_run_summary(
        ifile=ifile,
        objectname=objectname,
        exposure=exposure,
        ene_min=ene_min,
        ene_max=ene_max,
        bin_width=bin_width,
        itypenames=itypenames,
        ratioflag=ratioflag,
        ylogflag=ylogflag,
        commonymax=commonymax,
        g_ymax=g_ymax,
        outfile=outfile,
        total_counts_by_itype=total_counts_by_itype,
        hp_counts_by_pixel=hp_counts_by_pixel,
        verbose=verbose,
    )

    if plotflag:
        plt.show()


def parse_itypenames(value):
    itypenames = [int(v.strip()) for v in value.split(",") if v.strip() != ""]
    invalid = [itype for itype in itypenames if itype < 0 or itype >= len(ITYPE_NAMES)]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid ITYPE(s): {invalid}. Allowed values are 0--4.")
    return itypenames


def main():
    parser = argparse.ArgumentParser(
        description="Plot Resolve 6x6 energy spectra by ITYPE.",
        epilog="""
Example (1): plot spectra from 2 keV to 20 keV with 400 eV bin, Hp only
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype_colorfix_v2.py xa300049010rsl_p0px3000_cl.evt -b 400 -l 2000 -x 20000 -y 0

Example (2): plot spectral ratios from 2 keV to 20 keV with 400 eV bin, Hp only
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype_colorfix_v2.py xa300049010rsl_p0px3000_cl.evt -r -b 400 -l 2000 -x 20000 -y 0 -c -g

Example (3): dense multi-ITYPE plot without error bars
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype_colorfix_v2.py xa201060010rsl_p0px3000_cl.evt -b 5 -l 6000 -x 7200 -y 0,1,2,3,4 --no_errorbar
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("filename", type=str, help="Input FITS event file name")
    parser.add_argument("--bin_width", "-b", type=float, default=4, help="Bin width for histogram [eV]")
    parser.add_argument("--ene_min", "-l", type=float, default=6300, help="Minimum energy [eV]")
    parser.add_argument("--ene_max", "-x", type=float, default=6900, help="Maximum energy [eV]")
    parser.add_argument("--commonymax", "-c", action="store_false", help="Disable common global ymax")
    parser.add_argument("--commonymaxvalue", "-cval", type=float, default=None, help="Y max used when common-y mode is enabled")
    parser.add_argument("--ratioflag", "-r", action="store_true", help="Plot spectral ratio to average of central four pixels")
    parser.add_argument("--ylogflag", "-g", action="store_false", help="Use linear y-scale when specified; default is log scale")
    parser.add_argument("--itypenames", "-y", type=parse_itypenames, help="Comma-separated list of ITYPE values", default="0,1,2,3,4")
    parser.add_argument("--plotflag", "-p", action="store_true", help="Show plot interactively")
    parser.add_argument("--no_errorbar", action="store_true", help="Do not draw statistical error bars. Useful for dense multi-ITYPE plots.")
    parser.add_argument("--verbose", action="store_true", help="Print per-pixel Hp count table in addition to the compact summary.")

    args = parser.parse_args()
    itypenames = args.itypenames if isinstance(args.itypenames, list) else parse_itypenames(args.itypenames)
    itypeinfo = "_".join(str(itype) for itype in itypenames)

    plot_spec_6x6(
        ifile=args.filename,
        bin_width=args.bin_width,
        ene_min=args.ene_min,
        ene_max=args.ene_max,
        itypenames=itypenames,
        commonymax=args.commonymax,
        ratioflag=args.ratioflag,
        outtag=itypeinfo,
        ylogflag=args.ylogflag,
        plotflag=args.plotflag,
        commonymaxvalue=args.commonymaxvalue,
        show_errorbar=(not args.no_errorbar),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
