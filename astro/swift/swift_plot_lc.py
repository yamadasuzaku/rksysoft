#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from astropy.time import Time
import os
import sys
import argparse
from astropy.io import fits

# ------------------------------------------------------------
# データ読み込み & フィルタ
# ------------------------------------------------------------

def load_and_filter_data(
    filename,
    relative_error_threshold=10,
    apply_relerr_filter=True,
    debug=False,
):
    """
    Swift/BAT ライトカーブ FITS を読み込む。

    [1] one-day ファイル（RapidBurster.lc.fits 等）
        - BINTABLE に MJD カラムが無い
        - TIME カラムがそのまま MJD [d]
        -> TIME をそのまま mjd として扱う

    [2] one-orbit ファイル（RapidBurster.orbit.lc.fits 等）
        - BINTABLE に MJD カラムが存在する
        - TIME カラムは MET [s] （MJDREF からの経過秒）
        -> TIME, MJDREFI, MJDREFF から mjd を計算

    その上で、相対誤差の大きい点を簡易フィルタする。
    """

    print("=== load_and_filter_data ===")
    print(f"filename            : {filename}")
    print(f"relerr_threshold[%] : {relative_error_threshold}")
    print(f"apply_relerr_filter : {apply_relerr_filter}")

    with fits.open(filename) as hdul:
        hdu  = hdul[1]
        data = hdu.data
        hdr  = hdu.header

        colnames = [name.upper() for name in data.names]
        print(f"columns             : {colnames}")

        # -------------------------------
        # [1] one-day: TIME = MJD [d]
        # -------------------------------
        if 'MJD' not in colnames:
            mjd = np.array(data['TIME'], dtype=float)
            mode_str = "one-day (TIME = MJD [d])"
        # -------------------------------
        # [2] one-orbit: TIME = MET [s]
        # -------------------------------
        else:
            time_met = np.array(data['TIME'], dtype=float)  # [s]
            mjdrefi  = hdr.get('MJDREFI', 0.0)
            mjdreff  = hdr.get('MJDREFF', 0.0)
            timezero = hdr.get('TIMEZERO', 0.0)  # 無ければ 0 とみなす

            mjdref = mjdrefi + mjdreff
            mjd    = mjdref + (time_met + timezero) / 86400.0
            mode_str = "one-orbit (TIME = MET [s] → MJD via MJDREF)"

        rate  = np.array(data['RATE'], dtype=float)
        error = np.array(data['ERROR'], dtype=float)

        n_total = len(mjd)
        print(f"mode                : {mode_str}")
        print(f"N total rows        : {n_total}")
        print(f"MJD range           : {mjd.min():.3f} – {mjd.max():.3f}")
        print(
            "RATE stats          : "
            f"min={rate.min():.3e}, max={rate.max():.3e}, "
            f"mean={rate.mean():.3e}, median={np.median(rate):.3e}"
        )
        print(
            "ERROR stats         : "
            f"min={error.min():.3e}, max={error.max():.3e}, "
            f"mean={error.mean():.3e}"
        )
        print(f"N rate<=0           : {(rate <= 0).sum()}")

        # astropy Time を用いて datetime に変換
        time_objs = Time(mjd, format='mjd')  # 厳密な TT/UTC はここでは気にしない
        dates     = np.array(time_objs.to_datetime())

    # -------------------------------
    # 有限値・rate>0 カット
    # -------------------------------
    finite_mask = np.isfinite(rate) & np.isfinite(error)
    pos_mask    = rate > 0
    base_mask   = finite_mask & pos_mask

    print(f"N finite(rate,err)  : {finite_mask.sum()}")
    print(f"N rate>0            : {pos_mask.sum()}")
    print(f"N after finite&>0   : {base_mask.sum()}")

    mjd   = mjd[base_mask]
    dates = dates[base_mask]
    rate  = rate[base_mask]
    error = error[base_mask]

    if len(mjd) == 0:
        print("WARNING: No data points left after finite & rate>0 cut.")
        return {"dates": np.array([]), "mjd": np.array([]),
                "rate": np.array([]), "error": np.array([])}

    # -------------------------------
    # 相対誤差による簡易フィルタ
    # -------------------------------
    if apply_relerr_filter:
        x = relative_error_threshold / 100.0
        relative_error = error / rate

        # 基本統計は常に表示
        print(
            "Rel.err stats       : "
            f"min={relative_error.min():.3f}, "
            f"max={relative_error.max():.3f}"
        )
        if debug:
            # もう少し詳しい統計は debug のときだけ
            q10, q50, q90 = np.percentile(relative_error, [10, 50, 90])
            print(
                "Rel.err percentiles : "
                f"P10={q10:.3f}, P50={q50:.3f}, P90={q90:.3f}"
            )

        print(
            f"N rel.err <= {x:.3f} : "
            f"{(relative_error <= x).sum()} / {len(relative_error)}"
        )

        mask = (relative_error <= x)
    else:
        print("Relative-error filter: DISABLED (--no_relerr_filter).")
        mask = np.ones_like(rate, dtype=bool)

    mjd   = mjd[mask]
    dates = dates[mask]
    rate  = rate[mask]
    error = error[mask]

    print(f"N after relerr filt.: {len(mjd)}")
    print("=== end of load_and_filter_data ===")

    if len(mjd) == 0:
        print("WARNING: No data points left after all filtering.")
        return {"dates": np.array([]), "mjd": np.array([]),
                "rate": np.array([]), "error": np.array([])}

    filtered_data = {
        "dates": dates,
        "mjd":   mjd,
        "rate":  rate,
        "error": error,
    }
    return filtered_data

# ------------------------------------------------------------
# Plotly
# ------------------------------------------------------------

def create_plotly_subplots(dates, flux_data, base_filename,
                           start_date=None, end_date=None,
                           highlight_periods=None):

    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=('Time vs Flux',)
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=flux_data["rate"],
            mode='markers',
            name='15–50 keV',
            error_y=dict(type='data', array=flux_data["error"], visible=True)
        ),
        row=1, col=1
    )

    # 表示範囲の設定
    if start_date and end_date:
        fig.update_xaxes(range=[start_date, end_date], row=1, col=1)

    # ハイライトの設定
    if highlight_periods:
        for start, end in highlight_periods:
            print(f"..... highlight from {start} to {end}.")
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="orange", opacity=0.3, line_width=0,
                row=1, col=1
            )

    fig.update_layout(
        height=900,
        font_family="Times New Roman",
        title=f'Interactive Light Curve ({base_filename})',
        xaxis_title='Date',
        yaxis_title='Counts [/cm^2/s] (15–50 keV)',
        hovermode='x unified'
    )

    return fig


CRAB_RATE_15_50_KEV = 0.22  # [count/cm^2/s] で 1 Crab に相当

def create_matplotlib_subplots(
    dates,
    flux_data,
    base_filename,
    start_date=None,
    end_date=None,
    highlight_periods=None,
    yscale="linear",
):
    """
    15–50 keV BAT ライトカーブを matplotlib で描画する。

    左軸  : count/cm^2/s
    右軸  : Crab 単位（1 Crab = 0.22 count/cm^2/s）
    yscale: "linear" または "log"
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- メイン（左）y軸: count/cm^2/s ---
    ax.errorbar(
        dates,
        flux_data["rate"],
        yerr=flux_data["error"],
        fmt='.',
        label='15–50 keV',
        color='red',
    )
    ax.set_ylabel('Flux [count/cm$^2$/s] (15–50 keV)')
    ax.legend()
    ax.grid(True, alpha=0.6, linestyle="--")

    # yスケール設定（linear / log）
    ax.set_yscale(yscale)

    # 表示範囲の設定（x軸）
    if start_date and end_date:
        ax.set_xlim(start_date, end_date)

    # ハイライトの設定（左軸のスケールに合わせて塗る）
    if highlight_periods:
        ymin = np.min(flux_data["rate"])
        ymax = np.max(flux_data["rate"])
        # ログスケールで ymin<=0 だと困るので、少しだけ底上げ
        if yscale == "log" and ymin <= 0:
            positive = flux_data["rate"][flux_data["rate"] > 0]
            if len(positive) > 0:
                ymin = positive.min() * 0.5
            else:
                ymin = 1e-6
        ylow = ymin
        yhigh = ymax

        for start, end in highlight_periods:
            ax.fill_between(
                dates,
                ylow,
                yhigh,
                where=((dates >= start) & (dates <= end)),
                color='orange',
                alpha=0.3,
            )

    # --- 右 y軸: Crab 単位（secondary_yaxis を使用） ---
    def counts_to_crab(y_counts):
        return y_counts / CRAB_RATE_15_50_KEV

    def crab_to_counts(y_crab):
        return y_crab * CRAB_RATE_15_50_KEV

    # 左軸と同じスケール（linear/log）で、単位だけ変換する副軸
    ax_crab = ax.secondary_yaxis(
        'right',
        functions=(counts_to_crab, crab_to_counts),
    )
    ax_crab.set_ylabel('Flux [Crab] (15–50 keV)')

    plt.suptitle(f'Light Curve ({base_filename})')
    plt.tight_layout()

    return fig

# ------------------------------------------------------------
# 時間ビニング
# ------------------------------------------------------------

def bin_flux_by_days(flux_data, bin_size_days):
    """
    指定した日数 bin_size_days で時間ビニングしたライトカーブを作る。
    ビン内の rate は 1/σ^2 で重み付き平均、誤差は 1/sqrt(sum(1/σ^2)) とする。
    """

    mjd   = np.array(flux_data["mjd"])
    rate  = np.array(flux_data["rate"])
    error = np.array(flux_data["error"])

    if len(mjd) == 0:
        raise ValueError("No data points to bin.")

    t_min = mjd.min()
    t_max = mjd.max()
    bin_edges = np.arange(t_min, t_max + bin_size_days, bin_size_days)

    # bin_index: 各データ点がどのビンに属するか
    bin_index = np.digitize(mjd, bin_edges) - 1  # 0-origin に

    binned_mjd  = []
    binned_rate = []
    binned_err  = []

    for k in range(len(bin_edges) - 1):
        mask = (bin_index == k)
        if not np.any(mask):
            continue

        mjd_bin = mjd[mask].mean()

        sigma = error[mask]
        # 誤差 0 の点があると困るので、安全側に小さな値を入れておく
        if np.any(sigma <= 0):
            pos = sigma[sigma > 0]
            if len(pos) > 0:
                sigma[sigma <= 0] = np.median(pos)
            else:
                sigma[sigma <= 0] = 1.0  # 最悪の保険

        weights = 1.0 / (sigma ** 2)
        rate_bin = np.sum(rate[mask] * weights) / np.sum(weights)
        err_bin  = 1.0 / np.sqrt(np.sum(weights))

        binned_mjd.append(mjd_bin)
        binned_rate.append(rate_bin)
        binned_err.append(err_bin)

    binned_mjd  = np.array(binned_mjd)
    binned_rate = np.array(binned_rate)
    binned_err  = np.array(binned_err)

    time_objs = Time(binned_mjd, format="mjd")
    dates     = np.array(time_objs.to_datetime())

    binned_data = {
        "dates": dates,
        "mjd":   binned_mjd,
        "rate":  binned_rate,
        "error": binned_err,
    }

    return binned_data

# ------------------------------------------------------------
# argparse
# ------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Plot Swift/BAT 15–50 keV light curve using matplotlib or plotly."
    )
    parser.add_argument(
        "filename",
        help="Path to the input data file (e.g., CygX-1.lc.fits)",
    )
    parser.add_argument(
        "--plotter",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Choose the plotting library: 'matplotlib' or 'plotly' (default: matplotlib)",
    )
    parser.add_argument(
        "--start_date",
        help="Start date in ISOT format (e.g., '2023-10-15T00:00:00')",
        default=None,
    )
    parser.add_argument(
        "--end_date",
        help="End date in ISOT format (e.g., '2024-05-01T00:00:00')",
        default=None,
    )
    parser.add_argument(
        "--highlight",
        nargs=2,
        action='append',
        metavar=('START', 'END'),
        help="Highlight period in ISOT format "
             "(e.g., '2023-11-03T23:51:04 2023-11-05T00:00:00')",
    )

    # ビン幅（日）を指定（複数指定可）
    parser.add_argument(
        "--bin_days",
        nargs="+",
        type=float,
        help="Time bin size(s) in days for heavy binning (e.g., 10 30 60)",
        default=None,
    )

    # 相対誤差しきい値（パーセント）
    parser.add_argument(
        "--relerr_thresh",
        type=float,
        default=50.0,
        help=(
            "Relative error threshold in percent (default: 50). "
            "Points with ERROR/RATE larger than this will be discarded."
        ),
    )

    # 相対誤差フィルタを無効化するフラグ
    parser.add_argument(
        "--no_relerr_filter",
        action="store_true",
        help="If set, skip relative-error-based filtering.",
    )

    # y軸スケール（linear or log）
    parser.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for flux (default: linear).",
    )

    # デバッグ情報を出すフラグ（より詳細な統計）
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug information (percentiles of relative errors etc.).",
    )

    return parser.parse_args()

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(
    filename,
    plot_library,
    start_date=None,
    end_date=None,
    highlight_periods=None,
    bin_days=None,
    relerr_thresh=50.0,
    no_relerr_filter=False,
    debug=False,
    yscale="linear",
):
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # データの読み込みとフィルタリング
    flux_data = load_and_filter_data(
        filename,
        relative_error_threshold=relerr_thresh,
        apply_relerr_filter=not no_relerr_filter,
        debug=debug,
    )

    # フィルタ後にデータが残っているか確認
    if len(flux_data["dates"]) == 0:
        print("No data to plot after filtering. "
              "Try increasing --relerr_thresh or using --no_relerr_filter.")
        return

    # 時間範囲の処理
    if start_date:
        start_date = Time(start_date.strip(), format="isot").datetime
    if end_date:
        end_date = Time(end_date.strip(), format="isot").datetime

    # ハイライト期間の処理
    if highlight_periods:
        highlight_periods = [
            (
                Time(start.strip(), format="isot").datetime,
                Time(end.strip(),   format="isot").datetime
            )
            for start, end in highlight_periods
        ]

    # メインのライトカーブ
    if plot_library == 'plotly':
        fig = create_plotly_subplots(
            flux_data["dates"], flux_data,
            base_filename,
            start_date, end_date, highlight_periods
        )
        html_filename = f"{base_filename}_plotly.html"
        pio.write_html(fig, file=html_filename)
        fig.show()
    elif plot_library == 'matplotlib':
        fig = create_matplotlib_subplots(
            flux_data["dates"], flux_data,
            base_filename,
            start_date, end_date, highlight_periods,
            yscale=yscale,
        )
        out_png = f"{base_filename}_matplotlib.png"
        print(f"Saving matplotlib figure to {out_png}")
        plt.savefig(out_png, dpi=300)
        plt.show()

    # ハイライト期間の flux 内挿（従来どおり）
    if highlight_periods:
        dates_num = np.array([date.timestamp() for date in flux_data["dates"]])
        highlight_times = np.array(
            [date.timestamp() for period in highlight_periods for date in period]
        )
        highlight_rate = np.interp(highlight_times, dates_num, flux_data["rate"])
        for i, period in enumerate(highlight_periods):
            print(f"Period {period[0]} to {period[1]}:")
            print(f"  Interpolated flux at {period[0]}: {highlight_rate[2*i]}")
            print(f"  Interpolated flux at {period[1]}: {highlight_rate[2*i+1]}")

    # 重い時間ビニング（10, 30, 60 日など）の追加プロット
    if bin_days is not None:
        for bd in bin_days:
            print(f"==== Time binning with {bd} days ====")
            binned_data = bin_flux_by_days(flux_data, bd)
            bin_label   = f"{base_filename}_bin{bd:.1f}d"

            if plot_library == 'plotly':
                fig_bin = create_plotly_subplots(
                    binned_data["dates"], binned_data,
                    bin_label,
                    start_date, end_date, highlight_periods
                )
                html_filename = f"{bin_label}_plotly.html"
                pio.write_html(fig_bin, file=html_filename)
                fig_bin.show()
            elif plot_library == 'matplotlib':
                fig_bin = create_matplotlib_subplots(
                    binned_data["dates"], binned_data,
                    bin_label,
                    start_date, end_date, highlight_periods,
                    yscale=yscale,
                )
                out_png = f"{bin_label}_matplotlib.png"
                print(f"Saving binned matplotlib figure to {out_png}")
                plt.savefig(out_png, dpi=300)
                plt.show()

# ------------------------------------------------------------
# Colab 用
# ------------------------------------------------------------

def run_in_colab():
    filename = "CygX-1.lc.fits"  # ここは適宜変更

    plot_library = "plotly"
    start_date   = "2023-11-01T00:00:00"
    end_date     = "2024-06-01T00:00:00"

    highlight_periods = [
        ("2023-11-03T23:51:04", "2023-11-05T14:01:04"),
    ]

    bin_days = [10, 30, 60]

    main(
        filename,
        plot_library,
        start_date,
        end_date,
        highlight_periods,
        bin_days,
        relerr_thresh=50.0,
        no_relerr_filter=False,
        debug=False,
        yscale="linear",
    )

# ------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------

if __name__ == "__main__":
    if "google.colab" in sys.modules:
        print("Running in Google Colab environment. Using predefined settings.")
        run_in_colab()
    else:
        args = get_args()
        main(
            args.filename,
            args.plotter,
            args.start_date,
            args.end_date,
            args.highlight,
            args.bin_days,
            args.relerr_thresh,
            args.no_relerr_filter,
            args.debug,
            args.yscale,
        )