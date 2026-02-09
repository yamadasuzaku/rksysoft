#!/usr/bin/env python
"""
calc_lscluster_rates.py

FITS イベントファイルから、GTI を考慮したカウントレートを
ICLUSTERL / ICLUSTERS の 4 パターンごとに計算して CSV に書き出し、
同時に簡易サマリプロット (2x2 サブプロット) を作るスクリプト。

使い方例:
    # 単一観測 (イベントファイルを列挙)
    python calc_lscluster_rates.py evt1.evt evt2.evt \
        -o lscluster_rates_cygx1.csv

    # ファイルリストを使う場合
    python calc_lscluster_rates.py @evt_list.txt \
        -o lscluster_rates_perseus.csv

    # cal pixel(12) も含めたい場合
    python calc_lscluster_rates.py @evt_list.txt \
        -o lscluster_rates_allpix.csv --include-cal-pixel12
"""

import argparse
import os
import csv

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# ========================================
# Pixel 配置 (DETX/DETY -> pixel) のマップ
#   resolve_plot_detxdety.py の定義を流用
# ========================================
PIXEL_FROM_DETX_DETY = [
    [12, 11, 9, 19, 21, 23],
    [14, 13, 10, 20, 22, 24],
    [16, 15, 17, 18, 25, 26],
    [8, 7, 0, 35, 33, 34],
    [6, 4, 2, 28, 31, 32],
    [5, 3, 1, 27, 29, 30],
]

FIGSIZE_PPT = (10.5, 7)  # 横長 (ppt スライド向け)
FIGSIZE_A4 = (8.27, 11.69)  # 参考: A4 portrait


def build_pixel_map_from_group(group, value_col, fill_value=np.nan):
    """
    group: df.groupby("pixel").agg(...) の結果 (index が pixel)
    value_col: group 内のカラム名 (例: "rate_all_mean", "rate_large_total_mean" など)

    戻り値:
        6x6 の numpy 配列 (DETX, DETY 方向)
    """
    g = group.copy()
    g.index = g.index.astype(int)

    arr = np.full((6, 6), fill_value=fill_value, dtype=float)

    for detx_idx in range(6):  # 0..5
        for dety_idx in range(6):  # 0..5
            pix = PIXEL_FROM_DETX_DETY[detx_idx][dety_idx]
            if pix in g.index:
                arr[detx_idx, dety_idx] = g.loc[pix, value_col]

    return arr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ICLUSTERL / ICLUSTERS に基づく 4 パターンのカウントレートを CSV に出力し、"
            "簡易サマリプロットを作るスクリプト"
        )
    )
    parser.add_argument(
        "events",
        nargs="+",
        help=(
            "処理するイベント FITS ファイル群。"
            "@filelist.txt のように先頭に @ を付けると、そのファイル中のパスを読み込む。"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default=None,
        help=(
            "出力 CSV ファイル名。"
            "指定しない場合は、最初の入力 EVT ファイル名から "
            "'*_lsrates.csv' を自動生成する。"
        ),
    )
    parser.add_argument(
        "--no-use-gti",
        dest="use_gti",
        action="store_false",
        help="GTI を使わず、TIME の範囲から露光時間を計算する (デフォルト: GTI を使う)",
    )
    parser.add_argument(
        "--min-events-per-pixel",
        type=int,
        default=0,
        help="この数未満のイベントしかないピクセルは CSV に出力しない（デフォルト 0 = 全て出す）",
    )
    parser.add_argument(
        "--include-cal-pixel12",
        action="store_true",
        help="cal pixel (pixel=12) も含めて出力する（デフォルトでは除外）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="簡易サマリプロットを作らない",
    )
    parser.add_argument(
        "--summary-plot",
        help=(
            "サマリプロットの出力ファイル名。指定しない場合は "
            "<output_csv_basename>_summary.png になる。"
            " --no-plot が指定されている場合は無視される。"
        ),
    )
    return parser.parse_args()


def infer_output_csv(event_files):
    """
    出力 CSV 名を自動生成する。

    ルール:
      - 最初のイベントファイル名を使う
      - basename.evt → basename_lsrates.csv
    """
    first_evt = os.path.basename(event_files[0])

    if first_evt.endswith(".evt"):
        base = first_evt[:-4]
    else:
        base = first_evt

    return base + "_lsrates.csv"


def expand_event_files(args_events):
    """@filelist 記法を展開してイベントファイルリストを返す。"""
    event_files = []
    for token in args_events:
        if token.startswith("@"):
            list_path = token[1:]
            print(f"[INFO] loading event list from {list_path}")
            with open(list_path, "r") as f:
                for line in f:
                    path = line.strip()
                    if path and not path.startswith("#"):
                        event_files.append(path)
        else:
            event_files.append(token)
    return event_files


def compute_exposure(fname, use_gti=True):
    """
    有効露光時間 [s] を計算する。

    - use_gti=True かつ HDU[2] が GTI を持っていれば sum(STOP - START)
    - それ以外は TIME の max - min
    """
    with fits.open(fname) as hdul:
        evt = hdul[1].data
        time = evt["TIME"]

        if (
            use_gti
            and len(hdul) > 2
            and "START" in hdul[2].data.names
            and "STOP" in hdul[2].data.names
        ):
            gti = hdul[2].data
            t_exp = float(np.sum(gti["STOP"] - gti["START"]))
            print(f"    [INFO] exposure from GTI: {t_exp:.3f} s")
        else:
            if len(time) < 2:
                t_exp = 0.0
                print("    [WARN] TIME が十分でないため露光時間を 0 とみなします")
            else:
                t_exp = float(time.max() - time.min())
                print(f"    [INFO] exposure from TIME range: {t_exp:.3f} s")

    return t_exp


def process_one_file(
    fname,
    use_gti=True,
    min_events_per_pixel=0,
    include_cal_pixel12=False,
):
    """
    1 つのイベント FITS から、ピクセルごとのレート情報を dict のリストとして返す。
    ここで pixel=12 (cal pixel) はデフォルトで除外される。
    """
    rows = []

    with fits.open(fname) as hdul:
        hdr = hdul[1].header
        evt = hdul[1].data

        obs_id = hdr.get("OBS_ID", "")
        obj = hdr.get("OBJECT", "")
        event_file = os.path.basename(fname)

        print(f"[INFO] file={event_file}, OBS_ID={obs_id}, OBJECT={obj}")

        # 必須カラムの存在チェック
        required_cols = ["TIME", "PIXEL", "ICLUSTERL", "ICLUSTERS"]
        for col in required_cols:
            if col not in evt.names:
                raise RuntimeError(f"{fname}: 必須カラム '{col}' が見つかりません。")

        time = evt["TIME"]
        pixel = evt["PIXEL"]
        icl = evt["ICLUSTERL"]
        ics = evt["ICLUSTERS"]

        n_evt_total = len(time)
        print(f"    [INFO] total events in file: {n_evt_total}")

        # 観測開始・終了・span
        if len(time) > 0:
            time_start = float(time.min())
            time_end = float(time.max())
            time_span = time_end - time_start
            print(
                f"    [INFO] TIME start={time_start:.3f}, "
                f"end={time_end:.3f}, span={time_span:.3f} s"
            )
        else:
            time_start = 0.0
            time_end = 0.0
            time_span = 0.0
            print("    [WARN] TIME が空です")

        # 露光時間 (GTI or TIME range)
        t_exp = compute_exposure(fname, use_gti=use_gti)
        if t_exp <= 0.0 or len(time) == 0:
            print(f"    [WARN] skip (no exposure/events): {fname}")
            return rows  # 空

        unique_pixels = np.unique(pixel.astype(int))
        print(f"    [INFO] number of unique pixels in this file: {len(unique_pixels)}")
        if not include_cal_pixel12:
            print(
                "    [INFO] excluding cal pixel 12 "
                "(use --include-cal-pixel12 to include)"
            )
        else:
            print("    [INFO] including cal pixel 12")

        # ファイル全体でのクラスター統計の合計（ざっくり表示用）
        file_sum_n_all = 0
        file_sum_large = 0
        file_sum_small = 0
        file_sum_both_flag = 0
        file_sum_clean = 0

        for pix in unique_pixels:
            if (not include_cal_pixel12) and (pix == 12):
                continue

            mask_pix = pixel == pix
            n_all = int(np.sum(mask_pix))
            if n_all < min_events_per_pixel:
                continue

            icl_pix = icl[mask_pix]
            ics_pix = ics[mask_pix]

            # (0,0): 両方 pass (clean)
            mask_both_pass = (icl_pix == 0) & (ics_pix == 0)
            # (1,0): large のみ flag
            mask_large_only = (icl_pix > 0) & (ics_pix == 0)
            # (0,1): small のみ flag
            mask_small_only = (icl_pix == 0) & (ics_pix > 0)
            # (1,1): 両方 flag
            mask_both_flag = (icl_pix > 0) & (ics_pix > 0)

            n_both_pass = int(np.sum(mask_both_pass))
            n_large_only = int(np.sum(mask_large_only))
            n_small_only = int(np.sum(mask_small_only))
            n_both_flag = int(np.sum(mask_both_flag))

            # ファイル合計に加算
            file_sum_n_all += n_all
            file_sum_large += n_large_only
            file_sum_small += n_small_only
            file_sum_both_flag += n_both_flag
            file_sum_clean += n_both_pass

            # レート [count/s]
            rate_all = n_all / t_exp
            rate_both_pass = n_both_pass / t_exp  # clean
            rate_large_only = n_large_only / t_exp
            rate_small_only = n_small_only / t_exp
            rate_both_flag = n_both_flag / t_exp

            row = {
                "obs_id": obs_id,
                "object": obj,
                "event_file": event_file,
                "pixel": int(pix),
                "t_exp_s": t_exp,
                # 時間情報
                "time_start": time_start,
                "time_end": time_end,
                "time_span": time_span,
                # カウント数 & レート
                "n_all": n_all,
                "rate_all": rate_all,
                "n_both_pass": n_both_pass,
                "rate_both_pass": rate_both_pass,
                "n_large_only": n_large_only,
                "rate_large_only": rate_large_only,
                "n_small_only": n_small_only,
                "rate_small_only": rate_small_only,
                "n_both_flag": n_both_flag,
                "rate_both_flag": rate_both_flag,
            }
            rows.append(row)

        print(
            "    [INFO] file summary: n_all={:d}, clean(0,0)={:d}, "
            "large_only(>0,0)={:d}, small_only(0,>0)={:d}, both_flag(>0,>0)={:d}".format(
                file_sum_n_all,
                file_sum_clean,
                file_sum_large,
                file_sum_small,
                file_sum_both_flag,
            )
        )

    return rows


def make_summary_plot(df, outpath, output_csv):
    """
    サマリプロットを作成する。

    Figure 1 (outpath):
      (a) 6x6 pixel map (mean rate_all)
      (b) rate_all vs large/small で落ちたイベントレートの散布図
      (c) rate_all vs large/small の割合 (frac) の散布図
      (d) pixel ごとの large/small 割合の棒グラフ

    Figure 2 (<base>_pixmaps.png):
      2x2 サブプロットの 6x6 pixel map:
        (a) mean rate_all
        (b) mean rate_clean (ICLUSTERL=0 & ICLUSTERS=0)
        (c) mean rate_large_total
        (d) mean rate_small_total
    """
    if df.empty:
        print("[WARN] no data for plotting. skip summary plot.")
        return

    # --- メタ情報の集約 ---
    obs_ids = df["obs_id"].dropna().unique()
    objects = df["object"].dropna().unique()

    if len(obs_ids) == 1:
        obsid_str = obs_ids[0]
    else:
        obsid_str = f"multiple ({len(obs_ids)})"

    if len(objects) == 1:
        object_str = objects[0]
    else:
        object_str = f"multiple ({len(objects)})"

    # TIME start/end/span は、複数ファイルをまとめる可能性も考えて
    # 全 row の min/max を使う
    t_start = float(df["time_start"].min())
    t_end = float(df["time_end"].max())
    t_span = t_end - t_start
    t_exp_mean = float(df["t_exp_s"].mean())

    meta_text = (
        f"object: {object_str} | OBS_ID: {obsid_str} | "
        f"TIME start: {t_start:.3f} | TIME end: {t_end:.3f} | "
        f"ΔT (start→end): {t_span:.1f} s | "
        f"exposure (t_exp_s, mean over rows): {t_exp_mean:.1f} s"
    )

    # --- 必要な列を追加 ---
    df = df.copy()
    # clean = ICLUSTERL==0 & ICLUSTERS==0
    df["rate_clean"] = df["rate_both_pass"]

    df["rate_large_total"] = df["rate_large_only"] + df["rate_both_flag"]
    df["rate_small_total"] = df["rate_small_only"] + df["rate_both_flag"]
    df["frac_large"] = (df["n_large_only"] + df["n_both_flag"]) / df["n_all"]
    df["frac_small"] = (df["n_small_only"] + df["n_both_flag"]) / df["n_all"]

    print("[INFO] making summary plot ...")

    # pixel ごとの平均値（複数 OBS/ファイルが混在している場合に備えて mean を取る）
    group = df.groupby("pixel").agg(
        rate_all_mean=("rate_all", "mean"),
        rate_clean_mean=("rate_clean", "mean"),
        rate_large_total_mean=("rate_large_total", "mean"),
        rate_small_total_mean=("rate_small_total", "mean"),
        frac_large_mean=("frac_large", "mean"),
        frac_small_mean=("frac_small", "mean"),
    )

    # 6x6 pixel map 用の配列を作成
    pixmap_rate_all = build_pixel_map_from_group(group, "rate_all_mean")
    pixmap_rate_clean = build_pixel_map_from_group(group, "rate_clean_mean")
    pixmap_rate_large_total = build_pixel_map_from_group(group, "rate_large_total_mean")
    pixmap_rate_small_total = build_pixel_map_from_group(group, "rate_small_total_mean")

    # =========================================================
    # Figure 1: pixel map + scatter & frac & per-pixel bar
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_PPT)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (a) rate_all pixel map (平均レートの空間分布)
    xbins = np.linspace(0.5, 6.5, 7)
    ybins = np.linspace(0.5, 6.5, 7)

    pcm = ax1.pcolormesh(xbins, ybins, np.nan_to_num(pixmap_rate_all.T), cmap="plasma")
    cbar = plt.colorbar(pcm, ax=ax1, fraction=0.04, shrink=0.8, pad=0.02)
    cbar.set_label("mean rate_all [count/s]", fontsize=9)

    # pixel 番号と値を表示
    for detx_idx in range(6):
        for dety_idx in range(6):
            pix = PIXEL_FROM_DETX_DETY[detx_idx][dety_idx]
            val = pixmap_rate_all[detx_idx, dety_idx]
            ax1.text(
                detx_idx + 1 - 0.3,
                dety_idx + 1 + 0.3,
                str(pix),
                ha="center",
                va="center",
                color="0.9",
                fontsize=8,
            )
            if not np.isnan(val):
                ax1.text(
                    detx_idx + 1,
                    dety_idx + 1,
                    f"{val:.1e}",
                    ha="center",
                    va="center",
                    color="k",
                    fontsize=7,
                )

    ax1.set_xlabel("DETX")
    ax1.set_ylabel("DETY")
    ax1.set_aspect("equal")
    ax1.set_title("(a) mean rate_all per pixel")

    # (b) rate_all vs large/small で落ちたイベントレート
    ax2.scatter(df["rate_all"], df["rate_large_total"], s=15, alpha=0.7, label="large")
    ax2.scatter(df["rate_all"], df["rate_small_total"], s=15, alpha=0.7, label="small")
    ax2.set_xlabel("rate_all [count/s]")
    ax2.set_ylabel("LS cluster-affected rate [count/s]")
    ax2.set_title("(b) LS cluster-affected rate vs rate_all")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # (c) rate_all vs large/small の frac
    ax3.scatter(df["rate_all"], df["frac_large"], s=15, alpha=0.7, label="large frac")
    ax3.scatter(df["rate_all"], df["frac_small"], s=15, alpha=0.7, label="small frac")
    ax3.set_xlabel("rate_all [count/s]")
    ax3.set_ylabel("fraction of events")
    ax3.set_title("(c) Fraction of LS cluster-affected events")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # (d) pixel ごとの large/small 割合 (mean) の棒グラフ
    pixels = group.index.values
    x = np.arange(len(pixels))
    width = 0.4

    ax4.bar(
        x - width / 2,
        group["frac_large_mean"],
        width,
        alpha=0.7,
        label="large frac (mean)",
    )
    ax4.bar(
        x + width / 2,
        group["frac_small_mean"],
        width,
        alpha=0.7,
        label="small frac (mean)",
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(pixels, rotation=90)
    ax4.set_xlabel("pixel")
    ax4.set_ylabel("mean fraction")
    ax4.set_title("(d) Mean LS fraction per pixel")
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend()

    fig.tight_layout()

    # 図全体の下部にメタ情報を小さく表示
    fig.text(
        0.01,
        0.98,
        output_csv,
        fontsize=7,
        color='gray'
        )    
    fig.text(
        0.01,
        0.01,
        meta_text,
        fontsize=7,
        ha="left",
        va="bottom",
        color='gray'
    )

    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[INFO] summary plot saved to {outpath}")

    # =========================================================
    # Figure 2: 4 パターンの 6x6 pixel map を 2x2 でまとめた図
    # =========================================================
    base, _ = os.path.splitext(outpath)
    pixmap_out = base + "_pixmaps.png"

    fig2, axes2 = plt.subplots(2, 2, figsize=FIGSIZE_PPT)
    axA, axB, axC, axD = axes2.flatten()

    def _plot_pixmap(ax, arr, title, cbar_label):
        pcm = ax.pcolormesh(xbins, ybins, np.nan_to_num(arr.T), cmap="plasma")
        cbar_local = plt.colorbar(pcm, ax=ax, fraction=0.04, shrink=0.8, pad=0.02)
        cbar_local.set_label(cbar_label, fontsize=9)
        ax.set_aspect("equal")
        for detx_idx in range(6):
            for dety_idx in range(6):
                pix = PIXEL_FROM_DETX_DETY[detx_idx][dety_idx]
                val = arr[detx_idx, dety_idx]
                ax.text(
                    detx_idx + 1 - 0.3,
                    dety_idx + 1 + 0.3,
                    str(pix),
                    ha="center",
                    va="center",
                    color="0.9",
                    fontsize=8,
                )
                if not np.isnan(val):
                    ax.text(
                        detx_idx + 1,
                        dety_idx + 1,
                        f"{val:.1e}",
                        ha="center",
                        va="center",
                        color="k",
                        fontsize=7,
                    )
        ax.set_xlabel("DETX")
        ax.set_ylabel("DETY")
        ax.set_title(title)

    _plot_pixmap(axA, pixmap_rate_all, "(a) mean rate_all", "mean rate_all [count/s]")
    _plot_pixmap(
        axB,
        pixmap_rate_clean,
        "(b) mean rate_clean (ICLUSTERL=0 & ICLUSTERS=0)",
        "mean rate_clean [count/s]",
    )
    _plot_pixmap(
        axC,
        pixmap_rate_large_total,
        "(c) mean rate_large_total",
        "mean large-affected rate [count/s]",
    )
    _plot_pixmap(
        axD,
        pixmap_rate_small_total,
        "(d) mean rate_small_total",
        "mean small-affected rate [count/s]",
    )

    fig2.tight_layout()
    # こちらの図にも同じメタ情報を付与
    fig2.text(
        0.01,
        0.98,
        output_csv,
        fontsize=7,
        color='gray'
        )
    fig2.text(
        0.01,
        0.01,
        meta_text,
        fontsize=7,
        ha="left",
        va="bottom",
        color='gray'
    )
    fig2.savefig(pixmap_out, dpi=200)
    plt.close(fig2)
    print(f"[INFO] pixel-map figure saved to {pixmap_out}")


def main():
    args = parse_args()
    event_files = expand_event_files(args.events)

    if not event_files:
        print("[ERROR] no event files to process.")
        return

    # --- 出力 CSV 名の自動決定 ---
    if args.output_csv is None:
        args.output_csv = infer_output_csv(event_files)
        print(f"[INFO] output CSV not specified. auto-generated: {args.output_csv}")
    else:
        print(f"[INFO] output CSV specified by user: {args.output_csv}")

    print(f"[INFO] number of event files to process: {len(event_files)}")

    # 出力 CSV のヘッダ
    fieldnames = [
        "obs_id",
        "object",
        "event_file",
        "pixel",
        "t_exp_s",
        "time_start",
        "time_end",
        "time_span",
        "n_all",
        "rate_all",
        "n_both_pass",
        "rate_both_pass",
        "n_large_only",
        "rate_large_only",
        "n_small_only",
        "rate_small_only",
        "n_both_flag",
        "rate_both_flag",
    ]

    all_rows = []  # プロット用に全 rows を保持

    if os.path.exists(args.output_csv):
        print(f"[WARN] overwriting existing CSV: {args.output_csv}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        n_files_ok = 0
        n_rows_written = 0

        for fname in event_files:
            print(f"[INFO] processing {fname}")
            if not os.path.exists(fname):
                print(f"[WARN] file not found: {fname}")
                continue

            try:
                rows = process_one_file(
                    fname,
                    use_gti=args.use_gti,
                    min_events_per_pixel=args.min_events_per_pixel,
                    include_cal_pixel12=args.include_cal_pixel12,
                )
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")
                continue

            if not rows:
                print(f"[INFO] no valid pixels for {fname}")
                continue

            for row in rows:
                writer.writerow(row)
                all_rows.append(row)
                n_rows_written += 1

            n_files_ok += 1

    print("[INFO] finished processing.")
    print(f"       files processed successfully: {n_files_ok} / {len(event_files)}")
    print(f"       total pixel-rows written    : {n_rows_written}")

    # プロット作成
    if args.no_plot:
        print("[INFO] --no-plot specified. skip summary plot.")
        return

    if not all_rows:
        print("[WARN] no data rows collected. skip summary plot.")
        return

    df = pd.DataFrame(all_rows)
    if args.summary_plot:
        plot_path = args.summary_plot
    else:
        base, _ = os.path.splitext(args.output_csv)
        plot_path = base + "_summary.png"

    make_summary_plot(df, plot_path, args.output_csv)


if __name__ == "__main__":
    main()
