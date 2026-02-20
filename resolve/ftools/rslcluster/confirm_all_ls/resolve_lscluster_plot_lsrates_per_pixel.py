#!/usr/bin/env python
"""
resolve_lscluster_plot_lsrates_per_pixel.py

calc_lscluster_rates.py が出力した *_lsrates.csv を複数読み込み、
pixel ごとに

    rate_all
    rate_both_pass
    rate_large_only
    rate_small_only

を OBS_ID に対して可視化するスクリプト。

さらに、全 pixel・全 OBS をまとめた

    - 横軸 rate_all vs 縦軸 rate_both_pass / rate_large_only / rate_small_only
      の scatter plot を生成する。

    - 1パターン目: stdcut / no_stdcut を色分けした scatter
    - 2パターン目: stdcut / no_stdcut ごとにファイルを分け、
      その中で OBS_ID ごとに色を変え、OBS_ID + object 名を同色のテキストで表示
    - 3パターン目: 2 と同様だが pixel ごとにファイルを分けてプロット

を行う。

(注) 「stdcut」はファイル名に '_stdcut_' を含むかどうかで判定。
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"


# ============================================================
# 引数処理
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve small/large cluster lsrates CSV をまとめて読み込み、"
            "pixel ごと & rate_all vs その他の rate を可視化するスクリプト"
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help=(
            "入力 CSV ファイル（*_lsrates.csv）。"
            "シェルのワイルドカード（small_large_*lsrates*.csv など）も可。"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="lscluster_pixel_plots",
        help="出力図を保存するディレクトリ（デフォルト: lscluster_pixel_plots）",
    )
    parser.add_argument(
        "--pixels",
        default=None,
        help=(
            "プロット対象の pixel をカンマ区切りで指定（例: '0,1,2,35'）。"
            "省略時は CSV に含まれる全 pixel を対象とする。"
        ),
    )
    parser.add_argument(
        "--include-cal-pixel12",
        action="store_true",
        help="cal pixel (12) も含めてプロットする（デフォルトでは除外）",
    )
    parser.add_argument(
        "--scatter-xscale",
        choices=["linear", "log"],
        default="linear",
        help="scatter 図の x 軸スケール（linear / log, デフォルト: linear）",
    )
    parser.add_argument(
        "--scatter-yscale",
        choices=["linear", "log"],
        default="linear",
        help="scatter 図の y 軸スケール（linear / log, デフォルト: linear）",
    )
    return parser.parse_args()


def expand_csv_files(patterns):
    """ワイルドカードを展開して CSV ファイルのリストを返す。"""
    paths = []
    for pat in patterns:
        matched = glob.glob(pat)
        if not matched:
            # pat 自体がファイル名の場合も受け付ける
            if os.path.exists(pat):
                matched = [pat]
        paths.extend(sorted(matched))
    return paths


# ============================================================
# CSV 読み込み & 前処理
# ============================================================

def load_all_csv(csv_paths):
    """
    すべての CSV を読み込んで 1 つの DataFrame にまとめる。

    追加列:
      - csv_path : 元ファイルパス
      - csv_name : basename
      - stdcut_flag : "stdcut" or "no_stdcut"
    """
    dfs = []
    for path in csv_paths:
        print(f"[INFO] reading {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN]   failed to read {path}: {e}")
            continue

        base = os.path.basename(path)
        stdcut_flag = "stdcut" if "_stdcut_" in base else "no_stdcut"

        df["csv_path"] = path
        df["csv_name"] = base
        df["stdcut_flag"] = stdcut_flag

        dfs.append(df)

    if not dfs:
        raise RuntimeError("有効な CSV が 1 つも読み込めませんでした。")

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def parse_pixel_list(arg_pixels, df, include_cal_pixel12=False):
    """
    pixel のリストを決める。

    - arg_pixels が指定されていればそれを使う
    - 指定なしなら DataFrame 内のユニーク pixel を全部使う
    - include_cal_pixel12=False の場合は pixel=12 を除外
    """
    if arg_pixels is not None:
        pixels = []
        for token in arg_pixels.split(","):
            token = token.strip()
            if not token:
                continue
            pixels.append(int(token))
        pixels = sorted(set(pixels))
    else:
        pixels = sorted(int(p) for p in df["pixel"].unique())

    if not include_cal_pixel12:
        pixels = [p for p in pixels if p != 12]

    print(f"[INFO] number of pixels to plot: {len(pixels)}")
    print(f"[INFO] pixels: {pixels}")
    return pixels


# ============================================================
# pixel ごとの OBS_ID vs rate の図
# ============================================================

def make_pixel_plot(df_pix, pixel, outdir):
    """
    ある 1 pixel について、4 つの rate を OBS_ID に対してプロットする。
    df_pix: すでに pixel=特定値 で絞り込んだ DataFrame
    """
    df_pix = df_pix.copy()
    df_pix["obs_id"] = pd.to_numeric(df_pix["obs_id"], errors="coerce")

    rate_cols = [
        "rate_all",
        "rate_both_pass",
        "rate_large_only",
        "rate_small_only",
    ]

    # obs_id, stdcut_flag ごとに平均を取る
    grouped = (
        df_pix.groupby(["obs_id", "stdcut_flag"])[rate_cols]
        .mean()
        .reset_index()
    )

    if grouped.empty:
        print(f"[WARN] pixel {pixel}: no data, skip plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax11, ax12, ax21, ax22 = axes.flatten()

    ax_map = {
        "rate_all": ax11,
        "rate_both_pass": ax12,
        "rate_large_only": ax21,
        "rate_small_only": ax22,
    }
    ylabel_map = {
        "rate_all": "rate_all [count/s]",
        "rate_both_pass": "rate_both_pass [count/s]",
        "rate_large_only": "rate_large_only [count/s]",
        "rate_small_only": "rate_small_only [count/s]",
    }
    style_map = {
        "no_stdcut": dict(marker="o", linestyle="-", label="no_stdcut"),
        "stdcut": dict(marker="s", linestyle="--", label="stdcut"),
    }

    for stdcut_flag, sub in grouped.groupby("stdcut_flag"):
        sub = sub.sort_values("obs_id")
        x = sub["obs_id"].values
        style = style_map.get(
            stdcut_flag,
            dict(marker="o", linestyle="-", label=stdcut_flag),
        )

        for col in rate_cols:
            ax = ax_map[col]
            y = sub[col].values
            ax.plot(x, y, **style)

    for col, ax in ax_map.items():
        ax.set_ylabel(ylabel_map[col])
        ax.grid(alpha=0.3)

    ax21.set_xlabel("OBS_ID")
    ax22.set_xlabel("OBS_ID")

    ax11.set_title(f"pixel {pixel} : rate_all")
    ax12.set_title("rate_both_pass (clean)")
    ax21.set_title("rate_large_only")
    ax22.set_title("rate_small_only")

    ax11.legend(loc="best", fontsize=9)

    fig.suptitle(f"Resolve LS-cluster rates per pixel (pixel={pixel})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = os.path.join(outdir, f"pixel_{pixel:02d}_lsrates_vs_obsid.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[INFO] saved {outpath}")


# ============================================================
# 追加: rate_all vs 他の rate の scatter 図
# ============================================================

def _apply_log_mask(sub, x_col, y_col, xscale, yscale):
    """log スケール指定時に非正の値を落とすためのヘルパ。"""
    mask = np.ones(len(sub), dtype=bool)
    if xscale == "log":
        mask &= sub[x_col] > 0
    if yscale == "log":
        mask &= sub[y_col] > 0
    if not np.all(mask):
        n_drop = (~mask).sum()
        if n_drop > 0:
            print(f"[INFO] drop {n_drop} rows for log-scale (x={xscale}, y={yscale})")
    return sub[mask]


def scatter_rateall_vs_others_by_stdcut(df, outdir, xscale="linear", yscale="linear"):
    """
    全 pixel・全 OBS をまとめて、
    x=rate_all, y=rate_both_pass / rate_large_only / rate_small_only
    を stdcut_flag ごとに色分けした scatter plot を生成する。
    """
    rate_pairs = [
        ("rate_both_pass", "both_pass"),
        ("rate_large_only", "large_only"),
        ("rate_small_only", "small_only"),
    ]

    for y_col, tag in rate_pairs:
        fig, ax = plt.subplots(figsize=(7, 6))

        for stdcut_flag, sub in df.groupby("stdcut_flag"):
            sub = sub[["rate_all", y_col]].dropna()
            if sub.empty:
                continue
            sub = _apply_log_mask(sub, "rate_all", y_col, xscale, yscale)
            if sub.empty:
                continue
            ax.scatter(
                sub["rate_all"].values,
                sub[y_col].values,
                alpha=0.5,
                s=15,
                label=stdcut_flag,
            )

        ax.set_xlabel("rate_all [count/s]")
        ax.set_ylabel(f"{y_col} [count/s]")
        ax.set_title(f"rate_all vs {y_col} (color: stdcut_flag)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        fig.tight_layout()
        outpath = os.path.join(outdir, f"scatter_rateall_vs_{tag}_by_stdcut.png")
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"[INFO] saved {outpath}")


def scatter_rateall_vs_others_by_obsid_with_labels_global(
    df, outdir, xscale="linear", yscale="linear"
):
    """
    stdcut / no_stdcut ごとにファイルを分け、
    その中で OBS_ID ごとに色を変えて scatter、さらに

        OBS_ID + object 名

    を同じ色のテキストで近傍に書き込む（全 pixel まとめ）。
    """
    rate_pairs = [
        ("rate_both_pass", "both_pass"),
        ("rate_large_only", "large_only"),
        ("rate_small_only", "small_only"),
    ]

    for stdcut_flag, df_flag in df.groupby("stdcut_flag"):

        # OBS_ID x object の組み合わせごとにグループ化
        groups = list(df_flag.groupby(["obs_id", "object"]))

        if not groups:
            print(f"[WARN] stdcut_flag={stdcut_flag}: no data, skip")
            continue

        n_grp = len(groups)
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / max(1, n_grp - 1)) for i in range(n_grp)]

        for y_col, tag in rate_pairs:
            fig, ax = plt.subplots(figsize=(7.5, 6))

            for idx, ((obs_id, obj), sub) in enumerate(groups):
                sub = sub[["rate_all", y_col]].dropna()
                if sub.empty:
                    continue

                sub = _apply_log_mask(sub, "rate_all", y_col, xscale, yscale)
                if sub.empty:
                    continue

                col = colors[idx]
                ax.scatter(
                    sub["rate_all"].values,
                    sub[y_col].values,
                    color=col,
                    alpha=0.7,
                    s=20,
                )

                # クラスタの代表位置として median を取ってラベルを置く
                x_med = sub["rate_all"].median()
                y_med = sub[y_col].median()

                # object 名も含めたラベル（例: "161000 MCG-6-30-15"）
                label_text = f"{int(obs_id)} {obj}"
                ax.text(
                    x_med,
                    y_med,
                    label_text,
                    color=col,
                    fontsize=8,
                    alpha=0.9,
                )

            ax.set_xlabel("rate_all [count/s]")
            ax.set_ylabel(f"{y_col} [count/s]")
            ax.set_title(
                f"rate_all vs {y_col} "
                f"(stdcut_flag={stdcut_flag}, color: OBS_ID+object, all pixels)"
            )
            ax.grid(alpha=0.3)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            fig.tight_layout()
            outname = (
                f"scatter_rateall_vs_{tag}_stdcut-{stdcut_flag}_by_obsid.png"
            )
            outpath = os.path.join(outdir, outname)
            fig.savefig(outpath, dpi=200)
            plt.close(fig)
            print(f"[INFO] saved {outpath}")


def scatter_rateall_vs_others_by_obsid_with_labels_per_pixel(
    df, outdir, pixels, xscale="linear", yscale="linear"
):
    """
    pixel ごとにファイルを分けて、stdcut / no_stdcut ごとに、
    OBS_ID+object ごとに色を変えた scatter + ラベルを生成する。
    """
    rate_pairs = [
        ("rate_both_pass", "both_pass"),
        ("rate_large_only", "large_only"),
        ("rate_small_only", "small_only"),
    ]

    for pix in pixels:
        df_pix = df[df["pixel"] == pix]
        if df_pix.empty:
            print(f"[INFO] pixel {pix}: no data for scatter-by-obsid, skip")
            continue

        for stdcut_flag, df_flag in df_pix.groupby("stdcut_flag"):

            groups = list(df_flag.groupby(["obs_id", "object"]))
            if not groups:
                print(
                    f"[WARN] pixel={pix}, stdcut_flag={stdcut_flag}: no data, skip"
                )
                continue

            n_grp = len(groups)
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i / max(1, n_grp - 1)) for i in range(n_grp)]

            for y_col, tag in rate_pairs:
                fig, ax = plt.subplots(figsize=(7.5, 6))

                for idx, ((obs_id, obj), sub) in enumerate(groups):
                    sub = sub[["rate_all", y_col]].dropna()
                    if sub.empty:
                        continue

                    sub = _apply_log_mask(sub, "rate_all", y_col, xscale, yscale)
                    if sub.empty:
                        continue

                    col = colors[idx]
                    ax.scatter(
                        sub["rate_all"].values,
                        sub[y_col].values,
                        color=col,
                        alpha=0.7,
                        s=20,
                    )

                    x_med = sub["rate_all"].median()
                    y_med = sub[y_col].median()

                    label_text = f"{int(obs_id)} {obj}"
                    ax.text(
                        x_med,
                        y_med,
                        label_text,
                        color=col,
                        fontsize=8,
                        alpha=0.9,
                    )

                ax.set_xlabel("rate_all [count/s]")
                ax.set_ylabel(f"{y_col} [count/s]")
                ax.set_title(
                    "rate_all vs {y} (pixel={pix}, stdcut_flag={flag}, "
                    "color: OBS_ID+object)".format(
                        y=y_col, pix=pix, flag=stdcut_flag
                    )
                )
                ax.grid(alpha=0.3)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)

                fig.tight_layout()
                outname = (
                    f"scatter_rateall_vs_{tag}_stdcut-{stdcut_flag}"
                    f"_by_obsid_pixel-{pix:02d}.png"
                )
                outpath = os.path.join(outdir, outname)
                fig.savefig(outpath, dpi=200)
                plt.close(fig)
                print(f"[INFO] saved {outpath}")


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()

    csv_paths = expand_csv_files(args.csv_files)
    if not csv_paths:
        raise RuntimeError("入力 CSV が見つかりませんでした。パターンを確認してください。")

    print(f"[INFO] number of CSV files: {len(csv_paths)}")
    for p in csv_paths:
        print(f"       {p}")

    os.makedirs(args.outdir, exist_ok=True)

    df_all = load_all_csv(csv_paths)

    # 対象 pixel
    pixels = parse_pixel_list(
        args.pixels, df_all, include_cal_pixel12=args.include_cal_pixel12
    )

    # 1) pixel ごとの OBS_ID vs 各 rate
    for pix in pixels:
        df_pix = df_all[df_all["pixel"] == pix]
        if df_pix.empty:
            print(f"[INFO] pixel {pix}: no rows, skip")
            continue
        make_pixel_plot(df_pix, pix, args.outdir)

    # 2) 全体で rate_all vs 他の rate の scatter（stdcut 色分け）
    scatter_rateall_vs_others_by_stdcut(
        df_all, args.outdir, xscale=args.scatter_xscale, yscale=args.scatter_yscale
    )

    # 3) stdcut ごとに分け、OBS_ID+object ごとに色 + ラベル付き scatter（全 pixel）
    scatter_rateall_vs_others_by_obsid_with_labels_global(
        df_all, args.outdir, xscale=args.scatter_xscale, yscale=args.scatter_yscale
    )

    # 4) pixel ごとにファイルを分けた scatter
    scatter_rateall_vs_others_by_obsid_with_labels_per_pixel(
        df_all,
        args.outdir,
        pixels=pixels,
        xscale=args.scatter_xscale,
        yscale=args.scatter_yscale,
    )

    print("[INFO] done.")


if __name__ == "__main__":
    main()