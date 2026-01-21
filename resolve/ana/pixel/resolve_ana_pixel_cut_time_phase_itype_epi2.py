#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
resolve_ana_pixel_cut_time_phase_itype_epi2.py

FITS イベントファイルから EPI2 のヒストグラムを作る簡単なスクリプト。

[1] TIME カラムを使って、観測開始からの相対時刻で
    timecutstart ～ timecutstop の範囲だけを選ぶ
[2] 軌道周期 Porb [s] と基準時刻 T0 [MJD] から orbital phase を自前で計算し、
    phasestart ～ phasestop の範囲だけを選ぶ
[3] ITYPE (0〜4) でイベントを選ぶ
[4] これらの条件をすべて満たすイベントだけで EPI2 のヒストグラムを作成

使い方（例）:
  python resolve_ana_pixel_cut_time_phase_itype_epi2.py \
      xa300065010rsl_p0px1000_cl.evt \
      --timecutstart 1000 --timecutstop 2000 \
      --porb 295000.0 --t0 50100.12345 \
      --phasestart 0.4 --phasestop 0.6 \
      --itype 0 1 2 \
      --emin 3000 --emax 9000 --de 10

学生向けポイント:
- 「相対時刻」は TIME - TIME.min() として計算している
- orbital phase は
    MJD = MJDREF + TIME/86400
    phase = ((MJD - T0)/(Porb/86400)) % 1
  で計算している
- ITYPE は 0〜4 の整数を一つ以上指定できる（例: --itype 0 1 4）
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def parse_arguments():
    """コマンドライン引数を定義・解釈する関数"""
    parser = argparse.ArgumentParser(
        description="TIME・orbital phase・ITYPE でイベントをカットして "
                    "EPI2 のヒストグラムを作るスクリプト"
    )

    # 入力 FITS イベントファイル
    parser.add_argument(
        "filename",
        help="入力する FITS イベントファイル名（例: xa300065010rsl_p0px1000_cl.evt）"
    )

    # [1] 相対時刻によるカット
    parser.add_argument(
        "--timecutstart",
        type=float,
        default=None,
        help="観測開始からの相対時刻 [s] の下限 (この値以上を採用)。指定しなければ制限なし。"
    )
    parser.add_argument(
        "--timecutstop",
        type=float,
        default=None,
        help="観測開始からの相対時刻 [s] の上限 (この値以下を採用)。指定しなければ制限なし。"
    )

    # [2] 軌道位相のカットに必要なパラメータ
    parser.add_argument(
        "--porb",
        type=float,
        default=None,
        help="軌道周期 Porb [秒]。phase カットを行う場合は必須。"
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=None,
        help="基準時刻 T0 [MJD]。phase カットを行う場合は必須。"
    )
    parser.add_argument(
        "--phasestart",
        type=float,
        default=None,
        help="軌道位相 (0–1) の下限 (この値以上を採用)。指定しなければ制限なし。"
    )
    parser.add_argument(
        "--phasestop",
        type=float,
        default=None,
        help="軌道位相 (0–1) の上限 (この値未満を採用)。指定しなければ制限なし。"
    )

    # [3] ITYPE によるカット（0〜4 の整数を一つ以上）
    parser.add_argument(
        "--itype",
        type=int,
        choices=range(5),
        nargs="+",
        default=None,
        help=("選択する ITYPE の値（0〜4 の整数）。"
              "複数指定可（例: --itype 0 1 4）。指定しなければ ITYPE でカットしない。")
    )

    # EPI2 ヒストグラムの設定
    parser.add_argument(
        "--emin",
        type=float,
        default=0.0,
        help="ヒストグラムの最小 EPI2 値（デフォルト: 0）"
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=20000.0,
        help="ヒストグラムの最大 EPI2 値（デフォルト: 20000）"
    )
    parser.add_argument(
        "--de",
        type=float,
        default=10.0,
        help="ビン幅 ΔE（EPI2 の単位そのまま。デフォルト: 10）"
    )

    # 出力ファイル名
    parser.add_argument(
        "--outpng",
        type=str,
        default=None,
        help="出力 PNG ファイル名（デフォルト: 自動生成）"
    )

    # 画面にプロットを表示するかどうか
    parser.add_argument(
        "--show",
        action="store_true",
        help="指定すると plt.show() で図を表示する（Jupyter などで確認用）"
    )

    return parser.parse_args()


def get_mjdref(header):
    """
    FITS ヘッダから MJDREF を取得する。

    - MJDREF があればそれを使う
    - なければ MJDREFI + MJDREFF を使う
    """
    if "MJDREF" in header:
        return header["MJDREF"]

    mjdrefi = header.get("MJDREFI", None)
    mjdreff = header.get("MJDREFF", None)

    if mjdrefi is None or mjdreff is None:
        print("[ERROR] MJDREF (または MJDREFI/MJDREFF) がヘッダにありません。")
        sys.exit(1)

    return float(mjdrefi) + float(mjdreff)


def compute_orbital_phase(time, header, porb_sec, t0_mjd):
    """
    TIME [s] と FITS ヘッダから orbital phase を計算する関数。

    手順:
      MJDREF = get_mjdref(header)
      MJD = MJDREF + TIME / 86400
      Porb_day = Porb_sec / 86400
      phase = ((MJD - T0) / Porb_day) % 1.0
    """
    mjdref = get_mjdref(header)
    time_day = time / 86400.0
    mjd = mjdref + time_day

    porb_day = porb_sec / 86400.0
    if porb_day <= 0:
        print("[ERROR] Porb は正の値で指定してください。")
        sys.exit(1)

    phase_raw = (mjd - t0_mjd) / porb_day
    phase = np.mod(phase_raw, 1.0)  # 0〜1 に折りたたむ
    return phase


def main():
    args = parse_arguments()

    # FITS ファイルを開く（拡張 1 = EVENTS テーブルを想定）
    try:
        hdul = fits.open(args.filename)
    except FileNotFoundError:
        print(f"[ERROR] ファイルが見つかりません: {args.filename}")
        sys.exit(1)

    if len(hdul) < 2:
        print("[ERROR] 拡張 1 (EVENTS テーブル) が存在しないようです。")
        sys.exit(1)

    hdu = hdul[1]
    header = hdu.header
    data = hdu.data

    # TIME, EPI2 カラムを取得
    try:
        time = data["TIME"]
    except KeyError:
        print("[ERROR] 'TIME' カラムが見つかりません。")
        sys.exit(1)

    try:
        epi2 = data["EPI2"]
    except KeyError:
        print("[ERROR] 'EPI2' カラムが見つかりません。")
        sys.exit(1)

    # ITYPE カラム（存在しない場合はエラー）
    try:
        itype = data["ITYPE"]
    except KeyError:
        print("[ERROR] 'ITYPE' カラムが見つかりません。Resolve イベントファイルか確認してください。")
        sys.exit(1)

    # ---------------------------
    # [1] 相対時刻によるカット
    # ---------------------------
    # 観測開始時刻 = TIME の最小値とする
    t0_obs = time.min()
    time_rel = time - t0_obs  # 観測開始からの相対時刻 [s]

    # まず全部 True のマスクを用意して、条件ごとに & で絞っていく
    mask = np.ones(len(time), dtype=bool)

    # timecutstart / timecutstop のチェック
    if args.timecutstart is not None and args.timecutstop is not None:
        if args.timecutstart >= args.timecutstop:
            print("[ERROR] timecutstart < timecutstop となるように指定してください。")
            sys.exit(1)

    if args.timecutstart is not None:
        mask &= (time_rel >= args.timecutstart)
    if args.timecutstop is not None:
        mask &= (time_rel <= args.timecutstop)

    # ----------------------------
    # [2] 軌道位相によるカット
    # ----------------------------
    if args.phasestart is not None or args.phasestop is not None:
        # Porb と T0 が一緒に指定されているかチェック
        if args.porb is None or args.t0 is None:
            print("[ERROR] phase カットを行う場合は --porb と --t0 を両方指定してください。")
            sys.exit(1)

        # phasestart / phasestop のチェック
        if (args.phasestart is not None and
                (args.phasestart < 0.0 or args.phasestart > 1.0)):
            print("[ERROR] phasestart は 0〜1 の範囲で指定してください。")
            sys.exit(1)

        if (args.phasestop is not None and
                (args.phasestop < 0.0 or args.phasestop > 1.0)):
            print("[ERROR] phasestop は 0〜1 の範囲で指定してください。")
            sys.exit(1)

        if (args.phasestart is not None and args.phasestop is not None and
                args.phasestart >= args.phasestop):
            print("[ERROR] phasestart < phasestop となるように指定してください。")
            sys.exit(1)

        # orbital phase を自前で計算
        phase = compute_orbital_phase(time, header, args.porb, args.t0)

        # 位相の範囲でマスクを更新
        if args.phasestart is not None:
            mask &= (phase >= args.phasestart)
        if args.phasestop is not None:
            mask &= (phase < args.phasestop)

    # ----------------------------
    # [3] ITYPE によるカット
    # ----------------------------
    if args.itype is not None:
        # args.itype は [0, 1, 4] のようなリスト
        itype_list = np.array(args.itype, dtype=int)
        mask &= np.isin(itype, itype_list)
        print(f"[INFO] ITYPE での選択: {itype_list.tolist()}")

    # ----------------------------
    # [4] マスクをかけて EPI2 を取得
    # ----------------------------
    selected_epi2 = epi2[mask]
    selected_time_rel = time_rel[mask]  # 必要ならデバッグに使える

    print(f"[INFO] 全イベント数      : {len(time)}")
    print(f"[INFO] 選択されたイベント: {len(selected_epi2)}")

    if len(selected_epi2) == 0:
        print("[WARNING] 条件を満たすイベントが 0 です。ヒストグラムを作れません。")
        sys.exit(0)

    # ----------------------------
    # EPI2 ヒストグラムの作成
    # ----------------------------
    emin = args.emin
    emax = args.emax
    de = args.de

    if emin >= emax:
        print("[ERROR] emin < emax となるように指定してください。")
        sys.exit(1)

    if de <= 0:
        print("[ERROR] de は正の数で指定してください。")
        sys.exit(1)

    nbins_float = (emax - emin) / de
    nbins = int(nbins_float)

    if nbins <= 0:
        print("[ERROR] (emax - emin) / de が 1 未満になっています。設定を確認してください。")
        sys.exit(1)

    print(f"[INFO] ヒストグラム設定: emin={emin}, emax={emax}, de={de}, nbins={nbins}")

    hist, bin_edges = np.histogram(selected_epi2, bins=nbins, range=(emin, emax))

    # 図の作成
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.step(bin_centers, hist, where="mid")
    ax.set_xlabel("EPI2")
    ax.set_ylabel("Counts")
    ax.set_title("EPI2 Histogram with Time / Phase / ITYPE Cuts")

    ax.grid(alpha=0.3)

    # 選択範囲の情報を図に追記（学生向けデバッグ用）
    info_lines = []
    info_lines.append(f"Rel time: "
                      f"{args.timecutstart if args.timecutstart is not None else 'min'}"
                      f"–"
                      f"{args.timecutstop if args.timecutstop is not None else 'max'} [s]")
    if args.porb is not None and args.t0 is not None:
        info_lines.append(f"Porb={args.porb:.1f} s, T0={args.t0:.5f} MJD")
    info_lines.append(f"Phase: "
                      f"{args.phasestart if args.phasestart is not None else 'min'}"
                      f"–"
                      f"{args.phasestop if args.phasestop is not None else 'max'}")
    if args.itype is not None:
        info_lines.append(f"ITYPE: {', '.join(str(v) for v in args.itype)}")
    else:
        info_lines.append("ITYPE: all")

    info_text = "\n".join(info_lines)
    ax.text(
        0.98, 0.95, info_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.tight_layout()

    # 出力ファイル名を決める
    if args.outpng is not None:
        out_png = args.outpng
    else:
        # 元ファイル名から拡張子を取って簡単な名前を作る
        base = args.filename.replace(".fits", "").replace(".evt", "").replace(".gz", "")
        out_png = f"{base}_epi2_hist.png"

    plt.savefig(out_png)
    print(f"[INFO] ヒストグラムを {out_png} に保存しました。")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
