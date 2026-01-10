#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys, os 
from matplotlib.cm import get_cmap
import csv
from scipy.optimize import minimize_scalar

# Type Information
g_itypename = [0, 1, 2, 3, 4]
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]

# 40色の色リストを複数のカラーマップから作成
cmap1 = get_cmap('tab20')
cmap2 = get_cmap('tab20b')
cmap3 = get_cmap('tab20c')
g_pixcolors = [cmap1(i/20) for i in range(20)] + [cmap2(i/20) for i in range(10)] + [cmap3(i/20) for i in range(10)]

# コマンドライン引数を解析する関数
def parse_args():
    """
    コマンドライン引数を解析する。
    """
    parser = argparse.ArgumentParser(description='',
                                     usage='python resolve_ana_pixel_mklc_branch.py f.list -y 0 -p 0 -g')

    parser = argparse.ArgumentParser(
      description='FITSファイルから光度曲線をプロットするスクリプト。',
      epilog='''
        Example 1) 中心４ピクセルの Hpだけ(-y0)、GTIを使う(-u)場合の例:
        resolve_ana_pixel_mklc_branch.py f.list -l -u -y 0 -p 0,17,18,35 -t 256 -o p0_17_18_35 -s 
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('filelist', help='処理するFITSファイルリストの名前。')
    parser.add_argument('--timebinsize', '-t', type=float, help='光度曲線の時間ビンサイズ', default=100.0)
    parser.add_argument('--itypenames', '-y', type=str, help='カンマ区切りのitypeリスト', default='0,1,2,3,4')
    parser.add_argument('--plotpixels', '-p', type=str, help='プロットするピクセルのカンマ区切りリスト', default=','.join(map(str, range(36))))
    parser.add_argument('--output', '-o', type=str, help='出力ファイル名のプレフィックス', default='mklc')
    parser.add_argument('--plot_lightcurve', '-l', action='store_true', help='光度曲線をプロットする')
    parser.add_argument('--plot_rate_vs_grade', '-g', action='store_true', help='rate_vs_gradeをプロットする')
    parser.add_argument('--gtiuse', '-u', action='store_true', help='GTIを使用して光度曲線を生成する')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか。defaultはplotしない。')
    parser.add_argument('--nonstop', '-n', action='store_true', help='GTIが同時刻の部分で区切らない。')
    parser.add_argument('--lcthresh', '-e', type=float, default=0.8, help='fractional exposure の閾値')

    # added for Pereus 
    parser.add_argument('--rate_max_ingratio', '-rmax', type=float, default=10.0, help='max rate for bratio')
    parser.add_argument('--yscale_ingratio', '-yscaleing', type=str, default="linear", help='log or linear for bratio')

    args = parser.parse_args()

    # 引数の確認をプリント
    print("----- 設定 -----")
    print("  plot_lightcurve    = ", args.plot_lightcurve)
    print("  plot_rate_vs_grade = ", args.plot_rate_vs_grade)
    print("  gtiuse             = ", args.gtiuse)
    print("  show               = ", args.show)
    print("-----------------")

    # どちらのプロットフラグも設定されていない場合はエラーを表示
    if (not args.plot_lightcurve) and (not args.plot_rate_vs_grade):
        parser.error("少なくとも--plot_lightcurveまたは--plot_rate_vs_gradeのどちらかを指定してください。")
    
    return args

# matplotlibのプロットパラメータを設定する関数
def setup_plot():
    """
    matplotlibのプロットパラメータを設定する。
    """
    params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12}
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update(params)

# 分岐比を計算する関数
def calc_branchingratios(rate, debug=False):
    """
    分岐比を計算する。武田さんSPIEに準じるが検証が必要な式。
    """
    clock = 12500 # 12.5kHz = 80us       
    dtHR = (1.0 / clock) * 874 
    dtMR = (1.0 / clock) * 219   
    dtLR = (1.0 / clock) * 15   

    exp_hr = np.exp(-1.0 * dtHR * rate)
    exp_mr = np.exp(-1.0 * dtMR * rate)             
    exp_lr = np.exp(-1.0 * dtLR * rate)             

    Hp_bl_cor = exp_hr * (exp_hr + 1 - exp_lr)
    Mp_bl = exp_hr * (exp_mr - exp_hr)
    Ms_bl = exp_mr * (exp_mr - exp_hr)
    Lp_bl_cor = exp_hr * (exp_lr - exp_mr)
    Ls_bl_cor = (1 - exp_mr) * (1 + exp_mr - exp_hr) - exp_hr * (1 - exp_lr)

    checksum = Hp_bl_cor + Mp_bl + Ms_bl + Lp_bl_cor + Ls_bl_cor

    Hp_bl_cor /= checksum
    Mp_bl /= checksum
    Ms_bl /= checksum
    Lp_bl_cor /= checksum
    Ls_bl_cor /= checksum

    checksum_confirm = Hp_bl_cor + Mp_bl + Ms_bl + Lp_bl_cor + Ls_bl_cor

    if debug:
        print("checksum", checksum)
        print("checksum_confirm", checksum_confirm)
    return Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, Ls_bl_cor

# FITSファイルからデータを処理する関数
def process_data(fname, ref_time, TRIGTIME_FLAG=False, AC_FLAG=False):
    """
    FITSファイルからデータを処理する。
    """
    print(f"{fname}のデータを処理中...")
    data = fits.open(fname)[1].data
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]

    if len(time) == 0:
        print("エラー: データが空です", time)
        sys.exit()

    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]
    pha, rise_time, deriv_max, pixel = data["PHA"], data["RISE_TIME"], data["DERIV_MAX"], data["PIXEL"]

    # 時間順にデータをソート
    sortid = np.argsort(time)
    time = time[sortid]
    pha = pha[sortid]
    itype = itype[sortid]
    rise_time = rise_time[sortid]
    deriv_max = deriv_max[sortid]
    pixel = pixel[sortid]
    dtime = np.array([ref_time.datetime + datetime.timedelta(seconds=float(t)) for t in time])

    print(f"データの範囲: {dtime[0]} から {dtime[-1]} まで")

    dt = np.diff(time)
    time, dtime, pha, itype, rise_time, deriv_max, pixel = time[:-1], dtime[:-1], pha[:-1], itype[:-1], rise_time[:-1], deriv_max[:-1], pixel[:-1]

    return dt, time, dtime, pha, itype, rise_time, deriv_max, pixel

# 重複期間を見つける関数
def find_overlaps(start1, stop1, start2, stop2, nonstop=False):
    """
    重複期間を見つける。
    """
    # すべての開始イベントと終了イベントを1つのリストにまとめる
    events = [(x, 1, i) for i, x in enumerate(start1)] + \
             [(x, 2, i) for i, x in enumerate(stop1)] +  \
             [(x, 3, i) for i, x in enumerate(start2)] + \
             [(x, 4, i) for i, x in enumerate(stop2)]    

    # イベントを時間 (x[0]) とイベントの優先順位 (x[1]) に基づいてソートする         
    if nonstop:
        # 同時刻なら start の方を優先する
        events.sort(key=lambda x: (x[0], {1: 1, 2: 2, 3: 3, 4: 4}[x[1]]))
    else:
        # 同時刻なら stop の方を優先する
        events.sort(key=lambda x: (x[0], {2: 1, 4: 2, 1: 3, 3: 4}[x[1]]))

    # 各セットは現在アクティブな時間区間を追跡するために使用される
    inside1 = set()  # start1-stop1 間のアクティブなインデックス
    inside2 = set()  # start2-stop2 間のアクティブなインデックス

    # 重複期間の開始と終了を記録するリスト
    overlaps_start = []
    overlaps_stop = []
    current_overlap_start = None  # 現在の重複期間の開始時間を記録する変数

    # ソートされたイベントリストを順に処理する
    for x, kind, idx in events:
        if kind == 1:
            inside1.add(idx)  # start1 イベントの場合、インデックスを inside1 に追加
        elif kind == 2:
            inside1.remove(idx)  # stop1 イベントの場合、インデックスを inside1 から削除
        elif kind == 3:
            inside2.add(idx)  # start2 イベントの場合、インデックスを inside2 に追加
        else:  # kind == 4
            inside2.remove(idx)  # stop2 イベントの場合、インデックスを inside2 から削除

        # 両方のセットに要素が含まれている場合、重複が始まったと見なす
        if inside1 and inside2 and current_overlap_start is None:
            current_overlap_start = x  # 重複の開始時間を記録
        # どちらか一方のセットが空の場合、重複が終了したと見なす
        elif (not inside1 or not inside2) and current_overlap_start is not None:
            overlaps_start.append(current_overlap_start)  # 重複の開始時間をリストに追加
            overlaps_stop.append(x)  # 重複の終了時間をリストに追加
            current_overlap_start = None  # 現在の重複開始時間をリセット

    return np.array(overlaps_start), np.array(overlaps_stop)  # 重複期間の開始時間と終了時間のリストを返す

# 固定時間ビンで光度曲線を計算する関数
def fast_lc(tstart, tstop, binsize, x, debug=True, gtiuse = False, overlaps_start=None, overlaps_stop=None, lcthresh = 0.8):
    """
    固定時間ビンで光度曲線を計算する。
    """

    if not gtiuse:
        times = np.arange(tstart, tstop, binsize)[:-1]
        n_bins = len(times)

        x_lc, y_lc = np.zeros(n_bins), np.zeros(n_bins)
        x_err = np.full(n_bins, 0.5 * binsize)
        y_err = np.zeros(n_bins)

        x = np.sort(x)

        for i, t in enumerate(times):
            start, end = np.searchsorted(x, [t, t + binsize])
            count = end - start

            x_lc[i] = t + 0.5 * binsize
            y_lc[i] = count / binsize
            y_err[i] = np.sqrt(count) / binsize

        if debug:
            print(f"length = {len(x_lc)}")
    else:
        n_bins = len(overlaps_start)
        x_lc, y_lc = np.zeros(n_bins), np.zeros(n_bins)
        x_err = np.full(n_bins, 0.5 * (overlaps_stop - overlaps_start))
        y_err = np.zeros(n_bins)
        x = np.sort(x)

        for i, (tpre, tafter) in enumerate(zip(overlaps_start,overlaps_stop)):
            start, end = np.searchsorted(x, [tpre, tafter])
            count = end - start
            real_timebinsize = tafter - tpre
            fractional_exposure = real_timebinsize/binsize
            if fractional_exposure > lcthresh:
                x_lc[i] = tpre + 0.5 * real_timebinsize
                y_lc[i] = count / real_timebinsize
                y_err[i] = np.sqrt(count) / real_timebinsize
            else:
                if fractional_exposure >0:
                    print(f"..... not filled due to fractional_exposure is {fractional_exposure} lower than {lcthresh}.")
        if debug:
            print(f"length = {len(x_lc)}")

    return x_lc, x_err, y_lc, y_err

# GTIを使用してデータを処理する関数
def process_data_wgti(fname, ref_time, timebinsize, debug=False, nonstop = False):
    """
    GTIを使用してFITSファイルからデータを処理する。
    """

    outftag = fname.replace(".evt", "").replace(".gz", "")
    head = fits.open(fname)[1].header
    obsid = head["OBS_ID"]
    oname = head["OBJECT"]
    dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, ref_time)

    gtistart = fits.open(fname)[2].data["START"]
    gtistop = fits.open(fname)[2].data["STOP"]

    tstart = time[0] 
    tstop = time[-1]
    times = np.arange(tstart, tstop, timebinsize)[:-1]
    n_bins = len(times)
    bintimes_start = times[:-1]
    bintimes_stop = times[1:]

    overlaps_start, overlaps_stop = find_overlaps(bintimes_start, bintimes_stop, gtistart, gtistop, nonstop = nonstop)

    if debug:
        # 新しい図としてfig_debug, ax_debugを作成
        fig_debug, ax_debug = plt.subplots(1, 1, figsize=(8, 4))

        # GTIのプロット
        for i, (x1, x2) in enumerate(zip(gtistart, gtistop)):
            ax_debug.hlines(y=2, xmin=x1, xmax=x2, lw=10 + 10*(i % 8), color="red", label='GTI' if x1 == gtistart[0] else "", alpha=0.8)

        # Time Binのプロット
        for i, (x1, x2) in enumerate(zip(bintimes_start, bintimes_stop)):
            ax_debug.hlines(y=1.5, xmin=x1, xmax=x2, lw=10 + 10*(i % 8), color="blue", label='bintime' if x1 == bintimes_start[0] else "", alpha=0.8)

        # Overlapのプロット
        for i, (x1, x2) in enumerate(zip(overlaps_start, overlaps_stop)):
            print(i, x1, x2)
            ax_debug.hlines(y=1, xmin=x1, xmax=x2, lw=10 + 10*(i % 8), color="green", label='overlap' if x1 == overlaps_start[0] else "", alpha=0.8)

        # 凡例の設定
        ax_debug.legend()

        # 軸とその他の設定
        ax_debug.set_ylim(0.5, 2.5)
        ax_debug.set_yticks([1, 1.5, 2])
        ax_debug.set_yticklabels(['GTI', 'Time Bin', 'Overlap'])
        ax_debug.set_xlabel('Time')
        ax_debug.set_title('(debug) 重複期間の可視化')
        ax_debug.grid(alpha=0.7)

        # 図の表示とクローズ
        plt.show()
        plt.close(fig_debug)

        print("警告: 処理をここで停止します")
        print("以下の図は、pltのグローバル変数の競合のため作成されません。")
        sys.exit()

    return dt, time, dtime, pha, itype, rise_time, deriv_max, pixel, np.array(overlaps_start), np.array(overlaps_stop)

# 光度曲線をプロットする関数
def plot_lightcurve(event_list, plotpixels, itypenames, timebinsize, output, ref_time, \
                       gtiuse = False, debug=False, show = False, nonstop = False, lcthresh = 0.8):
    """
    イベントリストから光度曲線をプロットする。
    """
    colors = plt.cm.get_cmap('tab10', len(plotpixels)).colors
    ishape = [".", "s", "D", "*", "x"]

    type_colors = plt.cm.Set1(np.linspace(0, 1, 9))

    for fname in event_list:
        outftag = fname.replace(".evt", "").replace(".gz", "")
        head = fits.open(fname)[1].header
        obsid = head["OBS_ID"]
        oname = head["OBJECT"]

        if gtiuse:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel, overlaps_start, overlaps_stop = process_data_wgti(fname, ref_time, timebinsize, debug=debug, nonstop = nonstop)
        else:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, ref_time)

        for j, pix in enumerate(plotpixels):

            fig, (ax, branching_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_ylabel(f"Counts/s (binsize = {timebinsize}s)")
            ax.grid(alpha=0.2)

            branching_ax.set_xscale("linear")
            branching_ax.set_yscale("linear")
            branching_ax.set_ylabel("Branching Ratios")
            branching_ax.set_xlabel("TIME")
            branching_ax.grid(alpha=0.2)

            for k, itype_ in enumerate(itypenames):
                type_color = type_colors[k % len(type_colors)]

                print(f"ピクセル{pix}とタイプ{itype_}のデータを処理中 (obsid {obsid})")

                pixel_cutid = np.where((pixel == pix))[0]
                pixel_time_ = time[pixel_cutid]            

                if len(pixel_time_) == 0:
                    print(f"エラー: ピクセル{pix}とタイプ{itype_}のデータが空です。")
                    continue

                cutid = np.where((itype == itype_) & (pixel == pix))[0]
                time_ = time[cutid]
                if len(time_) == 0:
                    print(f"エラー: ピクセル{pix}とタイプ{itype_}のデータが空です。")
                    continue

                if gtiuse:
                    pixel_x_lc, pixel_x_err, pixel_y_lc, pixel_y_err = fast_lc(time[0], time[-1], timebinsize, pixel_time_, \
                        gtiuse = gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop, lcthresh = lcthresh)            

                    x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_, \
                        gtiuse = gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop, lcthresh = lcthresh)

                else:
                    pixel_x_lc, pixel_x_err, pixel_y_lc, pixel_y_err = fast_lc(time[0], time[-1], timebinsize, pixel_time_)                                
                    x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)

                zcutid = np.where(y_lc > 0)[0]
                pixel_x_lc  = pixel_x_lc[zcutid]
                pixel_x_err = pixel_x_err[zcutid]
                pixel_y_lc  = pixel_y_lc[zcutid]
                pixel_y_err = pixel_y_err[zcutid]

                bratios = calc_branchingratios(pixel_y_lc)

                x_lc  = x_lc[zcutid]
                x_err = x_err[zcutid]
                y_lc  = y_lc[zcutid]
                y_err = y_err[zcutid]

                dtime_lc = [ref_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]

                if k == 0:
                    ax.errorbar(dtime_lc, pixel_y_lc, yerr=pixel_y_err, fmt=ishape[0], label=f"p={pix},t=all,id={obsid}")

                ax.errorbar(dtime_lc, y_lc, yerr=y_err, fmt=ishape[0], label=f"p{pix},t={itype_},id={obsid}", color=type_color)
                ax.errorbar(dtime_lc, y_lc/bratios[itype_], yerr=y_err/bratios[itype_], fmt=ishape[0], label=f"p={pix},t={itype_} (rate/bratio)", \
                                    color=type_color, alpha=0.4)

                # 分岐比
                if k == 0:
                    branching_ax.errorbar(dtime_lc, bratios[0], fmt="-",label=f"pix={pix}, type=Hp", alpha=0.4, color=type_colors[0])
                    branching_ax.errorbar(dtime_lc, bratios[1], fmt="-",label=f"pix={pix}, type=Mp", alpha=0.4, color=type_colors[1])
                    branching_ax.errorbar(dtime_lc, bratios[2], fmt="-",label=f"pix={pix}, type=Ms", alpha=0.4, color=type_colors[2])
                    branching_ax.errorbar(dtime_lc, bratios[3], fmt="-",label=f"pix={pix}, type=Lp", alpha=0.4, color=type_colors[3])
                    branching_ax.errorbar(dtime_lc, bratios[4], fmt="-",label=f"pix={pix}, type=Ls", alpha=0.4, color=type_colors[4])                
                branching_ax.errorbar(dtime_lc, y_lc/pixel_y_lc, fmt="o",label=f"pix={pix}, type={itype_} (data)",ms=2, color=type_color)

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            branching_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            outpng = f"{output}_lightcurve_pixel{pix:02d}_{outftag}.png"
            plt.savefig(outpng)
            print(f"..... {outpng} is created. ")
            if show:
                plt.show()

# 光度曲線をプロットする関数
def plot_sumlightcurve(event_list, plotpixels, itypenames, timebinsize, output, ref_time, \
                       gtiuse = False, debug=False, show = False, nonstop = False, pfactor = 2, lcthresh = 0.8):
    """
    イベントリストから光度曲線をプロットする。
    """
    colors = plt.cm.get_cmap('tab10', len(plotpixels)).colors
    ishape = [".", "s", "D", "*", "x"]

    type_colors = plt.cm.Set1(np.linspace(0, 1, 9))

    for fname in event_list:
        outftag = fname.replace(".evt", "").replace(".gz", "")
        head = fits.open(fname)[1].header
        obsid = head["OBS_ID"]
        oname = head["OBJECT"]

        if gtiuse:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel, overlaps_start, overlaps_stop = process_data_wgti(fname, ref_time, timebinsize, debug=debug, nonstop = nonstop)
        else:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, ref_time)

        for k, itype_ in enumerate(itypenames):

            fig, (ax, branching_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_ylabel(f"Counts/s (binsize = {timebinsize}s)")
            ax.grid(alpha=0.2)
            ax.set_title(f"{obsid} {oname}")

            branching_ax.set_xscale("linear")
            branching_ax.set_yscale("linear")
            branching_ax.set_ylabel("Branching Ratios")
            branching_ax.set_xlabel("TIME")
            branching_ax.grid(alpha=0.2)


            # plot all pixels (itype < 5)
            cutid = np.where(itype < 5)[0]
            time_ = time[cutid]

            if gtiuse:
                x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_, \
                    gtiuse = gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop, lcthresh = lcthresh)
            else:
                x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)

            zcutid = np.where(y_lc > 0)[0]
            x_lc  = x_lc[zcutid]
            x_err = x_err[zcutid]
            y_lc  = y_lc[zcutid]
            y_err = y_err[zcutid]

            dtime_lc = [ref_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]
            # all lightcurve in all types
            ax.errorbar(dtime_lc, y_lc, yerr=y_err, fmt=".", label=f"all pixel,all types", color="r")

            # plot all pixels (itype == itype_)
            cutid = np.where(itype == itype_)[0]
            time_ = time[cutid]

            if gtiuse:
                x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_, \
                    gtiuse = gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop, lcthresh = lcthresh)
            else:
                x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)

            zcutid = np.where(y_lc > 0)[0]
            x_lc  = x_lc[zcutid]
            x_err = x_err[zcutid]
            y_lc  = y_lc[zcutid]
            y_err = y_err[zcutid]

            dtime_lc = [ref_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]
            # all lightcurve 
            ax.errorbar(dtime_lc, y_lc, yerr=y_err, fmt=".", label=f"all pixel,{g_typename[itype_]}", color="k")

            for j, pix in enumerate(plotpixels):

                type_color = type_colors[k % len(type_colors)]

                print(f"ピクセル{pix}とタイプ{itype_}のデータを処理中 (obsid {obsid})")

                pixel_cutid = np.where((pixel == pix))[0]
                pixel_time_ = time[pixel_cutid]            

                if len(pixel_time_) == 0:
                    print(f"エラー: ピクセル{pix}とタイプ{itype_}のデータが空です。")
                    continue

                cutid = np.where((itype == itype_) & (pixel == pix))[0]
                time_ = time[cutid]
                if len(time_) == 0:
                    print(f"エラー: ピクセル{pix}とタイプ{itype_}のデータが空です。")
                    continue

                if gtiuse:
                    pixel_x_lc, pixel_x_err, pixel_y_lc, pixel_y_err = fast_lc(time[0], time[-1], timebinsize, pixel_time_, \
                        gtiuse = gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop, lcthresh = 0.8)            

                    x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_, \
                        gtiuse = gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop, lcthresh = 0.8)

                else:
                    pixel_x_lc, pixel_x_err, pixel_y_lc, pixel_y_err = fast_lc(time[0], time[-1], timebinsize, pixel_time_)                                
                    x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)

                zcutid = np.where(y_lc > 0)[0]
                pixel_x_lc  = pixel_x_lc[zcutid]
                pixel_x_err = pixel_x_err[zcutid]
                pixel_y_lc  = pixel_y_lc[zcutid]
                pixel_y_err = pixel_y_err[zcutid]

                bratios = calc_branchingratios(pixel_y_lc)

                x_lc  = x_lc[zcutid]
                x_err = x_err[zcutid]
                y_lc  = y_lc[zcutid]
                y_err = y_err[zcutid]

                dtime_lc = [ref_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]

                # each lightcurve 
                ax.errorbar(dtime_lc, y_lc * pfactor  + 2*j, yerr=y_err * pfactor, fmt=".", label=f"p{pix},{g_typename[itype_]}, x {pfactor} + {j}", color=g_pixcolors[pix])

                # branching ratios 
                branching_ax.errorbar(dtime_lc, y_lc/pixel_y_lc, fmt="o",label=f"p{pix},{g_typename[itype_]} (data)",ms=2, color=g_pixcolors[pix])

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            branching_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            outpng = f"{output}_sumlightcurve_type_{g_typename[itype_]}_{outftag}.png"
            plt.savefig(outpng)
            print(f"..... {outpng} is created. ")
            if show:
                plt.show()

def plot_rate_vs_grade(
    event_list,
    plotpixels,
    itypenames,
    timebinsize,
    output,
    ref_time,
    gtiuse=False,
    debug=False,
    show=True,
    nonstop=False,
    lcthresh=0.8,
    rate_max_ingratio=10,
    yscale_ingratio="log",
    # ---- new knobs ----
    ratio_mode="percent_residual",  # "percent_residual" or "obs_over_model"
#    ratio_mode="obs_over_model",  # "percent_residual" or "obs_over_model"
    ratio_ylim=(-200, 200),           # for percent_residual
#    ratio_ylim=(0, 10),           # for obs_over_model
    save_summary_csv=True,
):
    """
    イベントリストから rate_vs_grade をプロットする（改訂版）。

    上段: each-grade rate vs all-grade rate（観測 + 理論直線）
    下段: 観測と理論の比の診断（デフォルトは %残差）

    ratio_mode:
      - "percent_residual": (obs-model)/model * 100 [%]
      - "obs_over_model":   obs/model [dimensionless]
    """

    # plotting colors
    type_colors = plt.cm.Set1(np.linspace(0, 1, 9))

    # marker for ratio panel: logだと点がスカスカに見えやすいので丸、linearなら点でもOK
    if yscale_ingratio == "log":
        ms_gratio = "o"
    else:
        ms_gratio = "."

    # For model curves
    npoints = 10000
    rate_y = np.linspace(0, rate_max_ingratio, num=npoints)
    bratios = calc_branchingratios(rate_y)  # expected branching ratio as function of total rate

    # summary accumulator (list of dicts)
    summary_rows = []

    for fname in event_list:
        outftag = fname.replace(".evt", "").replace(".gz", "")
        head = fits.open(fname)[1].header
        obsid = head.get("OBS_ID", "UNKNOWN")
        oname = head.get("OBJECT", "UNKNOWN")

        if gtiuse:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel, overlaps_start, overlaps_stop = process_data_wgti(
                fname, ref_time, timebinsize, debug=debug, nonstop=nonstop
            )
        else:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, ref_time)

        # If no events at all, skip safely
        if len(time) == 0:
            print(f"[WARN] empty event list: {fname}")
            continue

        t0, t1 = time[0], time[-1]

        for j, pix in enumerate(plotpixels):

            # -------------------------
            # Figure: 2 panels
            # -------------------------
            fig, (ax, ax_ratio) = plt.subplots(
                2, 1,
                figsize=(11, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            )

            # Upper panel
            ax.set_xscale(yscale_ingratio)
            ax.set_yscale(yscale_ingratio)
            ax.set_xlabel("")  # sharex: xlabel on lower
            ax.set_ylabel("each grade rate (c/s/pixel)")
            ax.grid(alpha=0.2)
            ax.set_title(f"OBSID={obsid} OBJECT={oname} pixel={pix:02d} timebinsize={timebinsize} (s)")

            if yscale_ingratio == "log":
                ax.set_ylim(1e-5, rate_max_ingratio)
                ax.set_xlim(1e-2, rate_max_ingratio)
            else:
                ax.set_ylim(0, rate_max_ingratio)
                ax.set_xlim(0, rate_max_ingratio)

            # Lower panel
            ax_ratio.set_xscale(yscale_ingratio)
            ax_ratio.set_xlabel("all grade rate (c/s/pixel)")
            ax_ratio.grid(alpha=0.2)

            if ratio_mode == "percent_residual":
                ax_ratio.set_yscale("linear")
                ax_ratio.set_ylabel("Δ from model (%)")
                ax_ratio.axhline(0.0, lw=1.0, alpha=0.8)
                for p in (10, 20, 50):
                    ax_ratio.axhline(+p, lw=0.8, alpha=0.4, ls="--")
                    ax_ratio.axhline(-p, lw=0.8, alpha=0.4, ls="--")
                if ratio_ylim is not None:
                    ax_ratio.set_ylim(*ratio_ylim)
            elif ratio_mode == "obs_over_model":
                ax_ratio.set_yscale("linear")
                ax_ratio.set_ylabel("obs / model")
                ax_ratio.axhline(1.0, lw=1.0, alpha=0.8)
                for r in (0.8, 0.9, 1.1, 1.2, 1.5, 2.0):
                    ax_ratio.axhline(r, lw=0.8, alpha=0.35, ls="--")
            else:
                raise ValueError(f"Unknown ratio_mode: {ratio_mode}")

            itype_counts_text = []
            per_pix_all_ratios = []  # collect per itype ratios for pixel summary
            per_pix_all_weights = [] # optional weights (N or 1); here we just keep unweighted list

            # Pixel cut (all types)
            pixel_cutid = np.where(pixel == pix)[0]
            pixel_time_ = time[pixel_cutid]

            if len(pixel_time_) == 0:
                print(f"[WARN] pixel {pix:02d} has no events in {fname}")
                plt.close(fig)
                continue

            # Pre-compute pixel all-grade LC x-axis from pixel_time_
            if gtiuse:
                pixel_x_lc, pixel_x_err, pixel_y_lc, pixel_y_err = fast_lc(
                    t0, t1, timebinsize, pixel_time_,
                    gtiuse=gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop,
                    lcthresh=lcthresh
                )
            else:
                pixel_x_lc, pixel_x_err, pixel_y_lc, pixel_y_err = fast_lc(
                    t0, t1, timebinsize, pixel_time_
                )

            # -------------------------
            # Loop over event grades
            # -------------------------
            for k, itype_ in enumerate(itypenames):
                print(f"processing pix={pix:02d}, itype={itype_} (obsid={obsid})")
                type_color = type_colors[k % len(type_colors)]

                cutid = np.where((itype == itype_) & (pixel == pix))[0]
                time_ = time[cutid]

                if len(time_) == 0:
                    print(f"[INFO] pix={pix:02d}, itype={itype_} is empty -> skip")
                    continue

                itype_counts_text.append(f"{itype_}:{len(time_)}")

                # Grade LC
                if gtiuse:
                    x_lc, x_err, y_lc, y_err = fast_lc(
                        t0, t1, timebinsize, time_,
                        gtiuse=gtiuse, overlaps_start=overlaps_start, overlaps_stop=overlaps_stop,
                        lcthresh=lcthresh
                    )
                else:
                    x_lc, x_err, y_lc, y_err = fast_lc(t0, t1, timebinsize, time_)

                # Keep only bins where grade LC has >0
                zcutid = np.where(y_lc > 0)[0]
                if len(zcutid) == 0:
                    continue

                # Align to those bins
                x_plot = pixel_y_lc[zcutid]          # x = all-grade rate
                y_obs = y_lc[zcutid]                 # y = this grade rate
                yerr_obs = y_err[zcutid]

                # Model: y_model = x_all * branching_ratio(itype)
                # NOTE: bratios is returned as an array keyed by itype name,
                #       and can be a function of x via rate_y grid inside calc_branchingratios.
                # In your current code you used bratios[itype_] with rate_y, so we keep same behavior:
                # treat bratios[itype_] as a scalar or an array broadcastable.
                # Here, we assume it can accept x_plot by interpolation if needed.
                # If bratios[itype_] is scalar: OK.
                # If bratios[itype_] is array over rate_y: we interpolate onto x_plot.
                br = bratios[itype_]
                if np.ndim(br) == 0:
                    br_x = np.full_like(x_plot, float(br))
                else:
                    br_x = np.interp(x_plot, rate_y, br, left=np.nan, right=np.nan)

                y_model = x_plot * br_x

                # Valid points
                valid = np.isfinite(x_plot) & np.isfinite(y_obs) & np.isfinite(yerr_obs) & np.isfinite(y_model) & (y_model > 0)
                x_plot = x_plot[valid]
                y_obs = y_obs[valid]
                yerr_obs = yerr_obs[valid]
                y_model = y_model[valid]

                if len(x_plot) == 0:
                    continue

                # Upper: model curve + data
                # model curve: y = x * bratio(x)
                ax.plot(rate_y, rate_y * bratios[itype_], "--", alpha=0.7, color=type_color)
                ax.errorbar(
                    x_plot, y_obs, yerr=yerr_obs,
                    fmt=ms_gratio, color=type_color,
                    label=f"itype={itype_}"
                )

                # Lower: ratio diagnostics
                if ratio_mode == "percent_residual":
                    ratio_val = (y_obs - y_model) / y_model * 100.0
                    ratio_err = (yerr_obs / y_model) * 100.0
                else:  # "obs_over_model"
                    ratio_val = y_obs / y_model
                    ratio_err = yerr_obs / y_model

                ax_ratio.errorbar(
                    x_plot, ratio_val, yerr=ratio_err,
                    fmt=ms_gratio, color=type_color, alpha=0.9
                )

                # Collect for summary
                per_pix_all_ratios.append(ratio_val)
                # if you want to weight by inverse variance etc., store ratio_err too; for now unweighted
                per_pix_all_weights.append(np.ones_like(ratio_val))

                # ---- per (pix, itype) summary row ----
                def _q(a, q):
                    return float(np.nanquantile(a, q)) if len(a) > 0 else np.nan

                row = {
                    "obsid": obsid,
                    "object": oname,
                    "file": os.path.basename(fname),
                    "pixel": int(pix),
                    "itype": str(itype_),
                    "ratio_mode": ratio_mode,
                    "n_bins": int(len(ratio_val)),
                    "median": float(np.nanmedian(ratio_val)),
                    "mean": float(np.nanmean(ratio_val)),
                    "std": float(np.nanstd(ratio_val)),
                    "p16": _q(ratio_val, 0.16),
                    "p84": _q(ratio_val, 0.84),
                }
                summary_rows.append(row)

            # Legend only in upper
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))            
            # ---- per-pixel summary (all itypes merged) ----
            if len(per_pix_all_ratios) > 0:
                merged = np.concatenate(per_pix_all_ratios)
                if len(merged) > 0 and np.any(np.isfinite(merged)):
                    med = float(np.nanmedian(merged))
                    mea = float(np.nanmean(merged))
                    std = float(np.nanstd(merged))
                    p16 = float(np.nanquantile(merged, 0.16))
                    p84 = float(np.nanquantile(merged, 0.84))
                    nbin = int(np.sum(np.isfinite(merged)))

                    # Put a small text box (upper panel)
                    if ratio_mode == "percent_residual":
                        box = (
                            f"ALL itypes (pixel summary)\n"
                            f"median={med:+.1f}%  mean={mea:+.1f}%\n"
                            f"p16={p16:+.1f}%  p84={p84:+.1f}%  N={nbin}"
                        )
                    else:
                        box = (
                            f"ALL itypes (pixel summary)\n"
                            f"median={med:.3f}  mean={mea:.3f}\n"
                            f"p16={p16:.3f}  p84={p84:.3f}  N={nbin}"
                        )

                    ax.text(
                        0.98, 0.02, box,
                        transform=ax.transAxes,
                        ha="right", va="bottom",
                        fontsize=8,
                        color="black",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, lw=0.5),
                    )

                    # Also store per-pixel summary row (itype="ALL")
                    summary_rows.append({
                        "obsid": obsid,
                        "object": oname,
                        "file": os.path.basename(fname),
                        "pixel": int(pix),
                        "itype": "ALL",
                        "ratio_mode": ratio_mode,
                        "n_bins": nbin,
                        "median": med,
                        "mean": mea,
                        "std": std,
                        "p16": p16,
                        "p84": p84,
                    })

            # ---- counts text (bottom-left, gray) ----
            count_text = f"{fname} # " + " ".join(itype_counts_text)
            fig.text(0.03, 0.02, count_text, va="top", fontsize=6, color="gray", alpha=0.7)
            fig.subplots_adjust(right=0.85)
            fig.tight_layout()

            outpng = f"{output}_rate_vs_grade_pixel{pix:02d}_{outftag}.png"
            plt.savefig(outpng, dpi=200)
            print(f"[OK] {outpng} is created.")

            if show:
                plt.show()
            else:
                plt.close(fig)

        # end pix loop
    # end file loop

    # -------------------------
    # Save summary CSV
    # -------------------------
    if save_summary_csv and len(summary_rows) > 0:
        import csv

        outcsv = f"{output}_rate_vs_grade_summary.csv"
        fieldnames = ["obsid", "object", "file", "pixel", "itype", "ratio_mode",
                      "n_bins", "median", "mean", "std", "p16", "p84"]

        # append if exists, else write header
        write_header = not os.path.exists(outcsv)
        with open(outcsv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        print(f"[OK] summary CSV saved: {outcsv}")
    elif save_summary_csv:
        print("[INFO] summary CSV not saved (no summary rows).")

def estimate_lp_ls_from_hpmpms_profilelik(
    *,
    event_list,
    plotpixels,
    itypenames,
    timebinsize,
    output,
    ref_time,
    # pipeline flags (kept to match your existing signature style)
    gtiuse=False,
    debug=False,
    nonstop=False,
    lcthresh=0.8,  # kept for compatibility; not used in this “total-count” estimator
    # model / inference knobs
    fit_grades=("Hp", "Mp", "Ms"),
    target_grades=("Lp", "Ls"),
    rate_max=10.0,          # [c/s/pixel] for λ bounds + branching grid
    rate_grid_n=20001,      # dense enough for smooth interp
    ci_delta_2logL=1.0,     # 68% for 1 parameter (Wilks): 2ΔlogL<=1
    ci_scan_points=2000,    # profile scan resolution
    # outputs
    save_csv=True,
    make_plot=True,
    show=True,
):
    """
    (A) Pixel総カウント版・最小実装（関数1本）

    方針:
      1) process_data(_wgti) で event-level を取得し、pixel×grade の総カウント K_{i,g} を作る
      2) 露光 T は
           - gtiuse=False: T = time[-1] - time[0]
           - gtiuse=True : T = sum(overlaps_stop - overlaps_start)  (process_data_wgti の返り値を利用)
      3) 信頼できる grade (fit_grades = Hp/Mp/Ms) のみで λ_i を最尤推定:
           K_{i,g} ~ Poisson( T * λ_i * p_g(λ_i) )
         ここで p_g(λ) は calc_branchingratios(rate_grid) を補間して得る
      4) profile likelihood で λ_i の 68%CI を求める（2ΔlogL<=1）
      5) その λ_i,CI から Lp/Ls の期待カウントを予測し、観測との差（bias[%]）とCIを出す
      6) CSV保存 + 図（pixelごとのbias[%]）を作る

    依存:
      - process_data / process_data_wgti
      - calc_branchingratios
      - (fast_lc は使いません：total-count 推定なので event-level count を直接使うのが最小&堅牢)
    """

    # -------------------------
    # Helpers
    # -------------------------
    def _safe_log(x):
        return np.log(np.clip(x, 1e-300, None))

    def _build_branching_functions(rate_max_, ngrid_, itypenames_):
        """
        calc_branchingratios(rate_grid) の戻り値が
          - dict
          - numpy structured array / recarray
          - list/ndarray（itypenames順）
        のどれでも動くようにする。
        """
        rate_grid = np.linspace(0.0, float(rate_max_), int(ngrid_))
        br = calc_branchingratios(rate_grid)

        def _get_br_array(g):
            # 1) dict / structured array: br[g] が通る場合
            try:
                arr = br[g]
                return arr
            except Exception:
                pass

            # 2) list/ndarray: itypenames の順番対応で引く
            try:
                idx = list(itypenames_).index(g)
            except ValueError as e:
                raise KeyError(
                    f"Grade '{g}' not found. Available itypenames={list(itypenames_)}"
                ) from e

            try:
                return br[idx]
            except Exception as e:
                raise KeyError(
                    f"Cannot index branchingratios by idx={idx} for grade='{g}'. "
                    f"type(br)={type(br)}"
                ) from e

        def p_of(g, lam):
            lam = float(lam)
            if lam <= 0.0:
                return 0.0

            arr = _get_br_array(g)

            # scalar
            if np.ndim(arr) == 0:
                val = float(arr)
                return val if (np.isfinite(val) and val > 0) else 0.0

            # array over rate_grid
            val = float(np.interp(lam, rate_grid, np.asarray(arr), left=np.nan, right=np.nan))
            if not np.isfinite(val) or val <= 0.0:
                return 0.0
            return val

        return p_of

    def _logL_pixel(lam, T, K_by_grade, p_of):
        """
        log L(lam) = Σ_{g in fit_grades} [ K_g log( T lam p_g(lam) ) - T lam p_g(lam) ] + const
        factorial terms omitted (do not affect argmax / profile thresholds by lam)
        """
        lam = float(lam)
        if lam <= 0.0 or not np.isfinite(lam):
            return -np.inf

        ll = 0.0
        for g in fit_grades:
            Kg = int(K_by_grade.get(g, 0))
            pg = p_of(g, lam)
            mu = T * lam * pg
            if mu <= 0.0 or not np.isfinite(mu):
                return -np.inf
            ll += Kg * _safe_log(mu) - mu
        return float(ll)

    def _mle_and_ci_profile(T, K_by_grade, p_of, lam_min, lam_max):
        """
        1D MLE via bounded scalar minimization of -logL,
        then profile CI via scanning and 2ΔlogL<=ci_delta_2logL.
        """
        # MLE
        def nll(x):
            return -_logL_pixel(x, T, K_by_grade, p_of)

        res = minimize_scalar(nll, bounds=(lam_min, lam_max), method="bounded")
        lam_hat = float(res.x) if res.success else np.nan
        ll_hat = _logL_pixel(lam_hat, T, K_by_grade, p_of) if np.isfinite(lam_hat) else -np.inf

        if not np.isfinite(lam_hat) or not np.isfinite(ll_hat):
            return np.nan, (np.nan, np.nan), (np.nan, np.nan)

        # Profile scan grid (log-ish around lam_hat, but bounded)
        # Make a dense grid that covers [lam_min, lam_max], with extra concentration near lam_hat.
        # Simple robust choice: logspace between bounds (avoid 0).
        eps = 1e-12
        lo = max(lam_min, eps)
        hi = max(lam_max, lo * 1.001)
        grid = np.logspace(np.log10(lo), np.log10(hi), int(ci_scan_points))

        ll_grid = np.array([_logL_pixel(x, T, K_by_grade, p_of) for x in grid], dtype=float)
        d2 = 2.0 * (ll_hat - ll_grid)  # 2ΔlogL
        ok = np.isfinite(d2) & (d2 <= ci_delta_2logL)

        if not np.any(ok):
            # CI not found within bounds
            return lam_hat, (np.nan, np.nan), (ll_hat, np.nan)

        # Find lower/upper by edge of ok region
        idx = np.where(ok)[0]
        i_lo = idx[0]
        i_hi = idx[-1]

        lam_lo = float(grid[i_lo])
        lam_hi = float(grid[i_hi])

        # Optional refinement: interpolate where d2 crosses threshold near edges
        thr = float(ci_delta_2logL)

        def _interp_cross(i1, i2):
            # linear interpolation in (log lam, d2) space for stability
            x1, x2 = np.log(grid[i1]), np.log(grid[i2])
            y1, y2 = d2[i1], d2[i2]
            if not (np.isfinite(y1) and np.isfinite(y2)) or y1 == y2:
                return float(grid[i2])
            a = (thr - y1) / (y2 - y1)
            x = x1 + a * (x2 - x1)
            return float(np.exp(x))

        # refine lower
        if i_lo > 0 and np.isfinite(d2[i_lo - 1]) and d2[i_lo - 1] > thr:
            lam_lo = _interp_cross(i_lo - 1, i_lo)
        # refine upper
        if i_hi < len(grid) - 1 and np.isfinite(d2[i_hi + 1]) and d2[i_hi + 1] > thr:
            lam_hi = _interp_cross(i_hi + 1, i_hi)

        return lam_hat, (lam_lo, lam_hi), (ll_hat, None)

    # -------------------------
    # Precompute branching p_g(lam)
    # -------------------------
    p_of = _build_branching_functions(rate_max, rate_grid_n, itypenames)

    # -------------------------
    # Main loop over files
    # -------------------------
    results = []  # list[dict]

    for fname in event_list:
        head = fits.open(fname)[1].header
        obsid = head.get("OBS_ID", "UNKNOWN")
        obj = head.get("OBJECT", "UNKNOWN")
        outftag = fname.replace(".evt", "").replace(".gz", "")

        # event-level load
        if gtiuse:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel, overlaps_start, overlaps_stop = process_data_wgti(
                fname, ref_time, timebinsize, debug=debug, nonstop=nonstop
            )
            # exposure from overlaps (total good time)
            if overlaps_start is None or overlaps_stop is None or len(overlaps_start) == 0:
                # fallback: use span
                T = float(time[-1] - time[0]) if len(time) > 1 else 0.0
            else:
                T = float(np.sum(np.asarray(overlaps_stop) - np.asarray(overlaps_start)))
        else:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, ref_time)
            T = float(time[-1] - time[0]) if len(time) > 1 else 0.0

        if T <= 0.0 or len(time) == 0:
            print(f"[WARN] skip (no exposure/events): {fname}")
            continue

        print("[DBG] itype dtype:", getattr(itype, "dtype", None), "example:", itype[:10])
        print("[DBG] unique itype (head):", np.unique(itype)[:20])
        print("[DBG] pixel dtype:", getattr(pixel, "dtype", None), "min/max:", np.min(pixel), np.max(pixel))
        print("[DBG] unique pixel (head):", np.unique(pixel)[:20])
        print("[DBG] itypenames passed:", itypenames)
        print("[DBG] fit_grades:", fit_grades, "target_grades:", target_grades)


        # per-pixel inference
        for pix in plotpixels:
            pix = int(pix)
            cut_pix = (pixel == pix)
            if not np.any(cut_pix):
                # no events in this pixel
                continue

            # total counts per grade (direct from event-level)
            K_by_grade = {}
            for g in itypenames:
                # itype array might be strings or codes; your current code compares (itype == itype_)
                K_by_grade[g] = int(np.sum(cut_pix & (itype == g)))

            # Only proceed if fit grades have some counts (otherwise λ is weakly identified)
            K_fit_sum = sum(int(K_by_grade.get(g, 0)) for g in fit_grades)
            if K_fit_sum <= 0:
                continue

            # MLE + profile CI for lambda
            lam_min = 1e-8
            lam_max = float(rate_max)

            lam_hat, (lam_lo, lam_hi), _ = _mle_and_ci_profile(T, K_by_grade, p_of, lam_min, lam_max)

            # Predict target grades (expected counts)
            pred = {}
            pred_lo = {}
            pred_hi = {}
            bias = {}
            bias_lo = {}
            bias_hi = {}

            for g in target_grades:
                Kobs = int(K_by_grade.get(g, 0))

                def _M(lam):
                    pg = p_of(g, lam)
                    return T * float(lam) * float(pg)

                if np.isfinite(lam_hat):
                    Mhat = float(_M(lam_hat))
                else:
                    Mhat = np.nan

                if np.isfinite(lam_lo):
                    Mlo = float(_M(lam_lo))
                else:
                    Mlo = np.nan
                if np.isfinite(lam_hi):
                    Mhi = float(_M(lam_hi))
                else:
                    Mhi = np.nan

                pred[g] = Mhat
                pred_lo[g] = Mlo
                pred_hi[g] = Mhi

                # bias[%] relative to model expectation
                if Mhat > 0:
                    b = (Kobs - Mhat) / Mhat * 100.0
                else:
                    b = np.nan

                # bias CI from mapped pred CI (treat pred interval as uncertainty of model expectation)
                # Use conservative mapping: compute bias if model=upper/lower
                # (If model expectation is higher -> bias smaller; if lower -> bias larger)
                if np.isfinite(Mlo) and Mlo > 0:
                    b_hi = (Kobs - Mlo) / Mlo * 100.0
                else:
                    b_hi = np.nan
                if np.isfinite(Mhi) and Mhi > 0:
                    b_lo = (Kobs - Mhi) / Mhi * 100.0
                else:
                    b_lo = np.nan

                bias[g] = b
                bias_lo[g] = b_lo
                bias_hi[g] = b_hi

            row = {
                "obsid": obsid,
                "object": obj,
                "file": os.path.basename(fname),
                "pixel": pix,
                "T_exposure_s": T,
                "K_fit_sum": K_fit_sum,
                "lambda_hat_cps": lam_hat,
                "lambda_lo_cps": lam_lo,
                "lambda_hi_cps": lam_hi,
            }
            # attach observed + predicted + bias for each target grade
            for g in target_grades:
                row[f"Kobs_{g}"] = int(K_by_grade.get(g, 0))
                row[f"Mpred_{g}"] = pred[g]
                row[f"Mpred_lo_{g}"] = pred_lo[g]
                row[f"Mpred_hi_{g}"] = pred_hi[g]
                row[f"biaspct_{g}"] = bias[g]
                row[f"biaspct_lo_{g}"] = bias_lo[g]
                row[f"biaspct_hi_{g}"] = bias_hi[g]

            results.append(row)

    # -------------------------
    # Save CSV
    # -------------------------
    outcsv = f"{output}_lp_ls_infer_profilelik.csv"
    if save_csv and len(results) > 0:
        # stable header
        base_fields = [
            "obsid", "object", "file", "pixel", "T_exposure_s", "K_fit_sum",
            "lambda_hat_cps", "lambda_lo_cps", "lambda_hi_cps",
        ]
        extra_fields = []
        for g in target_grades:
            extra_fields += [
                f"Kobs_{g}",
                f"Mpred_{g}", f"Mpred_lo_{g}", f"Mpred_hi_{g}",
                f"biaspct_{g}", f"biaspct_lo_{g}", f"biaspct_hi_{g}",
            ]
        fieldnames = base_fields + extra_fields

        with open(outcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        print(f"[OK] CSV saved: {outcsv}")
    elif save_csv:
        print("[WARN] No results -> CSV not written.")

    # -------------------------
    # Plot: pixel vs bias[%] for target grades
    # -------------------------
    outpng = f"{output}_lp_ls_bias_profilelik.png"
    if make_plot and len(results) > 0:
        # If multiple files are included, this plot overlays all rows;
        # For “minimal” implementation, we just plot all points (obsid labels in legend).
        fig, axes = plt.subplots(len(target_grades), 1, figsize=(10, 3.2 * len(target_grades)), sharex=True)
        if len(target_grades) == 1:
            axes = [axes]

        # sort by (file, pixel) for nicer x
        results_sorted = sorted(results, key=lambda r: (r["file"], int(r["pixel"])))

        # x positions: use pixel number; if multiple files, points overlap intentionally
        xs = np.array([int(r["pixel"]) for r in results_sorted], dtype=int)

        for ax, g in zip(axes, target_grades):
            ys = np.array([r.get(f"biaspct_{g}", np.nan) for r in results_sorted], dtype=float)
            ylo = np.array([r.get(f"biaspct_lo_{g}", np.nan) for r in results_sorted], dtype=float)
            yhi = np.array([r.get(f"biaspct_hi_{g}", np.nan) for r in results_sorted], dtype=float)

            # errorbars: asymmetric
            yerr_low = ys - ylo
            yerr_high = yhi - ys
            # guard
            yerr = np.vstack([
                np.where(np.isfinite(yerr_low), yerr_low, 0.0),
                np.where(np.isfinite(yerr_high), yerr_high, 0.0),
            ])

            ax.axhline(0.0, lw=1.0, alpha=0.8)
            for p in (10, 20, 50):
                ax.axhline(+p, lw=0.8, alpha=0.35, ls="--")
                ax.axhline(-p, lw=0.8, alpha=0.35, ls="--")

            ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=2, alpha=0.9)
            ax.set_ylabel(f"{g}: (obs-model)/model [%]")
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("pixel")
        fig.suptitle("Lp/Ls bias[%] vs model (λ inferred from Hp/Mp/Ms) with profile-likelihood CI", y=0.995)
        fig.tight_layout()
        plt.savefig(outpng, dpi=200)
        print(f"[OK] Plot saved: {outpng}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    elif make_plot:
        print("[WARN] No results -> plot not created.")

    return results, outcsv, outpng


# メイン関数
def main():
    """
    スクリプトを実行するメイン関数。
    """
    args = parse_args()
    setup_plot()
    
    MJD_REFERENCE_DAY = 58484
    ref_time = Time(MJD_REFERENCE_DAY, format='mjd')
    
    plotpixels = list(map(int, args.plotpixels.split(',')))
    itypenames = list(map(int, args.itypenames.split(',')))

    with open(args.filelist, 'r') as file:
        event_list = [line.strip() for line in file]

    if args.gtiuse:
        if args.plot_lightcurve:
            print("----- plot_lightcurve ----- ")
            plot_lightcurve(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time,\
                gtiuse = args.gtiuse, debug= args.debug, show = args.show, nonstop = args.nonstop, lcthresh = args.lcthresh)

            print("----- plot_sumlightcurve ----- ")
            plot_sumlightcurve(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time,\
                gtiuse = args.gtiuse, debug= args.debug, show = args.show, nonstop = args.nonstop, lcthresh = args.lcthresh)

        if args.plot_rate_vs_grade:
            print("----- plot_rate_vs_grade ----- ")

            plot_rate_vs_grade(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time,\
                gtiuse = args.gtiuse, debug= args.debug, show = args.show, nonstop = args.nonstop, lcthresh = args.lcthresh, \
                                            rate_max_ingratio = args.rate_max_ingratio, yscale_ingratio = args.yscale_ingratio)

            results, outcsv, outpng = estimate_lp_ls_from_hpmpms_profilelik(event_list=event_list,\
                plotpixels=plotpixels, \
                itypenames=itypenames, \
                timebinsize=args.timebinsize, \
                output=args.output, \
                ref_time=ref_time,\
                gtiuse=args.gtiuse, \
                rate_max=args.rate_max_ingratio, \
                fit_grades=(0, 1, 2),
                target_grades=(3, 4),
                show=False, \
            )


    else:
        if args.plot_lightcurve:
            plot_lightcurve(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time)

        if args.plot_rate_vs_grade:
            plot_rate_vs_grade(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time, \
                debug= args.debug, show = args.show, rate_max_ingratio = args.rate_max_ingratio, yscale_ingratio = args.yscale_ingratio)

if __name__ == "__main__":
    main()
