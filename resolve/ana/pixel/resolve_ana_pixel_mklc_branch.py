#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys
from matplotlib.cm import get_cmap

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


# rate_vs_gradeをプロットする関数
def plot_rate_vs_grade(event_list, plotpixels, itypenames, timebinsize, output, ref_time, \
                            gtiuse = False, debug=False, show = True, nonstop = False, lcthresh = 0.8, rate_max_ingratio=10, yscale_ingratio = "log"):
    """
    イベントリストからrate_vs_gradeをプロットする。
    """
    colors = plt.cm.get_cmap('tab10', len(plotpixels)).colors
    ishape = [".", "s", "D", "*", "x"]
    type_colors = plt.cm.Set1(np.linspace(0, 1, 9))

    if yscale_ingratio == "log":
        ms_gratio="o"
    else:
        ms_gratio="."

    for fname in event_list:
        outftag = fname.replace(".evt", "").replace(".gz", "")
        head = fits.open(fname)[1].header
        obsid = head["OBS_ID"]
        oname = head["OBJECT"]

        if gtiuse:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel, overlaps_start, overlaps_stop = process_data_wgti(fname, ref_time, timebinsize, debug=debug, nonstop = nonstop)
        else:
            dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, ref_time)

        # 分岐比を計算
        npoints = 10000
        rate_y = np.linspace(0, rate_max_ingratio, num=npoints)
        bratios = calc_branchingratios(rate_y)

        for j, pix in enumerate(plotpixels):

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set_xscale(yscale_ingratio)
            ax.set_yscale(yscale_ingratio)
            ax.set_xlabel("all grade rate (c/s/pixel)")
            ax.set_ylabel("each grade rate (c/s/pixel)")
            ax.grid(alpha=0.2)
            ax.set_title(f"OBSID={obsid} pixel={pix:02d} timebinsize={timebinsize} (s)")
            if yscale_ingratio == "log":
                ax.set_ylim(1e-5,rate_max_ingratio)
                ax.set_xlim(1e-2,rate_max_ingratio)

            itype_counts = []
            for k, itype_ in enumerate(itypenames):
                print(f"ピクセル{pix}とタイプ{itype_}のデータを処理中 (obsid {obsid})")
                type_color = type_colors[k % len(type_colors)]

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

                # save count
                itype_counts.append(f"{itype_}: {len(time_)}")

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
                x_lc  = x_lc[zcutid]
                x_err = x_err[zcutid]
                y_lc  = y_lc[zcutid]
                y_err = y_err[zcutid]

                ax.plot(rate_y,rate_y*bratios[itype_], "--", alpha=0.7,label=f"pix={pix}, itype={itype_}", color=type_color)
                ax.errorbar(pixel_y_lc, y_lc, fmt=ms_gratio, label=f"pix={pix}, itype={itype_}", color=type_color)

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            # make text showing all counts
            count_text = f"{fname}" + "# are " + " ".join(itype_counts)
            fig.text(0.03, 0.02, count_text, va='top', fontsize=6, color="gray", alpha=0.7)
            plt.tight_layout()
            outpng = f"{output}_rate_vs_grade_pixel{pix:02d}_{outftag}.png"
            plt.savefig(outpng)
            print(f"..... {outpng} is created. ")
            print("show",show)
            if show:
                plt.show()

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
    else:
        if args.plot_lightcurve:
            plot_lightcurve(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time)

        if args.plot_rate_vs_grade:
            plot_rate_vs_grade(event_list, plotpixels, itypenames, args.timebinsize, args.output, ref_time, \
                debug= args.debug, show = args.show, rate_max_ingratio = args.rate_max_ingratio, yscale_ingratio = args.yscale_ingratio)

if __name__ == "__main__":
    main()
