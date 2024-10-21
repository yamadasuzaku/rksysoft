#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from pathlib import Path
import os 
from matplotlib.gridspec import GridSpec

# 環境変数 "RESOLVETOOLS" を取得
resolve_tools = os.getenv("RESOLVETOOLS")
# 環境変数が設定されていない場合にエラーを出して終了
if resolve_tools is None:
    print("Error: RESOLVETOOLS environment variable is not set.", file=sys.stderr)
    sys.exit(1)
# 環境変数が正しく取得された場合に処理を続行
print(f"RESOLVETOOLS is set to: {resolve_tools}")

# Set plot parameters for consistent styling
params = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 6}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
# Set default linewidth to 2
plt.rcParams['lines.linewidth'] = 0.8

# Define a pixel map for the 6x6 grid
pixel_map = np.array([
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],  # DETY
    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],  # DETX
    [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18, 25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27, 29, 30]  # PIXEL
])

# Maximum number of colors to be used in plots
CMAX = 20
colors = plt.cm.tab20(np.linspace(0, 1, CMAX))

# グローバル変数として宣言
MEM_AVGS_TO_SUB_ADDR = {}
MEM_DERIV_TO_SUB_ADDR = {}
MEM_DERIV_TO_SUB_ADDR_DERIVMAX = {}

# according to 
RECORD_LEN=1024
# /Users/syamada/work/ana/resolve/resolve_git/eclipse/2307_XRISM_PFT_TC11/main2/main2-TC11_01z_VAB_230705_RT.cps
# 4034    # CSXS3070R (1) SET_PRE_TRIG_LEN_HM 140 27 (nominal), SET_MIN_MAX_SHIFT
PRE_TRIG_LEN_H = 140
PRE_TRIG_LEN_M = 27
PULSE_THRES = 120
SECOND_THRES_MIN = 120 
SECOND_TRIG_GAP_LEN = 25 # 2 ms
SLOPE_DETECT_LEN = 20 # 1.6 ms
SLOPE_SKIP_LEN =0 # 0 ms
SECOND_THRES_FRAC = 20 # 1.6 ms
SECOND_THRES_USE_LEN = 75 # 6 ms
SPARE_LEN=8 # offset 
CLIPTHRES=12235
FALL_END_THRES=0

px_offset_avg_len_pow = 3  # 平均化するサンプル数の指数
px_offset_avg_len = 2 ** px_offset_avg_len_pow  # サンプル数 = 8
px_offset_avg_gap = 2  # オフセット

def shift_array(array, m):
    """
    配列を m 要素だけシフトする関数。範囲外の部分はゼロで埋める。
    
    Parameters:
    array (numpy.ndarray): シフト対象の配列
    m (int): シフトする要素数（正: 前方向、負: 後方向）
    
    Returns:
    numpy.ndarray: シフト後の配列
    """
    shifted_array = np.zeros_like(array)  # ゼロで初期化した配列

    if m > 0:
        shifted_array[m:] = array[:-m]  # 前方向にシフト（右方向）
    elif m < 0:
        shifted_array[:m] = array[-m:]  # 後方向にシフト（左方向）
    else:
        shifted_array = array  # シフトなし

    return shifted_array


import numpy as np

def align_peak(array, peakindex, outputlength):
    """
    配列の最大値の位置を peakindex に揃えてシフトし、指定された長さでゼロパディングする関数。

    Parameters:
    array (numpy.ndarray): 入力の一次元配列
    peakindex (int): シフト後の最大値のインデックス
    outputlength (int): 出力配列の長さ

    Returns:
    numpy.ndarray: シフトされ、ゼロパディングされた配列（長さは outputlength）
    """
    # 最大値のインデックスを取得
    max_index = np.argmax(array)

    # シフト量を計算 (正: 右シフト, 負: 左シフト)
    shift_amount = peakindex - max_index

    # 出力配列の初期化 (ゼロでパディング)
    shifted_array = np.zeros(outputlength, dtype=array.dtype)

    # シフト後の有効範囲を計算
    start_in = max(-shift_amount, 0)  # 入力配列の開始位置
    end_in = min(len(array) - shift_amount, len(array))  # 入力配列の終了位置

    start_out = max(shift_amount, 0)  # 出力配列の開始位置
    end_out = start_out + (end_in - start_in)  # 出力配列の終了位置

    # 有効範囲の要素をコピー
    shifted_array[start_out:end_out] = array[start_in:end_in]

    return shifted_array

# --- 使用例 ---
array = np.array([0, 1, 2, 3, 10, 2, 1, 0])  # サンプル配列
peakindex = 5  # 最大値の位置を 5 に揃える
outputlength = 10  # 出力配列の長さ

result = align_peak(array, peakindex, outputlength)
print(result)


def calc_lo_res_base(adc_sample, px_offset_avg_len_pow, px_offset_avg_len, px_offset_avg_gap, trigpnt = PRE_TRIG_LEN_H):
    """
    信号波形からbaselineレベルを計算するPython関数。

    Parameters:
    adc_sample (list of int): ADCサンプルのリスト。
    px_offset_avg_len_pow (int): 平均化するサンプル数の指数。
    px_offset_avg_len (int): 平均化するサンプル数。
    px_offset_avg_gap (int): サンプルのオフセット。

    Returns:
    int: 計算されたbaselineレベル。
    """
    lo_res_base = 0

    # オフセットの位置から計算を開始
    start_index = 0 - px_offset_avg_gap - px_offset_avg_len + trigpnt
    for i in range(px_offset_avg_len):
        lo_res_base += adc_sample[start_index + i]

    # 平均のシフト演算
    lo_res_base = (lo_res_base + px_offset_avg_len // 2) >> px_offset_avg_len_pow

    return lo_res_base


def check_quickdouble(xdata, ydata, xmin=140, xmax=200):
    """
    - src/150615b/psp_task_pxp_calc.c
        if ( derivative - deriv_pre >= quick_double_thres ) {
            quick_double = 1;
    """
    cutxdata = xdata[xmin:xmax]
    cutydata = ydata[xmin:xmax]

    # 負の値が存在するかを確認
    negative_indices = np.where(cutydata < 0)[0]

    if len(negative_indices) > 0:
        # 負の値がある場合、最初の負の値が現れるインデックスを取得してカット
        negative_index = negative_indices[0]
        cutxdata = cutxdata[:negative_index]
        cutydata = cutydata[:negative_index]
    else:
        # 負の値がない場合、何かおかしいが特に何もしない。
        pass

    ydiff = cutydata[1:] - cutydata[0:-1]
    cutid = np.where(ydiff >= 1)[0]
    if len(cutid) == 0:
        return 0, -1, -1
    else:        
        qd_x = cutxdata[cutid]
        qd_y = cutydata[cutid]
        npoint = len(qd_x)
        return npoint,qd_x, qd_y

def find_first_positive_to_negative(npydatalc, deriv_max_i, search_max = 300):
    for i in range(deriv_max_i, search_max):
#    for i in range(deriv_max_i, len(npydatalc) - 1):
        if npydatalc[i] >= 0 and npydatalc[i + 1] < 0:
            return i + 1  # 正から負に変わった位置のインデックス
    return 0  # 変化が見つからない場合

def get_deriv(ydata, step=8, search_i_min = 100, search_i_max = 200):
    # Initialize an empty list to store the derivative values
    ydatalc = []
    
    # Create an array of indices from 0 to the length of ydata
    numarray = np.arange(0, len(ydata), 1)

    ipos_fall_end = 0
    # Iterate over each element in ydata
    for onei, oneydata in enumerate(ydata):
        # Get the previous 'step' elements
        if onei >= step:
            prey = ydata[onei-step:onei]
        else:
            prey = []

        # Get the next 'step' elements
        if onei + step < len(ydata):
            posty = ydata[onei:onei+step]
        else:
            posty = []

        # Calculate the long derivative as the difference of means multiplied by 8 (not STEP)
        if len(prey) > 0 and len(posty) > 0:
            derivLong = (np.mean(posty) - np.mean(prey)) * 8
        else:
            derivLong = 0
        
        # Calculate the final derivative, using floor function for adjustment
        derivative = np.floor((derivLong + 2.) / 4.)

        # if ipos_fall_end == 0:
        #     if derivative < FALL_END_THRES:
        #         if (onei > search_i_min) and (onei < search_i_max):
        #             ipos_fall_end = onei
        # # append the derivative to the list
        ydatalc.append(derivative)

    # Convert the list to a numpy array
    npydatalc = np.array(ydatalc)

    # Find the maximum and minimum derivative values within the range of search_i_min to search_i_max indices
    deriv_max = np.amax(npydatalc[np.where((numarray > search_i_min) & (numarray < search_i_max))])
    deriv_min = np.amin(npydatalc[np.where((numarray > search_i_min) & (numarray < search_i_max))])

    # Find the indices of the maximum and minimum derivative values
    deriv_max_i = numarray[np.where(npydatalc == deriv_max)][0]
    deriv_min_i = numarray[np.where(npydatalc == deriv_min)][0]

    # Determine the peak derivative value
    if np.abs(deriv_max) > np.abs(deriv_min):
        peak = deriv_max
    else:
        peak = deriv_min

    # Find the index of the peak derivative value
    deriv_peak_i = numarray[np.where(npydatalc == peak)][0]

    # Return the computed derivatives and their related values
    return (npydatalc, deriv_max, deriv_max_i, deriv_min, deriv_min_i, peak, deriv_peak_i)


def plot_deriv(prevt, itypes, dumptext=False, plotflag=False, usetime=False, prevflag=False, xlims=None, ylims=None, \
                  usederiv=False, step=8, check_qd = False, selected_pixels=None):
    """
    Plots the pulse record data from a FITS file in a 6x6 grid format.

    Args:
    prevt (str): Path to the input FITS file.
    itypes (list): List of itype values to filter the data.
    dumptext (bool): Flag to dump x_time and pulse data to NPZ files.
    plotflag (bool): Flag to display the plot.
    usetime (bool): Flag to use time for the x-axis.
    prevflag (bool): Flag to plot previous interval values as text.
    xlims (tuple): Tuple specifying the x-axis limits (xmin, xmax).
    ylims (tuple): Tuple specifying the y-axis limits (ymin, ymax).
    usederiv (tuple): Flag to use derivative.
    """
    # Open the FITS file and extract relevant data

    # Select all pixels if none are specified
    if selected_pixels is None:
        selected_pixels = range(36)

    data = fits.open(prevt)
    print(f'Load file name = {prevt}')


    header = data[1].header
    obsid = header["OBS_ID"]
    target = header["OBJECT"]
    dateobs = header["DATE-OBS"]

    pulse_list = data[1].data['PULSEREC']

    time_list = data[1].data['TIME']
    pixel_list = data[1].data['PIXEL']
    itype_list = data[1].data['ITYPE']
    prev_list = data[1].data['PREV_INTERVAL']
    next_list = data[1].data['NEXT_INTERVAL']
    dmax_list = data[1].data['DERIV_MAX']
    pha_list = data[1].data['PHA']
    pi_list = data[1].data['PI']
    risetime_list = data[1].data['RISE_TIME']
    tickshift_list = data[1].data['TICK_SHIFT']

    qd_list = data[1].data['QUICK_DOUBLE']
    sd_list = data[1].data['SLOPE_DIFFER']
    lores_list = data[1].data['LO_RES_PH']
    status_list = data[1].data['STATUS']

    # ビット型 (bool) を int 型に変換
    qd_list_int = qd_list.astype(int)
    sd_list_int = sd_list.astype(int)
    status_list_int = status_list.astype(int)
    # # 変換後のデータを確認
    # print(qd_list_int)
    # print(sd_list_int)
    # print(status_list_int)

    data.close()

    # Define the time resolution
    dt = 80.0e-6
    x_time = np.arange(0, pulse_list.shape[-1], 1) * dt
    xadc_time = np.arange(0, pulse_list.shape[-1], 1) * 1

    for itype in itypes:

        itype_str = get_itype_str(itype)
        path = Path(prevt)
        oname = f'{path.stem}_{itype_str}'
        print(f'{oname}')

        for pixel in selected_pixels:

            cutid = np.where((pixel_list == pixel) & (itype_list == itype))
            pulses = pulse_list[cutid]       

            times = time_list[cutid]
            prevs = prev_list[cutid]
            nexts = next_list[cutid]
            dmaxs = dmax_list[cutid]
            phas = pha_list[cutid]
            pis = pi_list[cutid]

            risetimes = risetime_list[cutid]
            tickshifts = tickshift_list[cutid]

            qds = qd_list_int[cutid]
            sds = sd_list_int[cutid]
            los = lores_list[cutid]
            statuss = status_list_int[cutid]

            num_of_evt = len(pulses)        
            if num_of_evt > 0:
                print(f'PIXEL={pixel:02d}, N={num_of_evt}')

            for k, (pulse, time, pt, nt, dmax, pha, pi, rs, ts, qd, sd, lo, status) in enumerate(zip(pulses,times, prevs,nexts, dmaxs, phas, pis, risetimes, tickshifts,qds,sds,los, statuss)):
                print("pulse =", pulse)
                print(f"k, time, pt, nt, dmax, pha, pi, rs, ts, qd, sd, lo, status = {k}, {time}, {pt}, {nt}, {dmax}, {pha}, {pi}, {rs}, {ts}, {qd}, {sd}, {lo}, {status}")
                slow_pulse = 1 if rs > 127 else 0  
                rs = rs - 128 if rs > 127 else rs                
#                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=False)
#                plt.subplots_adjust(right=0.9, top=0.85)

                fig = plt.figure(figsize=(12, 8))
                gs = GridSpec(3, 1, height_ratios=[1, 4, 1])  # 真ん中の行を2倍の高さに設定
                # axesをリストに格納して管理
                # 最初のsubplotを作成
                axes = [fig.add_subplot(gs[0, 0])]

                # 2つ目以降のsubplotに最初のsubplotのx軸を共有する
                for i in range(1, 3):
                    axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0]))
#                axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

                pulse = pulse[SPARE_LEN:]

                if itype == 2 or itype == 4:
                    deriv_to_use, deriv_max, deriv_max_i, _, _, _, _ = get_deriv(pulse, search_i_min = 0, search_i_max = 150)
                else:
                    deriv_to_use, deriv_max, deriv_max_i, _, _, _, _ = get_deriv(pulse)                    

                ipos_fall_end = find_first_positive_to_negative(deriv_to_use, deriv_max_i)

                print(f"{k} :  (deriv_max_i,deriv_max)={deriv_max_i},{deriv_max}")
                adcmax = MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"]
                print (f"{k} : MEM_AVGS_TO_SUB_ADDR_MAX = {adcmax} lo = {lo}")

#                scaled_pulse_to_sub = lo * MEM_AVGS_TO_SUB_ADDR[pixel]/(MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"])
                print(f"lo = {lo}")
                scaled_pulse_to_sub = float(lo) * np.array(MEM_AVGS_TO_SUB_ADDR[pixel]) / float(MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"])
                print (f"scaled_pulse_to_sub = {scaled_pulse_to_sub}")


                scaled_deriv_to_sub = float(deriv_max) * np.array(MEM_DERIV_TO_SUB_ADDR[pixel])/float((MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax"]))
                # ゼロパディングを追加して1040 - 8(offset) 要素にする
                padded_array_pulse = np.pad(scaled_pulse_to_sub, (0, 8), 'constant') # for pulse
                padded_array = np.pad(scaled_deriv_to_sub, (0, 8), 'constant') # for deriv

                # 整数 m の要素だけシフトする
                m = deriv_max_i - MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax_i"] 

                # 配列をシフトして結果を代入
                pulse_to_sub = shift_array(padded_array_pulse, m)
                deriv_to_sub = shift_array(padded_array, m)

                mycalc_slow_pulse = 0
                # 現在のパルスについて, slope check を実施
                if SLOPE_DETECT_LEN <= deriv_max_i - PRE_TRIG_LEN_H or \
                                2 * SLOPE_DETECT_LEN <= ipos_fall_end - PRE_TRIG_LEN_H:
                    mycalc_slow_pulse = 1  # slow_pulse フラグ (MSB of rise_time) をセット

                xindex = np.arange(len(deriv_to_sub))[1:-1] # common for pulse and deriv 
                thres_margin = np.abs((deriv_to_sub[2:]-deriv_to_sub[:-2])/2)

                fig.text(0.1,0.94,f"time={time} pt={pt} nt={nt} dmax={dmax} pha={pha} pi={pi} rs={rs} ts={ts}")
                fig.text(0.1,0.92,f"qd={qd} sd={sd} status={status} slow={slow_pulse}") 
                if lo > CLIPTHRES:
                    fig.text(0.8,0.92,f"lo={lo} > {CLIPTHRES}", color="red")
                else:
                    fig.text(0.8,0.92,f"lo={lo}")
                fig.text(0.1,0.90,f"[recalc] (deriv_max_i, deriv_max)=({deriv_max_i}, {deriv_max})")

                axes[0].plot(deriv_to_use,"b-",label=f"itype={itype}", alpha=0.8)
                axes[0].plot(deriv_to_sub,"r--",label="adjusted deriv", alpha=0.6)
                axes[0].vlines(x=deriv_max_i, ymin=0, ymax=deriv_max, color='c', linestyle='--',  label=f"deriv_max_i={deriv_max_i}", lw=1)
#                axes[0].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:],thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:], "-", color='magenta', label="thres_margin", alpha=0.5)
                axes[0].axvline(ipos_fall_end, color='g', linestyle='--', alpha=0.5, label=f"ipos_fall_end = {ipos_fall_end} calc. SP = {mycalc_slow_pulse}", lw=1)

                axes[0].set_ylabel("DERIV")
                axes[0].legend()


                # for pulse, plotting at 2nd axis 
#                ax2 = axes[1].twinx()  # ここで第2軸を作成 
                ax2 = axes[1] # 左に戻す。

                ################### calc lo_res_base (start) ################################
                lo_res_base = calc_lo_res_base(pulse, px_offset_avg_len_pow, px_offset_avg_len, px_offset_avg_gap)
                print(f"[calc lo_res_base] lo_res_base={lo_res_base}")
                lo_res_base_sub = calc_lo_res_base(pulse_to_sub.astype(int), px_offset_avg_len_pow, px_offset_avg_len, px_offset_avg_gap)
                print(f"[calc lo_res_base] lo_res_base_sub={lo_res_base_sub}")
                ################### calc lo_res_base (end) ##################################
                pulse_cor_lores = pulse - lo_res_base # e.g., pulse - (-4060)  = pulse + 4060


                # plot primary pulse
                ax2.plot(pulse_cor_lores, "k-", label=f"pulse - (lo_res_base={lo_res_base})", alpha=0.3, lw=2)
                ax2.plot(pulse_to_sub,"k--",label="adjusted pulse", alpha=0.7)
                pulse_after_sub = pulse_cor_lores - pulse_to_sub
                ax2.plot(pulse_after_sub,"k-",label="pulse - adjusted pulse", alpha=0.7)

                # plot secondary pulse
                # 配列を25シフトして結果を代入(Ls1)
                next_interval = 25
                derivmax_pnt = deriv_max_i + next_interval

                lo_Ls1 = int(pulse_after_sub[derivmax_pnt])
                scaled_pulse_to_sub_Ls1 = float(lo_Ls1) * np.array(MEM_AVGS_TO_SUB_ADDR[pixel]) / float(MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"])
                padded_array_pulse_Ls1 = align_peak(scaled_pulse_to_sub_Ls1, deriv_max_i + next_interval, len(pulse_after_sub))
                ax2.plot(padded_array_pulse_Ls1,"g--",label="adjusted (Ls1)", alpha=0.7)
                pnew_Ls1 = pulse_after_sub - padded_array_pulse_Ls1
                ax2.plot(pnew_Ls1,"g-",label=f"sub pulse (Ls1) lo={int(lo_Ls1)}", alpha=0.7)
                print(f"[Ls1] deriv_max_i {deriv_max_i} lo={lo_Ls1} amax(pnew_Ls1)={len(pnew_Ls1)}")
                ################### calc lo_res_base (start) ################################
                lo_res_base_a = calc_lo_res_base(pulse_cor_lores.astype(int), \
                                              px_offset_avg_len_pow, px_offset_avg_len, px_offset_avg_gap, trigpnt=derivmax_pnt)
                print(f"[calc lo_res_base] lo_res_base_a={lo_res_base_a} at derivmax_pnt {derivmax_pnt}")
                lo_res_base_b = calc_lo_res_base(pulse_to_sub.astype(int), \
                                              px_offset_avg_len_pow, px_offset_avg_len, px_offset_avg_gap, trigpnt=derivmax_pnt)
                print(f"[calc lo_res_base] lo_res_base_b={lo_res_base_b} at derivmax_pnt {derivmax_pnt}")
                ################### calc lo_res_base (end) ##################################                
                lo_res_ph_Ls1 = pulse_cor_lores[derivmax_pnt] - (lo_res_base_a - lo_res_base_b) 
                print(f"[calc lo_res_base] lo_res_ph_Ls1={lo_res_ph_Ls1} (={lo_Ls1} - ({lo_res_base_a} - {lo_res_base_b}), (={lo_Ls1} - {lo_res_base_a - lo_res_base_b})")

                # Ls2
                next_interval = 25 + 25
                lo_Ls2 = int(pnew_Ls1[deriv_max_i + next_interval])
                scaled_pulse_to_sub_Ls2 = float(lo_Ls2) * np.array(MEM_AVGS_TO_SUB_ADDR[pixel]) / float(MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"])
                padded_array_pulse_Ls2 = align_peak(scaled_pulse_to_sub_Ls2, deriv_max_i + next_interval, len(pulse_after_sub))
                ax2.plot(padded_array_pulse_Ls2,"m--",label=f"adjusted (Ls2) lo={lo_Ls2}, ", alpha=0.7)
                pnew_Ls2 = pnew_Ls1 - padded_array_pulse_Ls2
                ax2.plot(pnew_Ls2,"m-",label="sub pulse (Ls2)", alpha=0.7)
                print(f"[Ls2] deriv_max_i {deriv_max_i} lo={lo_Ls2} amax(pnew_Ls2)={len(pnew_Ls2)}")

                # Ls3
                next_interval = 25 + 25 + 25
                lo_Ls3 = int(pnew_Ls2[deriv_max_i + next_interval])
                scaled_pulse_to_sub_Ls3 = float(lo_Ls3) * np.array(MEM_AVGS_TO_SUB_ADDR[pixel]) / float(MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"])
                padded_array_pulse_Ls3 = align_peak(scaled_pulse_to_sub_Ls3, deriv_max_i + next_interval, len(pulse_after_sub))
                ax2.plot(padded_array_pulse_Ls3,"c--",label=f"adjusted (Ls3) lo={lo_Ls3}", alpha=0.7, lw=1)
                pnew_Ls3 = pnew_Ls2 - padded_array_pulse_Ls3
                ax2.plot(pnew_Ls3,"c-",label="sub pulse (Ls3)", alpha=0.7, lw=1)
                print(f"[Ls3] deriv_max_i {deriv_max_i} lo={lo_Ls3} amax(pnew_Ls3)={len(pnew_Ls3)}")

                # Ls4
                next_interval = 25 + 25 + 25 + 25
                lo_Ls4 = int(pnew_Ls3[deriv_max_i + next_interval])
                scaled_pulse_to_sub_Ls4 = float(lo_Ls4) * np.array(MEM_AVGS_TO_SUB_ADDR[pixel]) / float(MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"])
                padded_array_pulse_Ls4 = align_peak(scaled_pulse_to_sub_Ls4, deriv_max_i + next_interval, len(pulse_after_sub))
                ax2.plot(padded_array_pulse_Ls4,"b--",label=f"adjusted (Ls4) lo={lo_Ls4}", alpha=0.7)
                pnew_Ls4 = pnew_Ls3 - padded_array_pulse_Ls4
                ax2.plot(pnew_Ls4,"b-",label="sub pulse (Ls4)", alpha=0.7)
                print(f"[Ls4] deriv_max_i {deriv_max_i} lo={lo_Ls4} amax(pnew_Ls4)={len(pnew_Ls4)}")


                # calc lo_res_base
#                ax2.hlines(y=lo_res_base, xmin=0, xmax=RECORD_LEN, color='c', linestyle='--', alpha=0.4, label="lo_res_base")
#                ax2.hlines(y=lo_res_base_sub, xmin=0, xmax=RECORD_LEN, color='c', linestyle='--', alpha=0.4, label="lo_res_base_sub")
                ax2.set_ylabel("Pulse")
                ax2.legend(loc='best')


                ############### prep. for derib_sub #######################################################
                deriv_after_sub = deriv_to_use - deriv_to_sub

                thre_2nd = deriv_max/SECOND_THRES_FRAC if deriv_max/SECOND_THRES_FRAC > PULSE_THRES else PULSE_THRES

                trig_state = "STATE_PXP_ARMED"
                ipos_init_2nd = deriv_max_i + SECOND_TRIG_GAP_LEN
                ipos_limit = ipos_init_2nd + SLOPE_DETECT_LEN # Initialize 
                print(f"trig_state is set {trig_state}.")
                deriv_pre = deriv_after_sub[ipos_limit - 1]                

                derivmax_2nd = 0.

                for ipos in np.arange(ipos_init_2nd,RECORD_LEN):
                    derivative = deriv_after_sub[ipos]                                        
                    print(f"ipos, deriv_pre, derivative, ipos_limit = {ipos}, {deriv_pre}, {derivative}, {ipos_limit}")
                    if trig_state == "STATE_PXP_ARMED":
                        if ipos >= ipos_limit:
                            break
                        elif derivative < deriv_pre:
                            derivmax_2nd = deriv_pre
                            derivmax_2nd_i = ipos - 1 
                            trig_state = "STATE_PXP_FALL"
                            print(f"trig_state is set {trig_state}.")
                            ipos_limit += SLOPE_DETECT_LEN
                        else:
                            deriv_pre = derivative
                    
                    elif trig_state == "STATE_PXP_FALL":
                        if ipos >= ipos_limit:
                            break
                        elif derivative < FALL_END_THRES:
                            trig_state = "STATE_PXP_READY"
                            print(f"trig_state is set {trig_state}.")
                            break

#                anum = 1 
                anum = 2

                axes[anum].plot(deriv_after_sub,color='skyblue', label="deriv - adj.deriv")

                axes[anum].set_xlabel("TIME (TICKS)")
                axes[anum].set_ylabel("DERIV - adjusted DERIV")

                axes[anum].plot(derivmax_2nd_i, derivmax_2nd, "o", color='g', alpha=0.5, label=f"2nd derivmax at ({derivmax_2nd_i:d},{derivmax_2nd:.1f})")

                if derivative > FALL_END_THRES:
                    axes[anum].plot(ipos, derivative, "o", color='b', alpha=0.5, label=f"last point of 2nd search ({ipos:d},{derivative:.1f})")
                else:
                    axes[anum].plot(ipos, derivative, "o", color='r', alpha=0.5, label=f"last point of 2nd search ({ipos:d},{derivative:.1f})")                    

                axes[anum].axvline(ipos_limit, color='r', linestyle='--', alpha=0.5, label=f"ipos_limit with trig_state = {trig_state}")
#                axes[1].vlines(x=ipos_limit, ymin=0, ymax=derivmax_2nd + 1000, color='r', linestyle='--', alpha=0.5, label=f"ipos_limit with trig_state = {trig_state}")

                axes[anum].hlines(y=deriv_max/SECOND_THRES_FRAC, xmin=deriv_max_i + SECOND_TRIG_GAP_LEN, xmax=deriv_max_i+SECOND_THRES_USE_LEN, color='c', linestyle='--', alpha=0.8, label="deriv_max/SECOND_THRES_FRAC")
                axes[anum].hlines(y=PULSE_THRES, xmin=deriv_max_i + SECOND_TRIG_GAP_LEN, xmax=1024, color='y', linestyle='--', alpha=0.8, label = "PULSE_THRES")

                axes[anum].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:deriv_max_i+SECOND_THRES_USE_LEN],\
                             deriv_max/SECOND_THRES_FRAC + thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:deriv_max_i+SECOND_THRES_USE_LEN], \
                              "-", color='c', label="thres_margin + deriv_max/SECOND_THRES_FRAC", alpha=0.4)

                axes[anum].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:],\
                             PULSE_THRES + thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:], \
                              "-", color='y', label="thres_margin + PULSE_THRES", alpha=0.4)

                nega_thre = -2 * (PULSE_THRES + thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:])
                axes[anum].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:],\
                             nega_thre, "-", color='y', label="-2 * (thres_margin + PULSE_THRES)", alpha=0.4)

                axes[0].grid(alpha=0.3, ls=':')
                axes[anum].grid(alpha=0.3, ls=':')
                axes[anum].legend()
                axes[anum].legend(loc='lower right')

                if xlims:
                    axes[0].set_xlim(xlims)
                if ylims:
                    axes[0].set_ylim(ylims)

                fig.text(0.75,0.05,f"SLOPE_DETECT_LEN={SLOPE_DETECT_LEN}", fontsize=8,color="gray")
                fig.text(0.75,0.03,f"SECOND_TRIG_GAP_LEN={SECOND_TRIG_GAP_LEN}", fontsize=8,color="gray")
                fig.text(0.75,0.01,f"SECOND_THRES_USE_LEN={SECOND_THRES_USE_LEN}", fontsize=8,color="gray")
                fig.text(0.05,0.02,f"fname = {prevt}", fontsize=8)

                plt.suptitle(f"ID={obsid} {target} DATE-OBS={dateobs}   PIXEL={pixel} ITYPE={itype_str}")
#                plt.tight_layout()
                ofile = f"{oname}_{k:05d}_pixel{pixel:02d}.png"
                plt.savefig(ofile)
                print(f'.... {ofile} is saved.')

                if plotflag: 
                    plt.show()

def get_itype_str(itype):
    """
    Returns a string representation for a given itype.

    Args:
    itype (int): The itype value.

    Returns:
    str: The string representation of the itype.
    """
    itype_dict = {-1: 'all', 0: 'Hp', 1: 'Mp', 2: 'Ms', 3: 'Lp', 4: 'Ls', 5: 'BL', 6: 'EL', 7: '--'}
    return itype_dict.get(itype, 'unknown')

def dump_to_npz(filename, x_time, pulses):
    """
    Dumps the x_time and pulse data to a NPZ file.

    Args:
    filename (str): The output filename.
    x_time (numpy.ndarray): The time array.
    pulses (numpy.ndarray): The pulse data.
    """
    np.savez(filename, x_time=x_time, pulses=pulses)

def load_from_npz(filename):
    """
    Loads the x_time and pulse data from a NPZ file.

    Args:
    filename (str): The input filename.

    Returns:
    tuple: The time array and pulse data.
    """
    data = np.load(filename)
    return data['x_time'], data['pulses']

def parse_limits(limit_str):
    """
    Parses a comma-separated string into a tuple of floats.

    Args:
    limit_str (str): Comma-separated string of limits.

    Returns:
    tuple: Tuple of float limits.
    """
    try:
        print(limit_str)
        return tuple(map(float, limit_str.split(',')))
    except ValueError:
        print(limit_str)
        raise argparse.ArgumentTypeError("Limits must be comma-separated floats")

def parse_pixel_range(pixel_range_str):
    """
    Parse a comma-separated list of pixel numbers into a list of integers.

    Args:
        pixel_range_str (str): Comma-separated pixel numbers.

    Returns:
        list: List of pixel numbers as integers.
    """
    if pixel_range_str is None:
        return None
    return [int(p) for p in pixel_range_str.split(',')]


def load_hk2(hk2_file, debug=False):
    global MEM_DERIV_TO_SUB_ADDR, MEM_DERIV_TO_SUB_ADDR_DERIVMAX  # グローバル変数の宣言
    global MEM_AVGS_TO_SUB_ADDR, MEM_AVGS_TO_SUB_ADDR_MAX  # グローバル変数の宣言

    print("Opening FITS file...")
    hdu = fits.open(hk2_file)
    ftag = hk2_file.replace(".hk2", "")    
    # Extract metadata
    header = hdu[1].header
    date_obs = header["DATE-OBS"]
    
    # Read average pulse and derivative data from FITS
    print("Reading average pulse and derivative data...")
    data = hdu[4].data
    avgs = data["AVGPULSE"]
    derivs = data["AVGDERIV"]
    times = data["TIME"]
    pixels = data["PIXEL"]

    # グローバル変数に代入
    MEM_AVGS_TO_SUB_ADDR = {pixel: avgs[np.where(pixels == pixel)[0]][0] for pixel in range(36)}

    MEM_DERIV_TO_SUB_ADDR = {pixel: derivs[np.where(pixels == pixel)[0]][0] for pixel in range(36)}

    # 各ピクセルの波形データに基づく特徴量（最大値）を計算して別の辞書で管理
    MEM_AVGS_TO_SUB_ADDR_MAX = {pixel: {"adc_sample_max": np.max(waveform),"adc_sample_max_i": np.argmax(waveform)} for pixel, waveform in MEM_AVGS_TO_SUB_ADDR.items()}
    MEM_DERIV_TO_SUB_ADDR_DERIVMAX = {pixel: {"derivmax": np.max(waveform),"derivmax_i": np.argmax(waveform)} for pixel, waveform in MEM_DERIV_TO_SUB_ADDR.items()}

    if debug:
        # MEM_DERIV_TO_SUB_ADDR のデータをプロット（ピクセルごとの波形）
        plt.figure(figsize=(10, 6))
        for pixel in range(36):
            plt.plot(MEM_AVGS_TO_SUB_ADDR[pixel], label=f'Pixel {pixel}', alpha=0.5)

        plt.title('Average waveform Data for Each Pixel')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.show()

        plt.figure(figsize=(10, 6))
        for pixel in range(36):
            plt.plot(MEM_DERIV_TO_SUB_ADDR[pixel], label=f'Pixel {pixel}', alpha=0.5)

        plt.title('Derivative for Each Pixel')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.show()

        # MEM_AVGS_TO_SUB_ADDR_DERIVMAX の最大値（adc_sample_max）をピクセルごとにプロット
        adc_sample_max_values = [MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max"] for pixel in range(36)]
        derivmax_values = [MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax"] for pixel in range(36)]
        plist = np.arange(36)
        plt.figure(figsize=(10, 6))
        plt.plot(plist, adc_sample_max_values, "s", color='red', label="adc_sample_max_values")
        plt.plot(plist, derivmax_values, "o", color='skyblue', label="derivmax_values")
        plt.title('Maximum AVGS Value for Each Pixel')
        plt.xlabel('Pixel')
        plt.ylabel('Maximum Value')
        plt.xticks(plist)
        plt.legend()
        plt.show()

        # MEM_DERIV_TO_SUB_ADDR_DERIVMAX 最大値（derivmax）の時の i をピクセルごとにプロット
        derivmax_values_i = [MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax_i"] for pixel in range(36)]
        adc_sample_max_i = [MEM_AVGS_TO_SUB_ADDR_MAX[pixel]["adc_sample_max_i"] for pixel in range(36)]

        plist = np.arange(36)
        plt.figure(figsize=(10, 6))
        plt.plot(plist, adc_sample_max_i, "s", color='red',label="adc_sample_max_i")
        plt.plot(plist, derivmax_values_i, "o", color='skyblue',label="derivmax_values_i")
        plt.title('Maximum Deriv Value of i for Each Pixel')
        plt.xlabel('Pixel')
        plt.ylabel('Maximum Value')
        plt.xticks(plist)
        plt.legend()
        plt.show()
    # global variables are set : MEM_DERIV_TO_SUB_ADDR, MEM_DERIV_TO_SUB_ADDR_DERIVMAX, MEM_AVGS_TO_SUB_ADDR, MEM_AVGS_TO_SUB_ADDR_MAX


def main():
    """
    Main function to parse arguments and call the plot_data_6x6 function.
    """
    parser = argparse.ArgumentParser(
      description='This program is to plot pulserecord',
      epilog='''
        Example 1) just plot 
        resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext.evt
        Example 2) plot with ranges:
        resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_below30.evt --prevflag --xlims 0,500 --ylims=-5000,-2000      
        [Note] --ylims=-5000,-2000 should work but --ylims -5000,-2000 NOT work. 
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('prevt', type=str, help='Input FITS file')
    parser.add_argument('--itypelist', '-i', type=str, default='0,1,2,3,4', help='Comma-separated list of itype values (default: 0,1,2,3,4)')
    parser.add_argument('--dumptext', action='store_true', help='Flag to dump x_time and pulse data to NPZ files')
    parser.add_argument('--plot', '-p', action='store_true', help='Flag to plot')
    parser.add_argument('--xlims', '-x', type=parse_limits, help='Comma-separated x-axis limits (xmin,xmax)')
    parser.add_argument('--ylims', '-y', type=parse_limits, help='Comma-separated y-axis limits (ymin,ymax)')
    parser.add_argument('--prevflag', '-pr', action='store_true', help='Flag to plot previous interval values as text')
    parser.add_argument('--deriv', '-dr', action='store_true', help='Flag to plot derivative')
    parser.add_argument('--usetime', '-t', action='store_true', help='Flag to usetime')
    parser.add_argument('--step', '-s', type=int, default='8', help='Step size to created deriavtive')
    parser.add_argument('--check_qd', '-c', action='store_true', help='check qd by myself')
    parser.add_argument('--pixels', type=str, help='Comma-separated list of pixel numbers to process')
    parser.add_argument('--hk2', default=resolve_tools +'/resolve/ana/pixel/xa035315064rsl_a0.hk2', type=str, help='Comma-separated list of pixel numbers to process')
    parser.add_argument('--debug', '-d', action='store_true', help='debug flag')

    args = parser.parse_args()
    itypes = [int(itype) for itype in args.itypelist.split(',')]

    selected_pixels = parse_pixel_range(args.pixels)

    load_hk2(args.hk2, debug=args.debug)
    
    plot_deriv(args.prevt, itypes, args.dumptext, \
        plotflag=args.plot, usetime=args.usetime, prevflag=args.prevflag, xlims=args.xlims, ylims=args.ylims, \
                    usederiv=args.deriv, step=args.step, check_qd=args.check_qd, selected_pixels=selected_pixels)

if __name__ == "__main__":
    main()
