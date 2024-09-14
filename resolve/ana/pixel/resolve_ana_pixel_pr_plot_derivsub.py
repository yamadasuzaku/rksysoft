#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from pathlib import Path
import os 

# 環境変数 "RESOLVETOOLS" を取得
resolve_tools = os.getenv("RESOLVETOOLS")
# 環境変数が設定されていない場合にエラーを出して終了
if resolve_tools is None:
    print("Error: RESOLVETOOLS environment variable is not set.", file=sys.stderr)
    sys.exit(1)
# 環境変数が正しく取得された場合に処理を続行
print(f"RESOLVETOOLS is set to: {resolve_tools}")

# Set plot parameters for consistent styling
params = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

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

def get_deriv(ydata, step=8):
    # Initialize an empty list to store the derivative values
    ydatalc = []
    
    # Create an array of indices from 0 to the length of ydata
    numarray = np.arange(0, len(ydata), 1)

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

        # Append the derivative to the list
        ydatalc.append(derivative)

    # Convert the list to a numpy array
    npydatalc = np.array(ydatalc)

    # Find the maximum and minimum derivative values within the range of 100 to 200 indices
    deriv_max = np.amax(npydatalc[np.where((numarray > 100) & (numarray < 200))])
    deriv_min = np.amin(npydatalc[np.where((numarray > 100) & (numarray < 200))])

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

    pulse_list = data[1].data['PULSEREC']

    time_list = data[1].data['TIME']
    pixel_list = data[1].data['PIXEL']
    itype_list = data[1].data['ITYPE']
    prev_list = data[1].data['PREV_INTERVAL']
    next_list = data[1].data['NEXT_INTERVAL']
    dmax_list = data[1].data['DERIV_MAX']
    pha_list = data[1].data['PHA']
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
            risetimes = risetime_list[cutid]
            tickshifts = tickshift_list[cutid]

            qds = qd_list_int[cutid]
            sds = sd_list_int[cutid]
            los = lores_list[cutid]
            statuss = status_list_int[cutid]

            num_of_evt = len(pulses)        
            if num_of_evt > 0:
                print(f'PIXEL={pixel:02d}, N={num_of_evt}')

            for k, (pulse, time, pt, nt, dmax, pha, rs, ts, qd, sd, lo, status) in enumerate(zip(pulses,times, prevs,nexts, dmaxs, phas, risetimes, tickshifts,qds,sds,los, statuss)):
                print("pulse =", pulse)
                print(f"k, time, pt, nt, dmax, pha, rs, ts, qd, sd, lo, status = {k}, {time}, {pt}, {nt}, {dmax}, {pha}, {rs}, {ts}, {qd}, {sd}, {lo}, {status}")
                slow_pulse = 1 if rs > 127 else 0                
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
                plt.subplots_adjust(right=0.9, top=0.85)

                pulse = pulse[SPARE_LEN:]

                deriv_to_use, deriv_max, deriv_max_i, _, _, _, _ = get_deriv(pulse)
                print(f"{k} :  (deriv_max_i,deriv_max)={deriv_max_i},{deriv_max}")
                scaled_deriv_to_sub = deriv_max * MEM_DERIV_TO_SUB_ADDR[pixel]/(MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax"])
                # ゼロパディングを追加して1040 - 8(offset) 要素にする
                padded_array = np.pad(scaled_deriv_to_sub, (0, 8), 'constant')
                # 整数 m の要素だけシフトする（m = 50 で例を示します）
                m = deriv_max_i - MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax_i"]
                shifted_array = np.zeros_like(padded_array)  # ゼロで初期化した配列
                # シフトの処理（はみ出た部分はゼロ）
                if m > 0:
                    shifted_array[m:] = padded_array[:-m]  # 前方向にシフトして、はみ出た部分はゼロ
                elif m < 0:                    
                    shifted_array[:m] = padded_array[-m:]  # 後ろ方向にシフトして、はみ出た部分はゼロ
                else: # m = 0
                    shifted_array = padded_array
                deriv_to_sub = shifted_array

                xindex = np.arange(len(deriv_to_sub))[1:-1]
                thres_margin = np.abs((deriv_to_sub[2:]-deriv_to_sub[:-2])/2)

                fig.text(0.1,0.93,f"time={time} pt={pt} nt={nt} dmax={dmax} pha={pha} rs={rs} ts={ts}")
                fig.text(0.1,0.90,f"qd={qd} sd={sd} status={status} slow={slow_pulse}") 
                if lo > CLIPTHRES:
                    fig.text(0.8,0.90,f"lo={lo} > {CLIPTHRES}", color="red")
                else:
                    fig.text(0.8,0.90,f"lo={lo}")
                fig.text(0.1,0.87,f"[recalc] (deriv_max_i, deriv_max)=({deriv_max_i}, {deriv_max})")

                axes[0].plot(deriv_to_use,"b-",label=f"itype={itype}", alpha=0.8)
                axes[0].plot(deriv_to_sub,"r--",label="adjusted deriv", alpha=0.6)
                axes[0].vlines(x=deriv_max_i, ymin=0, ymax=deriv_max, color='c', linestyle='--')
                axes[0].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:],thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:], "-", color='magenta', label="thres_margin", alpha=0.5)

                axes[0].set_ylabel("DERIV")
                axes[0].legend()

                # 第二軸の追加
                ax2 = axes[0].twinx()  # ここで第2軸を作成
                ax2.plot(pulse, "y-", label="pulse", alpha=0.6)
                ax2.set_ylabel("Pulse")
                ax2.legend(loc='lower right')

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

                axes[1].plot(deriv_after_sub,color='skyblue', label="deriv - adj.deriv")

                axes[1].set_xlabel("TIME (TICKS)")
                axes[1].set_ylabel("DERIV - adjusted DERIV")

                axes[1].plot(derivmax_2nd_i, derivmax_2nd, "o", color='g', alpha=0.5, label=f"2nd derivmax at ({derivmax_2nd_i:d},{derivmax_2nd:.1f})")

                if derivative > FALL_END_THRES:
                    axes[1].plot(ipos, derivative, "o", color='b', alpha=0.5, label=f"last point of 2nd search ({ipos:d},{derivative:.1f})")
                else:
                    axes[1].plot(ipos, derivative, "o", color='r', alpha=0.5, label=f"last point of 2nd search ({ipos:d},{derivative:.1f})")                    

                axes[1].axvline(ipos_limit, color='r', linestyle='--', alpha=0.5, label=f"ipos_limit with trig_state = {trig_state}")
#                axes[1].vlines(x=ipos_limit, ymin=0, ymax=derivmax_2nd + 1000, color='r', linestyle='--', alpha=0.5, label=f"ipos_limit with trig_state = {trig_state}")


                axes[1].hlines(y=deriv_max/SECOND_THRES_FRAC, xmin=deriv_max_i + SECOND_TRIG_GAP_LEN, xmax=deriv_max_i+SECOND_THRES_USE_LEN, color='c', linestyle='--', alpha=0.8, label="deriv_max/SECOND_THRES_FRAC")
                axes[1].hlines(y=PULSE_THRES, xmin=deriv_max_i + SECOND_TRIG_GAP_LEN, xmax=1024, color='y', linestyle='--', alpha=0.8, label = "PULSE_THRES")

                axes[1].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:deriv_max_i+SECOND_THRES_USE_LEN],\
                             deriv_max/SECOND_THRES_FRAC + thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:deriv_max_i+SECOND_THRES_USE_LEN], \
                              "-", color='c', label="thres_margin + deriv_max/SECOND_THRES_FRAC", alpha=0.4)

                axes[1].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:],\
                             PULSE_THRES + thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:], \
                              "-", color='y', label="thres_margin + PULSE_THRES", alpha=0.4)

                nega_thre = -2 * (PULSE_THRES + thres_margin[deriv_max_i + SECOND_TRIG_GAP_LEN:])
                axes[1].plot(xindex[deriv_max_i + SECOND_TRIG_GAP_LEN:],\
                             nega_thre, "-", color='y', label="-2 * (thres_margin + PULSE_THRES)", alpha=0.4)


                axes[1].grid(alpha=0.5)
                axes[1].legend()
                axes[1].legend(loc='upper right')

                if xlims:
                    axes[0].set_xlim(xlims)
                if ylims:
                    axes[0].set_ylim(ylims)

                fig.text(0.55,0.02,f"SECOND_TRIG_GAP_LEN={SECOND_TRIG_GAP_LEN}", fontsize=8,color="gray")
                fig.text(0.75,0.02,f"SECOND_THRES_USE_LEN={SECOND_THRES_USE_LEN}", fontsize=8,color="gray")
                fig.text(0.05,0.02,f"fname = {prevt}", fontsize=8)

                plt.suptitle(f"obs={obsid} tname={target} pix={pixel} itype={itype_str}")
#                plt.tight_layout()
                ofile = f"{oname}_{k:05d}.png"
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
    MEM_DERIV_TO_SUB_ADDR = {pixel: derivs[np.where(pixels == pixel)[0]][0] for pixel in range(36)}

    # 各ピクセルの波形データに基づく特徴量（最大値）を計算して別の辞書で管理
    MEM_DERIV_TO_SUB_ADDR_DERIVMAX = {pixel: {"derivmax": np.max(waveform),"derivmax_i": np.argmax(waveform)} for pixel, waveform in MEM_DERIV_TO_SUB_ADDR.items()}

    if debug:
        # MEM_DERIV_TO_SUB_ADDR のデータをプロット（ピクセルごとの波形）
        plt.figure(figsize=(10, 6))
        for pixel in range(36):
            plt.plot(MEM_DERIV_TO_SUB_ADDR[pixel], label=f'Pixel {pixel}', alpha=0.5)

        plt.title('Waveform Data for Each Pixel')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.show()

        # MEM_DERIV_TO_SUB_ADDR_DERIVMAX の最大値（derivmax）をピクセルごとにプロット
        derivmax_values = [MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax"] for pixel in range(36)]
        plist = np.arange(36)
        plt.figure(figsize=(10, 6))
        plt.plot(plist, derivmax_values, "o", color='skyblue')
        plt.title('Maximum Deriv Value for Each Pixel')
        plt.xlabel('Pixel')
        plt.ylabel('Maximum Value')
        plt.xticks(plist)
        plt.show()

        # MEM_DERIV_TO_SUB_ADDR_DERIVMAX 最大値（derivmax）の時の i をピクセルごとにプロット
        derivmax_values_i = [MEM_DERIV_TO_SUB_ADDR_DERIVMAX[pixel]["derivmax_i"] for pixel in range(36)]
        plist = np.arange(36)
        plt.figure(figsize=(10, 6))
        plt.plot(plist, derivmax_values_i, "o", color='green')
        plt.title('Maximum Deriv Value of i for Each Pixel')
        plt.xlabel('Pixel')
        plt.ylabel('Maximum Value')
        plt.xticks(plist)
        plt.show()


    return MEM_DERIV_TO_SUB_ADDR, MEM_DERIV_TO_SUB_ADDR_DERIVMAX


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

    args = parser.parse_args()
    itypes = [int(itype) for itype in args.itypelist.split(',')]

    selected_pixels = parse_pixel_range(args.pixels)

    load_hk2(args.hk2)
    
    plot_deriv(args.prevt, itypes, args.dumptext, \
        plotflag=args.plot, usetime=args.usetime, prevflag=args.prevflag, xlims=args.xlims, ylims=args.ylims, \
                    usederiv=args.deriv, step=args.step, check_qd=args.check_qd, selected_pixels=selected_pixels)

if __name__ == "__main__":
    main()
