#!/usr/bin/env python 

import ast
import argparse
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator, LogLocator, ScalarFormatter
import glob
import sys
import pandas as pd
import subprocess
import astropy.io.fits as fits
import os
import numpy as np

def ev_to_pi(ev):
    # PI = (E [eV] -0.5)*2
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    # E [eV] = PI*0.5+0.5
    return pi * 0.5 + 0.5

def pha_to_csv(input_pha):
    data = fits.open(input_pha)[1].data
    pi = data["CHANNEL"]
    grouping = data["GROUPING"]
    df = pd.DataFrame({
        'PI': pi,
        'GROUPING': grouping
    })
    input_pha_name = os.path.basename(input_pha).replace(".pha", "")
    output_csv = f"grouping_csv_for_{input_pha_name}.csv"
    df.to_csv(output_csv, index=False)

    print("-----------------------")
    print(f"Created ---> {output_csv}")
    print("-----------------------")


""" 指定したエネルギー範囲を指定したエネルギービンでビンまとめするための関数 """  #単体でも使えるように引数のdefaultの値は関数の引数でも設定
# ftgrouppha を使ってbaseのbinnningを作成
def grouping_by_ftgrouppha(input_pha, grouptype="opt", input_rmf=None, backfile=None, groupscale=None):
    outputname = input_pha.replace(".pha", "").replace("_v0", "")
    output_pha = f"{outputname}_{grouptype}.pha"

    try:
        print("-------> run ftgrouppha")
        print(f"input pha file : {input_pha}")
        print(f"output pha file : {output_pha}")
        print(f"grouptype : {grouptype}")
        ftgrouppha_cmd = f"ftgrouppha infile={input_pha} outfile={output_pha} grouptype={grouptype} clobber=True"
        if grouptype == "opt":
            data_header = fits.open(input_pha)[1].header
            respfile = data_header["RESPFILE"]
            
            if input_rmf is None:
                if respfile == "none":
                    print("Input rmf file")
                    sys.exit()
                input_rmf = respfile
            else:
                input_rmf = respfile if respfile != "none" else input_rmf

            print(f"rmf file : {input_rmf}")
            ftgrouppha_cmd += f" respfile={input_rmf}"
        
        if grouptype == "snmin" or grouptype == "bmin":
            if backfile is not None:
                print("input back file")
                sys.exit()
            else:
                print(f"back file : {backfile}")
                ftgrouppha_cmd = ftgrouppha_cmd + f" backfile={backfile}"

        if grouptype == "min" or grouptype == "constant" or grouptype == "snmin" or grouptype == "bmin":
            if groupscale is None:
                print("input groupscale")
                sys.exit()
            else:
                print(f"groupscale : {groupscale}")
                ftgrouppha_cmd = ftgrouppha_cmd + f" groupscale={groupscale}"
        result = subprocess.run(ftgrouppha_cmd, shell=True, capture_output=True, check=True)
        print("ftgouppha successfully")

    except subprocess.CalledProcessError as e:
        print("標準エラー:", e.stderr)
        sys.exit()
    
    print("---------------")
    return output_pha

def binning(start_ene, stop_ene, ebinsize, base_binning, all_grouping_flag_array):
    allpilist = np.arange(0, 60000, 1)
    base_bin_start_pi = np.where(base_binning==1)[0]
    
    start_pi = ev_to_pi(start_ene)
    stop_pi = ev_to_pi(stop_ene)
    
    # 指定したエネルギーのstartまでoptのgroupingを採用
    start_pi = base_bin_start_pi[base_bin_start_pi <= start_pi][-1] #start_piの更新

    # 指定したエネルギー範囲を指定したエネルギー幅でgrouping
    rbin_factor = int(ebinsize/0.5)
    rbin_start_pi_list = np.arange(start_pi, 60000+rbin_factor, rbin_factor)

    # base binnningと指定したエネルギー範囲のrbinと繋ぐ
    basebin_and_rbin_start_pi = base_bin_start_pi[np.isin(base_bin_start_pi, rbin_start_pi_list)] 
    stop_pi = basebin_and_rbin_start_pi[stop_pi<=basebin_and_rbin_start_pi][0]

    print("reset")
    print(f"start : {pi_to_ev(start_pi)} eV   {start_pi} channel")
    print(f"stop : {pi_to_ev(stop_pi)} eV   {stop_pi} channel")

    rbin_pilist_new = np.arange(start_pi, stop_pi+rbin_factor, rbin_factor) # 改めて、start、stopのpiを取得し、rbin_factorでbinning
    
    # ビンまとめのstartのchannelのflagを1にかえる
    all_grouping_flag_array[np.isin(allpilist, rbin_pilist_new)] = 1

    # startとstopの間を-1にかえる
    zero_flag_pi = np.where(all_grouping_flag_array==0)[0]
    zero_flag_pi = zero_flag_pi[(start_pi <= zero_flag_pi) & (zero_flag_pi <= stop_pi)]
    all_grouping_flag_array[zero_flag_pi] = -1
    
    return all_grouping_flag_array, pi_to_ev(start_pi), pi_to_ev(stop_pi)

def change_0flag_to_originalbinning(all_grouping_flag_array, base_binning):
    zero_flag_pi = np.where(all_grouping_flag_array==0)[0]
    all_grouping_flag_array[zero_flag_pi] = base_binning[zero_flag_pi]
    return all_grouping_flag_array

""" 本体 """
# baseのbinningから「指定したエネルギー範囲を指定したエネルギービンでビンまとめ」に更新する 出力はcsv
def main_create_binned_spectrum_in_csv(input_pha, start_enes=6200, stop_enes=8100, ebinsizes=2.0, base_binning_pha=None, grouptype="opt", input_rmf=None, backfile=None, groupscale=None):
    if start_enes is None and stop_enes is None:
        base_binning_pha = grouping_by_ftgrouppha(input_pha=input_pha, grouptype=grouptype, input_rmf=input_rmf, backfile=backfile, groupscale=groupscale)
        base_spectrum = fits.open(base_binning_pha)[1].data
        base_binning = base_spectrum["GROUPING"]

        suf = f"{grouptype}"
        all_grouping_flag_array = base_binning
    else:
            
        if not isinstance(start_enes, list):
            start_enes = [start_enes]
        if not isinstance(stop_enes, list):
            stop_enes = [stop_enes]
        if not isinstance(ebinsizes, list):
            ebinsizes = [ebinsizes]

        if not ((len(start_enes) == len(stop_enes)) & (len(start_enes) == len(ebinsizes))):
            print("start_ene, stop_ene and ebinsize must be same size")
            sys.exit()
        
        print("-----------------------")
        print(f"input pha file : {input_pha}")
        print(f"start : {start_enes} eV   {ev_to_pi(np.array(start_enes))} channel")
        print(f"stop : {stop_enes} eV   {ev_to_pi(np.array(stop_enes))} channel")
        print(f"Energy binsize : {ebinsizes}")
        print("-----------------------")

        # opt binnningを基にgroupingを作成していく
        if base_binning_pha is None:
            base_binning_pha = grouping_by_ftgrouppha(input_pha=input_pha, grouptype=grouptype, input_rmf=input_rmf, backfile=backfile, groupscale=groupscale)
        base_spectrum = fits.open(base_binning_pha)[1].data
        base_binning = base_spectrum["GROUPING"]
        

        # 初期値
        all_grouping_flag_array = np.zeros_like(np.arange(0, 60000, 1))
        reset_start_ene = start_enes[0]
        reset_stop_ene = stop_enes[0]
        suf = f"{grouptype}"
        for i, (start_ene, stop_ene, ebinsize) in enumerate(zip(start_enes, stop_enes, ebinsizes)):
            if (0< i) and (start_ene < reset_stop_ene):
                start_ene = reset_stop_ene
            
            ebinsuf = str(ebinsize).replace(".", "p")
            suf = suf + f"ene{start_ene}to{stop_ene}binned{ebinsuf}eV"            

            all_grouping_flag_array, reset_start_ene, reset_stop_ene = binning(start_ene=start_ene, stop_ene=stop_ene, ebinsize=ebinsize, base_binning=base_binning, all_grouping_flag_array=all_grouping_flag_array)

        all_grouping_flag_array = change_0flag_to_originalbinning(all_grouping_flag_array=all_grouping_flag_array, base_binning=base_binning)
    
    ## CSVの作成
    input_pha_name = os.path.basename(input_pha).replace(".pha", "").replace("_v0", "")
    df = pd.DataFrame({
        'Energy [eV]': pi_to_ev(np.arange(0, 60000, 1)),
        'PI': np.arange(0, 60000, 1),
        'GROUPING': all_grouping_flag_array
    })
    output_csv = f"grouping_csv_{suf}_for_{input_pha_name}.csv"
    # output_csv = f"grouping_csv_{suf}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Created ---> {output_csv}")
    print("-----------------------")

    return output_csv, suf


""" csvからpha、grpphaスクリプトを作る関数 """
def make_binned_pha_from_csv(input_pha, input_csv, suffix="output"):
    df = pd.read_csv(input_csv)
    grouping = df['GROUPING'].values
    
    name = os.path.basename(input_pha).replace(".pha", "").replace("_v0", "")
    output_pha = f"{name}_{suffix}.pha"
    
    
    # データの取得
    hdulist = fits.open(input_pha) #全HDUを取得
    spectrum = hdulist[1].data

    # 現在のカラムを取得して、新しいカラムを追加
    cols = spectrum.columns
    new_col = fits.Column(name='GROUPING', array=grouping, format='K')  # 'K'は整数型
    new_cols = cols + new_col  # 既存のカラムと新しいカラムを結合


    # 新しいデータテーブルを作成し、元のヘッダー情報を保持しながら差し替え
    new_hdu = fits.BinTableHDU.from_columns(new_cols, header=hdulist[1].header, name=hdulist[1].name)

    # 新しいHDUListを作成し、元の全HDUをコピー、編集したHDUを差し替え
    new_hdulist = fits.HDUList(hdulist)
    new_hdulist[1] = new_hdu

    new_hdulist.writeto(output_pha, overwrite=True)
    hdulist.close()
    new_hdulist.close()

    print(f"Created ----> {output_pha}")
    return output_pha

def make_grppha_script_from_csv(input_csv, suffix="output"):
    df = pd.read_csv(input_csv)
    grouping = df['GROUPING'].values

    output_grppha_script = f"grppha_script_{suffix}.sh"

    allpilist = np.arange(0, 60000, 1)
    frouping_start_pi = allpilist[np.where(grouping==1)[0]]
    with open(output_grppha_script, "w") as fout:
        fout.write("#!/bin/sh\n")
        fout.write("infile=$1\n")
        fout.write("outfile=$2\n")
        fout.write("grppha << EOF\n")
        fout.write("$infile\n")
        fout.write("$outfile\n")
        for i in range(len(frouping_start_pi)-1):
            fout.write(f"group {frouping_start_pi[i]} {frouping_start_pi[i+1]-1} {frouping_start_pi[i+1]-frouping_start_pi[i]}"+"\n")
        fout.write("exit\n")
        fout.write("quit\n")
        fout.write("EOF\n")
    
    return output_grppha_script


""" 結果を確認する関数 """
def check_binnning(phafile, set_start_ene, set_stop_ene, set_ebinsize):
    allpilist = np.arange(0, 60000, 1)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.dpi"] = 150
    inch = 25.4  # 1インチ=25.4 mm
    height = 80 #defo
    wide = 90 #defo

    fig, axs = plt.subplots(1, 1, figsize=(wide / inch, height / inch))
    fig.subplots_adjust(left=0.102, right=0.96, top=0.94, bottom=0.15)

    spectrum = fits.open(phafile)[1].data
    grouping = spectrum["GROUPING"]
    
    start_pi = np.where(grouping==1)[0]
    start_ene = pi_to_ev(pi=start_pi)
    
    binsize = start_ene[1:] - start_ene[0:-1]
    center = (start_ene[1:] + start_ene[0:-1])/2

    axs.plot(center, binsize, "black", linestyle="", marker="o", label="input pha", ms=2.0)

    if not isinstance(set_start_ene, list):
        set_start_ene = [set_start_ene]
    if not isinstance(set_stop_ene, list):
        set_stop_ene = [set_stop_ene]
    if not isinstance(ebinsize, list):
        set_ebinsize = [set_ebinsize]
    
    for s, p, z in zip(set_start_ene, set_stop_ene, set_ebinsize):
        energy_mask = (s <= center) & (center <= p)
        axs.plot(center[energy_mask], binsize[energy_mask], "red", linestyle="", marker="o", label="binned", ms=2.0)
        axs.hlines(xmin=s, xmax=p, y=z, linestyles='solid', color = "lime", label="set ebinsize")

    axs.legend()
    axs.set_xlabel("Energy [eV]")
    axs.set_ylabel("enegy bin size [eV]")
    plt.show()

# 引数がリストかどうかを判定する関数
def parse_list_or_number(value):
    if value.lower() in ["none", "None"]:  # "none"を None に変換
        return None
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, (int, float)):
            return parsed
        elif isinstance(parsed, None):
            return parsed
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid input: {value}. Expected a number or a list.")



# Argument parser
parser = argparse.ArgumentParser(description='Create binned spectrum in specific enegy.', 
                                 usage='%(prog)s <input pha file> --start_ene=6200 or "[2000, 6200]" (--se) --stop_ene=8100 or "[3000, 8100]" (--pe) --ebinsize=2.0 or "[2.0, 2.0"]')
parser.add_argument('input_pha', help='input no binned spectrum')
parser.add_argument('--start_ene', '--se', default=6200, type=parse_list_or_number, help='Beginning of energy range to group in a specific energy bin (in eV).')
parser.add_argument('--stop_ene', '--pe', default=8100, type=parse_list_or_number, help='End of energy range to group in a specific energy bin (in eV).')
parser.add_argument('--ebinsize', '--es', default=2.0, type=parse_list_or_number, help='Energy bin size (in eV).')

"""ftgroupphaのための引数"""
parser.add_argument('--grouptype', default="opt", help='input group type for ftgrouppha, (bmin/constant/snmin/opt)')
parser.add_argument('--input_rmf', '--respfile', default=None, help='input rmf file for ftgrouppha')
parser.add_argument('--backfile', default=None, help='input backgorund pha file for ftgrouppha')
parser.add_argument('--groupscale', default=None, help='input groupscale file for ftgrouppha')
args = parser.parse_args()

input_pha = args.input_pha
start_ene = args.start_ene
stop_ene = args.stop_ene
ebinsize = args.ebinsize

"""ftgroupphaのための引数"""
grouptype = args.grouptype
input_rmf = args.input_rmf
backfile = args.backfile
groupscale = args.groupscale


"""
実行方法
python create_binnedspectrum_in_specific_energy.py rsl_source_Hp_pixCentre4x4_tave_v0.pha --se=6200 --pe=8100 --es=2.5
--> rsl_source_Hp_pixCentre4x4_tave_optene6200to8100binned2p0eV.pha

--start_ene, --stop_ene, --ebinsizeは同じサイズの配列に対応

デフォルトでは、baseのbinnningはftgroupphaを使用。
group typeはopt(rmfを読むので、phaのkeywordに入っているのが理想)
他にも、bmin/constant/snmin が使用可能
grouptypeによって--groupscaleの設定が必要

中間ファイルはcsv
まず、指定したgroupingでcsvを作成
csvから、phaのgroupingを編集-->新しいbinnned phaを作成

csvから任意のphaをgroupingすることも可能 (複数のphaを共通のgroupingにしたい時など)
"""




"""基本の使い方"""
output_csv, output_suf = main_create_binned_spectrum_in_csv(input_pha=input_pha, start_enes=start_ene, stop_enes=stop_ene, ebinsizes=ebinsize, grouptype=grouptype, input_rmf=input_rmf, backfile=backfile, groupscale=groupscale)
output_grppha_script = make_grppha_script_from_csv(input_csv=output_csv, suffix=output_suf+"_for_"+os.path.basename(input_pha).replace(".pha", ""))
output_pha = make_binned_pha_from_csv(input_pha=input_pha, input_csv=output_csv, suffix=output_suf)
# check_binnning(phafile=output_pha, set_start_ene=start_ene, set_stop_ene=stop_ene, set_ebinsize=ebinsize)


""" こういう使い方も可能 """
# phafiles = glob.glob("rsl_source_Hp_pixCentre4x4_phase*_v0.pha")


# phafile = f"rsl_source_Hp_pixCentre4x4_phase0p95to1p0_v0.pha"
# output_csv, output_suf = main_create_binned_spectrum_in_csv(input_pha=phafile, start_enes=start_ene, stop_enes=stop_ene, ebinsizes=ebinsize, grouptype=grouptype, groupscale=groupscale)
# # output_pha = make_binned_pha_from_csv(input_pha=phafile, input_csv=output_csv, suffix=output_suf)




# phafiles = glob.glob("*_v0.pha")
# for phafile in phafiles:
#     output_pha = make_binned_pha_from_csv(input_pha=phafile, input_csv=output_csv, suffix=output_suf)


# orbital_phase_start = np.arange(0.65, 1.11, 0.05)
# orbital_phase_end = np.arange(0.7, 1.21, 0.05)
# ebinsizelist = [3.5]
# for n in range(len(orbital_phase_start)):
#     name = str(round(orbital_phase_start[n], 3)).replace(".", "p")+"to"+str(round(orbital_phase_end[n], 3)).replace(".", "p")
#     phafile = f"rsl_source_Hp_pixCentre4x4_phase{name}_v0.pha"
#     for ebinsize in ebinsizelist:
#         output_csv, output_suf = main_create_binned_spectrum_in_csv(input_pha=phafile, start_enes=start_ene, stop_enes=stop_ene, ebinsizes=ebinsize, grouptype=grouptype, groupscale=groupscale)
#         output_pha = make_binned_pha_from_csv(input_pha=phafile, input_csv=output_csv, suffix=output_suf)

