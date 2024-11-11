#!/usr/bin/env python

import os
import subprocess
import argparse
from astropy.io import fits
import sys
import matplotlib.pyplot as plt

def generate_xspec_data(flist):
    data_lines = '\n'.join([f"data {i+1}:{i+1} {pha}" for i, pha in enumerate(flist)])
    return data_lines

def generate_xspec_data_we(flist, emin, emax):
    data_lines = '\n'.join([f"data {i+1}:{i+1} {pha}" for i, pha in enumerate(flist)])
    ignore_lines = '\n'.join([f"ignore {i+1}:**-{emin} {emax}-**" for i, pha in enumerate(flist)])
    return data_lines, ignore_lines

def generate_xspec_qdp_label(flist, laboffset=5):

    flabel_list = [pha.replace("rsl_source_","") for i, pha in enumerate(flist)]    
    label_lines = '\n'.join([f"LABEL {i+1 + laboffset} VPosition 0.2 {0.98 - 0.02*i} \"{pha}\"" for i, pha in enumerate(flabel_list)])
    label_cols = '\n'.join([f"LABEL {i+1 + laboffset} color {i+1}" for i, pha in enumerate(flist)])

    return label_lines, label_cols

def check_xspec_data(flist):
    headerlist = []
    for i, pha in enumerate(flist):
        header1 = fits.open(pha)[1].header
        rmf = header1["RESPFILE"]
        check_file_exists(rmf)
        arf = header1["ANCRFILE"]
        check_file_exists(arf)
        obsid = header1["OBS_ID"]
        dateobs = header1["DATE-OBS"]
        objname = header1["OBJECT"]
        exposure = header1["EXPOSURE"]
        headerlist.append(header1)
    return headerlist

def color_text(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(color_text(f"Error: {filepath} does not exist.", "red"))
        sys.exit(1)

def check_command_exists(command):
    if subprocess.call(f"type {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print(color_text(f"Error: {command} is not available in the PATH.", "red"))
        sys.exit(1)

def run_shell_script(script_content, script_name):
    with open(script_name, "w") as script_file:
        script_file.write(script_content)
    os.chmod(script_name, 0o755)
    subprocess.run(f"./{script_name}", shell=True)
#    os.remove(script_name)

def convert_ps_to_pdf(ps_file, output_pdf=None):
    check_command_exists("ps2pdf")
    if not os.path.exists(ps_file):
        print(f"Error: {ps_file} が見つかりません。")
        return

    if not output_pdf:
        # デフォルトのPDF名を .ps ファイルの名前に基づいて生成
        output_pdf = os.path.splitext(ps_file)[0] + ".pdf"

    try:
        # ps2pdf コマンドの実行
        result = subprocess.run(
            ["ps2pdf", ps_file, output_pdf],
            check=True,  # エラー時に例外を投げる
            capture_output=True,  # 標準出力とエラーを取得
            text=True
        )
        print(f"変換成功: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"変換失敗: {e.stderr}")

def convert_pdf_to_png(pdf_file, output_png=None):
    check_command_exists("convert")    
    """PDF ファイルを PNG に変換し、背景を白に設定します。"""
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} が見つかりません。")
        return

    if not output_png:
        output_png = os.path.splitext(pdf_file)[0] + ".png"

    try:
        # 背景を白にして PNG に変換
        subprocess.run(
            ["convert", "-density", "300", pdf_file, 
             "-background", "white", "-alpha", "remove", 
             "-alpha", "off", output_png],
            check=True, capture_output=True, text=True
        )
        print(f"PNG 変換成功: {output_png}")
    except subprocess.CalledProcessError as e:
        print(f"PNG 変換失敗: {e.stderr}")

def read_xspec_log(filename):
    """テキストファイルから行ごとに読み込む"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines

def text_to_image(text_lines, output_file, width=800, height=2000, font_size=7):
    """テキストを画像に描画しPNG形式で保存"""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # DPI 100基準でサイズ調整
    ax.axis('off')  # 軸は非表示

    # 行の高さを計算して、何行表示できるか決定
    line_height = font_size + 4  # 行間を少し空ける
    max_lines = height // line_height

    # 描画する行を決定（収まる範囲のみ）
    display_lines = text_lines[:max_lines]

    # 各行を上から順に描画
    for i, line in enumerate(display_lines):
        ax.text(0, 1 - 1.0*(i + 1) * (line_height / height), line.strip(), 
                fontsize=font_size, ha='left', va='top', family='monospace')

    # 画像として保存
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"{output_file} に保存しました。")

# コマンドライン引数を解析する関数
def parse_args():
    """
    コマンドライン引数を解析する。
    """
    parser = argparse.ArgumentParser(
      description='Resolve QL spectral fit',
      epilog='''
        Example 
       (1) standard   : resolve_ana_qlfit.py phafile
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    

    parser.add_argument('phalist', help='list for pha file that includes rmf, arf file paths.')
    # カンマ区切りの数値列を受け取る
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか。defaultはplotしない。')    
    parser.add_argument('--emin', '-l', type=float, default=2.0, help='Minimum energy (keV)')
    parser.add_argument('--emax', '-x', type=float, default=10.0, help='Maximum energy (keV)')
    parser.add_argument("--rmf", "-rmf", default=None, help="response file")
    parser.add_argument("--arf", "-arf", default=None, help="arf file")
    parser.add_argument('--progflags', "-pg", type=str, help='Comma-separated flags for plld, fitpl,..e (e.g. 0,1,0)', default="1,1,1")
    parser.add_argument('--xscale', "-xs", choices=['off', 'on'], default="on", help='Choose xscale for linear or log')
    parser.add_argument('--fname', '-f', type=str, help='output filename tag', default='mkspec')

    args = parser.parse_args()    
    # 引数の確認をプリント
    print("----- 設定 -----")
    # Print the command-line arguments to help with debugging
    args_dict = vars(args)
    print("Command-line arguments:")
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")    
    print("-----------------")
    return args

def main():
################### setting for arguments ###################################################################
    args = parse_args()
    emin = args.emin
    emax = args.emax
    rmf = args.rmf
    arf = args.arf
    xscale = args.xscale
    fname = args.fname

    with open(args.phalist, 'r') as file:
        phalist = [line.strip() for line in file]

    headerlist = check_xspec_data(phalist)

    enetag = f"emin{int(1e3*emin)}_emax{int(1e3*emax)}" 

    # カンマで分割して、数値に変換
    # ユーザーの入力をパースし、整数に変換
    progflags = args.progflags
    progflags = progflags or ""
    flag_values = [int(x) for x in progflags.split(',')]
    # N個未満の場合は、0 で埋める
    nprog=2 # pleff, plld, fitpl as of 2024.11.11
    if len(flag_values) < nprog:
        flag_values += [0] * (nprog - len(flag_values))    

    # 数値列をTrue/Falseに変換し、flagが1の時だけ実行
    procdic = {
        "pleff": bool(flag_values[0]),
        "plld": bool(flag_values[1]),
        "fitpl": bool(flag_values[2]),
    }
    print(f"procdic = {procdic}")    

    header1 = headerlist[0]
    obsid = header1["OBS_ID"]
    dateobs = header1["DATE-OBS"]
    objname = header1["OBJECT"]
    exposure = header1["EXPOSURE"]

    title=f"{obsid} {objname} {dateobs} {exposure:0.2e}s" 

    if procdic["pleff"]:

        datadef = generate_xspec_data(phalist)
        label_lines, label_cols = generate_xspec_qdp_label(phalist)

        # xspec commands
        print(color_text(f"Stage 0: Running xspec to check effective area of {phalist}", "blue"))
        xspec_script = f"""#!/bin/sh  
rm -rf {fname}_eff.ps {fname}_eff.pdf
xspec << EOF

{datadef}

plot efficien

iplot
{label_lines}
{label_cols}
r x 0.5 20.0
r y 0.1 500
LOG Y ON 
font roman
label rotate 
label titile {title}
LOG X {xscale} 1
plot
hard {fname}_eff.ps/cps
q

exit

EOF
        """
        run_shell_script(xspec_script, f"run_xspec_{fname}_eff.sh")   
        convert_ps_to_pdf(f"{fname}_eff.ps")
        convert_pdf_to_png(f"{fname}_eff.pdf")

    if procdic["plld"]:
        # xspec commands
        datadef, ignoredef = generate_xspec_data_we(phalist, emin, emax)
        label_lines, label_cols = generate_xspec_qdp_label(phalist)

        print(color_text(f"Stage 1: Running xspec to plot raw spectrum of {phalist}", "blue"))
        xspec_script = f"""#!/bin/sh  
rm -rf {fname}.ps {fname}.pdf {fname}.pco {fname}.qdp {fname}.fi {fname}.xcm
xspec << EOF

{datadef}
{ignoredef}

show rate 
plot ldata
setplot energy

save file {fname}.fi
save all {fname}.xcm

iplot
{label_lines}
{label_cols}
font roman
label rotate 
label titile {title}
LOG X {xscale} 1
plot
hard {fname}.ps/cps
we {fname}
q

exit

EOF
        """
        run_shell_script(xspec_script, f"run_xspec_{fname}.sh")   
        convert_ps_to_pdf(f"{fname}.ps")
        convert_pdf_to_png(f"{fname}.pdf")


    if procdic["fitpl"]:
        # xspec commands
        datadef, ignoredef = generate_xspec_data_we(phalist, emin, emax)        
        label_lines, label_cols = generate_xspec_qdp_label(phalist)

        print(color_text(f"Stage 2: Running xspec to plot a powerlaw fit of {phalist}", "blue"))
        xspec_script = f"""#!/bin/sh  
rm -rf {fname}_fitpl.ps {fname}_fitpl.pdf {fname}_fitpl.pco {fname}_fitpl.qdp {fname}_fitpl.fi {fname}_fitpl.xcm {fname}_fitpl_xspecfitlog.txt
xspec << EOF

{datadef}
{ignoredef}
show rate 
plot ldata
setplot energy

model tbabs * power 
















renorm 

fit 100

plot ldata delchi eeuf ratio 

save file {fname}_fitpl.fi 

save all {fname}_fitpl.xcm

log {fname}_fitpl_xspecfitlog.txt

show file 
show rate 
flux 2.0 10.0 
show rate
show model 
show fit 
show pa

log none 

iplot
{label_lines}
{label_cols}

font roman
label rotate 
label titile {title}

LOG X {xscale} 1 2 3 4

plot 

hard {fname}_fitpl.ps/cps
we {fname}_fitpl
q

exit

EOF
        """
        run_shell_script(xspec_script, f"run_xspec_{fname}_fitpl.sh")   
        convert_ps_to_pdf(f"{fname}_fitpl.ps")
        convert_pdf_to_png(f"{fname}_fitpl.pdf")

        text_lines = read_xspec_log(f"{fname}_fitpl_xspecfitlog.txt")
        output_file = f"{fname}_fitpl_xspecfitlog.png"
        text_to_image(text_lines, output_file, width=800, height=2000, font_size=8)

if __name__ == "__main__":
    main()
