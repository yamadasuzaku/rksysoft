#!/usr/bin/env python 
import argparse
import glob
import os
import shutil
from datetime import datetime
import sys

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


def generate_html_for_pngs(obsid, output_dir, keyword="check_", ver="v0", fitkeyword="xspecfitlog"):
    # ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # OBSIDディレクトリ以下のPNGファイルを収集
    # PNGファイルの取得と生成時刻でソート
    png_files = glob.glob(os.path.join(f"./{obsid}", "**", "*.png"), recursive=True)
    png_files = sorted(png_files, key=os.path.getctime)
    # パス全体に "check" が含まれるものだけをフィルタリング
    png_files = [f for f in png_files if keyword in f]

    if len(png_files) == 0:
        print(f"[Finish] No png files are found", png_files)
        sys.exit()

    # 出力HTMLファイルのパスを作成
    output_html = os.path.join(output_dir, f"A_QL_html_{obsid}_{keyword}{ver}.html")

    # 現在の日付を秒まで取得
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # HTMLのヘッダー部分を作成
    html_content = f"""
    <html>
    <head>
        <title>XRISM QL : OBSID={obsid} - {current_date} ({keyword} {ver})</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 10px;
            }}
        </style>
    </head>
    <body>
    <h1>XRISM QL : OBSID={obsid} </h1>
    <p>Generated on {current_date} ({keyword} {ver})</p>
    """
    
    # PNGファイルをコピーして、HTMLに埋め込む
    for png in png_files:

        file_name = os.path.basename(png)
        destination_png = os.path.join(output_dir, file_name)
        # ファイルを出力ディレクトリにコピー
        shutil.copy(png, destination_png)

        # find xspec log file 
        if fitkeyword in png:
            logfile=png.replace(".png",".txt")        
            logfile_name = os.path.basename(logfile)
            # ファイルを出力ディレクトリにコピー
            destination_log = os.path.join(output_dir, logfile_name)
            shutil.copy(logfile, destination_log)
            print("..... logfile is found", logfile)    
        else:
            logfile = None

        if logfile is None:
            # コピーされたファイルをHTMLに埋め込む（クリックでフルサイズ画像が開くように）
            html_content += f'''
            <div>
                <h5><a href="{file_name}">{png}</a></h5>
                <a href="{file_name}" target="_blank">
                    <img src="{file_name}" alt="{file_name}" style="max-width:600px; height:auto;">
                </a>
            </div>\n
            '''        
        else:
            # コピーされたファイルをHTMLに埋め込む（クリックでフルサイズ画像が開くように）
            html_content += f'''
            <div>
                <h5><a href="{file_name}">{png}</a></h5>
                <h5><a href="{logfile_name}">{logfile}</a></h5>
                <a href="{logfile_name}" target="_blank">
                    <img src="{file_name}" alt="{file_name}" style="max-width:600px; height:auto;">
                </a>
            </div>\n
            '''        

#         html_content += f'<div><h2>{file_name}</h2><img src="{file_name}" alt="{file_name}"></div>\n'

    # HTMLのフッター部分を作成
    html_content += """
    </body>
    </html>
    """

    # HTMLファイルに書き出し
    with open(output_html, "w") as html_file:
        html_file.write(html_content)
    
    print(f"HTML file generated: {output_html}")

def main():
    parser = argparse.ArgumentParser(description="Generate HTML for PNG files")
    parser.add_argument('obsid', help='OBSID to define the directory name')
    parser.add_argument('--output-dir', default='A_QL_html', help='Output directory (default: QLhtml)')
    parser.add_argument('--keyword', default='A_QL_html', help='keyword for png (default: check_)')
    parser.add_argument('--ver', default='A_QL_html', help='version (default: v0)')

    args = parser.parse_args()
    
    # HTMLファイルを生成
    generate_html_for_pngs(args.obsid, args.output_dir + "_" + args.obsid, keyword = args.keyword, ver=args.ver)

if __name__ == "__main__":
    main()
