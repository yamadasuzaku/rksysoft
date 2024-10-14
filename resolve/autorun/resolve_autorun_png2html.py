#!/usr/bin/env python 
import argparse
import glob
import os
import shutil
from datetime import datetime

def generate_html_for_pngs(obsid, output_dir):
    # ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # OBSIDディレクトリ以下のPNGファイルを収集
    png_files = glob.glob(os.path.join(f"./{obsid}", "**", "*.png"), recursive=True)
    
    # 出力HTMLファイルのパスを作成
    output_html = os.path.join(output_dir, f"QLhtml_{obsid}.html")

    # 現在の日付を秒まで取得
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    
    # HTMLのヘッダー部分を作成
    # HTMLのヘッダー部分を作成
    html_content = f"""
    <html>
    <head>
        <title>XRISM QL : OBSID={obsid} - {current_date}</title>
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
    <h1>PNG Files for OBSID={obsid}</h1>
    <p>Generated on {current_date}</p>
    """
    
    # PNGファイルをコピーして、HTMLに埋め込む
    for png in png_files:
        file_name = os.path.basename(png)
        destination_png = os.path.join(output_dir, file_name)
        
        # ファイルを出力ディレクトリにコピー
        shutil.copy(png, destination_png)

        # コピーされたファイルをHTMLに埋め込む（クリックでフルサイズ画像が開くように）
        html_content += f'''
        <div>
            <h5><a href="{file_name}">{png}</a></h5>
            <a href="{file_name}" target="_blank">
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
    parser.add_argument('--output-dir', default='QLhtml', help='Output directory (default: QLhtml)')
    
    args = parser.parse_args()
    
    # HTMLファイルを生成
    generate_html_for_pngs(args.obsid, args.output_dir + "_" + args.obsid)

if __name__ == "__main__":
    main()
