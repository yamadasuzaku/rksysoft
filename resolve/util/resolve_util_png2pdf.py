import os
import argparse
from PIL import Image
from PyPDF2 import PdfMerger

def convert_images_to_pdf(directory, output_pdf):
    # ディレクトリ内のファイルを取得
    files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    
    images_per_page = 6
    images = [Image.open(os.path.join(directory, f)).convert('RGB') for f in files]

    # 各ページの画像配置を設定
    page_images = []
    while images:
        page = images[:images_per_page]
        images = images[images_per_page:]

        # 画像のサイズと配置を設定
        image_width, image_height = page[0].size
        canvas_width = image_width * 2
        canvas_height = image_height * 3
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        for i, image in enumerate(page):
            x = (i % 2) * image_width
            y = (i // 2) * image_height
            canvas.paste(image, (x, y))
        
        page_images.append(canvas)

    # 一時的なPDFファイルを保存
    temp_pdf_files = []
    for i, page_image in enumerate(page_images):
        temp_pdf_path = os.path.join(directory, f'temp_{i}.pdf')
        page_image.save(temp_pdf_path)
        temp_pdf_files.append(temp_pdf_path)
    
    # PyPDF2 を使って一つのPDFに統合
    merger = PdfMerger()
    for pdf in temp_pdf_files:
        merger.append(pdf)
    
    # 結果のPDFを保存
    merger.write(output_pdf)
    merger.close()

    # 一時的なPDFファイルを削除
    for temp_pdf in temp_pdf_files:
        os.remove(temp_pdf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PNG images to a single PDF file.')
    parser.add_argument('directory', type=str, help='The directory containing PNG files.')
    parser.add_argument('output_pdf', type=str, help='The name of the output PDF file.')
    
    args = parser.parse_args()
    convert_images_to_pdf(args.directory, args.output_pdf)

