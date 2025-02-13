#!/usr/bin/env python 

import argparse
import bibtexparser
import re
import os

# 特殊文字をLaTeXのエスケープ表現に変換するためのマッピング
TEX_REPLACEMENTS = {
    "ä": r"{\"a}", "ö": r"{\"o}", "ü": r"{\"u}", "ß": r"{\ss}",
    "Ä": r"{\"A}", "Ö": r"{\"O}", "Ü": r"{\"U}",
    "é": r"{\'e}", "è": r"{\`e}", "ê": r"{\^e}", "ë": r"{\"e}",
    "É": r"{\'E}", "È": r"{\`E}", "Ê": r"{\^E}", "Ë": r"{\"E}",
    "á": r"{\'a}", "à": r"{\`a}", "â": r"{\^a}", "ã": r"{\~a}",
    "Á": r"{\'A}", "À": r"{\`A}", "Â": r"{\^A}", "Ã": r"{\~A}",
    "í": r"{\'i}", "ì": r"{\`i}", "î": r"{\^i}", "ï": r"{\"i}",
    "Í": r"{\'I}", "Ì": r"{\`I}", "Î": r"{\^I}", "Ï": r"{\"I}",
    "ó": r"{\'o}", "ò": r"{\`o}", "ô": r"{\^o}", "õ": r"{\~o}",
    "Ó": r"{\'O}", "Ò": r"{\`O}", "Ô": r"{\^O}", "Õ": r"{\~O}",
    "ú": r"{\'u}", "ù": r"{\`u}", "û": r"{\^u}",
    "Ú": r"{\'U}", "Ù": r"{\`U}", "Û": r"{\^U}",
    "ñ": r"{\~n}", "Ñ": r"{\~N}", "ç": r"{\c c}", "Ç": r"{\c C}"
}

def texify(text):
    """文字列内の特殊文字をLaTeX形式に変換する関数"""
    return re.sub(
        '|'.join(map(re.escape, TEX_REPLACEMENTS.keys())),  # 変換対象の文字を正規表現でマッチさせる
        lambda match: TEX_REPLACEMENTS[match.group(0)],  # マッチした文字をLaTeX表現に変換
        text
    )

def process_bibtex(input_file, output_file):
    """BibTeXファイルを読み込み、特殊文字をLaTeX形式に変換して保存する関数"""
    with open(input_file, encoding="utf-8") as bib_file:
        bib_database = bibtexparser.load(bib_file)  # BibTeXファイルを読み込む

    # 各エントリのすべてのフィールドを変換
    for entry in bib_database.entries:
        for key, value in entry.items():
            entry[key] = texify(value)  # LaTeX形式に変換

    # 変換後のデータを新しいBibTeXファイルとして保存
    with open(output_file, "w", encoding="utf-8") as bib_file:
        bibtexparser.dump(bib_database, bib_file)

    print(f"変換完了: {output_file}")  # 完了メッセージを表示

def main():
    """コマンドライン引数を処理し、BibTeXファイルを変換するメイン関数"""
    parser = argparse.ArgumentParser(
        description="BibTeXファイル内の特殊文字をLaTeX形式に変換するツール"
    )
    parser.add_argument("input", help="入力するBibTeXファイルのパス")
    parser.add_argument("--output", help="出力ファイルのパス（省略時は自動生成）")
    
    args = parser.parse_args()
    
    # 出力ファイル名を決定（指定がない場合は元のファイル名に "_cvt" を付ける）
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_file = f"{base}_cvt{ext}"
    
    process_bibtex(args.input, output_file)  # BibTeXファイルを変換

if __name__ == "__main__":
    main()  # メイン関数を実行
