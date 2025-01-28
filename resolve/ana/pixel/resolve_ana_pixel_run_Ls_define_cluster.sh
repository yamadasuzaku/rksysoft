#!/bin/bash

# デフォルト値の設定
pixel=19
evtfile=""
allpixel=false

# 使用方法の表示
usage() {
    echo "Usage: $0 -f <evtfile> [-p <pixel_number>] [-a]"
    echo "  -f <evtfile>       Input event file (required)"
    echo "  -p <pixel_number>  Pixel number (default: 19, range: 0-35)"
    echo "  -a                 Process all pixels (ignores pixel number)"
    exit 1
}

# 引数の解析
while getopts ":f:p:a" opt; do
    case $opt in
        f)
            evtfile="$OPTARG"
            ;;
        a)
            allpixel=true
            ;;
        p)
            pixel="$OPTARG"
            if ! [[ $pixel =~ ^[0-9]+$ ]] || [ "$pixel" -lt 0 ] || [ "$pixel" -gt 35 ]; then
                echo "Error: Pixel number must be an integer between 0 and 35."
                exit 1
            fi
            ;;
        *)
            usage
            ;;
    esac
done

# 必須引数 evtfile の確認
if [ -z "$evtfile" ]; then
    echo "Error: evtfile is required."
    usage
fi

# ファイルの存在確認
if [ ! -f "$evtfile" ]; then
    echo "Error: File '$evtfile' not found."
    exit 1
fi

# 処理の分岐
if $allpixel; then
    echo "Processing all pixels for evtfile: $evtfile"
    resolve_ana_pixel_Ls_define_cluster.py "$evtfile"
    
    # 全ピクセルの結果ファイル名
    cluster_outfile="addcluster_${evtfile%.evt}.evt"
    cluster_outfile_imposi="addcluster_${evtfile%.evt}_imposi.evt"
    cluster_outfile_im1="addcluster_${evtfile%.evt}_im1.evt"
else
    # ファイル名用に pixel を 2 桁にゼロ埋め
    pixel_padded=$(printf "%02d" "$pixel")
    # 個別ピクセルの結果ファイル名    
    pixelcut_outfile="${evtfile%.evt}_p${pixel_padded}.evt"
    cluster_outfile="addcluster_${evtfile%.evt}_p${pixel_padded}.evt"
    cluster_outfile_imposi="addcluster_${evtfile%.evt}_p${pixel_padded}_imposi.evt"
    cluster_outfile_im1="addcluster_${evtfile%.evt}_p${pixel_padded}_im1.evt"

    echo "Processing pixel $pixel for evtfile: $evtfile (outfile: $outfile)"
    ftselect infile="$evtfile" outfile="$pixelcut_outfile" expr="PIXEL==$pixel" chatter=5 clobber=yes

    resolve_ana_pixel_Ls_define_cluster.py "$pixelcut_outfile"
fi

# 共通処理
ftselect infile="$cluster_outfile" \
         outfile="$cluster_outfile_imposi" \
         expr="IMEMBER>0" chatter=5 clobber=yes

ftselect infile="$cluster_outfile" \
         outfile="$cluster_outfile_im1" \
         expr="IMEMBER==1" chatter=5 clobber=yes

echo "Processing completed. Output files:"
echo "  $cluster_outfile"
echo "  $cluster_outfile_imposi"
echo "  $cluster_outfile_im1"

if $allpixel; then
    echo "  finish"
else
    echo "  $pixelcut_outfile"
    echo "  finish"
fi