#!/bin/bash

# スクリプトの開始
echo "[INFO] Starting the script..."

# 引数のチェック (2つのみ許可)
if [ $# -ne 2 ]; then
    echo "[ERROR] Invalid number of arguments."
    echo "Usage: $0 <OBSID> <SUFFIX>"
    echo "Example: $0 300049010 20241023"
    exit 1
fi

# 引数の設定
obsid=$1
suffix=$2
outdir=repro_${obsid}_${suffix}

# rslpipelineの開始メッセージ
echo "[INFO] Starting rslpipeline for OBSID: $obsid with suffix: $suffix"

# rslpipelineコマンドの実行
rslpipeline indir=${obsid} outdir=${outdir} entry_stage=1 exit_stage=2 steminputs=xa${obsid} stemoutputs=DEFAULT \
           attitude=${obsid}/auxil/xa${obsid}.att \
           orbit=${obsid}/auxil/xa${obsid}.orb \
           obsgti=${obsid}/auxil/xa${obsid}_gen.gti \
           housekeeping=${obsid}/resolve/hk/xa${obsid}rsl_a0.hk1 \
           makefilter=${obsid}/auxil/xa${obsid}.mkf \
           extended_housekeeping=${obsid}/auxil/xa${obsid}.ehk \
           timfile=${obsid}/auxil/xa${obsid}.tim \
           calc_gtilost=yes \
           clobber=yes \
           cleanup=no

# rslpipelineの終了メッセージ
echo "[INFO] Finished rslpipeline for OBSID: $obsid"

# cleanupの開始メッセージ
echo "[INFO] Starting cleanup for $obsid"

# 出力ディレクトリを作成し、生成されたファイルを移動
mkdir -p ${outdir}/cleanup_output
mv *evt ${outdir}/cleanup_output 
mv *log ${outdir}/cleanup_output 
mv *hk1 ${outdir}/cleanup_output 
mv *gti ${outdir}/cleanup_output 

# cleanupの終了メッセージ
echo "[INFO] Finished cleanup for $obsid"

# スクリプトの終了を示すメッセージ
echo "[INFO] Script completed successfully."
