#!/bin/bash

# ファイルの存在をチェックする関数
check_file_exists() {
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            echo "[ERROR] File not found: $file"
            exit 1
        fi
    done
}

# 引数のチェック (3つまたは4つを許可)
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "[ERROR] Invalid number of arguments."
    echo "Usage: $0 <ufevt> <minevent> <hkevent> [pixels]"
    echo "Example: $0 xa300049010rsl_p0px5000_uf.evt 200 xa300049010rsl_a0.hk1 0,1,2,3"
    exit 1
fi

# 引数の取得
ufevt=$1
minevent=$2
hkevent=$3
pixels=${4:-"0,17,18,35"}  # デフォルトは 0,17,18,35

outghf="min${minevent}_${ufevt%.evt}.ghf"
logfile="min${minevent}_${ufevt%.evt}.log"
outdir=min${minevent}

# 不要なファイルの削除
rm -f $outghf $logfile fe.gti

# GTIファイルの作成
ftmgtime "${ufevt}[2],${ufevt}[6]" fe.gti AND

# 必要なファイルへのシンボリックリンクを作成
ln -fs $CALDB/data/xrism/gen/bcf/xa_gen_linefit_20190101vx001.fits .
ln -fs $CALDB/data/xrism/resolve/bcf/gain/xa_rsl_gainpix_20190101v006.fits .

# 必要なファイルの存在をチェック
check_file_exists "$ufevt" "$hkevent" \
    "./xa_gen_linefit_20190101vx001.fits" \
    "./xa_rsl_gainpix_20190101v006.fits"

# rslgainの実行
echo "[start rslgain] $ufevt $outghf $logfile"

rslgain infile=${ufevt} outfile=${outghf} gainfile=./xa_rsl_gainpix_20190101v006.fits tempidx=NOM \
        gaincoeff=H linefitfile=./xa_gen_linefit_20190101vx001.fits linetocorrect=Mnka itypecol=ITYPE \
        ntemp=3 calmethod=Fe55 numevent=1000 minevent=${minevent} gtifile=fe.gti \
        gapdt=-1 grpoverlap=50 startenergy=-1 stopenergy=-1 extraspread=40 pxphaoffset=0 \
        broadening=1 gridprofile=no fitwidth=yes background=CONST spangti=no usemp=no \
        ckrisetime=yes ckclip=yes calcerr=yes writeerrfunc=yes ckant=yes ckctrec=yes ckctel=yes \
        ckctel2=no extrap=no avgwinrad=30 minwidth0=8 maxitcycle=5 r2tol=0.001 searchstepshift=2 \
        maxdshift=10 bisectolshift=0.001 searchstepwidth=5 maxdwidth=10 bisectolwidth=0.001 \
        minwidth=0.5 nerrshift=100 nerrwidth=100 shifterrfac=3 widtherrfac=4 buffer=-1 clobber=yes \
        chatter=2 logfile=${logfile} debug=no history=yes mode=hl

echo "[finish rslgain]"

# cleanupの開始メッセージ
echo "[INFO] Starting cleanup for $ufevt"

# 出力ディレクトリの作成とファイルの移動
mkdir -p ${outdir}
mv ${outghf} ${outdir}/
mv fe.gti ${outdir}/
mv ${logfile} ${outdir}/
cd ${outdir}

# 必要なシンボリックリンクの作成
ln -fs ../${hkevent} .

# GHFプロットスクリプトの実行
resolve_ecal_plot_ghf_with_FWE.py  ${outghf} --hk1 ${hkevent}
resolve_ecal_plot_ghf_detail.py ${outghf}
resolve_ecal_plot_ghf_detail.py ${outghf} --pixels ${pixels} --detail

# cleanupの終了メッセージ
echo "[INFO] Finished cleanup for $ufevt"

# スクリプトの終了を示すメッセージ
echo "[INFO] Script completed successfully."
