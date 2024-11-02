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

# 引数のチェック (1つから4つを許可)
if [ $# -lt 1 ] || [ $# -gt 4 ]; then
    echo "[ERROR] Invalid number of arguments."
    echo "Usage: $0 OBSID [minevent] [maxdshift] [pixels]"
    echo "Example: $0 300049010 200 15 0,1,2,3 "
    exit 1
fi

# 引数の取得
obsid=$1
ufevt=xa${obsid}rsl_p0px5000_uf.evt
hkevt=xa${obsid}rsl_a0.hk1 
ehk=xa${obsid}.ehk
fe55ehk=xa${obsid}rsl_fe55_ehk.gti
telgti=xa${obsid}rsl_tel.gti
gengti=xa${obsid}_gen.gti
minevent=${2:-200}         # デフォルトは 20
maxdshift=${3:-10}         # デフォルトは 10
pixels=${4:-"0,17,18,35"}  # デフォルトは 0,17,18,35

# 必要なファイルの存在をチェック
check_file_exists "$ufevt" "$hkevt" "$ehk" "$telgti" "$gengti" 

# 出力ファイル名とディレクトリ名を maxdshift を含めて作成
outghf="min${minevent}_dshift${maxdshift}_${ufevt%.evt}.ghf"
logfile="min${minevent}_dshift${maxdshift}_${ufevt%.evt}.log"
outdir="min${minevent}_dshift${maxdshift}"
outrslgti=xa${obsid}_rlsgain.gti

# 不要なファイルの削除
rm -f $outghf $logfile $outrslgti

# fe55ehkの生成
rm -f ${fe55ehk}
echo "[ahgtigen] start : create fe55ehk"
ahgtigen infile=${ehk} outfile=${fe55ehk} gtifile=none gtiexpr=SAA_SXS.EQ.0 mergegti=AND telescop=XRISM
echo "[ahgtigen] end : created ${fe55ehk}"

# GTIファイルの作成
echo "[ftmgtime] start : create $outrslgti"
ftmgtime "${ufevt}[GTI],${ufevt}[GTIADROFF],${ufevt}[GTIMKF],${telgti}[GTITEL],${gengti}[GTIOBS],${fe55ehk}[GTI]" $outrslgti AND
echo "[ftmgtime] end   : created $outrslgti"

# 必要なファイルへのシンボリックリンクを作成
ln -fs $CALDB/data/xrism/gen/bcf/xa_gen_linefit_20190101vx001.fits .
ln -fs $CALDB/data/xrism/resolve/bcf/gain/xa_rsl_gainpix_20190101v006.fits .

# 必要なファイルの存在をチェック
check_file_exists "./xa_gen_linefit_20190101vx001.fits" "./xa_rsl_gainpix_20190101v006.fits" "$fe55ehk"

# rslgainの実行
echo "[start rslgain] $ufevt $outghf $logfile"

rslgain infile=${ufevt} outfile=${outghf} gainfile=./xa_rsl_gainpix_20190101v006.fits tempidx=NOM \
        gaincoeff=H linefitfile=./xa_gen_linefit_20190101vx001.fits linetocorrect=Mnka itypecol=ITYPE \
        ntemp=3 calmethod=Fe55 numevent=1000 minevent=${minevent} gtifile=${outrslgti} \
        gapdt=-1 grpoverlap=50 startenergy=-1 stopenergy=-1 extraspread=40 pxphaoffset=0 \
        broadening=1 gridprofile=no fitwidth=yes background=CONST spangti=no usemp=no \
        ckrisetime=yes ckclip=yes calcerr=yes writeerrfunc=yes ckant=yes ckctrec=yes ckctel=yes \
        ckctel2=no extrap=no avgwinrad=30 minwidth0=8 maxitcycle=5 r2tol=0.001 searchstepshift=2 \
        maxdshift=${maxdshift} bisectolshift=0.001 searchstepwidth=5 maxdwidth=10 bisectolwidth=0.001 \
        minwidth=0.5 nerrshift=100 nerrwidth=100 shifterrfac=3 widtherrfac=4 buffer=-1 clobber=yes \
        chatter=2 logfile=${logfile} debug=no history=yes mode=hl

echo "[finish rslgain]"

# cleanupの開始メッセージ
echo "[INFO] Starting cleanup for $ufevt"

# 出力ディレクトリの作成とファイルの移動
mkdir -p ${outdir}
mv ${outghf} ${outdir}/
mv ${outrslgti} ${outdir}/
mv ${logfile} ${outdir}/
cd ${outdir}

# 必要なシンボリックリンクの作成
ln -fs ../${hkevt} .

# GHFプロットスクリプトの実行
resolve_ecal_plot_ghf_with_FWE.py ${outghf} --hk1 ${hkevt}
resolve_ecal_plot_ghf_detail.py ${outghf}
resolve_ecal_plot_ghf_detail.py ${outghf} --pixels ${pixels} --detail

# cleanupの終了メッセージ
echo "[INFO] Finished cleanup for $ufevt"

# スクリプトの終了を示すメッセージ
echo "[INFO] Script completed successfully."
