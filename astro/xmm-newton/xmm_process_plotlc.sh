#!/bin/sh 


#20180407 Shinya Yamada, TMU, ver1 
# 
# [Usage] 

if [ _$3 = _ ];
then
echo ./xmm_process_ql.sh f1 f2 f3 
exit
fi

#parameters
f1=$1
f2=$2
f3=$3

f1tag=`echo $f1 | awk -F. '{print $1}'`
f2tag=`echo $f2 | awk -F. '{print $1}'`
f3tag=`echo $f3 | awk -F. '{print $1}'`

outfname=${f1tag}_${f2tag}_${f3tag}

echo "f1 = " $f1 $f1tag
echo "f2 = " $f2 $f2tag
echo "f3 = " $f3 $f3tag

pwd=$PWD


# func:setpass 
plotlc(){
echo "[plotlc] : all"
mkdir -p log
rm -f $outfname plotlc.fits log/plotlc.log

lcurve <<EOF 
3
$f1
$f2
$f3
-
100
INDEF
plotlc.fits
Y
/ps
3
la t ${outfname}
hard ${outfname}.ps/cps
q

EOF

rm -f ${outfname}.pdf
ps2pdf ${outfname}.ps ${outfname}.pdf

}

# 3 # Number of time series for this task
# $f1 # Ser. 1 filename +options (or @file of filenames +options)[mos1_filt_lc.fits] 
# $f2 # Ser. 2 filename +options (or @file of filenames +options)[mos2_filt_lc.fits] 
# $f3 # Ser. 3 filename +options (or @file of filenames +options)[pn_filt_lc.fits] 
# - # Name of the window file ('-' for default window)[-] 
# 100 # Newbin Time or negative rebinning[100] 
# INDEF # Number of Newbins/Interval[311] 
# plotlc.fits # Name of output file[all.lc] 
# Y # Do you want to plot your results?[Y] 
# /ps # Enter PGPLOT device[FILE] /ps
# 3 # Enter PLOT style number (default=1)[3] 
# hard ${outfname}/cps # PLT> hard all.ps/cps 
# q # PLT> q


run()
{
plotlc
}


# main program start from here
run 
