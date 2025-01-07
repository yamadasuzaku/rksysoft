#!/bin/bash

if [ _$1 = _ ];
then
cat << EOF
usage : resolve_ana_pixel_fitMnKa.sh (1:cleaned event file)
EOF
exit
fi

# Function to check if a file exists
check_file_exists() {
  local file=$1
  if [ ! -f "$file" ]; then
    echo "Error: File '$file' not found."
    exit 1
  fi
}

evtfile=$1
check_file_exists $evtfile
outpha="${evtfile%.evt}_Hp_src.pha"

# echo "1. Make a diagonal response matrix (newdiag.rmf)"

# ln -sf ${CALDB}/data/xrism/resolve/bcf/response/xa_rsl_rmfparam_20190101v005.fits.gz .
# rm -f xa_rsl_rmfparam_fordiagrmf.fits newdiag.rmf
# ftcopy 'xa_rsl_rmfparam_20190101v005.fits.gz[GAUSFWHM1]' xa_rsl_rmfparam_fordiagrmf.fits
# ftcalc 'xa_rsl_rmfparam_fordiagrmf.fits[GAUSFWHM1]' xa_rsl_rmfparam_fordiagrmf.fits PIXEL0 0.000000001 rows=- clobber=yes
# rslrmf NONE newdiag whichrmf=S rmfparamfile=xa_rsl_rmfparam_fordiagrmf.fits

echo "2. Extract the spectrum"


tmp=tmp.log
rm -f $tmp $outpha

xselect << EOF 2>&1 | tee $tmp

no

read event $evtfile ./
yes


filter column "PIXEL=0:11,13:35"
filter GRADE "0:0"

show filter 

extract spectrum 
save spectrum $outpha


exit
no

EOF



echo "3. Fit the spectrum of Mn Ka with Holzer"

outtag=55Fe_hp
xcmfile=${outtag}.xcm
qdpfile=${outtag}.qdp
pcofile=${outtag}.pco
psfile=${outtag}.ps
pdffile=${outtag}.pdf
rm -f $xcmfile $qdpfile $pcofile $psfile $pdffile
tmp=tmpxspec.log
rm -f $tmp

xspec << EOF 2>&1 | tee $tmp

data 1:1 $outpha
response 1 newdiag.rmf

ignore 1:1-11701,11860-60000
statistic cstat

setplot energy 

gain fit 1:1
1 -1
0 0.01


model gsmooth((lorentz + lorentz + lorentz + lorentz + lorentz + lorentz+ lorentz + lorentz)constant)
0.0019
0 -1
5.898882 -1
0.0017145 -1
0.3523 -1
5.897898 -1
0.0020442 -1
0.1409 -1
5.894864 -1
0.0044985 -1
0.07892 -1
5.896566 -1
0.0026616 -1
0.06624 -1
5.899444 -1
0.00097669 -1
0.01818 -1
5.902712 -1
0.0015528 -1
0.004475 -1
5.887772 -1
0.0023604 -1
0.2283 -1
5.886528 -1
0.0042168 -1
0.1106 -1
1 0.01


query yes

fit

error 1. 1

rerror 1. 2


save all $xcmfile

pl d ra

iplot
la t $outpha
log x off 
col 8 on 2 
win 2 
log x off 

we ${outtag}
hard ${psfile}/cps 

q

exit 


EOF

ps2pdf $psfile