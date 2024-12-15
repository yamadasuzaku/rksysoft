#!/bin/sh 


#20180407 Shinya Yamada, TMU, ver1 
#[Usage] ./xmm_process_ql.sh 
#[Usage] ./xmm_process_ql.sh cx cy rin rout 

#20180718 Shinya Yamada, TMU, ver1.1, for Qiita
#[Usage] ./xmm_process_ql.sh cx cy rin rout rin2 (when rin2 is for background)

#20180811 Shinya Yamada, TMU, ver1.2, for Qiita, to change the inner radios for background 
#[Usage] ./xmm_process_ql.sh cx cy rin rout rin2 (when rin2, cx2, cy2 is for background)

#20180812 Shinya Yamada, TMU, ver1.3, for Qiita, to change binsize 
#[Usage] ./xmm_process_ql.sh cx cy rin rout rin2 binsize (when rin2, cx2, cy2 is for background, binsize is group min value)
#


if [ _$4 = _ ];
then
cx=25286.2
cy=18506.7
rin=1200 
rout=3600
else
cx=$1
cy=$2
rin=$3
rout=$4

if [ _$5 = _ ];
then
rin2=$rin
else
rin2=$5
fi

if [ _$6 = _ ];
then
cx2=$cx
cy2=$cy
else
cx2=$6
cy2=$7
fi


if [ _$8 = _ ];
then
grpphabin=10 # default 20
else
grpphabin=$8
fi


fi




#parameters
echo "cx       = " $cx
echo "cy       = " $cy
echo "rin      = " $rin
echo "rout     = " $rout
echo "rin2     = " $rin2
echo "cx2      = " $cx2
echo "cy2      = " $cy2
echo "grpphabin= " $grpphabin

pwd=$PWD

# func:setpass 
setpass(){

export SAS_ODF=`/bin/ls -1 *SUM.SAS`
export SAS_CCF=ccf.cif

}


## according to https://heasarc.gsfc.nasa.gov/docs/xmm/abc/node8.html

# 6.1 Rerun the Pipeline
repro(){

mkdir -p log
echo "[repro] process the MOS data"
rm -f log/emchain.log
#perl `which emchain` > log/emchain.log 2>&1
emchain > log/emchain.log 2>&1

echo "[repro] process the PN data"
rm -f log/epchain.log
#perl `which epchain` > log/epchain.log 2>&1
epchain > log/epchain.log 2>&1

}

# 6.1 (optional) re name event files
everename(){
evem1=`ls *M1**MIEVLI*`
evem2=`ls *M2**MIEVLI*`
evepn=`ls *PN**PIEVLI*`

echo "[everename] please check event file name"
echo "evem1 = " $evem1
echo "evem2 = " $evem2
echo "evepn = " $evepn

cp -f ${evem1} mos1.fits
cp -f ${evem2} mos2.fits
cp -f ${evepn} pn.fits

}


#6.2 Create and Display an Image
createDisplayImage(){

mkdir -p log

echo "[createDisplayImage] : mos1"
rm -f mos1_image.fits log/mos1_image.log
evselect table=mos1.fits withimageset=yes imageset=mos1_image.fits \
         xcolumn=X ycolumn=Y imagebinning=imageSize ximagesize=600 yimagesize=600 > log/mos1_image.log 2>&1

echo "[createDisplayImage] : mos2"
rm -f mos2_image.fits log/mos2_image.log
evselect table=mos2.fits withimageset=yes imageset=mos2_image.fits \
         xcolumn=X ycolumn=Y imagebinning=imageSize ximagesize=600 yimagesize=600 > log/mos2_image.log 2>&1

echo "[createDisplayImage] : pn"
rm -f pn_image.fits log/pn_image.log
evselect table=pn.fits withimageset=yes imageset=pn_image.fits \
         xcolumn=X ycolumn=Y imagebinning=imageSize ximagesize=600 yimagesize=600 > log/pn_image.log 2>&1
}


#6.3 Applying Standard Filters the Data
stdfilt(){

mkdir -p log

echo "[stdfilt] : mos1"
rm -f mos1_filt.fits log/mos1_filt.log
evselect table=mos1.fits withfilteredset=yes expression='(PATTERN $<=$ 12)&&(PI in [200:12000])&&#XMMEA_EM' filteredset=mos1_filt.fits filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes > log/mos1_filt.log 2>&1

echo "[stdfilt] : mos2"
rm -f mos2_filt.fits log/mos2_filt.log
evselect table=mos2.fits withfilteredset=yes expression='(PATTERN $<=$ 12)&&(PI in [200:12000])&&#XMMEA_EM' filteredset=mos2_filt.fits filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes > log/mos2_filt.log 2>&1

echo "[stdfilt] : pn"
rm -f pn_filt.fits log/pn_filt.log
evselect table=pn.fits withfilteredset=yes expression='(PATTERN $<=$ 12)&&(PI in [200:12000])&&#XMMEA_EP' filteredset=pn_filt.fits filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes > log/pn_filt.log 2>&1

}


#6.4 Create and Display a Light Curve
mklc(){

echo "[mklc] : mos1"
rm -f mos1_filt_lc.fits log/mos1_filt_lc.log
evselect table=mos1_filt.fits withrateset=yes rateset=mos1_filt_lc.fits maketimecolumn=yes timecolumn=TIME timebinsize=100 makeratecolumn=yes > log/mos1_filt_lc.log 2>&1

echo "[mklc] : mos2"
rm -f mos2_filt_lc.fits log/mos2_filt_lc.log
evselect table=mos2_filt.fits withrateset=yes rateset=mos2_filt_lc.fits maketimecolumn=yes timecolumn=TIME timebinsize=100 makeratecolumn=yes > log/mos1_filt_lc.log 2>&1

echo "[mklc] : pn"
rm -f pn_filt_lc.fits log/pn_filt_lc.log
evselect table=pn_filt.fits withrateset=yes rateset=pn_filt_lc.fits maketimecolumn=yes timecolumn=TIME timebinsize=100 makeratecolumn=yes > log/mos1_filt_lc.log 2>&1

echo "[mklc] : all"
rm -f log/all_filt_lc.log
./xmm_process_plotlc.sh mos1_filt_lc.fits mos2_filt_lc.fits pn_filt_lc.fits > log/all_filt_lc.log 2>&1

}


#6.5 Applying Time Filters the Data
mkgti(){

echo "[mkgti] : create gtiset.fits"
echo "GTI is created using MOS1, if not, please modify here"
rm -f gtiset.fits log/gtigen.log
tabgtigen table=mos1_filt_lc.fits gtiset=gtiset.fits timecolumn=TIME expression='(RATE<=6)' > log/gtigen.log 2>&1

echo "[mkgti] : apply gtiset.fits into mos1 mos2 pn "

rm -f mos1_filt_time.fits log/mos1_filt_time.log
rm -f mos2_filt_time.fits log/mos2_filt_time.log
rm -f pn_filt_time.fits   log/pn_filt_time.log

evselect table=mos1_filt.fits withfilteredset=yes expression='GTI(gtiset.fits,TIME)' filteredset=mos1_filt_time.fits filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes > log/mos1_filt_time.log 2>&1

evselect table=mos2_filt.fits withfilteredset=yes expression='GTI(gtiset.fits,TIME)' filteredset=mos2_filt_time.fits filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes > log/mos1_filt_time.log 2>&1

evselect table=pn_filt.fits withfilteredset=yes expression='GTI(gtiset.fits,TIME)' filteredset=pn_filt_time.fits filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes > log/mos1_filt_time.log 2>&1


}


# Source Detection with edetect_chain
srcDetect(){

echo "[srcDetect]"
echo "[srcDetect] : First, make the attitude file"
rm -f attitude.fits log/atthkgen.log
atthkgen atthkset=attitude.fits timestep=1 > log/atthkgen.log 2>&1

# mos1
rm -f mos1-s.fits mos1-h.fits mos1-all.fits mos1-all_bin88.fits mos1-all_bin176.fits
rm -f log/mos1_s.log log/mos1_h.log  log/mos1_all.log  log/mos1_all_bin88.log  log/mos1_all_bin176.log 

echo "[srcDetect] : create mos1-s"
evselect table=mos1_filt_time.fits withimageset=yes imageset=mos1-s.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [300:2000])'  > log/mos1_s.log 2>&1

echo "[srcDetect] : create mos1-h"
evselect table=mos1_filt_time.fits withimageset=yes imageset=mos1-h.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [2000:10000])' > log/mos1_h.log 2>&1

echo "[srcDetect] : create mos1-all"
evselect table=mos1_filt_time.fits withimageset=yes imageset=mos1-all.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos1_all.log 2>&1

echo "[srcDetect] : create mos1-all_bin_88"
evselect table=mos1_filt_time.fits withimageset=yes imageset=mos1-all_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos1_all_bin88.log 2>&1

echo "[srcDetect] : create mos1-all_bin_176"
evselect table=mos1_filt_time.fits withimageset=yes imageset=mos1-all_bin176.fits imagebinning=binSize xcolumn=X ximagebinsize=176 ycolumn=Y yimagebinsize=176 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos1_all_bin176.log 2>&1


# mos2
rm -f mos2-s.fits mos2-h.fits mos2-all.fits mos2-all_bin88.fits mos2-all_bin176.fits
rm -f log/mos2_s.log log/mos2_h.log  log/mos2_all.log  log/mos2_all_bin88.log  log/mos2_all_bin176.log 

echo "[srcDetect] : create mos2-s"
evselect table=mos2_filt_time.fits withimageset=yes imageset=mos2-s.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [300:2000])'  > log/mos2_s.log 2>&1

echo "[srcDetect] : create mos2-h"
evselect table=mos2_filt_time.fits withimageset=yes imageset=mos2-h.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [2000:10000])' > log/mos2_h.log 2>&1

echo "[srcDetect] : create mos2-all"
evselect table=mos2_filt_time.fits withimageset=yes imageset=mos2-all.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos2_all.log 2>&1

echo "[srcDetect] : create mos2-all_bin_88"
evselect table=mos2_filt_time.fits withimageset=yes imageset=mos2-all_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos2_all_bin88.log 2>&1

echo "[srcDetect] : create mos2-all_bin_176"
evselect table=mos2_filt_time.fits withimageset=yes imageset=mos2-all_bin176.fits imagebinning=binSize xcolumn=X ximagebinsize=176 ycolumn=Y yimagebinsize=176 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos2_all_bin176.log 2>&1


# pn
rm -f pn-s.fits pn-h.fits pn-all.fits pn-all_bin88.fits pn-all_bin176.fits
rm -f log/pn_s.log log/pn_h.log  log/pn_all.log  log/pn_all_bin88.log  log/pn_all_bin176.log 

echo "[srcDetect] : create pn-s"
evselect table=pn_filt_time.fits withimageset=yes imageset=pn-s.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [300:2000])'  > log/pn_s.log 2>&1

echo "[srcDetect] : create pn-h"
evselect table=pn_filt_time.fits withimageset=yes imageset=pn-h.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [2000:10000])' > log/pn_h.log 2>&1

echo "[srcDetect] : create pn-all"
evselect table=pn_filt_time.fits withimageset=yes imageset=pn-all.fits imagebinning=binSize xcolumn=X ximagebinsize=22 ycolumn=Y yimagebinsize=22 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/pn_all.log 2>&1

echo "[srcDetect] : create pn-all_bin_88"
evselect table=pn_filt_time.fits withimageset=yes imageset=pn-all_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/pn_all_bin88.log 2>&1

echo "[srcDetect] : create pn-all_bin_176"
evselect table=pn_filt_time.fits withimageset=yes imageset=pn-all_bin176.fits imagebinning=binSize xcolumn=X ximagebinsize=176 ycolumn=Y yimagebinsize=176 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/pn_all_bin176.log 2>&1


#rm -f eboxlist_l.fits eboxlist_m.fits eboxlist.fits log/edet.log 
#echo "[srcDetect] : edetect"
#edetect_chain imagesets='mos1-s.fits mos1-h.fits' eventsets='mos1_filt_time.fits' attitudeset=attitude.fits pimin='300 2000' pimax='2000 10000' likemin=10 witheexpmap=yes ecf='0.878 0.220' eboxl_list=eboxlist_l.fits eboxm_list=eboxlist_m.fits eml_list=emllist.fits esp_withootset=no  > log/edet.log 2>&1
#
#rm -f regionfile.txt
#srcdisplay boxlistset=emllist.fits imageset=mos1-all.fits regionfile=regionfile.txt sourceradius=0.01 withregionfile=yes

}


# 6.7 Extract the Source and Background Spectra
mkspec(){

echo "[mkspec] : create src mos1"
rm -f mos1_filtered.fits mos1_pi.fits log/mos1_pi.log log/mos1_bgd_pi.log mos1_bgd_filtered.fits mos1_bgd_pi.fits
evselect table='mos1_filt_time.fits' energycolumn='PI' withfilteredset=yes filteredset='mos1_filtered.fits' keepfilteroutput=yes filtertype='expression' expression='((X,Y) in CIRCLE('${cx}','${cy}','${rin}'))' withspectrumset=yes spectrumset='mos1_pi.fits' spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=11999 > log/mos1_pi.log 2>&1
echo "[mkspec] : create bgd mos1"
evselect table='mos1_filt_time.fits' energycolumn='PI' withfilteredset=yes filteredset='mos1_bgd_filtered.fits' keepfilteroutput=yes filtertype='expression' expression='((X,Y) in CIRCLE('${cx2}','${cy2}','${rout}'))&&!((X,Y) in CIRCLE('${cx2}','${cy2}','${rin2}'))' withspectrumset=yes spectrumset='mos1_bgd_pi.fits' spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=11999 > log/mos1_bgd_pi.log 2>&1

rm -f mos1_filtered_bin88.fits mos1_bgd_filtered_bin88.fits log/mos1_filtered_bin88.log log/mos1_bgd_filtered_bin88.log
echo "[mkspec] : create mos1-all_bin_88 for src"
evselect table=mos1_filtered.fits withimageset=yes imageset=mos1_filtered_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos1_filtered_bin88.log 2>&1
echo "[mkspec] : create mos1-all_bin_88 for bgd"
evselect table=mos1_bgd_filtered.fits withimageset=yes imageset=mos1_bgd_filtered_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos1_bgd_filtered_bin88.log 2>&1


echo "[mkspec] : create src mos2"
rm -f mos2_filtered.fits mos2_pi.fits log/mos2_pi.log log/mos2_bgd_pi.log mos2_bgd_filtered.fits mos2_bgd_pi.fits
evselect table='mos2_filt_time.fits' energycolumn='PI' withfilteredset=yes filteredset='mos2_filtered.fits' keepfilteroutput=yes filtertype='expression' expression='((X,Y) in CIRCLE('${cx}','${cy}','${rin}'))' withspectrumset=yes spectrumset='mos2_pi.fits' spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=11999 > log/mos2_pi.log 2>&1
echo "[mkspec] : create bgd mos2"
evselect table='mos2_filt_time.fits' energycolumn='PI' withfilteredset=yes filteredset='mos2_bgd_filtered.fits' keepfilteroutput=yes filtertype='expression' expression='((X,Y) in CIRCLE('${cx2}','${cy2}','${rout}'))&&!((X,Y) in CIRCLE('${cx2}','${cy2}','${rin2}'))' withspectrumset=yes spectrumset='mos2_bgd_pi.fits' spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=11999 > log/mos2_bgd_pi.log 2>&1

rm -f mos2_filtered_bin88.fits mos2_bgd_filtered_bin88.fits log/mos2_filtered_bin88.log log/mos2_bgd_filtered_bin88.log
echo "[mkspec] : create mos2-all_bin_88 for src"
evselect table=mos2_filtered.fits withimageset=yes imageset=mos2_filtered_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos2_filtered_bin88.log 2>&1
echo "[mkspec] : create mos2-all_bin_88 for bgd"
evselect table=mos2_bgd_filtered.fits withimageset=yes imageset=mos2_bgd_filtered_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/mos2_bgd_filtered_bin88.log 2>&1


echo "[mkspec] : create src pn"
rm -f pn_filtered.fits pn_pi.fits log/pn_pi.log log/pn_bgd_pi.log pn_bgd_filtered.fits pn_bgd_pi.fits
evselect table='pn_filt_time.fits' energycolumn='PI' withfilteredset=yes filteredset='pn_filtered.fits' keepfilteroutput=yes filtertype='expression' expression='((X,Y) in CIRCLE('${cx}','${cy}','${rin}'))' withspectrumset=yes spectrumset='pn_pi.fits' spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=20479 > log/pn_pi.log 2>&1
echo "[mkspec] : create bgd pn"
evselect table='pn_filt_time.fits' energycolumn='PI' withfilteredset=yes filteredset='pn_bgd_filtered.fits' keepfilteroutput=yes filtertype='expression' expression='((X,Y) in CIRCLE('${cx2}','${cy2}','${rout}'))&&!((X,Y) in CIRCLE('${cx2}','${cy2}','${rin2}'))' withspectrumset=yes spectrumset='pn_bgd_pi.fits' spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=20479 > log/pn_bgd_pi.log 2>&1
rm -f pn_filtered_bin88.fits pn_bgd_filtered_bin88.fits log/pn_filtered_bin88.log log/pn_bgd_filtered_bin88.log
echo "[mkspec] : create pn-all_bin_88 for src"
evselect table=pn_filtered.fits withimageset=yes imageset=pn_filtered_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/pn_filtered_bin88.log 2>&1
echo "[mkspec] : create pn-all_bin_88 for bgd"
evselect table=pn_bgd_filtered.fits withimageset=yes imageset=pn_bgd_filtered_bin88.fits imagebinning=binSize xcolumn=X ximagebinsize=88 ycolumn=Y yimagebinsize=88 filtertype=expression expression='(FLAG == 0)&&(PI in [300:10000])'  > log/pn_bgd_filtered_bin88.log 2>&1


}

# 6.10 Determine the Spectrum Extraction Areas
mkarea(){

echo "[mkarea] : backscale for pi files"

rm -f log/bs_mos1.log  log/bs_mos1bgd.log 
backscale spectrumset=mos1_pi.fits badpixlocation=mos1_filt_time.fits      > log/bs_mos1.log 2>&1
backscale spectrumset=mos1_bgd_pi.fits badpixlocation=mos1_filt_time.fits  > log/bs_mos1bgd.log 2>&1

rm -f log/bs_mos2.log  log/bs_mos2bgd.log 
backscale spectrumset=mos2_pi.fits badpixlocation=mos2_filt_time.fits      > log/bs_mos2.log 2>&1
backscale spectrumset=mos2_bgd_pi.fits badpixlocation=mos2_filt_time.fits  > log/bs_mos2bgd.log 2>&1

rm -f log/bs_pn.log  log/bs_pnbgd.log 
backscale spectrumset=pn_pi.fits badpixlocation=pn_filt_time.fits          > log/bs_pn.log 2>&1
backscale spectrumset=pn_bgd_pi.fits badpixlocation=pn_filt_time.fits      > log/bs_pnbgd.log 2>&1

}

# 6.11 Create the Photon Redistribution Matrix (RMF) and Ancillary File (ARF)
mkrmfarf(){

echo "[mkrmfarf] : mos1"
rm -f mos1_rmf.fits log/mos1_rmf.log mos1_arf.fits
rmfgen rmfset=mos1_rmf.fits spectrumset=mos1_pi.fits > log/mos1_rmf.log 2>&1
arfgen arfset=mos1_arf.fits spectrumset=mos1_pi.fits withrmfset=yes rmfset=mos1_rmf.fits withbadpixcorr=yes badpixlocation=mos1_filt_time.fits > log/mos1_arf.log 2>&1

echo "[mkrmfarf] : mos2"
rm -f mos2_rmf.fits log/mos2_rmf.log mos2_arf.fits log/mos2_arf.log
rmfgen rmfset=mos2_rmf.fits spectrumset=mos2_pi.fits > log/mos2_rmf.log 2>&1
arfgen arfset=mos2_arf.fits spectrumset=mos2_pi.fits withrmfset=yes rmfset=mos2_rmf.fits withbadpixcorr=yes badpixlocation=mos2_filt_time.fits > log/mos2_arf.log 2>&1

echo "[mkrmfarf] : pn"
rm -f pn_rmf.fits log/pn_rmf.log pn_arf.fits log/pn_arf.log
rmfgen rmfset=pn_rmf.fits spectrumset=pn_pi.fits > log/pn_rmf.log 2>&1
arfgen arfset=pn_arf.fits spectrumset=pn_pi.fits withrmfset=yes rmfset=pn_rmf.fits withbadpixcorr=yes badpixlocation=pn_filt_time.fits > log/pn_arf.log 2>&1

}


# 13. Fitting an EPIC Spectrum in XSPEC
mkgrppha(){

if [ $# != 1 ]; then
  echo "ERROR: Too few arguments. need binsize" 1>&2
  return 1
fi

binsize=$1

echo "[mkgrppha] : binsize = " $binsize
echo "[mkgrppha] : mos1"

rm -f mos1_pi_grp${binsize}.fits log/mos1_grppha.log
grppha <<EOF > log/mos1_grppha.log 2>&1
mos1_pi.fits
mos1_pi_grp${binsize}.fits
chkey BACKFILE mos1_bgd_pi.fits
chkey RESPFILE mos1_rmf.fits
chkey ANCRFILE mos1_arf.fits
group min ${binsize}
exit
EOF

echo "[mkgrppha] : mos2"
rm -f mos2_pi_grp${binsize}.fits log/mos2_grppha.log
grppha <<EOF > log/mos2_grppha.log 2>&1
mos2_pi.fits
mos2_pi_grp${binsize}.fits
chkey BACKFILE mos2_bgd_pi.fits
chkey RESPFILE mos2_rmf.fits
chkey ANCRFILE mos2_arf.fits
group min ${binsize}
exit
EOF

echo "[mkgrppha] : pn"
rm -f pn_pi_grp${binsize}.fits log/pn_grppha.log
grppha <<EOF > log/pn_grppha.log 2>&1
pn_pi.fits
pn_pi_grp${binsize}.fits
chkey BACKFILE pn_bgd_pi.fits
chkey RESPFILE pn_rmf.fits
chkey ANCRFILE pn_arf.fits
group min ${binsize}
exit
EOF

}


# sas-thread-timing : EXTRACTION AND CORRECTION OF AN X-RAY LIGHT CURVE FOR A POINT-LIKE SOURCE
mklcps(){

if [ $# != 3 ]; then
  echo "ERROR: Too few arguments. energylow energyhigh timebin" 1>&2
  return 1
fi

lcenelow=$1
lcenehigh=$2
lctimebin=$3
ftag=${lcenelow}_${lcenehigh}_${lctimebin}
mkdir -p log

echo "[mklcps] : mos1 src"
rm -f mos1_filtered_lc_${ftag}.fits log/mos1_filtered_lc_${ftag}.log
evselect table=mos1_filtered.fits energycolumn=PI withrateset=yes rateset=mos1_filtered_lc_${ftag}.fits maketimecolumn=yes timecolumn=TIME timebinsize=${lctimebin} makeratecolumn=yes filtertype=expression expression='(FLAG == 0)&&(PI in ['${lcenelow}':'${lcenehigh}'])' > log/mos1_filtered_lc_${ftag}.log 2>&1
echo "[mklcps] : mos1 bgd"
rm -f mos1_bgd_filtered_lc_${ftag}.fits log/mos1_bgd_filtered_lc_${ftag}.log
evselect table=mos1_bgd_filtered.fits energycolumn=PI withrateset=yes rateset=mos1_bgd_filtered_lc_${ftag}.fits maketimecolumn=yes timecolumn=TIME timebinsize=${lctimebin} makeratecolumn=yes filtertype=expression expression='(FLAG == 0)&&(PI in ['${lcenelow}':'${lcenehigh}'])' > log/mos1_bgd_filtered_lc_${ftag}.log 2>&1


echo "[mklcps] : mos2 src"
rm -f mos2_filtered_lc_${ftag}.fits log/mos2_filtered_lc_${ftag}.log
evselect table=mos2_filtered.fits energycolumn=PI withrateset=yes rateset=mos2_filtered_lc_${ftag}.fits maketimecolumn=yes timecolumn=TIME timebinsize=${lctimebin} makeratecolumn=yes filtertype=expression expression='(FLAG == 0)&&(PI in ['${lcenelow}':'${lcenehigh}'])' > log/mos2_filtered_lc_${ftag}.log 2>&1

echo "[mklcps] : mos2 bgd"
rm -f mos2_bgd_filtered_lc_${ftag}.fits log/mos2_bgd_filtered_lc_${ftag}.log
evselect table=mos2_bgd_filtered.fits energycolumn=PI withrateset=yes rateset=mos2_bgd_filtered_lc_${ftag}.fits maketimecolumn=yes timecolumn=TIME timebinsize=${lctimebin} makeratecolumn=yes filtertype=expression expression='(FLAG == 0)&&(PI in ['${lcenelow}':'${lcenehigh}'])' > log/mos2_bgd_filtered_lc_${ftag}.log 2>&1


echo "[mklcps] : pn src"
rm -f pn_filtered_lc_${ftag}.fits log/pn_filtered_lc_${ftag}.log
evselect table=pn_filtered.fits energycolumn=PI withrateset=yes rateset=pn_filtered_lc_${ftag}.fits maketimecolumn=yes timecolumn=TIME timebinsize=${lctimebin} makeratecolumn=yes filtertype=expression expression='(FLAG == 0)&&(PI in ['${lcenelow}':'${lcenehigh}'])' > log/pn_filtered_lc_${ftag}.log 2>&1

echo "[mklcps] : pn bgd"
rm -f pn_bgd_filtered_lc_${ftag}.fits log/pn_bgd_filtered_lc_${ftag}.log
evselect table=pn_bgd_filtered.fits energycolumn=PI withrateset=yes rateset=pn_bgd_filtered_lc_${ftag}.fits maketimecolumn=yes timecolumn=TIME timebinsize=${lctimebin} makeratecolumn=yes filtertype=expression expression='(FLAG == 0)&&(PI in ['${lcenelow}':'${lcenehigh}'])' > log/pn_bgd_filtered_lc_${ftag}.log 2>&1


echo "[mklcps] : mos1 net"
rm -f mos1_net_filtered_lc_${ftag}.fits log/mos1_net_filtered_lc_${ftag}.log
epiclccorr srctslist=mos1_filtered_lc_${ftag}.fits eventlist=mos1_filt_time.fits outset=mos1_net_filtered_lc_${ftag}.fits bkgtslist=mos1_bgd_filtered_lc_${ftag}.fits withbkgset=yes applyabsolutecorrections=yes > log/mos1_net_filtered_lc_${ftag}.log 2>&1

echo "[mklcps] : mos2 net"
rm -f mos2_net_filtered_lc_${ftag}.fits log/mos2_net_filtered_lc_${ftag}.log
epiclccorr srctslist=mos2_filtered_lc_${ftag}.fits eventlist=mos2_filt_time.fits outset=mos2_net_filtered_lc_${ftag}.fits bkgtslist=mos2_bgd_filtered_lc_${ftag}.fits withbkgset=yes applyabsolutecorrections=yes > log/mos2_net_filtered_lc_${ftag}.log 2>&1

echo "[mklcps] : pn net"
rm -f pn_net_filtered_lc_${ftag}.fits log/pn_net_filtered_lc_${ftag}.log
epiclccorr srctslist=pn_filtered_lc_${ftag}.fits eventlist=pn_filt_time.fits outset=pn_net_filtered_lc_${ftag}.fits bkgtslist=pn_bgd_filtered_lc_${ftag}.fits withbkgset=yes applyabsolutecorrections=yes > log/pn_net_filtered_lc_${ftag}.log 2>&1

echo "[mklcps] : all"
rm -f log/all_net_filtered_lc_${ftag}.log
./xmm_process_plotlc.sh mos1_net_filtered_lc_${ftag}.fits mos2_net_filtered_lc_${ftag}.fits pn_net_filtered_lc_${ftag}.fits > log/all_net_filtered_lc_${ftag}.log 2>&1


}



run()
{

setpass # or setpass $basedir $odffile
repro               # 6.1 Rerun the Pipeline
everename            # 6.1 (optional) re name event files
createDisplayImage   # 6.2 Create and Display an Image
stdfilt              # 6.3 Applying Standard Filters the Data
mklcps                 # 6.4 Create and Display a Light Curve
mkgti                # 6.5 Applying Time Filters the Data
srcDetect            # 6.6 Source Detection with edetect_chain
mkspec               # 6.7 Extract the Source and Background Spectra
mkarea               # 6.10 Determine the Spectrum Extraction Areas
mkrmfarf             # 6.11 Create the Photon Redistribution Matrix (RMF) and Ancillary File (ARF)
mkgrppha $grpphabin          # 13. Fitting an EPIC Spectrum in XSPEC

## extra process 
mklcps 500   2000 100 # 500eV-2000eV dt=100sec, sas-thread-timing : EXTRACTION AND CORRECTION OF AN X-RAY LIGHT CURVE FOR A POINT-LIKE SOURCE
mklcps 2000 10000 100 # 500eV-2000eV dt=100sec, sas-thread-timing : EXTRACTION AND CORRECTION OF AN X-RAY LIGHT CURVE FOR A POINT-LIKE SOURCE
mklcps 500  10000 100 # 500eV-2000eV dt=100sec, sas-thread-timing : EXTRACTION AND CORRECTION OF AN X-RAY LIGHT CURVE FOR A POINT-LIKE SOURCE
#
}

# main program start from here
run 
