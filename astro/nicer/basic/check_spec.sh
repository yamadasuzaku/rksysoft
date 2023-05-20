#!/bin/sh 

# set working directory
cdir=`pwd` 

xmin=0.5
xmax=10
ymin=1e-1
ymax=100000

num=1

for obsid in `cat obsid.list`
do

cd ../$obsid 
pifile=`ls ../$obsid/*tot_*.pi`
if [ _${pifile} = _ ]; 
then
echo "pifile is not found : " $pifile 
cd $cdir
continue 
fi

dateobs=`fkeyprint $pifile DATE-OBS | tail -1 | awk '{printf("%s",$2)}' | sed -e "s/\'//g" `
dateend=`fkeyprint $pifile DATE-END | tail -1 | awk '{printf("%s",$2)}' | sed -e "s/\'//g" `
exposure=`fkeyprint $pifile EXPOSURE | tail -1 | awk '{printf("%.1f",$2)}'`

ftag=`basename $pifile .pi`
outputtag=${ftag}_plld

rm -f ${outputtag}*

xspec <<EOF

data 1:1 $pifile 

setplot e 

pl ld 

iplot 

la t $dateobs exp=$exposure

r x $xmin $xmax
r y $ymin $ymax

col $num on 1 

hard ${outputtag}.ps/cps
we $outputtag

exit 

EOF

ps2pdf ${outputtag}.ps

num=`echo $num | awk '{print $1+1}'`

cd $cdir 
done 