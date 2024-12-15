#!/bin/sh

copy_vla() 
{
cp -rf /Users/syamada/work/ana/ss433/ss433/ana/0075140501/ana/for_paper/VLA-remake_4_TAN_J2000.fits .
}

dolink(){

if [ $# != 2 ];
then
echo "[ERROR] need to specify directory name"
return 1
fi

dir=$1
name=$2
echo "..... mergeimg () : $dir $name "

for file in adapt-2000-7200-all.fits adapt-400-750-all.fits adapt-750-1300-all.fits size-2000-7200-all.fits size-400-750-all.fits   size-750-1300-all.fits
do

ln -sf $dir/$file ${name}_${file}

done 

}



mergeimg(){


if [ $# != 1 ];
then
echo "[ERROR] need to specify directory"
return 1
fi

dir=$1

echo "..... mergeimg () : $dir "

for file in adapt-2000-7200-all.fits adapt-400-750-all.fits adapt-750-1300-all.fits 
do

ftag=`basename ${file} .fits`
efile=`echo $file | sed -e 's/adapt/size/g' `
etag=`basename $efile .fits`

echo $file $ftag $efile $etag

rm -f ${ftag}.list
rm -f ${etag}.list
ls *_${file} > ${ftag}.list
ls *_${efile} > ${etag}.list

rm -f sum_${ftag}.fits
rm -f sumexp_${ftag}.fits
rm -f sumcor_${ftag}.fits


ecl <<EOF

imcombine @${ftag}.list sum_${ftag}.fits combine="sum" offsets="wcs"

imcombine @${etag}.list sumexp_${ftag}.fits combine="sum" offsets="wcs"

imarith sum_${ftag}.fits / sumexp_${ftag}.fits  sumcor_${ftag}.fits

logout 

EOF

done 


mkdir -p $dir

mv -f *.list $dir
mv -f *.list $dir
mv -f sum*.fits $dir

echo "..... mergeimg () : ouput is stored in $dir "

}


run(){

copy_vla # if needed  

# find image files 
dolink /Users/syamada/work/ana/ss433/ss433_since20180731/ana/image img1 # SS433-jet2、2004-10-04, eastern ear
dolink /Users/syamada/work/ana/ss433/ss433/ana/0075140501/ana img2 # SS433-jet2、2004-10-04, eastern ear
dolink ../copy_from_mpulx10_ss433image/mostleft img3
dolink ../copy_from_mpulx10_ss433image/center1 img4

#
mergeimg sumall 


}

run