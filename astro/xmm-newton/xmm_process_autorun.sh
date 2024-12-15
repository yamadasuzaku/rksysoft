#!/bin/sh

#20180803 Shinya Yamada, TMU, ver1 
#[Usave] ./xmm_process_autorun.sh obsid cx cy rin rout 

obsid=0075140501
cx=25286.2
cy=18506.7
rin=1200 
rout=3600

if [ _$1 = _ ];
then
echo "[ERROR] need to specify obsid cx cy rin rout"
echo "e.g., "
echo "./xmm_process_autorun.sh " $obsid $cx $cy $rin $rout
exit

else
obsid=$1

if [ _$2 = _ ];
then
echo "spetral analysis is not performed. "
else
cx=$2
cy=$3
rin=$4
rout=$5
echo "spetral analysis is performed. "
echo "cx       = " $cx
echo "cy       = " $cy
echo "rin      = " $rin
echo "rout     = " $rout
fi
fi

#parameters
echo "obsid    = " $obsid

pwd=$PWD

createdir(){

mkdir -p $obsid
mkdir -p $obsid/data # where to store original data
mkdir -p $obsid/ana  # where to analyze data
mkdir -p $obsid/ana/spec   # where to analyze spectral analysis
mkdir -p $obsid/ana/image  # where to analyze image analysis
mkdir -p $obsid/ana/timing  # where to analyze timing analysis

}


getdata(){

echo "..... getdata()"

cd $obsid/data

curl -o ${obsid}.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=${obsid}" 

cd $pwd

}

unpackdata(){

echo "..... unpackdata()"

cd $obsid/data

tar -xvf ${obsid}.tar

cd $obsid/odf

tar -zxvf ${obsid}.tar.gz
tar -xvf *.TAR

cd $pwd

}



set_cifsum(){

echo "..... set_sifsum()"

cd $obsid/data/${obsid}/odf 

export SAS_ODF=`pwd` # set current dir 

rm -f cifbuild.log 
cifbuild > cifbuild.log 2>&1
ls -thrl | tail -1 

export SAS_CCF=ccf.cif

rm -f odfingest.log 
odfingest > odfingest.log 2>&1
ls -thrl | tail -1

cd $pwd

}


set_ql(){

echo "..... set_ql()"

cd $obsid/ana/spec

ln -fs ../../data/${obsid}/odf/*FIT .
ln -fs ../../data/${obsid}/odf/*SUM.SAS .
ln -fs ../../data/${obsid}/odf/ccf.cif .

cd $pwd

}


do_ql(){

echo "..... do_ql()"

cd $obsid/ana/spec

xmm_process_ql.sh $cx $cy $rin $rout

cd $pwd

}



set_image(){

echo "..... set_image()"

cd $obsid/ana/image/

ln -fs ../../data/${obsid}/odf/*FIT .
ln -fs ../../data/${obsid}/odf/*SUM.SAS .
ln -fs ../../data/${obsid}/odf/ccf.cif .

cd $pwd

}


do_image(){

echo "..... do_image()"

cd $obsid/ana/image

xmm_process_image_each.sh

cd $pwd

}




run()
{

createdir # create analysis directoty tree
getdata # download data
unpackdata # unpack data
set_cifsum # set ccf.cif and odf 
#set_ql # set quick run 
#do_ql # do quick run 
set_image # set image run
do_image # do image run

}

# main program start from here
run 
