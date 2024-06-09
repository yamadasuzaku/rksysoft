#!/bin/sh

if [ _$1 = _ ];
then
    echo "usage : resolve_util_ftmgtime.sh obs.evt"
    echo "create GTI file using inputfile"
exit
fi

obs=$1
outfile="${obs%.evt}.gti"
rm -f $outfile

ftmgtime "${obs},${obs}" $outfile AND
