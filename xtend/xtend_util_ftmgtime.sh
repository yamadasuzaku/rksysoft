#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : xtend_ftmgtime.sh"
exit
fi

obs=$1
outfile="${obs%.evt}.gti"
rm -f $outfile

ftmgtime "${obs},${obs}" $outfile AND
