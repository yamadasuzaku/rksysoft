#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : resolve_util_check_pixelrate.sh"
exit
fi

obs=$1
ftlist $obs+1 K include=OBJECT; echo COUNTS PIXEL ITYPE;
ftlist $obs T column=PIXEL,ITYPE | awk 'NR>3 {print $2,$3}' | sort -k1,1n -k2,2n | uniq -c
