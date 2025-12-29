#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : resolve_util_check_quickdouble_slope_differ.sh"
exit
fi

obs=$1
ftlist $obs+1 K include=OBJECT
ftlist $obs+1 K include=DATE-OBS,DATE-END
ftlist $obs+1 K include=PROCVER,DATE,TLM2FITS,SOFTVER,CALDBVER
echo COUNTS QUICK_DOUBLE SLOPE_DIFFER;
ftlist $obs T column=QUICK_DOUBLE,SLOPE_DIFFER | awk 'NR>3 {print $2,$3}' | sort | uniq -c
