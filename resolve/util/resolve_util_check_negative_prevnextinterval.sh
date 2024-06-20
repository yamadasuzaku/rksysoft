#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : resolve_util_check_negative_prevnextinterval.sh"
exit
fi

obs=$1
ftlist $obs+1 K include=OBJECT
ftlist $obs+1 K include=DATE-OBS,DATE-END
ftlist $obs+1 K include=PROCVER,DATE,TLM2FITS,SOFTVER,CALDBVER
echo COUNTS PREV_INTERVAL NEXT_INTERVAL;
ftlist $obs T column=PREV_INTERVAL,NEXT_INTERVAL | awk '(NR>3)&&($2<0) {print $2,$3}' | sort | uniq -c
