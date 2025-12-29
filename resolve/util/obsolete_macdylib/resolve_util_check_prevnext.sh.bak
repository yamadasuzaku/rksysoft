#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : resolve_util_check_prevnext.sh"
exit
fi

obs=$1
ftlist $obs+1 K include=OBJECT; echo COUNTS PREV_INTERVAL NEXT_INTERVAL TICK_SHIFT;
ftlist $obs T column=PREV_INTERVAL,NEXT_INTERVAL,TICK_SHIFT | awk 'NR>3 {print $2,$3,$4}' | sort | uniq -c
