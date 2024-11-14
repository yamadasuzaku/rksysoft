#!/bin/bash

echo "[INFO] Starting the script..."

if [ $# -lt 2 ]; then
    echo "[ERROR] Invalid number of arguments."
    echo "Usage: $0 <clevt> <ufevt> "
    echo "Example: $0 xa300041010rsl_p0px1000_cl.evt xa300041010rsl_p0px1000_uf.evt"
    exit 1
fi

clevt=$1
ufevt=$2
clgti="${clevt%.evt}.gti"
ufcutevt="${ufevt%.evt}_clgti.evt"

resolve_util_ftmgtime.sh $clevt
resolve_util_ftselect.sh $ufevt "gtifilter(\"${clgti}\")" clgti

resolve_util_fplot.py ${ufcutevt} TIME 1,1,1 DERIV_MAX,LO_RES_PH,RISE_TIME 1,1,1 --filters "ITYPE==4,PIXEL==0"
resolve_util_fplot.py ${ufcutevt} DERIV_MAX 1,1 LO_RES_PH,RISE_TIME 1,1 --filters "ITYPE==4,PIXEL==0"

resolve_util_fplot.py ${ufcutevt} TIME 1,1,1 DERIV_MAX,LO_RES_PH,RISE_TIME 1,1,1 --filters "ITYPE==4,PIXEL==1"
resolve_util_fplot.py ${ufcutevt} DERIV_MAX 1,1 LO_RES_PH,RISE_TIME 1,1 --filters "ITYPE==4,PIXEL==1"
