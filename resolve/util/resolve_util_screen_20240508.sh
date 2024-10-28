#!/bin/sh

usage() {
    echo "Usage: $0 inputevt outputevt"
    exit 1
}

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    usage
fi

input=$1
output=$2
echo "input = " $input
echo "output = " $output

ftcopy infile="${input}[EVENTS][(PI>=600) && (((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4)) && STATUS[4]==b0 && (((TICK_SHIFT<4)&&(TICK_SHIFT>1))||(DERIV_MAX>=15000)||(DERIV_MAX<6000))&& (((TICK_SHIFT<4)&&(TICK_SHIFT>0))||(DERIV_MAX>=6000)||(DERIV_MAX<2000))&& (((TICK_SHIFT<3)&&(TICK_SHIFT>-1))||(DERIV_MAX>=2000)||(DERIV_MAX<1000))&& (((TICK_SHIFT<2)&&(TICK_SHIFT>-2))||(DERIV_MAX>=1000)||(DERIV_MAX<500))&& (((TICK_SHIFT<1)&&(TICK_SHIFT>-3))||(DERIV_MAX>=500)||(DERIV_MAX<400))&& (((TICK_SHIFT<0)&&(TICK_SHIFT>-4))||(DERIV_MAX>=400)||(DERIV_MAX<300))&& (((TICK_SHIFT<-1)&&(TICK_SHIFT>-5))||(DERIV_MAX>=300)||(DERIV_MAX<200))&& (((TICK_SHIFT<-3)&&(TICK_SHIFT>-7))||(DERIV_MAX>=200)||(DERIV_MAX<100))&& (((TICK_SHIFT<-4)&&(TICK_SHIFT>-8))||(DERIV_MAX>=100))]" \
       outfile=${output} copyall=yes clobber=yes history=yes chatter=5
