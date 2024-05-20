#!/bin/bash

if [ _$1 = _ ];
then
cat << EOF
usage : xtend_create_detimg.sh (1:cleaned event file)
EOF
exit
fi

evt_file=$1
new_filename="${evt_file%.evt}.img"

tmp=tmp.log
rm -f $tmp
rm -f $new_filename

xselect << EOF 2>&1 | tee $tmp

no

read event $evt_file ./
yes

filter pha_cutoff 83 1667

extract image

save image $new_filename
exit
no

EOF
