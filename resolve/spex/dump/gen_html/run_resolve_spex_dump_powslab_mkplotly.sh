#!/bin/bash

for qdpfile in `ls ../output_spexqdp/*qdp`
do
echo $qdpfile 
tralfile=../output_spextral/`basename $qdpfile .qdp | sed -e 's/qdp/tral/g' `.asc

if [ -e $tralfile ]; then
echo "File exists : $tralfile"
resolve_spex_dump_powslab_mkplotly.py $qdpfile -s $tralfile 
resolve_spex_dump_powslab_mkplotly.py $qdpfile -s $tralfile --plotly
else
echo "File does not exist, $tralfile"
resolve_spex_dump_powslab_mkplotly.py $qdpfile
resolve_spex_dump_powslab_mkplotly.py $qdpfile --plotly
fi

done 

