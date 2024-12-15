#!/bin/sh 

#for obsid in 0075140401 
#for obsid in 0075140401 0504010101 0694870201 0137151601
for obsid in 0504010101 0694870201 0137151601
do

echo $obsid 
date

mkdir -p log
rm -f log/autorun_${obsid}.log

./xmm_process_autorun.sh $obsid > log/autorun_${obsid}.log 2>&1

done 
