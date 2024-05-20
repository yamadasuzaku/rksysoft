#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : xtend_util_check_grade.sh"
exit
fi

obs=$1
ftlist $obs+1 K include=OBJECT; echo COUNTS GRADE;
ftlist $obs T column=GRADE | awk 'NR>3 {print $2}' | sort | uniq -c
