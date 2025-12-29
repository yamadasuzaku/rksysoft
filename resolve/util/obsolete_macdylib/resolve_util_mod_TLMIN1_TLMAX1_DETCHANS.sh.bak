#!/bin/sh

# Define colors for output using printf-compatible ANSI codes
GREEN=$(printf '\033[32m')
RED=$(printf '\033[31m')
CYAN=$(printf '\033[36m')
RESET=$(printf '\033[0m')

# Usage message
usage() {
    printf "${CYAN}Usage: %s <ufevt file>${RESET}\n" "$0"
    exit 1
}

# Check arguments
if [ $# -ne 1 ]; then
    usage
fi

# Variables
pha=$1
fparkey 0 ${pha} TLMIN1 
fparkey 59999 ${pha} TLMAX1 
fparkey 60000 ${pha} DETCHANS
ftselect ${pha} cut_${pha} '0<=CHANNEL&&CHANNEL<60000'

#fkeyprint uf_ls_clgti.pha DETCHANS
# DETCHANS=                60000 / total number possible channels
#fkeyprint uf_ls_clgti.pha TLMIN
# TLMIN1  =                    0 / Lowest legal channel number
#fkeyprint uf_ls_clgti.pha TLMAX
# TLMAX1  =                59999 / Highest legal channel number
#  fparkey 0 inv_uf_ls_clgti.pha[1] TLMIN1 
