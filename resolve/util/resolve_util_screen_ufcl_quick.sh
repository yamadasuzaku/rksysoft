#!/bin/bash

# Define colors for output using printf-compatible ANSI codes
GREEN=$(printf '\033[32m')
RED=$(printf '\033[31m')
CYAN=$(printf '\033[36m')
RESET=$(printf '\033[0m')


printf "${CYAN} [start] resolve_util_screen_ufcl_quick.sh ${RESET}\n"

# Function to check if a program is in PATH
check_program_in_path() {
    local program_name="$1"
    if ! command -v "$program_name" &> /dev/null; then
        printf "${RED}Error: %s not found in \$PATH.${RESET}\n" "$program_name"
        exit 1
    else
        printf "${GREEN}%s is found in \$PATH${RESET}\n" "$program_name"
    fi
}

# Check for required programs
check_program_in_path "resolve_util_ftmgtime.sh"
check_program_in_path "ftselect"

# Usage message
usage() {
    printf "${CYAN}Usage: %s <clevt file> <ufevt file>${RESET}\n" "$0"
    exit 1
}

# Check arguments
if [ $# -ne 2 ]; then
    usage
fi

# Variables
clevt=$1
ufevt=$2
ufclgtievt="${ufevt%.evt}_clgti.evt"

# Base file name
outgti="${clevt%.evt}.gti"


# File existence checks
if [ ! -f "$clevt" ]; then
    printf "${RED}Error: Input file %s does not exist.${RESET}\n" "$ufevt"
    exit 1
fi
if [ ! -f "$ufevt" ]; then
    printf "${RED}Error: Input file %s does not exist.${RESET}\n" "$ufevt"
    exit 1
fi

# Remove any existing outgti file
rm -f "$outgti"

# Execute commands with status messages
printf "${CYAN}Running resolve_util_ftmgtime.sh...${RESET}\n"
resolve_util_ftmgtime.sh "$clevt"
printf "${GREEN}Completed: resolve_util_ftmgtime.sh${RESET}\n"

printf "${CYAN}Running ftselect with $outgti...${RESET}\n"
rm -rf ${ufclgtievt}
ftselect infile=${ufevt} outfile=${ufclgtievt} expr="gtifilter('${outgti}')" chatter=5 clobber=yes
printf "${GREEN}Completed: ftselect ${RESET}\n"
