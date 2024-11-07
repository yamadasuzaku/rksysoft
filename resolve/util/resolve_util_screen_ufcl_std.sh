#!/bin/bash

# Define colors for output using printf-compatible ANSI codes
GREEN=$(printf '\033[32m')
RED=$(printf '\033[31m')
CYAN=$(printf '\033[36m')
RESET=$(printf '\033[0m')


printf "${CYAN} [start] resolve_util_screen_ufcl_std.sh ${RESET}\n"

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
check_program_in_path "resolve_util_ftselect.sh"
check_program_in_path "ftlist"
check_program_in_path "ftcopy"

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
ufevt=$1
ufclgtievt="${ufevt%.evt}_clgti.evt"
output1="${ufevt%.evt}_clgti_stdufcut1.evt"
output2="${ufevt%.evt}_clgti_stdufcut2.evt"

output_st2="${ufevt%.evt}_clgti_st2.evt"
output_st3="${ufevt%.evt}_clgti_st3.evt"
output_st4="${ufevt%.evt}_clgti_st4.evt"
output_st6="${ufevt%.evt}_clgti_st6.evt"


# Base file name
base_top=$(basename "$ufevt" _uf.evt)
clevt=${base_top}_cl.evt
outgti="${clevt%.evt}.gti"

# File existence checks
if [ ! -f "$ufevt" ]; then
    printf "${RED}Error: Input file %s does not exist.${RESET}\n" "$ufevt"
    exit 1
fi
if [ -f "$output1" ] || [ -f "$output2" ]; then
    printf "${RED}Warning: Output files %s or %s already exist. They will be overwritten.${RESET}\n" "$output1" "$output2"
fi

# Remove any existing outgti file
rm -f "$outgti"

# Execute commands with status messages
printf "${CYAN}Running resolve_util_ftmgtime.sh...${RESET}\n"
resolve_util_ftmgtime.sh "$clevt"
printf "${GREEN}Completed: resolve_util_ftmgtime.sh${RESET}\n"

printf "${CYAN}Running resolve_util_ftselect.sh...${RESET}\n"
resolve_util_ftselect.sh "$ufevt" "gtifilter(\"$outgti\")" clgti
printf "${GREEN}Completed: resolve_util_ftselect.sh${RESET}\n"

# Display input and output file paths
printf "${CYAN}Input file:${RESET} %s\n" "$ufevt"
printf "${CYAN}Output file 1:${RESET} %s\n" "$output1"
printf "${CYAN}Output file 2:${RESET} %s\n" "$output2"

# Input file check
printf "${CYAN}>>>>>>>>>>>> Input check : ${ufclgtievt} >>>>>>>>>>>>>>>${RESET}\n"
ftlist "$ufclgtievt" H

# Generate output 1
printf "${CYAN}>>>>>>>>>>>> Generating output : 1 ${output1} >>>>>>>>>>>>>>>${RESET}\n"
ftcopy infile="${ufclgtievt}[EVENTS][(ITYPE<5)&&((SLOPE_DIFFER==b0||PI>22000))&&(QUICK_DOUBLE==b0)&&(STATUS[3]==b0)&&(STATUS[6]==b0)&&(STATUS[2]==b0)&&(PI>200)&&(RISE_TIME<127)&&(PIXEL!=12)&&(TICK_SHIFT>-8&&TICK_SHIFT<7)]" \
       outfile="$output1" copyall=yes clobber=yes history=yes chatter=5
ftlist "$output1" H

# Generate output 2
printf "${CYAN}>>>>>>>>>>>> Generating output 2 : ${output2} >>>>>>>>>>>>>>>${RESET}\n"
ftcopy infile="${output1}[EVENTS][(PI>=600)&&(((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))&&STATUS[4]==b0]" \
       outfile="$output2" copyall=yes clobber=yes history=yes chatter=5
ftlist "$output2" H

echo "..... (cal) STATUS bit check "

# Generate output STATUS2
printf "${CYAN}>>>>>>>>>>>> Generating output_st2 ${output_st2} >>>>>>>>>>>>>>>${RESET}\n"
ftcopy infile="${ufclgtievt}[EVENTS][(ITYPE<5)&&(STATUS[2]==b1)&&(PIXEL!=12)]" \
       outfile="$output_st2" copyall=yes clobber=yes history=yes chatter=5
ftlist "$output_st2" H

# Generate output STATUS3
printf "${CYAN}>>>>>>>>>>>> Generating output_st3 ${output_st3} >>>>>>>>>>>>>>>${RESET}\n"
ftcopy infile="${ufclgtievt}[EVENTS][(ITYPE<5)&&(STATUS[3]==b1)&&(PIXEL!=12)]" \
       outfile="$output_st3" copyall=yes clobber=yes history=yes chatter=5
ftlist "$output_st3" H

# Generate output STATUS4
printf "${CYAN}>>>>>>>>>>>> Generating output_st4 ${output_st4} >>>>>>>>>>>>>>>${RESET}\n"
ftcopy infile="${ufclgtievt}[EVENTS][(ITYPE<5)&&(STATUS[4]==b1)&&(PIXEL!=12)]" \
       outfile="$output_st4" copyall=yes clobber=yes history=yes chatter=5
ftlist "$output_st4" H

# Generate output STATUS6
printf "${CYAN}>>>>>>>>>>>> Generating output_st6 ${output_st6} >>>>>>>>>>>>>>>${RESET}\n"
ftcopy infile="${ufclgtievt}[EVENTS][(ITYPE<5)&&(STATUS[6]==b1)&&(PIXEL!=12)]" \
       outfile="$output_st6" copyall=yes clobber=yes history=yes chatter=5
ftlist "$output_st6" H

printf "${CYAN} [end] resolve_util_screen_ufcl_std.sh ${RESET}\n"

