#!/bin/bash

# Define colors for output using printf-compatible ANSI codes
GREEN=$(printf '\033[32m')
RED=$(printf '\033[31m')
CYAN=$(printf '\033[36m')
RESET=$(printf '\033[0m')

# Check if obsid argument is provided
if [ -z "$1" ]; then
    echo "${RED}Error: obsid argument is required.${RESET}"
    exit 1
fi

obsid=$1

# Check if the required files exist
if [ ! -f "xa${obsid}.ehk" ]; then
    echo "${RED}Error: File xa${obsid}.ehk does not exist.${RESET}"
    exit 1
fi

if [ ! -f "xa${obsid}rsl_a0ac_uf.evt" ]; then
    echo "${RED}Error: File xa${obsid}rsl_a0ac_uf.evt does not exist.${RESET}"
    exit 1
fi

# Run maketime for low elevation
echo "${CYAN}Running maketime for low elevation (ELV < -5)...${RESET}"
maketime xa${obsid}.ehk xa${obsid}_elvlo.ehk "SAA_SXS==0&&ELV<-5" compact=no time=TIME clobber=yes

# Run maketime for high elevation
echo "${CYAN}Running maketime for high elevation (ELV > 5)...${RESET}"
maketime xa${obsid}.ehk xa${obsid}_elvhi.ehk "SAA_SXS==0&&ELV>5" compact=no time=TIME clobber=yes

# Apply ftselect for low elevation
echo "${CYAN}Applying ftselect for low elevation...${RESET}"
ftselect infile=xa${obsid}rsl_a0ac_uf.evt outfile=xa${obsid}rsl_a0ac_uf_elvlo.evt expr="gtifilter('xa${obsid}_elvlo.ehk')" chatter=5 clobber=yes

# Apply ftselect for high elevation
echo "${CYAN}Applying ftselect for high elevation...${RESET}"
ftselect infile=xa${obsid}rsl_a0ac_uf.evt outfile=xa${obsid}rsl_a0ac_uf_elvhi.evt expr="gtifilter('xa${obsid}_elvhi.ehk')" chatter=5 clobber=yes

# Generate GTI plot
echo "${CYAN}Generating GTI plot...${RESET}"
resolve_util_gtiplot.py xa${obsid}rsl_a0ac_uf.evt,xa${obsid}rsl_a0ac_uf_elvhi.evt,xa${obsid}rsl_a0ac_uf_elvlo.evt -e xa${obsid}rsl_a0ac_uf.evt,xa${obsid}rsl_a0ac_uf_elvhi.evt,xa${obsid}rsl_a0ac_uf_elvlo.evt

# Create list of files
echo "${CYAN}Creating list of event files...${RESET}"
ls xa${obsid}rsl_a0ac_uf.evt xa${obsid}rsl_a0ac_uf_elvhi.evt xa${obsid}rsl_a0ac_uf_elvlo.evt > antico_compelv.list

# Run histogram analysis
echo "${CYAN}Running histogram analysis...${RESET}"
resolve_ana_pixel_hist1d_many_eventfiles_antico.py antico_compelv.list --x_col PI --xmin 0 --xmax 2000 --rebin 1 -o antico

# Indicate completion
echo "${GREEN}Processing completed successfully.${RESET}"
