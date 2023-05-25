#!/usr/bin/env python 

#fname="hexdump_PSPA0_CPU.txt"
fname="hexdump_PSPA0_FPGA.txt"
f=open(fname)
aoffset = int("04000000",16) # 0x0400:0000
for i, one in enumerate(f):
	print(aoffset+i, int(one.strip(),16)) 

