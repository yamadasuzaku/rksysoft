#!/usr/bin/env python 

fname="ram-211213_Resolve_FM_TC4-PSPA0_TEMPLATE.cps"
f=open(fname)
for i, one in enumerate(f):
	if i == 0:
		continue # skip header		
#	print("one = ", one.split())
	addr = int(one.strip().split(" ")[0], 16)
	content = one.strip().split(" ")[2]

	it = iter(list(content)) 
	for j, (a, b) in enumerate(zip(it, it)):
		onedata = int(a,16) * 16 + int(b,16)
		print(addr+j, onedata)

