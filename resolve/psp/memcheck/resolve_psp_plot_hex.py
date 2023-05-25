#!/usr/bin/evn python 

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import numpy as np

debug = False

def convint(file):
	f = open(file)
	x=[]
	y=[]
	for one in f:
		x.append(int(one.strip().split()[0]))
		y.append(int(one.strip().split()[1]))
	return np.array(x), np.array(y)

a0_cpu_x,   a0_cpu_y  = convint("hexdump_PSPA0_CPU_addrow.txt")
a0_fpga_x, a0_fpga_y  = convint("hexdump_PSPA0_FPGA_addrow.txt")
a0_temp_x, a0_temp_y  = convint("ram-211213_Resolve_FM_TC4-PSPA0_TEMPLATE.txt")

offset = a0_cpu_x[0] 

offset2 = 4000
xmin = a0_temp_x[0] - offset2
xmax = a0_temp_x[-1] + offset2

# plt.figure(figsize=(12,8))
# plt.title("check contents of A0")
# plt.subplot(211)
# plt.plot(a0_cpu_x - offset, a0_cpu_y,   'k.', label = "A0 CPU", ms = 0.3, alpha = 0.8)
# plt.plot(a0_fpga_x - offset, a0_fpga_y, 'g.', label = "A0 FPGA", ms = 0.3, alpha = 0.8)
# plt.plot(a0_temp_x - offset, a0_temp_y, 'ro', label = "A0 TEMPLATE", ms = 0.8, alpha = 0.8)

# plt.legend(numpoints=1, loc="best")
# plt.grid(linestyle='dotted',alpha=0.5)

# plt.subplot(212)
# plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_fpga_y, 'r.', label = "A0 CPU - A0 FPGA", ms = 0.3)
# plt.xlabel("SRAM Data (Byte) from the offset of " + str(hex(offset)))

# plt.legend(numpoints=1, loc="best")
# plt.grid(linestyle='dotted',alpha=0.5)
# plt.savefig("comp_A0_all.png")
# plt.show()


# ---- Compare Entire SRAM

# cutid = np.where( (a0_cpu_x > xmin) & (a0_cpu_x < xmax) ) 
# a0_cpu_x = a0_cpu_x[cutid]
# a0_cpu_y =  a0_cpu_y[cutid]
# a0_fpga_x = a0_fpga_x[cutid] 
# a0_fpga_y = a0_fpga_y[cutid] 

plt.figure(figsize=(11,7))
plt.title("check contents of A0")
plt.subplot(211)
plt.ylabel("Data of 8 bits")
plt.plot(a0_cpu_x - offset, a0_cpu_y,   'k|', label = "A0 CPU", ms = 3, alpha = 0.8)
plt.plot(a0_fpga_x - offset, a0_fpga_y, 'b_', label = "A0 FPGA", ms = 3, alpha = 0.8)
plt.plot(a0_temp_x - offset, a0_temp_y, 'rx', label = "A0 TEMPLATE", ms = 3, alpha = 0.3)
#plt.xlim(xmin - offset,xmax - offset)
plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=5, borderaxespad=0.,fontsize=13)
plt.grid(linestyle='dotted',alpha=0.5)

# cutid = np.where( (a0_cpu_x >= a0_temp_x[0]) & (a0_cpu_x <= a0_temp_x[-1]) ) 
# a0_cpu_x = a0_cpu_x[cutid]
# a0_cpu_y =  a0_cpu_y[cutid]
# a0_fpga_x = a0_fpga_x[cutid] 
# a0_fpga_y = a0_fpga_y[cutid] 

plt.subplot(212)
plt.ylabel("Difference")
plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_fpga_y, 'r+', label = "A0 CPU - A0 FPGA", ms = 2)
# plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_temp_y, 'k+', label = "A0 CPU - A0 TEMPLATE", ms = 2)
# plt.plot(a0_cpu_x - offset, a0_fpga_y - a0_temp_y + 1, 'b+', label = "A0 FPGA - A0 TEMPLATE + 1", ms = 2)

#plt.xlim(xmin - offset,xmax - offset)
plt.xlabel("SRAM Data (Byte) from the offset of " + str(hex(offset)))
plt.legend(numpoints=1, loc="best")
plt.grid(linestyle='dotted',alpha=0.5)
plt.savefig("comp_A0_template_entireSRAM.png")
if debug: plt.show()


# ---- Compare TEMPLATE All 

cutid = np.where( (a0_cpu_x > xmin) & (a0_cpu_x < xmax) ) 
a0_cpu_x = a0_cpu_x[cutid]
a0_cpu_y =  a0_cpu_y[cutid]
a0_fpga_x = a0_fpga_x[cutid] 
a0_fpga_y = a0_fpga_y[cutid] 

plt.figure(figsize=(11,7))
plt.title("check contents of A0")
plt.subplot(211)
plt.ylabel("Data of 8 bits")
plt.plot(a0_cpu_x - offset, a0_cpu_y,   'k|', label = "A0 CPU", ms = 3, alpha = 0.8)
plt.plot(a0_fpga_x - offset, a0_fpga_y, 'b_', label = "A0 FPGA", ms = 3, alpha = 0.8)
plt.plot(a0_temp_x - offset, a0_temp_y, 'rx', label = "A0 TEMPLATE", ms = 3, alpha = 0.3)
plt.xlim(xmin - offset,xmax - offset)
plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=5, borderaxespad=0.,fontsize=13)
plt.grid(linestyle='dotted',alpha=0.5)

cutid = np.where( (a0_cpu_x >= a0_temp_x[0]) & (a0_cpu_x <= a0_temp_x[-1]) ) 
a0_cpu_x = a0_cpu_x[cutid]
a0_cpu_y =  a0_cpu_y[cutid]
a0_fpga_x = a0_fpga_x[cutid] 
a0_fpga_y = a0_fpga_y[cutid] 

plt.subplot(212)
plt.ylabel("Difference")
#plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_fpga_y, 'r+', label = "A0 CPU - A0 FPGA", ms = 2)
plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_temp_y, 'k+', label = "A0 CPU - A0 TEMPLATE", ms = 2)
plt.plot(a0_cpu_x - offset, a0_fpga_y - a0_temp_y + 1, 'b+', label = "A0 FPGA - A0 TEMPLATE + 1", ms = 2)

plt.xlim(xmin - offset,xmax - offset)
plt.xlabel("SRAM Data (Byte) from the offset of " + str(hex(offset)))
plt.legend(numpoints=1, loc="best")
plt.grid(linestyle='dotted',alpha=0.5)
plt.savefig("comp_A0_template_all.png")
if debug: plt.show()


# ---- Compare TEMPLATE zoom
a0_cpu_x,   a0_cpu_y  = convint("hexdump_PSPA0_CPU_addrow.txt")
a0_fpga_x, a0_fpga_y  = convint("hexdump_PSPA0_FPGA_addrow.txt")
a0_temp_x, a0_temp_y  = convint("ram-211213_Resolve_FM_TC4-PSPA0_TEMPLATE.txt")

xmin = np.median(a0_temp_x[0]) - offset2
xmax = np.median(a0_temp_x[0]) + offset2

cutid = np.where( (a0_cpu_x > xmin) & (a0_cpu_x < xmax) ) 
a0_cpu_x = a0_cpu_x[cutid]
a0_cpu_y =  a0_cpu_y[cutid]
a0_fpga_x = a0_fpga_x[cutid] 
a0_fpga_y = a0_fpga_y[cutid] 

plt.figure(figsize=(11,7))
#plt.title("check contents of A0")
plt.subplot(111)
plt.ylabel("Data of 8 bits")
plt.plot(a0_cpu_x - offset, a0_cpu_y,   'k|', label = "A0 CPU", ms = 3, alpha = 0.8)
plt.plot(a0_fpga_x - offset, a0_fpga_y, 'b_', label = "A0 FPGA", ms = 3, alpha = 0.8)
plt.plot(a0_temp_x - offset, a0_temp_y, 'rx', label = "A0 TEMPLATE", ms = 3, alpha = 0.3)
plt.xlim(xmin - offset,xmax - offset)
plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=5, borderaxespad=0.,fontsize=13)
plt.grid(linestyle='dotted',alpha=0.5)

# cutid = np.where( (a0_cpu_x > xmin) & (a0_cpu_x < xmax) ) 
# #cutid = np.where( (a0_cpu_x >= a0_temp_x[0]) & (a0_cpu_x <= a0_temp_x[-1]) ) 
# a0_cpu_x = a0_cpu_x[cutid]
# a0_cpu_y =  a0_cpu_y[cutid]
# a0_fpga_x = a0_fpga_x[cutid] 
# a0_fpga_y = a0_fpga_y[cutid] 

# plt.subplot(212)
# plt.ylabel("Difference")
# #plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_fpga_y, 'r+', label = "A0 CPU - A0 FPGA", ms = 2)
# plt.plot(a0_cpu_x - offset, a0_cpu_y - a0_temp_y, 'k+', label = "A0 CPU - A0 TEMPLATE", ms = 2)
# plt.plot(a0_cpu_x - offset, a0_fpga_y - a0_temp_y + 1, 'b+', label = "A0 FPGA - A0 TEMPLATE + 1", ms = 2)

# plt.xlim(xmin - offset,xmax - offset)
plt.xlabel("SRAM Data (Byte) from the offset of " + str(hex(offset)))
# plt.legend(numpoints=1, loc="best")
# plt.grid(linestyle='dotted',alpha=0.5)
plt.savefig("comp_A0_template_zoom.png")
if debug: plt.show()
