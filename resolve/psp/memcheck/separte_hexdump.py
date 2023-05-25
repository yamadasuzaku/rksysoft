#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


filename="230331_PSPDUMP.txt"


# In[3]:


with open (filename,"r") as f:
    linedata=f.readlines()


# In[4]:


# hexdata={"PSPA0_CPU":np.array(None),"PSPA1_CPU":np.array(None),"PSPA0_FPGA":np.array(None)}


# In[5]:


hexdata={"PSPA0_CPU":[],"PSPA1_CPU":[],"PSPA0_FPGA":[]}
h=""
hprev=""
l=""
lprev=""
fl={}
el={}
nl={}
for l in linedata:
    if l.startswith("2023-03-31"):
        hh=int(l[11:13])
        mm=int(l[14:16])
        if (hh==5) and (mm>18) and (mm<49):
            h="PSPA0_CPU" # command sent at 05:19:03.3
        elif ((hh==5) and (mm>49)) or ((hh==6) and (mm<6)):
            h="PSPA1_CPU" # command sent at 05:19:03.3
        elif ((hh==6) and (mm>6)):
            h="PSPA0_FPGA" # command sent at 06:07:35.3
        else:
            print ("Invalid hour and min ",l)
        if h!=hprev:
            fl[h]=l[0:25]
            if hprev!="":
                el[hprev]=lprev[0:25]
                nl[hprev]=n
            n=0
        hprev=h
        lprev=l
        n+=1
    else:
        ll=l.split(':')
        if (len(ll)>1) and (int(ll[0])==20):
            hexdata[h].append(ll[1].split()[5:])
        elif (len(ll)>1) and (int(ll[0])>20):
            hexdata[h].append(ll[1].split())
el[h]=lprev[0:25]
nl[h]=n
print(fl,"\n",el,"\n",nl)


# In[6]:


for h in hexdata.keys():
    hexdata[h]=np.array([item for sublist in hexdata[h] for item in sublist])


# In[7]:


for h in hexdata.keys():
    print("%d %x" % (hexdata[h].size,hexdata[h].size))


# In[8]:


print(hexdata['PSPA0_CPU'][0:30])


# In[14]:


for h in hexdata.keys():
    outfile="hexdump_%s.txt" % h
    np.savetxt(outfile,hexdata[h],fmt="%s", delimiter='')


# In[ ]:




