#!/usr/bin/env python
# coding: utf-8

# # Data Validation

# In[1]:


import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import gurobipy as gp
from   gurobipy import GRB
import time
import datetime
import math
import csv
import os
import warnings
warnings.filterwarnings('ignore') 


# ## 1. Reading the data

# In[2]:


with open('results/VolIn_3.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = {}
    for row in reader:
        data1[row[0]] = eval(row[1])
VolIn = data1 

with open('results/VolOut_3.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = {}
    for row in reader:
        data1[row[0]] = eval(row[1])
VolOut = data1  

with open('results/VolExist_3.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = {}
    for row in reader:
        data1[row[0]] = eval(row[1])
VolExist = data1   


# ## 2. First round of validations
# 
# Check that all the keys are the same. Note that: 
# 1. Keys correspond to the different <b> cycles </b>.
# 2. Keys of VolIn, VolOut & VolExist <b> should be the same. </b>

# In[3]:


keys1 = sorted(list(VolIn.keys()))
keys2 = sorted(list(VolOut.keys()))
keys3 = sorted(list(VolExist.keys()))
print("****** Check 1: Keys ******")
if keys1 == keys2 and keys1 == keys3:
    print("Success: Keys are the same")
else:
    print("Failure: Keys are not the same")
    print(sorted(keys1))
    print(sorted(keys2))
    print(sorted(keys3))   


# ## 3. Second round of validations
# 
# In the second round of validations we check:
# 1. Visually check the <b> lines </b>
# 2. Whether the <b> products </b> in VolIn & VolOut are the same

# In[4]:


def validation_metadata(VolExist, VolIn, VolOut):
    
    #--------------------------------------------------------
    # Check 1: Lines
    #
    lines = []
    for cycle in VolIn:
        lines.extend((list(VolIn[cycle].keys())))   
    lines = set(lines) 
    lines1 = sorted(lines)

    lines = []
    for cycle in VolOut:
        lines.extend((list(VolOut[cycle].keys())))   
    lines  = set(lines) 
    lines2 = sorted(lines)
    print("****** Check 1: Lines ******")
    print("VolIn  -> ", lines1)
    print("VolOut -> ", lines2)
    
    #--------------------------------------------------------
    # Check 2: Products
    #
    prod = []
    for cycle in VolIn:
        for line in VolIn[cycle]:
            prod.extend((list(VolIn[cycle][line].keys())))   
    prod = set(prod) 
    prod1 = sorted(prod)

    prod = []
    for cycle in VolOut:
        for line in VolOut[cycle]:
            prod.extend((list(VolOut[cycle][line].keys())))   
    prod  = set(prod) 
    prod2 = sorted(prod)
    
    prod = []
    for cycle in VolExist:
        for tank in VolExist[cycle]:
            prod.extend((list(VolExist[cycle][tank].keys())))   
    prod  = set(prod) 
    prod3 = sorted(prod)
    
    print("****** Check 2: Products ******")
    print("VolIn  -> ", prod1)
    print("VolOut -> ", prod2)
    print("VolExist -> ", prod3)
    if prod1 == prod2:
        print("Success: Products are the same")
    else:
        print("Failure: Products are not the same")


# In[5]:


validation_metadata(VolExist, VolIn, VolOut)


# ## 4. Third round
# 
# In the third round we check
# 1. The inflow & outflow of each <b> product </b> and whether the net is positive. Otherwise there is a data error in the original files that needs to be removed. 

# In[6]:


def validation_data(VolExist, VolIn, VolOut):

    ret = {}
    for cycle in VolIn:

        CycleVolIn    = VolIn[cycle]
        CycleVolOut   = VolOut[cycle]
        CycleVolExist = VolExist[cycle]

        data = []
        for line in CycleVolOut:
            for prod in CycleVolOut[line]:
                row = [prod, CycleVolOut[line][prod]]
                data.append(row)

        d  = pd.DataFrame(data, columns=['Product', 'Volume'])  
        d  = d.groupby(['Product']).agg({'Volume': 'sum'}) 
        d  = d.reset_index()
        dO = d

        data = []
        for line in CycleVolIn:
            for prod in CycleVolIn[line]:
                row = [prod, CycleVolIn[line][prod]]
                data.append(row)

        d  = pd.DataFrame(data, columns=['Product', 'Volume'])  
        d  = d.groupby(['Product']).agg({'Volume': 'sum'}) 
        d  = d.reset_index()
        dI = d

        data = []
        for line in CycleVolExist:
            for prod in CycleVolExist[line]:
                row = [prod, CycleVolExist[line][prod]]
                data.append(row)

        d  = pd.DataFrame(data, columns=['Product', 'Volume'])  
        d  = d.groupby(['Product']).agg({'Volume': 'sum'}) 
        d  = d.reset_index()
        dE = d

        dEI = pd.merge(dE, dI, on=['Product'], how = 'outer')
        df = pd.merge(dEI, dO, on=['Product'], how = 'outer')
        df = df.rename(columns={'Volume_x': 'Volume_Exists', 'Volume_y': 'Volume_In', 'Volume': 'Volume_Out'})
        df['Diff'] = df['Volume_Exists'] + df['Volume_In'] - df['Volume_Out']

        ret[cycle] = df
        
    return (ret)        


# In[7]:


ret = validation_data(VolExist, VolIn, VolOut)
for cycle in ret:
    print(cycle)
    print(ret[cycle])


# In[8]:


prods  = ['A', 'D', '54', '62']
#rods  = ['A']
cycles = ['041', '051', '061', '071', '081']

for prod in prods:
    vol = []
    df     = ret['031']
    a  = df.loc[df['Product'] == prod]['Volume_Exists'] + df.loc[df['Product'] == prod]['Volume_In'] - df.loc[df['Product'] == prod]['Volume_Out']
    vol.append(a)
    for cycle in cycles:
        df = ret[cycle]
        a  = a + df.loc[df['Product'] == prod]['Volume_In'] - df.loc[df['Product'] == prod]['Volume_Out']
        vol.append(a)
    print("For Product " + prod + " the sequence is " + str(vol))


# ## 5. Aligning with model inputs
# 
# In this section we make sure the data align with the <b> fixed </b> model inputs.

# In[9]:


get_ipython().run_line_magic('run', './Execution_1_fixedinputs.ipynb')


# ### Tanks

# In[10]:


tanks_ = []
for cycle in VolExist:
    row = list(VolExist[cycle].keys())
    tanks_.extend(row)

my_dict_data = sorted(set(tanks_))


# In[11]:


for tank in my_dict_data:
    print(tank, tank in Tanks)


# ### Tanks: Check products going through tanks 

# In[12]:


tanks2 = []
for cycle in VolExist:
    for tank in VolExist[cycle]:
        for prod in VolExist[cycle][tank]:
            tanks2.append([tank, prod])
tanks2 = [list(x) for x in set(tuple(x) for x in tanks2)]

from operator import itemgetter
tanks2 = sorted(tanks2, key=itemgetter(0))
    
my_dict = {}    
for key, value in tanks2:
    if key in my_dict:
        my_dict[key].append(value)
    else:
        my_dict[key] = [value] 
        
my_dict      


# ### Topology: Tank/Line/Product Routes
# 
# Check whether the data topology is a subset of the input topology.

# In[13]:


tanks2 = []
for tank in topo_o:
    for line in topo_o[tank]:
        for prod in topo_o[tank][line]:
            tanks2.append([tank, line, prod])
            
my_dict = {}            
for tank, line, prod in tanks2:
    if tank in my_dict:
        my_dict[tank].append([line,prod])
    else:
        my_dict[tank] = [[line,prod]]

mydict_input = my_dict


# In[14]:


with open('results/Topology_3.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = {}
    for row in reader:
        data1[row[0]] = eval(row[1])
mydict_data = data1  


# Compare the two topologies

# In[15]:


for tank in mydict_data:
    a = mydict_data[tank]
    if int(tank) not in mydict_input:
        b = [] 
    else:
        b = mydict_input[int(tank)]
    c = set(map(tuple, a)).issubset(set(map(tuple, b)))
    print(tank, ": ", c)
    if c is False:
        print(a, "***" , b)


# ### Topology: Line/Product

# In[16]:


lines2 = []
for cycle in VolOut:
    for line in VolOut[cycle]:
        for prod in VolOut[cycle][line]:
            lines2.append([line, prod])
lines2 = [list(x) for x in set(tuple(x) for x in lines2)]
from operator import itemgetter
sorted(lines2, key=itemgetter(0))

my_dict = {}

for key, value in lines2:
    if key in my_dict:
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
        
sorted_dict = {k: v for k, v in sorted(my_dict.items())}
mydict_data = sorted_dict
mydict_data


# In[17]:


lines2 = []
for tank in topo_o:
    for line in topo_o[tank]:
        for prod in topo_o[tank][line]:
            lines2.append([line, prod])
            
lines2 = [list(x) for x in set(tuple(x) for x in lines2)]
from operator import itemgetter
lines2 = sorted(lines2, key=itemgetter(0))  

my_dict = {}
for key, value in lines2:
    if key in my_dict:
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]

mydict_input = my_dict
mydict_input


# In[18]:


for line in mydict_data:
    a = mydict_data[line]
    if line not in mydict_input:
        b = [] 
    else:
        b = mydict_input[line]
    c = set(map(tuple, a)).issubset(set(map(tuple, b)))
    print(line, ": ", c)
    if c is False:
        print(a, "***" , b)   


# ## Notes
# 
# 1. The data has Tanks not present in the matrix: 310, 360, 370, 373
# 2. The data also has topology paths not present in the matrix:
#     2.1 ['20', 'A'] 
#     2.2 ['20', 'D']
# 

# In[19]:


#-------------------------------------------------
# 021
#
# cycle = '021' 
# CycleVolIn2   = VolIn[cycle]
# CycleVolOut2  = VolOut[cycle]
# CycleVolExist  = VolExist[cycle]

# temp_ = {}    
# for tank in CycleVolExist:
#      for prod in CycleVolExist[tank]:
#         if prod in temp_:
#              temp_[prod] = temp_[prod] + CycleVolExist[tank][prod]
#         else:
#              temp_[prod] = CycleVolExist[tank][prod]
# temp_ = {k:v for k,v in sorted(temp_.items())}
# temp_Exist = temp_

# temp_ = {}    
# for tank in CycleVolOut2:
#     for prod in CycleVolOut2[tank]:
#         if prod in temp_:
#             temp_[prod] = temp_[prod] + CycleVolOut2[tank][prod]
#         else:
#             temp_[prod] = CycleVolOut2[tank][prod]
# temp_ = {k:v for k,v in sorted(temp_.items())}
# temp_Out = temp_

# temp_ = {}    
# for tank in CycleVolIn2:
#     for prod in CycleVolIn2[tank]:
#         if prod in temp_:
#             temp_[prod] = temp_[prod] + CycleVolIn2[tank][prod]
#         else:
#             temp_[prod] = CycleVolIn2[tank][prod]
# temp_ = {k:v for k,v in sorted(temp_.items())}
# temp_In = temp_

# temp_In
# temp_Out
# temp_Exist
 
# temp_Exist_next = {}
# for prod in temp_Exist:
#     temp_Exist_next[prod] = temp_Exist[prod] + temp_In[prod] - temp_Out[prod]
  
# print("Load at the end of 021 - Theoretical")
# print(temp_Exist)
# print(temp_In)
# print(temp_Out)
# print("-------------")
# print(temp_Exist_next)
# print("-------------")
# #---------------------------------------------------------------------------------
# #
# #
# with open("results/opt_CycleVolExist_1.csv", "r") as file:
#     reader = csv.reader(file)
#     next(reader) # skip header row
#     my_dict = {int(rows[0]): eval(rows[1]) for rows in reader} 
    
# CycleVolExist = my_dict 
# temp_ = {}    
# for tank in CycleVolExist:
#      for prod in CycleVolExist[tank]:
#         if prod in temp_:
#              temp_[prod] = temp_[prod] + CycleVolExist[tank][prod]
#         else:
#              temp_[prod] = CycleVolExist[tank][prod]
# temp_ = {k:v for k,v in sorted(temp_.items())}
# print("Load at the end of 021 - Model")
# print(temp_)
# print("-------------")
# #---------------------------------------------------------------------------------
# #
# #
# print("Load at the beginning of 031")
# CycleVolExist  = VolExist['031']
# temp_ = {}    
# for tank in CycleVolExist:
#      for prod in CycleVolExist[tank]:
#         if prod in temp_:
#              temp_[prod] = temp_[prod] + CycleVolExist[tank][prod]
#         else:
#              temp_[prod] = CycleVolExist[tank][prod]
# temp_ = {k:v for k,v in sorted(temp_.items())}
# temp_Exist = temp_
# print(temp_)

