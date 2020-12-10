#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import os.path


# In[37]:


path = os.path.join("results","resnet_2_1d_test_results.txt") 

lines = []
labels = []
outputs = []

with open(path,"r") as f:
    for line in f:
        l = [float(x) for x in line[1:-3].split(", ")]
        lines.append(l)
        
labels, outputs = lines[::2], lines[1::2]


# In[41]:


for i in range(len(labels)):
    plt.figure()
    line_t, = plt.plot(labels[i], label="target")
    line_o, = plt.plot(outputs[i], label="output")
    axes = plt.gca()
    axes.set_xlim([0,12])
    plt.ylabel('Velocity')
    plt.xlabel('Column')
    plt.legend(handles=[line_t, line_o])
    plt.show()


# In[ ]:




