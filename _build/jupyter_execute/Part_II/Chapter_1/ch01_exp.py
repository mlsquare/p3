#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import os
import random
import time
import numpy as np
import pandas as pd

import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict

from scipy import stats

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import ipywidgets as widgets
from IPython.display import display, clear_output


# In[22]:


def sim_y(xa,xs,a,b):
    p = np.exp(-((xa*a)+(xs*b)))
    y = np.random.binomial(1,p)
    return y


# In[23]:


n = 5000
m = 25
high = 10
a = np.random.uniform(low=0,high=high,size=n)
b = np.random.uniform(low=0,high=high,size=n)
y = np.zeros(n)
for i in range(n):
    y[i] = sim_y(1,0,a[i],b[i])
plt.hist(y)
plt.show()
print(np.mean(y))
print((1-np.exp(-high))/high)


# In[28]:


n = 500
m = 25
high = 0.01
a = np.random.uniform(low=0,high=high,size=n)
b = np.random.uniform(low=0,high=high,size=n)
y = np.zeros((n,m))
for i in range(n):
    a_local = a[i]
    b_local = b[i]
    y[i,0] = np.random.binomial(1,0.2)
    for j in range(1,m):
        xs = np.sum(y[i,:(j-1)])
        xa = (j)-xs
        #print(i,j,xa,xs)
        y[i,j] = sim_y(xa,xs,a_local,b_local)

yb = np.mean(y,axis=0)
plt.plot(yb)
plt.show()


# In[17]:


yb


# In[14]:


xa


# In[15]:


xs


# In[ ]:




