
# coding: utf-8

# In this workbook we assemble the required features of the data set as per the observations from the exploratory data analysis.

# In[37]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
get_ipython().magic('matplotlib inline')


# In[4]:


wells_features = pd.read_csv('data/well_features.csv')
wells_labels = pd.read_csv('data/well_labels.csv')


# In[5]:


wells_features.info()


# In[6]:


wells_labels.info()


# In[8]:


wells_labels.status_group.value_counts()


# ### Encoding the labels
# 
# Encoding the labels into numeric values is simple since there are only three categories. Using a dictionary in such cases makes the intent also explicit

# In[10]:


status_group_to_numeric = {'functional needs repair' : 0,
                           'functional' : 1,
                           'non functional' : 2}


# In[11]:


wells_labels['status'] = wells_labels['status_group'].map(status_group_to_numeric)


# In[12]:


wells_labels.status.value_counts()


# ### Encoding the features

# In[16]:


wells_features.shape


# In[64]:


gps_ht_bins = pd.qcut(wells_features.gps_height, 4, labels=range(4))


# In[65]:


gps_ht_bins.value_counts()


# In[69]:


wells_features.construction_year.value_counts(ascending=True)


# In[71]:


def bin_construction_yr(c):
    if c >= 1960 and c < 1970:
        return 1
    elif c >= 1971 and c < 1980:
        return 2
    elif c >= 1981 and c < 1990:
        return 3
    elif c >= 1991 and c < 2000:
        return 4
    elif c >= 2001 and c < 2010:
        return 5
    elif c >= 2011 and c < 2020:
        return 6
    else:
        return 0


# In[75]:


construct_yr_bins = wells_features.construction_year.apply(bin_construction_yr)


# In[76]:


construct_yr_bins.value_counts()


# In[78]:


wells_features.amount_tsh.describe()


# In[79]:


def is_tsh_zero(tsh):
    if tsh == 0:
        return 1
    else:
        return 0


# In[84]:


def take_log(tsh):
    if tsh == 0:
        return 0
    else:
        return np.log(tsh)


# In[86]:


tsh_zero = wells_features.amount_tsh.apply(is_tsh_zero)


# In[88]:


def group_funded(funder):
    if funder == 'Government Of Tanzania': return 'Govt'
    elif funder == 'Danida': return 'F1'
    elif funder == 'Hesawa': return 'F2'
    elif funder == 'Rwssp': return 'F3'
    elif funder == 'World Bank': return 'F4'
    elif funder == 'Kkkt': return 'F5'
    elif funder == 'World Vision': return 'F6'
    elif funder == 'Unicef': return 'F7'
    elif funder == 'Tasaf': return 'F8'
    elif funder == 'District Council': return 'F9'
    else: 
        return 'Oth'


# In[89]:


funded_by = wells_features.funder.apply(group_funded)


# In[90]:


funded_by.value_counts()


# In[94]:


wells_features.info()


# In[93]:


wells_features = wells_features.assign(gps_ht_bin = pd.qcut(wells_features.gps_height, 4, labels=range(4)),
                                       construct_yr_bin = wells_features.construction_year.apply(bin_construction_yr),
                                       tsh = wells_features.amount_tsh.apply(take_log),
                                       tsh_zero = wells_features.amount_tsh.apply(is_tsh_zero),
                                       funded_by = wells_features.funder.apply(group_funded))

