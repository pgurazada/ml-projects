
# coding: utf-8

# In[1]:


# Base

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Plot settings

get_ipython().magic('matplotlib inline')
sns.set_context('talk')
sns.set_palette('gray')
sns.set_style('ticks', {'grid.color' : '0.9'})


# In[3]:


# Algorithms

from sklearn.ensemble import RandomForestClassifier


# In[4]:


# Model selection

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


# In[5]:


x_train = np.load('processed/wells_feature_matrix.npy')
y_train = pd.read_csv('processed/wells_labels_train.csv')['status'].tolist()


# We begin by looking at the variation in the performance as we hand-tune some of the parameters. We will then put together a parameter grid that will possiblly find the best combination among all the parameters. For now, we will look at accuracy as the metric. At the moment the focus is on observing the contours of accuracy as we change the hyperparameters

# ### Number of estimators

# In[9]:


n_estimators_list = [50, 100, 500, 1000]


# In[10]:


rf_classif = RandomForestClassifier(n_estimators=50,
                                    max_features='auto',
                                    random_state=20130810,
                                    n_jobs=3)


# In[12]:


rf_cv = GridSearchCV(estimator=rf_classif, 
                     param_grid={'n_estimators' : n_estimators_list},
                     scoring='accuracy',
                     n_jobs=3)


# In[13]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[18]:


rf_cv.cv_results_['mean_test_score']


# ### Maximum depth of the tree

# ### Maximum features is the number of features to consider at each split 

# ### Minimum samples to split an internal node

# ### Minimum samples to be a leaf node

# Now we tune the random forests model over a parameter grid

# In[ ]:


pgrid = {'criterion' : ['gini', 'entropy'],
         'min_samples_leaf' : [1, 10, 20, 50],
         'min_samples_split' : [2, 10, 20, 30],
         'n_estimators' : [100, 500, 1000]}


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators = 100,
                                    max_features = 'auto',
                                    oob_score = True,
                                    random_state = 20130810,
                                    n_jobs = -1)


# In[ ]:


rf_cv = RandomizedSearchCV(estimator=rf_classif,
                           param_distributions = pgrid,
                           n_iter = 25,
                           cv = 10,
                           n_jobs = -1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.best_params_

