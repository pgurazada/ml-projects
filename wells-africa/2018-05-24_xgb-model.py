
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

import xgboost as xgb


# In[4]:


# Model selection

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


# In[5]:


x_train = np.load('processed/wells_feature_matrix.npy')
y_train = pd.read_csv('processed/wells_labels_train.csv')['status'].tolist()


# In[19]:


xgb_clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.001, random_state=20130810)


# In[20]:


get_ipython().run_cell_magic('time', '', "scores = cross_val_score(xgb_clf,\n                         x_train,\n                         y_train, \n                         scoring = 'accuracy',\n                         cv = 5,\n                         n_jobs = 3,\n                         verbose = 3)")


# We begin by looking at the variation in the performance as we hand-tune some of the parameters. We will then put together a parameter grid that will possiblly find the best combination among all the parameters. For now, we will look at accuracy as the metric. At the moment the focus is on observing the contours of accuracy as we change the hyperparameters

# In[21]:


scores.mean(), scores.std()


# In[18]:


print(xgb_clf)


# In[29]:


pgrid = {'max_depth' : [10, 20],
         'n_estimators' : [300, 500]}


# In[32]:


xgb_classif = xgb.XGBClassifier(learning_rate=0.1, random_state=20130810, silent=True)


# In[33]:


xgb_cv = RandomizedSearchCV(estimator = xgb_classif,
                            param_distributions = pgrid,
                            n_iter = 2,
                            cv = 5,
                            n_jobs = 3, 
                            random_state = 20130810,
                            verbose = 1)


# In[34]:


get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(x_train, y_train)')


# In[35]:


xgb_cv.best_score_


# In[36]:


xgb_cv.best_estimator_


# ### Rerun with the final parameters

# In[37]:


xgb_clf = xgb.XGBClassifier(max_depth=10, 
                            n_estimators=750, 
                            learning_rate=0.1, 
                            random_state=20130810, 
                            silent=True,
                            nthread=3)


# In[38]:


cross_val_score(xgb_clf, 
                x_train, y_train, 
                cv=5, 
                n_jobs=-1, 
                verbose=1)

