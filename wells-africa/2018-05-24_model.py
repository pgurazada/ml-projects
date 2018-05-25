
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


# In[14]:


# Model selection

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


# In[4]:


x_train = np.load('processed/wells_feature_matrix.npy')
y_train = pd.read_csv('processed/wells_labels_train.csv')['status'].tolist()


# We begin by looking at the variation in the performance as we hand-tune some of the parameters. We will then put together a parameter grid that will possiblly find the best combination among all the parameters. For now, we will look at accuracy as the metric. At the moment the focus is on observing the contours of accuracy as we change the hyperparameters

# ### Number of estimators

# In[ ]:


n_estimators_list = [50, 100, 500, 1000]


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators=50,
                                    max_features='auto',
                                    random_state=20130810,
                                    n_jobs=3)


# In[ ]:


rf_cv = GridSearchCV(estimator=rf_classif, 
                     param_grid={'n_estimators' : n_estimators_list},
                     scoring='accuracy',
                     n_jobs=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.cv_results_['mean_test_score']


# ### Maximum depth of the tree

# In[ ]:


np.sqrt(x_train.shape[1])


# In[ ]:


max_depth_list = [5, 15, 30, 50, 100]


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators=50,
                                    max_features='auto',
                                    random_state=20130810,
                                    n_jobs=3)


# In[ ]:


rf_cv = GridSearchCV(estimator=rf_classif, 
                     param_grid={'max_depth' : max_depth_list},
                     scoring='accuracy',
                     n_jobs=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.cv_results_['mean_test_score']


# ### Maximum number of features to consider at each split 

# In[ ]:


max_features_list = [5, 15, 30, 50]


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators=50,
                                    max_features='auto',
                                    random_state=20130810,
                                    n_jobs=3)


# In[ ]:


rf_cv = GridSearchCV(estimator=rf_classif, 
                     param_grid={'max_features' : max_features_list},
                     scoring='accuracy',
                     n_jobs=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.cv_results_['mean_test_score']


# ### Minimum samples to split an internal node

# In[ ]:


min_samples_split_list = [2, 5, 10, 20]


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators=50,
                                    max_features='auto',
                                    random_state=20130810,
                                    n_jobs=3)


# In[ ]:


rf_cv = GridSearchCV(estimator=rf_classif, 
                     param_grid={'min_samples_split' : min_samples_split_list},
                     scoring='accuracy',
                     n_jobs=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.cv_results_['mean_test_score']


# ### Minimum samples to be a leaf node

# In[ ]:


min_samples_leaf_list = [1, 5, 10, 20]


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators=50,
                                    max_features='auto',
                                    random_state=20130810,
                                    n_jobs=3)


# In[ ]:


rf_cv = GridSearchCV(estimator=rf_classif, 
                     param_grid={'min_samples_leaf' : min_samples_leaf_list},
                     scoring='accuracy',
                     n_jobs=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.cv_results_['mean_test_score']


# ### Randomized grid search over parameter combinations

# Now we tune the random forests model over a parameter grid

# In[ ]:


pgrid = {'min_samples_leaf' : [5, 10],
         'min_samples_split' : [5, 10],
         'max_features' : [15, 30],
         'max_depth' : [30, 50 , 100],
         'n_estimators' : [100, 500]}


# In[ ]:


rf_classif = RandomForestClassifier(random_state = 20130810,
                                    n_jobs = 3)


# In[ ]:


rf_cv = RandomizedSearchCV(estimator = rf_classif,
                           param_distributions = pgrid,
                           n_iter = 10,
                           cv = 5,
                           n_jobs = 3, 
                           random_state = 20130810,
                           verbose = 2)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(x_train, y_train)')


# In[ ]:


rf_cv.best_score_


# In[ ]:


rf_cv.best_params_


# ### Running the model at the final configurations

# In[5]:


rf_classif = RandomForestClassifier(n_estimators=500,
                                    max_depth=50,
                                    max_features=30,
                                    min_samples_leaf=5,
                                    min_samples_split=10,
                                    random_state=20130810,
                                    n_jobs=3)


# In[ ]:


scores = cross_val_score(rf_classif,
                         x_train,
                         y_train, 
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs = 3,
                         verbose = 3)


# In[ ]:


scores.mean(), scores.std()

