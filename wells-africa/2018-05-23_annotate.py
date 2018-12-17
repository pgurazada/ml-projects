
# coding: utf-8

# This script accomplishes a few key tasks
#  - Convert all categorical features to numeric (using dummy variables)
#  - fill in missing values
#  - scale all the features

# In[33]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import TransformerMixin


# # Preprocessing the features
#
# We define the following class that imputes missing numerical features with the mean and the categorical features with the most often occuring value

# In[4]:


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """*Impute missing values*.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[14]:


wells_features_train = pd.read_csv('processed/wells_features_train.csv')


# In[15]:


wells_features_train.info()


# The initial shortlist of features did not deal with missing values. We check the situation and then apply the corerection.

# In[16]:


wells_features_train.isnull().sum()


# In[17]:


wells_features_train = DataFrameImputer().fit_transform(wells_features_train)


# We now check that the missing values have been taken care of

# In[18]:


wells_features_train.isnull().sum()


# Now we convert all categorical features to dummies

# In[19]:


wells_features_train = pd.get_dummies(wells_features_train)


# Now check that the dummies have been created

# In[22]:


wells_features_train.shape


# Next step is to scale and center the data

# In[23]:


wells_features_train = StandardScaler().fit_transform(wells_features_train)


# Now, we need to store this matrix and import it when we do model exploration

# In[34]:


np.save('processed/wells_feature_matrix', wells_features_train)


# # Preprocessing the labels

# In[25]:


wells_labels_train = pd.read_csv('processed/wells_labels_train.csv')


# In[26]:


wells_labels_train.info()


# There is nothing to do in this case
