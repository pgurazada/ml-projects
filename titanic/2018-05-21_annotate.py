"""
This script accomplishes a few key tasks
 - Convert all categorical features to numeric (using dummy variables)
 - fill in missing values
 - scale all the features

 Another common task is to split the processed data into training and testing, applying the transformations learned
 from the training data to the test data. This portion of the code is not relevant in this case, but generally the output
 of this script is 4 files - x_train, y_train, x_test, y_test
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import TransformerMixin

# We first define a simple class that fills in the mean for numeric features and the most occuring value
# for categorical features

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

def read():
    data_df = pd.read_csv('processed/titanic_train.csv')
    return data_df

def write(x_df, y_vec):
    x_df.to_csv('processed/x_train.csv', header=None, index=None)
    y_vec.to_csv('processed/y_train.csv', header=None, index=None)

def annotate(data_df):
    x_train_df = data_df.drop('Survived', axis = 1)
    y_train = data_df['Survived']

    x_train_df = DataFrameImputer().fit_transform(x_train_df) # Impute missing values

    x_train_df = pd.get_dummies(x_train_df) # Convert categorical features to dummies

    x_train = StandardScaler().fit_transform(x_train_df) # Scale and center the data

    return pd.DataFrame(x_train), y_train

if __name__ == '__main__':
    train_df = read()
    x, y = annotate(train_df)
    write(x, y)



