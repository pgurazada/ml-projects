'''

This script trains the final prediction algorithm and prints the prediction accuracy on a 10 fold cross validated
training set.

The shortlist of algorithms from the model-explorer jupyter notebook is run with the final settings.
'''

# Basic I/O
import pandas as pd
import numpy as np

# Algorithm
from sklearn.ensemble import RandomForestClassifier

# Model performance
from sklearn.model_selection import cross_val_score

def read():
    x_train = np.asarray(pd.read_csv('processed/x_train.csv'))
    y_train = np.asarray(pd.read_csv('processed/y_train.csv')).ravel()

    return x_train, y_train

def cross_validate(x_train, y_train):
    rf_classif = RandomForestClassifier(n_estimators=500,
                                        max_features='auto',
                                        criterion='gini',
                                        min_samples_leaf=1,
                                        min_samples_split=10,
                                        oob_score=True,
                                        random_state=20130810,
                                        n_jobs=-1)

    scores = cross_val_score(rf_classif, x_train, y_train,
                             scoring = 'accuracy',
                             cv = 10)

    return scores.mean()

if __name__ == '__main__':
    x_train, y_train = read()
    accuracy = cross_validate(x_train, y_train)

    print('Mean accuracy score: {}'.format(accuracy))


