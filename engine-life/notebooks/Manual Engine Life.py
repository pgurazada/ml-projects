
# coding: utf-8

# # Predicting Remaining Useful Life with Manual Feature Engineering
# 
# In this notebook, we will work with NASA provided data to accomplish a critical real-world task: predict the remaining useful life of an engine. Our first attempt will be to tackle this problem by hand, building features based on aggregations, domain knowledge (if applicable), and time-series based methods. The original data can be downloaded [here.](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

# ## Roadmap 
# 
# Following will be our approach to this problem:
# 
# 1. Specify the prediction problem
#     * For each engine, define the cutoff point and label
# 2. Subset data to before cutoff point for each engine
# 3. Establish a baseline performance measure
# 4. Perform basic aggregations for each engine to develop features
#     * Use feature selection to reduce the number of features
# 5. Evaluate performance of features 
#     * In cross validation on the training data 
#     * On the testing data
# 6. Try more sophisticated feature engineering
#     * Time-series analysis
#     * KMeans clustering
# 7. Evaluate performance of new features
#     * Use feature selection
# 8. Perform random search to tune the random forest model
# 
# This problem is a good basic introduction to time-series prediction. Even though we have all the data for all the engines, when we make features, we have to limit ourselves to data from before the cutoff time in order to ensure our model trains on valid data.

# In[5]:


import pandas as pd
import numpy as np

from tqdm import tqdm
from utils import plot_feature_importances, feature_selection


# In[4]:


np.random.seed(20130810)


# First we need to read in the data and set the correct columns headers as specified in the data documentation.

# In[2]:


operational_settings = ['operational_setting_{}'.format(i + 1) for i in range (3)]
sensor_columns = ['sensor_measurement_{}'.format(i + 1) for i in range(26)]
cols = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns
data = pd.read_csv('../input/train_FD002.txt', sep=' ', header=-1, names=cols)
data.head()


# In[9]:


np.unique(data['engine_no'])


# There are 3 operational settings and 26 sensor measurements. Ahead of time, we have no idea which of these are relevant!

# # Prediction Problem
# 
# The training data initially has no prediction problem: we are given the entire operational history of the engine with the final entry representing the last successful measurement before failure. Therefore, we have to come up with our own prediction problem which we do as follows:

# 1. Select an arbitrary starting date: January 1, 2010
# 2. Create a time column using the `time_in_cycles` and the knowledge that one cycle takes 10 minutes
# 3. For each engine, select a random time to use as the cutoff point
# 4. Find the number of cycles between the cutoff point and the end of life of the engine, this becomes the label
# 5. Subset the data for each engine to only the times before the prediction point. 

# At the end of this process, we will have a labeled training set where each row is one engine and the label is the number of cycles to failure from that point (the `prediction_point` also called the `cutoff_time`) in time. We can then use the past operating data from before the prediction point and the labels to train a machine learning model. 

# In[3]:


# Pick a starting date (this can be arbitrary)
starting_date = pd.Timestamp(2010, 1, 1)

# Create a time column using the time in cycles * 10 minutes per cycle
data['time'] = [starting_date + pd.Timedelta(x * 10, 'm') for x in data['time_in_cycles']]

data[['engine_no', 'time_in_cycles', 'time']].head()


# Now, for each engine, we need to pick a random time to serve as the prediction point. We will impose the limits that we need to have at least 10 measurements before the prediction point.

# In[7]:


# Dataframe to hold results
engines = pd.DataFrame(columns = ['engine_no', 'prediction_point', 'label'])
engine_list = data['engine_no'].unique()

# Iterate through each engine
for engine in tqdm(engine_list):
    
    # Subset to the engine
    subset = data[data['engine_no'] == engine].copy().sort_values('time')
    
    measurements = subset.shape[0]
    
    # Select a random index for the prediction point
    random_index = np.random.randint(10, measurements - 1)
    
    # Record the predictino point and the label which is the remaining number of cycles
    prediction_point = subset.iloc[random_index, :].copy()['time']
    label = measurements - random_index
    
    # Record the measurements
    engines = engines.append(pd.DataFrame({'engine_no': engine, 'prediction_point': prediction_point, 
                                           'label': label}, index = [0]), 
                             ignore_index = True, sort = True)


# In[ ]:


# Save prediction problem 
engines.to_csv('../input/engines_4.csv', index = False)
engines.head()


# # Limit Data to before prediction point
# 
# For each engine, we can only use data from before the `prediction_point` to predict when the engine will fail. We need to subset the `data` table for each engine until only times before the `prediction_point`. In Featuretools, the prediction point is called the `cutoff_time`.

# In[ ]:


legal_data = pd.DataFrame(columns = data.columns)

for i, engine in engines.iterrows():
    # Subset to times before the prediction point
    legal_data_subset = data[(data['time'] < engine['prediction_point']) & (data['engine_no'] == engine['engine_no'])].copy()
    
    legal_data = legal_data.append(legal_data_subset, ignore_index = True, sort = True)


# In[ ]:


train_obs = legal_data.copy()
train_obs.to_csv('../input/train_obs.csv', index = False)
train_obs.head()


# ## Metric and Baseline 
# 
# Before we go any further, it's important to establish the metric we will use to judge how well our model does and a baseline measure of performance. For this regression problem, we'll use the [__Mean Absolute Percentage Error__ (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error),  a common metric that is easy to calculate and interpretable. This is the average of the absolute value of (prediction - true value) / true value) and is expressed as a percentage. 
# 
# For the naive baseline, we can use two methods:
# 
# 1. Guess the average value of the label in the training data for all instances on the testing set. This will be called `average_guess`.
# 2. For each engine in the testing data, predict it has reached half its life at the end of the data and will continue to operate for however many cycles it has already operated. This will be called `half_life_guess`.
# 
# First we need to read in the testing data which is the same format as the training data.

# In[ ]:


test_obs = pd.read_csv('../input/test_FD002.txt', sep=' ', header=-1, names=cols)

# Pick a starting date (this can be arbitrary)
starting_date = pd.Timestamp(2010, 1, 1)

# Create a time column using the time in cycles * 10 minutes per cycle
test_obs['time'] = [starting_date + pd.Timedelta(x * 10, 'm') for x in test_obs['time_in_cycles']]

test_obs.to_csv('../input/test_obs.csv', index = False)
test_obs.head()


# In[ ]:


# Half life guess
train_half_life_guess = train_obs.groupby('engine_no').apply(lambda x: 2 * x.shape[0])

# Dataframe with both baselines
train_baseline = pd.DataFrame({'engine_no': train_obs['engine_no'].unique(), 
                               'half_life_guess': train_half_life_guess.values,
                               'average_guess': engines['label'].mean()})


# In[ ]:


# Make two baseline guesses
test_half_life_guess = test_obs.groupby('engine_no').apply(lambda x: 2 * x.shape[0])
test_baseline = pd.DataFrame({'engine_no': test_obs['engine_no'].unique(), 
                              'half_life_guess': test_half_life_guess.values,
                               'average_guess': engines['label'].mean()})


# The labels for the testing data are in a separate file that we can read in.

# In[ ]:


test_y = pd.read_csv('../input/RUL_FD002.txt', sep=' ', header=-1, names=['label'], index_col=False)


# We'll write a basic function to calculate the mean absolute percentage error and then apply it to the two baselines.

# In[ ]:


def mape(y_true, pred):
    mape = 100 * np.mean(abs(y_true - pred) / y_true)
    
    return mape


# In[ ]:


print('The average_guess train MAPE is: {:.2f}.'.format(mape(engines['label'], train_baseline['average_guess'])))
print("The average_guess  test MAPE is: {:.2f}.".format(mape(test_y['label'], test_baseline['average_guess'])))


# In[ ]:


print('The half_life train MAPE is: {:.2f}.'.format(mape(engines['label'], train_baseline['half_life_guess'])))
print('The half_life  test MAPE is: {:.2f}.'.format(mape(test_y['label'], test_baseline['half_life_guess'])))


# These baselines give us a target for our machine learning models. If our model cannot beat even a naive baseline, then maybe we want to rethink the machine learnign approach!

# # Approach to Feature Engineering
# 
# The `train_obs` is now a child table of `engines` because for each unique engine (identified by the `engine_no`), there are multiple rows in the `train_obs`. Our final train dataframe will need to have one unique row for every engine, with the features in the columns, so feature engineering will involve aggregating the `train_obs` for every engine. The same operations that are done to `train_obs` will have to be applied to `test_obs` as well because we need to have the same columns in both the training and testing set.

# In[ ]:


train_obs = train_obs.drop(columns = 'time')
train_obs.shape 


# ## Aggregations
# 
# As a simple first step, we can perform numerical aggregations of every column in `train_obs` table. If we have no idea what the columns represent, this is a good place to start because it will provide a thorough summary of every column.

# In[ ]:


# First deal with some annoying type issues
train_obs['sensor_measurement_17'] = train_obs['sensor_measurement_17'].astype(np.float32)
train_obs['sensor_measurement_18'] = train_obs['sensor_measurement_18'].astype(np.float32)
train_obs['time_in_cycles'] = train_obs['time_in_cycles'].astype(np.int32)

# Aggregate each column
train_agg = train_obs.groupby('engine_no').agg(['min', 'max', 'mean', 'sum', 'std'])
train_agg.head()


# In order to better keep track of the columns, we can rename them using a for loop with the original column and then the statistic.

# In[ ]:


new_cols = []

# Iterate through the columns and create new names
for col in train_agg.columns.levels[0]:
    for stat in train_agg.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')
        
train_agg.columns = new_cols
train_agg.head()


# That fairly simple operation gave us 150 features that we can use as a main training dataframe. 
# 
# ### Assess Performance
# 
# Let's assess the performance of just these features in a model. We need to make sure to apply the same operations to the testing data.

# In[ ]:


# Apply same operations to testing data
test_agg = test_obs.groupby('engine_no').agg(['min', 'max', 'mean', 'sum', 'std'])

new_cols = []

for col in test_agg.columns.levels[0]:
    for stat in test_agg.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')
        
test_agg.columns = new_cols
test_agg.head()


# ## Modeling
# 
# For our model, we'll use the capable [Random Forest algorithm](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) as implemented in Scikit-Learn. We'll assess performance by MAPE using 5-fold cross validation on the training data and on the testing data.
# 
# We can also calculate the feature importances from the Random Forest to see if these give us any insight into the problem.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

mape_scorer = make_scorer(mape, greater_is_better = False)

def evaluate(train, train_labels, test, test_labels):
    """Evaluate a training dataset in cross validation and on the test data"""
    
    # Use the same model for each training set for now
    model = RandomForestRegressor(n_estimators = 100, 
                                  random_state = 50, n_jobs = -1)
    
    train = train.replace({np.inf: np.nan})
    test = test.replace({np.inf: np.nan})
    
    feature_names = list(train.columns)
    
    # Impute the missing values
    imputer = Imputer(strategy = 'median', axis = 1)
    train = imputer.fit_transform(train)
    test = imputer.transform(test)
    
    # Fit on the training data and make predictions
    model.fit(train, train_labels)
    preds = model.predict(test)
    
    cv_score = -1 * cross_val_score(model, train, train_labels, 
                                    scoring = mape_scorer, cv = 5)
    
    # Calculate the performance
    mape_score = mape(test_labels, preds)
    print('5-fold CV MAPE: {:.2f} with std: {:.2f}'.format(cv_score.mean(), cv_score.std()))
    print('Test MAPE: {:.2f}.'.format(mape_score))
    
    # Record feature importances
    feature_importances = pd.DataFrame({'feature': feature_names, 
                                        'importance': model.feature_importances_})
    
    return preds, feature_importances


# In[ ]:


# Remove the engine number since it should not be predictive
train = train_agg.reset_index(drop = True)
test = test_agg.reset_index(drop = True)

train_labels = engines['label']
test_labels = test_y['label']

preds, fi = evaluate(train, train_labels, test, test_labels)


# Our initial try did not even better than the baseline on the testing data! We're probably severaly overfitting to the training data given the much lower cross validation error than testing error. To get a sense of what might be wrong, we can plot the predictions.

# In[ ]:


import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.hist(preds, bins = 20, color = 'blue', edgecolor = 'k')
plt.title('Prediction Distribution'); plt.xlabel('Remaining Life (cycles)');


# We can also plot the 10 most important features to see what the model thinks are the most relevant features we made.

# In[ ]:


norm_fi = plot_feature_importances(fi, 10)
norm_fi.head(10)


# It seems the model was not able to learn, predicting nearly the same value for all engines. One possible issue is the presence of too many irrelevant features.

# ## Feature Selection
# 
# Feature selection is nearly as important as feature engineering because irrelevant features can slow down model training, lead to poorer performance on the test set, and result in less model interpretability. Below, we apply four methods of feature selection (see the `utils.py` file and the `feature_selection` function for details) to the dataset and then re-evaluate. Feature selection here is composed of:
# 
# 1. Remove columns with more than 90% missing values
# 2. Remove columns with only a single unique value
# 3. Remove one of each pair of columns with a correlation greater than 0.95.
# 
# Removing some of the features can help the model generalize to the testing data.

# In[ ]:


train_fs = feature_selection(train)
test_fs = test[train_fs.columns]


# In[ ]:


preds, fi = evaluate(train_fs, train_labels, test_fs, test_labels)


# The performance increases significantly! Our model is now significantly better than the baseline in both cross validation and on the training data. This highlights the importance of proper feature selection. 

# In[ ]:


plt.hist(preds, bins = 20, color = 'blue', edgecolor = 'k')
plt.title('Prediction Distribution'); plt.xlabel('Remaining Life (cycles)');


# In[ ]:


norm_fi = plot_feature_importances(fi, 10)
norm_fi.head()


# The most important features now make more sense: _the total previous cycles of the engine is the second greatest predictor of how much longer the engine will last_. We also can see that the first operational setting is important as well as sensors 11 and 15. These can give us clues as to what to focus on when monitoring future engines.
# 
# __Through feature selection, we went from a model that was no better than a naive guess, to a model that reduced the error of the guess by more than 75%!__

# In[ ]:


train_fs.head()


# In[ ]:


train_fs['time_in_cycles-max'] = list(train_agg['time_in_cycles-max'])
train_fs.to_csv('../input/simple_manual_features.csv')


# ## More Advanced Feature Engineering
# 
# To try and build a better model, we can apply more advanced feature engineering techniques. Since our data is in a time-series, we can apply any time-series operation to each engine. For example, we can find the percentage change and the cumulative sum for each engine.

# In[ ]:


train_obs = train_obs.sort_values(['engine_no', 'time_in_cycles'])
train_obs.head()


# In[ ]:


train_exp = train_obs.copy()

# Find percentage change and cumulative sum for each engine
for col in train_obs:
    train_exp[f'{col}_pct_change'] = train_obs.groupby('engine_no')[col].apply(lambda x: x.pct_change())
    train_exp[f'{col}_cum_sum'] = train_obs.groupby('engine_no')[col].apply(lambda x: np.cumsum(x))
    
train_exp.head()


# Since this just created new columns of observations, we can take this data and aggregate it to get a single dataframe for testing with one row for each engine. The function below carries out the aggregations and renaming of columns.

# In[ ]:


def agg_and_rename(df, agg_variable):
    """Function to aggregate a dataframe"""
    
    df_agg = df.groupby(agg_variable).agg(['min', 'max', 'mean', 'sum', 'std'])

    new_cols = []

    # Create a rename set of columns
    for col in df_agg.columns.levels[0]:
        for stat in df_agg.columns.levels[1]:
            new_cols.append(f'{col}-{stat}')

    df_agg.columns = new_cols
    
    return df_agg


# In[ ]:


train_exp_agg = agg_and_rename(train_exp, 'engine_no')
train_exp_agg.head()


# Now we apply the same operation to the testing data, do feature selection on the training data, subset the testing data to the same columns as the training data, and evaluate!

# In[ ]:


test_obs.drop(columns = 'time', inplace = True)
test_exp = test_obs.copy()

# Apply operations to testing data
for col in test_obs:
    test_exp[f'{col}_pct_change'] = test_obs.groupby('engine_no')[col].apply(lambda x: x.pct_change())
    test_exp[f'{col}_cum_sum'] = test_obs.groupby('engine_no')[col].apply(lambda x: np.cumsum(x))

test_exp_agg = agg_and_rename(test_exp, 'engine_no')


# In[ ]:


# Feature selection and subsetting of test data
train_exp_agg_fs = feature_selection(train_exp_agg, 90, 0.95)
test_exp_agg_fs = test_exp_agg[train_exp_agg_fs.columns]


# In[ ]:


preds, fi = evaluate(train_exp_agg_fs, train_labels, test_exp_agg_fs, test_labels)


# It appears we may have reached the limits of what we can do with feature engineering given that we added more features and yet the performance did not increase. It's possible we are approaching [Bayes error](https://en.wikipedia.org/wiki/Bayes_error_rate), which is the lowest possible error its possible to get on a problem. This is a function of noise in the data and latent (hidden) variables that we cannot measure. The problem with Bayes error is we can never know when we have reached that point, because it would require a model that we know for sure achieves that error rate.

# In[ ]:


norm_fi = plot_feature_importances(fi, 10)
norm_fi.head()


# The model has about the same performance even with the added features. We can see some of the percentage change features we added as among the most important features. 
# 
# ## KMeans clustering and Time-Series Analysis
# 
# We can try one more manual feature engineering effort using clustering and time-series analysis methods. This time, we will not use the `pct_change` and `cum_sum` because these features did not improve the model. The basic outline is:
# 
# * Clustering: First we will cluster the observations into 10 unique clusters. We can then perform numerical aggregations on these clusters to get the information into our training dataframe. 
# * Time-series analysis: Using the `tsfresh` package, we can apply a number of functions such as `number_peaks` and `cid_ce`. 

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(n_clusters = 10)

# Need to impute the missing values
imputer = Imputer(strategy='median')
train_cluster = imputer.fit_transform(train_obs.drop(columns = ['engine_no', 'time_in_cycles']))

# Create a new column with the cluster label
train_obs['cluster'] = kmeans.fit_predict(train_cluster)


# Next we apply the same procedure to the testing data (but just `transform` since we can only `fit` to training data).

# The new column contains the cluster assigned to each observation. This will still require aggregating since it is on a per-observation level instead of for each engine.

# In[ ]:


test_cluster = imputer.transform(test_obs.drop(columns = ['engine_no', 'time_in_cycles']))
test_obs['cluster'] = kmeans.predict(test_cluster)


# ### Visualize Clusters
#  
# We can try to visualize the clusters by applying a [UMAP embedding](https://github.com/lmcinnes/umap) to the data. This reduces the dimension of data primarily for visualization.

# In[ ]:


import umap

reducer = umap.UMAP(n_components = 3)
train_embedding = reducer.fit_transform(train_cluster)


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111, projection='3d')

cmap = plt.get_cmap('tab10', 10)
p = ax.scatter(train_embedding[:, 0], train_embedding[:, 1], train_embedding[:, 2], 
               c = train_obs['cluster'], cmap = cmap)
plt.title('UMAP Embedding Showing KMeans Cluster Assignments')
fig.colorbar(p);


# The UMAP embedding shows the clusters pretty clearly separate the observations, so perhaps they can be useful. Through clustering, we're hoping that observations that have the same cluster encode similar information which can then be used to predict how much longer an engine will run. 
# 
# To get one observation per engine, we again have to aggregate by the `engine_no`. This time we will have five extra columns with the aggregations of the cluster assignments. 

# In[ ]:


train_agg = agg_and_rename(train_obs, 'engine_no')
train_agg.head()


# In[ ]:


test_agg = agg_and_rename(test_obs, 'engine_no')


# ### Time Series Operations using tsfresh
# 
# For the time series operations, we have any number of choices we can make from `tsfresh`. Each of these operations is applied to a single time-series and generates one number. Therefore, to apply them, we group by the the `engine_no` and then apply the operation to get a single observation per engine. We will choose five different operations:
# 
# 1. `cid_ce`: measures the complexity of a time series
# 2. `number_peaks`: measures the number of peaks where a peak is bigger than `n` neighbors to the right and left
# 3. `last_location_of_maximum`: locates the last occurrence of the maximum value in the time series
# 4. `skewness`: the Fisher-Pearson skewness of the time series
# 5. `sample_entropy`: the sample entropy of the time series
# 
# All of these will be calculated for each operational setting and each sensor measurement giving us 5 new columns for each of the 29 original features (145 total features).

# In[ ]:


from tsfresh.feature_extraction.feature_calculators import (cid_ce, number_peaks, 
                                                             last_location_of_maximum, 
                                                             skewness, sample_entropy)


# To avoid the issue of passing multiple functions to `agg` with the same name `lambda`, we have to create lambda functions and then give them custom names. `cid_ce` and `number_peaks` both have required arguments but the other functions only need a time-series.

# In[ ]:


cid_ce_func = lambda x: cid_ce(x, normalize=False)
cid_ce_func.__name__ = 'cid_ce'

n_peaks = lambda x: number_peaks(x, n = 5)
n_peaks.__name__ = 'number_peaks'

# Apply the five operations
ts_values = train_obs.drop(columns = ['time_in_cycles']).groupby('engine_no').agg([cid_ce_func, n_peaks, 
                                                                                   last_location_of_maximum,
                                                                                   skewness, sample_entropy])
ts_values.head()


# Below we rename the columns.

# In[ ]:


new_cols = []

# Iterate through columns
for col in ts_values.columns.levels[0]:
    for stat in ts_values.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')


# In[ ]:


ts_values.columns = new_cols


# This dataframe can then be joined to the training data because there is one observation for each engine.

# In[ ]:


train_obs['engine_no'] = train_obs['engine_no'].astype(np.int32)


# In[ ]:


train_agg = train_agg.merge(ts_values, on = 'engine_no', how = 'outer')
train_agg.head()


# We're up to 305 features (before feature selection). Now, we apply the same operations to the testing data.

# In[ ]:


# Apply the five operations
ts_values_test = test_obs.drop(columns = ['time_in_cycles']).groupby('engine_no').agg([cid_ce_func, n_peaks, 
                                                                                   last_location_of_maximum,
                                                                                   skewness, sample_entropy])

# Rename the columns
new_cols = []

for col in ts_values_test.columns.levels[0]:
    for stat in ts_values_test.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')
        
        
ts_values_test.columns = new_cols


# In[ ]:


test_agg = test_agg.merge(ts_values_test, on = 'engine_no', how = 'outer')


# Finally, we can apply feature selection and then evalute the new set of features.

# In[ ]:


train_agg_fs = feature_selection(train_agg, 90, 0.95)
final_features = list(train_agg_fs.columns)
test_agg_fs = test_agg[train_agg_fs.columns]


# In[ ]:


preds = evaluate(train_agg_fs, train_labels, test_agg_fs, test_labels)


# Again, we see that these features do not outperform the simple aggregations on the testing data. This might show that the new features are not helping the model.

# ## Random Search
# 
# The model we used to evaluate the features was an unoptimized random forest. In order to make sure we are getting the most out of the model, we should perform random search over the hyperparameters. To do this, we can use [`RandomizedSearchCV` from Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). The process is as follows:
# 
# 1. Select a metric: we already made a scorer using MAPE
# 2. Define a hyperparameter grid over the search domain for the following hyperparameters:
#     * `n_estimators`
#     * `max_depth`
#     * `min_samples_leaf`
#     * `max_features`
# 3. Run random search for 100 iterations
# 4. Extract the best hyperparameters and use these for the final model
# 
# __We will only use the random search on the final set of features because these had the best performance in cross validation.__

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

scorer = mape_scorer

# Hyperparameter grid
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(50, 1000, num = 100)],
    'max_depth': [None] + [int(x) for x in np.linspace(4, 20)],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['auto', 'sqrt', 0.5, 'log2', None]
}


# In[ ]:


# Make a model 
reg = RandomForestRegressor(n_jobs = -1, random_state = 50)

# RandomizedSearchCV object
random = RandomizedSearchCV(reg, param_grid, n_iter = 100, scoring = scorer, 
                            verbose = 1, n_jobs = -1, cv = 5, random_state = 50)

# Fit on the training data
random.fit(train_agg_fs, train_labels)


# In[ ]:


random.best_params_


# ### Evaluate Best Model
# 
# We'll use the set of feature from aggregations, kmeans clustering, and time-series analysis along with the optimal hyperparameters to train one final model. Through random search with cross validation, we are making the assumption that the hyperparameters that do the best in cross validation will translate to doing well on the testing data. 

# In[ ]:


best_score = -1 * random.best_score_
best_score_std = random.cv_results_['std_test_score'][np.argmax(random.cv_results_['mean_test_score'])]
best_model = random.best_estimator_

# Need to impute the values on the test data
imputer.fit(train_agg_fs)
test_agg_fs = test_agg_fs.replace({np.inf: np.nan})
test_agg_fs = imputer.transform(test_agg_fs) 


# In[ ]:


# Make predictions on the. test data
preds = best_model.predict(test_agg_fs)
final_mape = mape(test_labels, preds)

print('5-fold Cross Validation MAPE: {:.2f} with std: {:.2f}'.format(best_score, best_score_std))
print('Test MAPE: {:.2f}.'.format(final_mape))


# The 5-fold cross validation MAPE stayed about the same, but the Test MAPE has slightly decreased over the unoptimized random forest. 

# In[ ]:


fi = pd.DataFrame({'feature': final_features, 'importance': best_model.feature_importances_})
norm_fi = plot_feature_importances(fi)


# It does not appear that the KMeans clustering had much effect on the model because none of the top 15 most important features involve the cluster. The complexity and entropy both make it into the top features suggesting these may have a beneficial effect on the model. 
# 
# __Overall, the simple aggregations and the more complex time-series methods, along with random search to optimize the hyperparameters, yields the best model performance on the test data__.

# # Conclusions 
# 
# After several rounds of manual feature engineering following are the results (the scores may have changed slightly over runs of the notebook. Even setting a random seed for the model did not result in identical performance metrics):
# 
# | Feature Set                                                      	| Model                   	| Number of Features   (before selection) 	| Number of Features   (after selection) 	| Time to Build 	| 5-fold Train CV MAPE 	| Test MAPE 	|
# |------------------------------------------------------------------	|-------------------------	|-----------------------------------------	|----------------------------------------	|---------------	|----------------------	|-----------	|
# | Baseline Average Train Label Guess                               	| -                       	| -                                       	| -                                      	| 15 seconds    	| 158.89 (0.00)        	| 226.44    	|
# | Baseline Half Life Guess                                         	| -                       	| -                                       	| -                                      	| 30 seconds    	| 614.89 (0.00)        	| 926.51    	|
# | Simple Aggregations (No Feature Selection)                       	| Default Random Forest   	| 150                                     	| 150                                    	| 120 minutes    	| 44.56 (3.22)         	| 232.47    	|
# | Simple Aggregations (With Feature Selection)                      	| Default Random Forest   	| 150                                     	| 42                                     	| 120 minutes    	| 46.62 (4.46)         	| 51.48     	|
# | Simple Aggregations +  Percent Change and Cumuluative Sum        	| Default Random Forest   	| 455                                     	| 88                                     	| 150 minutes   	| 46.13 (4.93)         	| 53.11     	|
# | Simple Aggregations + KMeans Clustering and Time-Series Analysis 	| Default Random Forest   	| 305                                     	| 88                                     	| 180 minutes   	| 45.36 (5.35)         	| 50.33     	|
# | Simple Aggregations + KMeans Clustering and Time-Series Analysis 	| Optimized Random Forest 	| 305                                     	| 88                                     	| 180 minutes   	| 45.33 (5.98)         	| 48.54     	|
# 
# The best performing model used the simple aggregations, KMeans clustering, and time-series analysis features along with the optimized random forest model. These results highlight several important takeaways:
# 
# 1. Feature Engineering is critical. A simple baseline guess is very poor for this problem.
# 2. Feature Selection is critical. The simple aggregation set of features resulted in significant overfitting to the training set before proper feature selection was applied. 
# 3. Beyond a certain point, adding more complex features has diminishing returns to performance gains for this problem.
# 4. Model hyperparameter tuning can improve performance, but the gains are much smaller than those from feature engineering.
# 
# There were many more operations we could have applied to the data to generate features that we did not. For manual feature engineering, we are limited only by our imagination and patience. However, once we get to a certain level of performance, squeezing out every last bit of accuracy is secondary to [model interpretability](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime). We might want to forgo complex operations and sacrifice a minor bit of performance in order to create a more interpretable model. 
# 
# Overall, manual feature engineering was effective for this problem and resulted in a model that significantly outperforms the baseline. Nonetheless, manual feature engineering is still time-consuming, error-prone, and does not translate between problems because we have to completely re-write the code for each dataset. In the next notebook, we will implement automated feature engineering using [Featuretools](https://www.featuretools.com/), which is significantly more efficient, can be applied to any dataset with only minor changes in syntax, and will allow us to create hundreds or thousands of features which are not limited by our creavity or our time.
