#!/usr/bin/env python
# coding: utf-8

# # HW11 Task 2 - Linear model
# # KNN, LMNN, SVM, ENN

# In[1]:


#install packets
#!pip3 install pylmnn
# !pip install enn


# In[6]:


import numpy as np
import pandas as pd
import matplotlib
import sklearn
import time
import timeit

from enn.enn import ENN
from numba.decorators import autojit
from pylmnn import LargeMarginNearestNeighbor as LMNN
from scipy.spatial.distance import euclidean, mahalanobis, minkowski, chebyshev
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle

get_ipython().run_line_magic('matplotlib', 'inline')
# matplotlib.style.use('seaborn')

print (sklearn.__version__)


# In[7]:


# path to data
path='winequality-red.csv'
df=pd.read_csv(path, sep=";")
df.head()
# data size
print('\n shape: ',df.shape)

# data info
df.describe()

# target count
type=df['quality'].groupby(df['quality']).count()
print('\n type: ',type)


# In[8]:


df.head()


# In[9]:


# count target plot
type.plot('bar');


# In[10]:


# input data

# get column titles except the last column
features=df.columns[:-1].tolist()

# get data set features
X=df[features].values
# get labels
y=df['quality'].values

# split data to train data set and test data set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)


# In[6]:


get_ipython().run_cell_magic('time', '', "# KNN && Gridsearch\nn_neighbors_array = list(range(1,31))\nparam_grid={'n_neighbors':n_neighbors_array, 'algorithm':['kd_tree','ball_tree'],\\\n            'metric':['chebyshev','manhattan','minkowski'], 'weights':['uniform','distance']}\n\nknn = KNeighborsClassifier()\ngrid = GridSearchCV(knn, param_grid, cv=5, iid=True)\n\ngrid.fit(X_train, y_train)\n\nbest_neighbors = grid.best_params_.get('n_neighbors')\n\nprint(f'best score: {grid.best_score_:.3f}  nieghbor: {best_neighbors}')\nprint(f'prediction precision rate: {grid.score(X_test,y_test):.5f}')")


# In[7]:


get_ipython().run_cell_magic('time', '', "# KNN & LMNN metric GridSearchSV\n\n# LMNN metric\nscores = []\n# paraments for GridSearchCV\n#k_test_neighbors_array = list(range(1,10))\nparam_grid_knn={'n_neighbors':[1], 'algorithm':['kd_tree','ball_tree'],\\\n            'metric':['chebyshev','manhattan','minkowski'], 'weights':['uniform','distance']}\n\n# loop k from 1 to 5, and get cross validation score of each K value\nn_components, max_iter = X.shape[1], 5\n\nfor k in range(1,9):\n    start_time = timeit.default_timer()\n    # Instantiate the metric learner\n    lmnn = LMNN(n_neighbors=k, max_iter=max_iter, n_components=n_components)\n    # Train the metric learner\n    lmnn.fit(X_train, y_train)\n     \n    #KNN Algorytm\n    knn = KNeighborsClassifier()\n    grid = GridSearchCV(knn, param_grid_knn, cv=5, iid=True)\n    \n    grid.fit(lmnn.transform(X_train), y_train)\n    \n    best_neighbors = grid.best_params_.get('n_neighbors')\n    \n    end_time = timeit.default_timer() - start_time\n    \n    print(f'best score: {grid.best_score_:.3f}   nieghbor LMNN: {k}')\n    print(f'prediction precision rate: {grid.score(X_test,y_test):.5f}')\n    print(f'time: {end_time:.0f} s \\n')\n\n    scores.append([grid.best_score_, best_neighbors, grid.score(X_test,y_test), k])\n\nresult_scores = sorted(scores, key=lambda x: x[0], reverse=True)[0]\nprint('\\nResult Nieghbor LMNN algorytm')\nprint(f'nieghbor LMNN: {result_scores[3]} score: {result_scores[2]}  predict precision rate: {result_scores[0]:.3f}')\n\n# KNN model \nbest_k = result_scores[3]\n# Instantiate the metric learner\nlmnn = LMNN(n_neighbors=int(best_k), max_iter=max_iter, n_components=n_components)\n# Train the metric learner\nlmnn.fit(X_train, y_train)\n\n# generate KNN model & GridSearchCV\nknn=KNeighborsClassifier(n_neighbors=best_k)\n\nparam_grid_knn={'n_neighbors':[best_k], 'algorithm':['kd_tree','ball_tree'],\\\n            'metric':['chebyshev','manhattan','minkowski'], 'weights':['uniform','distance']}\n\ngrid_knn = GridSearchCV(knn, param_grid_knn, cv=5, iid=True)\n# fit with train data set\ngrid_knn.fit(lmnn.transform(X_train), y_train)\n\nprint(f'\\nprediction precision rate: {grid_knn.score(lmnn.transform(X_test),y_test):.5f}')")


# In[8]:


get_ipython().run_cell_magic('time', '', "# KNN & LMNN metric GridSearchSV less\n\n# LMNN metric\n#https://pylmnn.readthedocs.io/en/stable/api.html#module-pylmnn.lmnn\n\nscores=[]\n# loop k from 1 to 9, and get cross validation score of each K value\nk_test, n_components, max_iter = 1, X.shape[1], 10\n\nfor k in range(1,9):\n    start_time = timeit.default_timer()\n    # Instantiate the metric learner\n    lmnn = LMNN(n_neighbors=k, max_iter=max_iter, n_components=n_components)\n    # Train the metric learner\n    lmnn.fit(X_train, y_train)\n    # Fit the nearest neighbors classifier\n    knn = KNeighborsClassifier(n_neighbors=k_test, algorithm='kd_tree', metric='chebyshev', weights='uniform')\n    knn.fit(lmnn.transform(X_train), y_train)\n    score_val=cross_val_score(knn,lmnn.transform(X_train),y_train,scoring='accuracy',cv=5)\n    score_mean=score_val.mean()\n    scores.append(score_mean)\n    \n    end_time = timeit.default_timer() - start_time\n    \n    print(f'neibors: {k}  ->  score: {score_mean:.5f}  <-  time: {end_time:.0f} s')\n\n# KNN model \n\n# get index of maxium score along axis, default axis=0 for 1 dimensional array\nbest_k=np.argmax(scores)+1\nprint('\\nneibor:',best_k)\n# Instantiate the metric learner\nlmnn = LMNN(n_neighbors=int(best_k), max_iter=max_iter, n_components=n_components)\n# Train the metric learner\nlmnn.fit(X_train, y_train)\n# generate KNN model\nknn=KNeighborsClassifier(n_neighbors=best_k)\n# fit with train data set\nknn.fit(lmnn.transform(X_train), y_train)\n# get Modes presicion rate using test set\nprint(f'prediction precision rate: {knn.score(lmnn.transform(X_test),y_test):.5f}')")


# In[44]:


get_ipython().run_cell_magic('time', '', '# SVC model GridSearchSV\n\ngamma_aray = np.linspace(0.35, 0.55, 30).tolist()\nparameters_svc = {\'gamma\':gamma_aray, \'C\':[1,10]}\nsvc = SVC()\nclf = GridSearchCV(svc, parameters_svc, cv=5)\nclf.fit(X_train, y_train);\n\nprint(\'\\nevalueted paremetrs SVC: \',clf.best_params_)\nprint(f\'prediction precision rate: {clf.score(X_test,y_test):.5f} \\n\')\n\ny_pred = clf.predict(X_test)\n\nprint(\'Train scores: \\n\')\nprint(f\'accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\'precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\'recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\'f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')')


# In[41]:


get_ipython().run_cell_magic('time', '', '# SVC model GridSearchSV less\n\n# The implementation is based on libsvm. \n# The fit time complexity is more than quadratic with the number of samples \n# which makes it hard to scale to dataset with more than a couple of 10000 samples.\n\nclf = SVC(gamma=0.53, C=1)\nclf.fit(X_train, y_train)\nprint(f\'prediction precision rate: {clf.score(X_test,y_test):.5f} \\n\')\n\ny_pred = clf.predict(X_test)\n\nprint(\'Train scores: \\n\')\nprint(f\'accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\'precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\'recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\'f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')\n#print(\'roc_auc_score = \', roc_auc_score(y_train, y_pred))')


# In[11]:


get_ipython().run_cell_magic('time', '', '# ENN model GridSearchSV\n# https://github.com/timo-stoettner/ENN/blob/master/README.md\n# article https://mlbootcamp.ru/article/tutorial/\n\n# optimized function for euclidean metric in ENN\ndef euclidean(x,y):   \n    return np.sqrt(np.sum((x-y)**2))\n\noptimized_euclidean = autojit(euclidean)\n\nk_array = list(range(1,3))\nparameters_enn = {\'k\':k_array, \'distance_function\':[optimized_euclidean]}\nenn = ENN()\nclf = GridSearchCV(enn, parameters_enn, cv=5)\nclf.fit(X_train, y_train)\nprint(\'\\nevalueted paremetrs ENN: \',clf.best_params_)\n\ny_pred = clf.predict(X_test)\n\nprint(\'\\n Model ENN: \')\nprint(\' Train scores: \\n\')\nprint(f\' accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\' precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')')


# In[12]:


get_ipython().run_cell_magic('time', '', '# ENN model  GridSearchSV less\n# https://github.com/timo-stoettner/ENN/blob/master/README.md\n# article https://mlbootcamp.ru/article/tutorial/\n\nclf = ENN(k=1, distance_function=optimized_euclidean)\nclf.fit(X_train, y_train)\ny_pred = clf.predict(X_test)\n\nprint(\'\\n Model ENN: \')\nprint(\' Train scores: \\n\')\nprint(f\' accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\' precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')')


# In[13]:


# Scoring improvement

# Get balanced sample by oversampling
def balanced_koef(serie, element):
    '''evaluation multiply koefficient'''
    return (serie.max() // serie[element])

# balancing algorytm - primitive upsampling 

aray_df = []
for i in type.index:
    
    df_target = df[df['quality'] == i]
    
#     if balanced_koef(type, i) != 0:
#         df_target = pd.concat([df_target] * balanced_koef(type, i))
    df_target = pd.concat([df_target] * balanced_koef(type, i))
    aray_df.append(df_target)

df_balanced = pd.concat(aray_df)

print(f'df: {df.shape}  -->>  balanced df: {df_balanced.shape}')


# In[14]:


# Count balances type in dataset
type_balanced=df_balanced['quality'].groupby(df_balanced['quality']).count()
type_balanced.plot('bar');
df_balanced = shuffle(df_balanced)


# In[15]:


# Balanced dataframe

# df.columns is column labels property
features=df_balanced.columns[:-1].tolist()
X = df_balanced[features].values
y = df_balanced['quality']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)

# set y to arrays
y_train = y_train.values
y_test = y_test.values


# In[56]:


get_ipython().run_cell_magic('time', '', "# KNN && Gridsearch\nn_neighbors_array = list(range(1,11))\nparam_grid={'n_neighbors':n_neighbors_array, 'algorithm':['kd_tree','ball_tree'],\\\n            'metric':['chebyshev','manhattan','minkowski'], 'weights':['uniform','distance']}\n\nknn = KNeighborsClassifier()\ngrid = GridSearchCV(knn, param_grid, cv=5, iid=True)\n\ngrid.fit(X_train, y_train)\n\nbest_neighbors = grid.best_params_.get('n_neighbors')\n\nprint(f'best score: {grid.best_score_:.3f}  nieghbor: {best_neighbors}')\nprint(f'prediction precision rate: {grid.score(X_test,y_test):.5f}')")


# In[16]:


get_ipython().run_cell_magic('time', '', '# ENN model balanced DF & GridSearchSV\n\nk_array = list(range(1,3))\nparameters_enn = {\'k\':k_array, \'distance_function\':[optimized_euclidean]}\nenn = ENN()\nclf = GridSearchCV(enn, parameters_enn, cv=5)\nclf.fit(X_train, y_train)\nprint(\'\\nevalueted paremetrs ENN: \',clf.best_params_)\n\ny_pred = clf.predict(X_test)\n\nprint(\'\\n Model ENN: \')\nprint(\' Train scores: \\n\')\nprint(f\' accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\' precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')')


# In[ ]:





# In[ ]:





# In[ ]:




