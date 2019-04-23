#!/usr/bin/env python
# coding: utf-8

# # HW11 Task 1 - Linear model
# # KNN, LMNN, SVM, ENN

# In[1]:


#install packets
#!pip3 install pylmnn
# !pip install enn


# In[46]:


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


# In[2]:


# path to data
path='winequality-white.csv'
df=pd.read_csv(path, sep=";")
df.head()
# data size
print('\n shape: ',df.shape)

# data info
df.describe()

# target count
type=df['quality'].groupby(df['quality']).count()
print('\n type: ',type)


# In[9]:


df.head()


# In[11]:


# count target plot
type.plot('bar');


# In[12]:


# input data

# get column titles except the last column
features=df.columns[:-1].tolist()

# get data set features
X=df[features].values
# get labels
y=df['quality'].values

# split data to train data set and test data set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)


# In[7]:


get_ipython().run_cell_magic('time', '', "# KNN\n# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html\n# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html\n\nprint('\\n\\n We used kd_tree - sorting alhorytm, manhattan metric, with uniform weights, \\\nits a compromise on accuracy and speed \\n')\n\nprint('\\n For result KNN alhorytm we used chebyshev metric, that increase score for 0.2 points \\n\\n')\n\n\n# store scores of KNN model by K\nscores=[]\n\n# loop k from 1 to 8, and get cross validation score of each K value\nfor k in range(1,31):\n    knn=KNeighborsClassifier(k,algorithm='kd_tree',metric='manhattan',weights='uniform')\n    score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=5)\n    score_mean=score_val.mean()\n    print(f'neibors: {k} score: {score_mean:.3f}')\n    scores.append(score_mean)\n    \n# get index of maxium score along axis, default axis=0 for 1 dimensional array\nbest_k=np.argmax(scores)+1\nprint('\\nneibor:',best_k)\n# generate KNN model\nknn=KNeighborsClassifier(best_k,algorithm='kd_tree',metric='chebyshev',weights='uniform')\n# fit with train data set\nknn.fit(X_train,y_train)\n# get Modes presicion rate using test set\nprint(f'prediction precision rate: {knn.score(X_test,y_test):.5f}')")


# In[8]:


get_ipython().run_cell_magic('time', '', "# KNN & LMNN metric\n\nprint('\\n\\n For LMNN metric we used max_iter = 10 and k_test = 1 , that increase score for 0.2-0.3 points \\n')\nprint(' We used only to 5 neibors limit, because dataset is not balanced, later we ll fix that \\n')\nprint(' For result KNN alhorytm we used chebyshev metric, that increase score for 0.5-0.7 points \\n\\n')\n\n# LMNN metric\n#https://pylmnn.readthedocs.io/en/stable/api.html#module-pylmnn.lmnn\n\nscores=[]\n# loop k from 1 to 8, and get cross validation score of each K value\nk_test, n_components, max_iter = 1, X.shape[1], 10\n\nfor k in range(1,5):\n    start_time = timeit.default_timer()\n    # Instantiate the metric learner\n    lmnn = LMNN(n_neighbors=k, max_iter=max_iter, n_components=n_components)\n    # Train the metric learner\n    lmnn.fit(X_train, y_train)\n    # Fit the nearest neighbors classifier\n    knn = KNeighborsClassifier(n_neighbors=k_test, algorithm='kd_tree', metric='chebyshev', weights='uniform')\n    knn.fit(lmnn.transform(X_train), y_train)\n    score_val=cross_val_score(knn,lmnn.transform(X_train),y_train,scoring='accuracy',cv=5)\n    score_mean=score_val.mean()\n    scores.append(score_mean)\n    \n    end_time = timeit.default_timer() - start_time\n    \n    print(f'neibors: {k}  ->  score: {score_mean:.5f}  <-  time: {end_time:.0f} s')\n\n# KNN model \n\n# get index of maxium score along axis, default axis=0 for 1 dimensional array\nbest_k=np.argmax(scores)+1\nprint('\\nneibor:',best_k)\n# Instantiate the metric learner\nlmnn = LMNN(n_neighbors=int(best_k), max_iter=max_iter, n_components=n_components)\n# Train the metric learner\nlmnn.fit(X_train, y_train)\n# generate KNN model\nknn=KNeighborsClassifier(n_neighbors=best_k)\n# fit with train data set\nknn.fit(lmnn.transform(X_train), y_train)\n# get Modes presicion rate using test set\nprint(f'prediction precision rate: {knn.score(lmnn.transform(X_test),y_test):.5f}')")


# In[9]:


get_ipython().run_cell_magic('time', '', '# SVC & SVM model\n# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html\n# https://scikit-learn.org/stable/modules/svm.html#svm-classification\n\n# The implementation is based on libsvm. \n# The fit time complexity is more than quadratic with the number of samples \n# which makes it hard to scale to dataset with more than a couple of 10000 samples.\n\nprint(\'\\n\\n For SVC alhorytm we used gamma=0.53, that increase score for 0.2 points \\n\')\nprint(\' accuracy_score precision_score and recall_score have more scores when we used gamma=0.53 \\n\')\nprint(\' Chanhing kernels by  ‘linear’, ‘poly’, ‘sigmoid’ doesnt icrease score, so we used default rbf \\n\\n\')\n\nclf = SVC(gamma=0.53)\nclf.fit(X_train, y_train)\nprint(f\'prediction precision rate: {clf.score(X_test,y_test):.5f} \\n\')\n\ny_pred = clf.predict(X_test)\n\nprint(clf)\nprint(\'\\nTrain scores: \\n\')\n\nprint(f\'accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\'precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\'recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\'f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')\n#print(\'roc_auc_score = \', roc_auc_score(y_train, y_pred))')


# In[47]:


get_ipython().run_cell_magic('time', '', '# ENN model\n# https://github.com/timo-stoettner/ENN/blob/master/README.md\n# article https://mlbootcamp.ru/article/tutorial/\n\n# optimized function for euclidean metric in ENN\ndef euclidean(x,y):   \n    return np.sqrt(np.sum((x-y)**2))\n\noptimized_euclidean = autojit(euclidean)\n\nprint(\'\\n\\n For ENN alhorytm we used k=2 neihgbors, that increase f1_score for 0.5 points \\n\')\nprint(\' accuracy_score precision_score and recall_score have more scores when we used k = 2 \\n\')\n\nclf = ENN(k=2, distance_function = optimized_euclidean)\nclf.fit(X_train, y_train)\ny_pred = clf.predict(X_test)\n\nprint(\' Model ENN: \\n\',clf)\nprint(\'\\n Train scores: \\n\')\nprint(f\' accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\' precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')')


# In[19]:


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


# In[20]:


# Count balances type in dataset
type_balanced=df_balanced['quality'].groupby(df_balanced['quality']).count()
type_balanced.plot('bar');
df_balanced = shuffle(df_balanced)


# In[49]:


# Balanced dataframe

# df.columns is column labels property
features=df_balanced.columns[:-1].tolist()
X = df_balanced[features].values
y = df_balanced['quality']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)

# set y to arrays
y_train = y_train.values
y_test = y_test.values


# In[52]:


get_ipython().run_cell_magic('time', '', "# KNN && Gridsearch 1-variant\nn_neighbors_array = list(range(1,11))\nparam_grid={'n_neighbors':n_neighbors_array, 'algorithm':['kd_tree','ball_tree'],\\\n            'metric':['chebyshev','manhattan','minkowski'], 'weights':['uniform','distance']}\n\nknn = KNeighborsClassifier()\ngrid = GridSearchCV(knn, param_grid)\n\ngrid.fit(X_train, y_train)\n\nprint(f'best score: {grid.best_score_:.3f}  nieghbor: {grid.best_estimator_.n_neighbors}')\nprint(f'prediction precision rate: {grid.score(X_test,y_test):.5f}')")


# In[53]:


get_ipython().run_cell_magic('time', '', "# KNN 2-variant\n\n# store scores of KNN model by K\nscores=[]\n\n# loop k from 1 to 31, and get cross validation score of each K value\nfor k in range(1,5):\n    knn=KNeighborsClassifier(k,algorithm='kd_tree',metric='chebyshev',weights='uniform')\n    score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10)\n    score_mean=score_val.mean()\n    print(f'neibors: {k} score: {score_mean:.3f}')\n    scores.append(score_mean)  \n    \nbest_k=np.argmax(scores)+1\nprint('\\nneibor:',best_k)\n# generate KNN model\nknn = KNeighborsClassifier(best_k)\nparam_grid = {'algorithm' : ['kd_tree'],'metric':['chebyshev','manhattan','minkowski'],\\\n              'weights':['uniform','distance']}\ngrid = GridSearchCV(knn, param_grid)\n\n# fit with train data set\ngrid.fit(X_train,y_train)\n# get Modes presicion rate using test set\nprint(f'prediction precision rate: {grid.score(X_test,y_test):.5f}')")


# In[53]:


get_ipython().run_cell_magic('time', '', '# ENN model\n# https://github.com/timo-stoettner/ENN/blob/master/README.md\n# article https://mlbootcamp.ru/article/tutorial/\n\nprint(\'\\n For ENN alhorytm we used k=1 neihgbors, that return high value of scores \\n\')\n\nclf = ENN(k=1, distance_function = optimized_euclidean)\nclf.fit(X_train, y_train)\ny_pred = clf.predict(X_test)\n\nprint(\' Model ENN: \\n\',clf)\nprint(\'\\n Train scores: \\n\')\nprint(f\' accuracy_score = {accuracy_score(y_test, y_pred):.5f}\')\nprint(f\' precision_score = {precision_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' recall_score = {recall_score(y_test, y_pred, average="macro"):.5f}\')\nprint(f\' f1_score = {f1_score(y_test, y_pred, average="macro"):.5f} \\n\')')


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


# with out grid

# %%time
# # KNN

# # store scores of KNN model by K
# scores=[]

# # loop k from 1 to 8, and get cross validation score of each K value
# for k in range(1,31):
#     knn=KNeighborsClassifier(k,algorithm='kd_tree',metric='manhattan',weights='uniform')
#     score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10)
#     score_mean=score_val.mean()
#     print(f'neibors: {k} score: {score_mean:.3f}')
#     scores.append(score_mean)  
    
# best_k=np.argmax(scores)+1
# print('\nneibor:',best_k)
# # generate KNN model
# knn=KNeighborsClassifier(best_k,algorithm='kd_tree',metric='chebyshev',weights='uniform')
# # fit with train data set
# knn.fit(X_train,y_train)
# # get Modes presicion rate using test set
# print(f'prediction precision rate: {knn.score(X_test,y_test):.5f}')


# In[ ]:


# # Grid samples

# n_neighbors_array = [1, 2, 3, 4, 5]
# knn = KNeighborsClassifier()
# grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
# grid.fit(X_train, y_train)

# best_cv_err = 1 - grid.best_score_
# best_n_neighbors = grid.best_estimator_.n_neighbors
# print(best_cv_err, best_n_neighbors)

# clf = GridSearchCV(ENN(), {'k' : [1,5,7,8], "distance_function": [euclidean, mahalanobis]}) 
# clf.fit(X_train, y_train)
# pred_y = clf.predict(X_test)

