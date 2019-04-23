#!/usr/bin/env python
# coding: utf-8

# # <center> Capstone проект №1. Идентификация пользователей по посещенным веб-страницам
# 
# # <center>Неделя 2. Подготовка и первичный анализ данных

# ## Часть 1. Подготовка нескольких обучающих выборок для сравнения
# 
# Пока мы брали последовательности из 10 сайтов, и это было наобум. Давайте сделаем число сайтов в сессии параметром, чтоб в дальнейшем сравнить модели классификации, обученные на разных выборках – с 5, 7, 10 и 15 сайтами в сессии. Более того, пока мы брали по 10 сайтов подряд, без пересечения. Теперь давайте применим идею скользящего окна – сессии будут перекрываться. 
# 
# **Пример**: для длины сессии 10 и ширины окна 7 файл из 30 записей породит не 3 сессии, как раньше (1-10, 11-20, 21-30), а 5 (1-10, 8-17, 15-24, 22-30, 29-30). При этом в предпоследней сессии будет один ноль, а в последней – 8 нолей.
# 
# Создадим несколько выборок для разных сочетаний параметров длины сессии и ширины окна. Все они представлены в табличке ниже:
# 
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">session_length -&gt;<br>window_size <br></th>
#     <th class="tg-031e">5</th>
#     <th class="tg-031e">7</th>
#     <th class="tg-031e">10</th>
#     <th class="tg-031e">15</th>
#   </tr>
#   <tr>
#     <td class="tg-031e">5</td>
#     <td class="tg-031e">v</td>
#     <td class="tg-031e">v</td>
#     <td class="tg-031e">v</td>
#     <td class="tg-031e">v</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">7</td>
#     <td class="tg-031e"></td>
#     <td class="tg-031e">v</td>
#     <td class="tg-031e">v</td>
#     <td class="tg-031e">v</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">10</td>
#     <td class="tg-031e"></td>
#     <td class="tg-031e"></td>
#     <td class="tg-031e"><font color='green'>v</font></td>
#     <td class="tg-031e">v</td>
#   </tr>
# </table>
# 
# Итого должно получиться 18 разреженных матриц – указанные в таблице 9 сочетаний параметров формирования сессий для выборок из 10 и 150 пользователей. При этом 2 выборки мы уже сделали в прошлой части, они соответствуют сочетанию параметров: session_length=10, window_size=10, которые помечены в таблице выше галочкой зеленого цвета (done).

# Реализуйте функцию *prepare_sparse_train_set_window*.
# 
# Аргументы:
# - *path_to_csv_files* – путь к каталогу с csv-файлами
# - *site_freq_path* – путь к pickle-файлу с частотным словарем, полученным в 1 части проекта
# - *session_length* – длина сессии (параметр)
# - *window_size* – ширина окна (параметр) 
# 
# Функция должна возвращать 2 объекта:
# - разреженную матрицу *X_sparse* (двухмерная Scipy.sparse.csr_matrix), в которой строки соответствуют сессиям из *session_length* сайтов, а *max(site_id)* столбцов – количеству посещений *site_id* в сессии. 
# - вектор *y* (Numpy array) "ответов" в виде ID пользователей, которым принадлежат сессии из *X_sparse*
# 
# Детали:
# - Модифицируйте созданную в 1 части функцию *prepare_train_set*
# - Некоторые сессии могут повторяться – оставьте как есть, не удаляйте дубликаты
# - Замеряйте время выполнения итераций цикла с помощью *time* из *time*, *tqdm* из *tqdm* или с помощью виджета [log_progress](https://github.com/alexanderkuk/log-progress) ([статья](https://habrahabr.ru/post/276725/) о нем на Хабрахабре)
# - 150 файлов из *capstone_websites_data/150users/* должны обрабатываться за несколько секунд (в зависимости от входных параметров). Если дольше – не страшно, но знайте, что функцию можно ускорить. 

# In[1]:


from __future__ import division, print_function

import os
import itertools
import numpy as np
import pickle
import pandas as pd
import re
import timeit
import warnings

from glob import glob
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm_notebook

# disable warning in Anaconda
warnings.filterwarnings('ignore')


# In[2]:


# Functions

def dict_modify(d1, d2):
    '''Adding frequence count (second tuple element) from dict2 to dict1
    d1 ->  site_freq_users_all
    d2 ->  site_freq_common_users
    '''
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    return {sites : (d1[sites][0], d1[sites][1] + d2[sites][1]) for sites in intersect_keys}


def incomplete_condition(user_data_eval, session_length):
    '''Condition for determining incomplete session'''
    if not user_data_eval.shape[0] % session_length == 0:   
        # Fill zeros for incomplete session
        for index in range( user_data_eval.shape[0], (user_data_eval.shape[0] // session_length + 1)*session_length ):
            user_data_eval.loc[index] = [0, 0]

def intersect1d_set(A,B):
    '''Intersection elements for lists'''
    if not B:
        result = []
    else: result = list( set.intersection(set(B),set(A)) )    
    return result


def sparse_csr(array2D):
    '''
    make data, indices, indptr for sparse matrix
    array2D - input array'''
    data = []
    indices = []
    indptr = [0]
    for array in array2D:
        unique, counts = np.unique(array[array != 0], return_counts=True)
        indptr.append(indptr[-1] + len(unique))
        for u, c in zip(unique, counts):
            indices.append(u - 1)
            data.append(c)
    return data, indices, indptr


def setdiff1d_modify(A,B):
    '''Differets elements of lists'''
    if not B:
        result = A
    else: result = np.array( list( set(A) - set(B) ), dtype=object )
    return result


def window_session_items(session_length, window_length, array_size):
    '''index of windowed session length'''
    
    # first index
    start_index = (session_length + 1) - (session_length - window_length)
    tmp_start = start_index
    indexs = [1]
    indexs.append(start_index)
    while tmp_start < array_size:   
        tmp_start =  tmp_start +  window_length
        indexs.append(tmp_start)
    first_index = indexs[:-1]

        # second index
    step_index = session_length - start_index
    second_index = [session_length]
    tmp_end = [*map(lambda x: x + step_index, first_index[2:])]
    tmp_end = [*map(lambda x: x if x < array_size else array_size, tmp_end)]
    second_index = second_index + tmp_end
    second_index.append(array_size)

    return zip(first_index, second_index)

def prepare_sparse_train_set_window(path_to_csv_files, path_pckl_file, session_length=10, window_length=7):
    
    # Str numeration for site
    site_numeration = ['site' + str(index + 1) for index in range(session_length)]

    # Inicnjdhtial dataframe
    resultData_all = pd.DataFrame(columns=(site_numeration + ['user_id']))
    # Initial unique site and index for them
    unique_site_all = []
    index_list_all = []
    site_freq_users_all = dict()
    user_id = 0
    start_index = 0

    def adding_algorytm(user_data):
        '''adding sessions to the dataframe'''
    
        nonlocal user_id, resultData_all
        #User id value
        user_id = user_id + 1 
        # Copy of daraframe for main algorytm
        user_data_eval = user_data.copy() 
       
        # Condition for determining incomplete session
        incomplete_condition(user_data_eval, session_length)
    
        # Main alhortm of replacing str sites for numbers
        slice_index = [*map(lambda x: (x[0] - 1, x[1] - 1),                        window_session_items(session_length, window_length, user_data_eval['site'].shape[0]))]
        sessions_length = [*map(lambda x: x[1] - x[0] + 1, slice_index)]
        #sessions_range = [*map(np.arange, sessions_length)]

        all_sessions = [*map(lambda x: user_data_eval['site'].values[x[0]:x[1]+1], slice_index)]
        sessions = len(sessions_length)

        resultData = pd.DataFrame(data=all_sessions, index=np.arange(sessions), columns=site_numeration)
        resultData = resultData.fillna(0)

        # resultData add user id
        resultData['user_id'] = pd.Series(user_id, index=resultData.index)
    
        # Store to the global value
        resultData_all = resultData_all.append(resultData, ignore_index=True)
        
    def path_to_csv(path_to_csv_files, PATH_TO_PROJECT='~/geekhubds/HW09'):
        ''' Path to data - csv files'''
        # File operations
        file_quant = len(glob(path_to_csv_files))
        file_names = [file for file in glob(path_to_csv_files)]
        file_length = len(file_names)
        # Import csv files
        user_data = [pd.read_csv(os.path.join(PATH_TO_PROJECT,file)) for file in file_names]
        return user_data
    
    def path_to_plc(path_pckl_file):
        ''' Path to data - pckl file'''
        with open(path_pckl_file, 'rb') as site_freq:
            site_freq = pickle.load(site_freq)
        def replace_func(site_freq):
            '''Creation replace dic from site freq dictionary'''
            return {site:site_freq.get(site)[0] for site in site_freq.keys()}
        
        return replace_func(site_freq)

   
    # Main algorytm

    # Import csv files 
    user_data = path_to_csv(path_to_csv_files) 

    # Import replace dictionary
    replasment = path_to_plc(path_pckl_file)
            
    # Replacing str sites for numbers
    [*map(adding_algorytm, user_data)];

    #Replacing site_id in column and delete NANs
    for site in site_numeration:
        resultData_all[site] = resultData_all[site].map(replasment.get)
    resultData_all = resultData_all.fillna(0).astype(int)
    
    # crs matrix algorytm
    X_users, y_users = resultData_all.iloc[:, :-1].values, resultData_all.iloc[:, -1].values
    
    data_users, indices_users, indptr_users = sparse_csr(X_users)
    X_sparse_users = csr_matrix((data_users, indices_users, indptr_users), dtype=int)
    
    return X_sparse_users, y_users


# In[3]:


get_ipython().run_cell_magic('time', '', "path_to_csv_files = '3users\\*'\npath_pckl_file = 'site_freq_3users.pkl'\nX_sparse_users, y_users =  \\\n                prepare_sparse_train_set_window(path_to_csv_files, path_pckl_file, session_length=10, window_length=7)")


# In[4]:


X_sparse_users.todense()


# **Примените полученную функцию с параметрами *session_length=5* и *window_size=3* к игрушечному примеру. Убедитесь, что все работает как надо.**

# In[5]:


path_to_csv_files = '3users\*'
path_pckl_file = 'site_freq_3users.pkl'
X_toy_s5_w3, y_s5_w3 =  prepare_sparse_train_set_window(path_to_csv_files, path_pckl_file, 5, 3)


# In[6]:


X_toy_s5_w3.todense()


# In[7]:


y_s5_w3


# **Запустите созданную функцию 16 раз с помощью циклов по числу пользователей num_users (10 или 150), значениям параметра *session_length* (15, 10, 7 или 5) и значениям параметра *window_size* (10, 7 или 5). Сериализуйте все 16 разреженных матриц (обучающие выборки) и векторов (метки целевого класса – ID пользователя) в файлы `X_sparse_{num_users}users_s{session_length}_w{window_size}.pkl` и `y_{num_users}users_s{session_length}_w{window_size}.pkl`.**
# 
# **Чтоб убедиться, что мы все далее будем работать с идентичными объектами, запишите в список *data_lengths* число строк во всех полученных рареженных матрицах (16 значений). Если какие-то будут совпадать, это нормально (можно сообразить, почему).**
# 
# **На моем ноутбуке этот участок кода отработал за 26 секунд, хотя понятно, что все зависит от эффективности реализации функции *prepare_sparse_train_set_window* и мощности используемого железа. И честно говоря, моя первая реализация была намного менее эффективной (34 минуты), так что тут у Вас есть возможность оптимизировать свой код.**

# In[10]:


get_ipython().run_cell_magic('time', '', "\ndata_lengths_dic = []\ndata_lengths = []\n\n#inputdata\nnum_users = (('10users\\*', 'site_freq_10users.pkl'),('150users\\*', 'site_freq_150users.pkl'))\n\n\nfor users in num_users:\n    \n    # User number\n    number_users = int(re.findall(r'\\d+', users[0])[0])\n    \n    for window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):\n        \n        if window_size <= session_length and (window_size, session_length) != (10, 10):\n            \n            X_sparse, y = prepare_sparse_train_set_window(users[0], users[1], session_length, window_size)\n            \n            data_lengths_dic.append({(session_length,window_size):X_sparse.shape[0]})\n            data_lengths.append(X_sparse.shape[0])\n            \n            #store pickle files\n            with open(f'X_sparse_{number_users}users_s{session_length}_w{window_size}.pkl', 'wb') as X_pkl:\n                pickle.dump(X_sparse, X_pkl, protocol=2)\n            with open(f'y_{number_users}users_s{session_length}_w{window_size}.pkl', 'wb') as y_pkl:\n                pickle.dump(y, y_pkl, protocol=2)\n            \n            print(f'Users:{number_users} session:{session_length} window:{window_size} matrix length:{X_sparse.shape[0]}')")


# **<font color='red'> Вопрос 1. </font>Сколько всего уникальных значений в списке `data_lengths`?**

# In[18]:


print('   Unique values in the data_lengths list:', len( np.unique(data_lengths) ))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#old code


# In[ ]:


# # Second realization without import pickle file
# # Functions

# def dict_modify(d1, d2):
#     '''Adding frequence count (second tuple element) from dict2 to dict1
#     d1 ->  site_freq_users_all
#     d2 ->  site_freq_common_users
#     '''
#     d1_keys = set(d1.keys())
#     d2_keys = set(d2.keys())
#     intersect_keys = d1_keys.intersection(d2_keys)
#     return {sites : (d1[sites][0], d1[sites][1] + d2[sites][1]) for sites in intersect_keys}


# def incomplete_condition(user_data_eval, session_length):
#     '''Condition for determining incomplete session'''
#     if not user_data_eval.shape[0] % session_length == 0:   
#         # Fill zeros for incomplete session
#         for index in range( user_data_eval.shape[0], (user_data_eval.shape[0] // session_length + 1)*session_length ):
#             user_data_eval.loc[index] = [0, 0]

# def intersect1d_set(A,B):
#     '''Intersection elements for lists'''
#     if not B:
#         result = []
#     else: result = list( set.intersection(set(B),set(A)) )    
#     return result


# def sparse_csr(array2D):
#     '''
#     make data, indices, indptr for sparse matrix
#     array2D - input array'''
#     data = []
#     indices = []
#     indptr = [0]
#     for array in array2D:
#         unique, counts = np.unique(array[array != 0], return_counts=True)
#         indptr.append(indptr[-1] + len(unique))
#         for u, c in zip(unique, counts):
#             indices.append(u - 1)
#             data.append(c)
#     return data, indices, indptr


# def setdiff1d_modify(A,B):
#     '''Differets elements of lists'''
#     if not B:
#         result = A
#     else: result = np.array( list( set(A) - set(B) ), dtype=object )
#     return result


# def window_session_items(session_length, window_length, array_size):
#     '''index of windowed session length'''
    
#     # first index
#     start_index = (session_length + 1) - (session_length - window_length)
#     tmp_start = start_index
#     indexs = [1]
#     indexs.append(start_index)
#     while tmp_start < array_size:   
#         tmp_start =  tmp_start +  window_length
#         indexs.append(tmp_start)
#     first_index = indexs[:-1]

#         # second index
#     step_index = session_length - start_index
#     second_index = [session_length]
#     tmp_end = [*map(lambda x: x + step_index, first_index[2:])]
#     tmp_end = [*map(lambda x: x if x < array_size else array_size, tmp_end)]
#     second_index = second_index + tmp_end
#     second_index.append(array_size)

#     return zip(first_index, second_index)

# def prepare_sparse_train_set_window(path_to_csv_files, session_length=10, window_length=7):
    
#     # Str numeration for site
#     site_numeration = ['site' + str(index + 1) for index in range(session_length)]

#     # Inicnjdhtial dataframe
#     resultData_all = pd.DataFrame(columns=(site_numeration + ['user_id']))
#     # Initial unique site and index for them
#     unique_site_all = []
#     index_list_all = []
#     site_freq_users_all = dict()
#     user_id = 0
#     start_index = 0

#     def adding_algorytm(user_data):
#         '''adding sessions to the dataframe'''
    
#         nonlocal user_id, resultData_all
#         #User id value
#         user_id = user_id + 1 
#         # Copy of daraframe for main algorytm
#         user_data_eval = user_data.copy() 
       
#         # Condition for determining incomplete session
#         incomplete_condition(user_data_eval, session_length)
    
#         # Main alhortm of replacing str sites for numbers
#         slice_index = [*map(lambda x: (x[0] - 1, x[1] - 1),\
#                         window_session_items(session_length, window_length, user_data_eval['site'].shape[0]))]
#         sessions_length = [*map(lambda x: x[1] - x[0] + 1, slice_index)]
#         #sessions_range = [*map(np.arange, sessions_length)]

#         all_sessions = [*map(lambda x: user_data_eval['site'].values[x[0]:x[1]+1], slice_index)]
#         sessions = len(sessions_length)

#         resultData = pd.DataFrame(data=all_sessions, index=np.arange(sessions), columns=site_numeration)
#         resultData = resultData.fillna(0)

#         # resultData add user id
#         resultData['user_id'] = pd.Series(user_id, index=resultData.index)
    
#         # Store to the global value
#         resultData_all = resultData_all.append(resultData, ignore_index=True)
        
#     def path_to_csv(path_to_csv_files, PATH_TO_PROJECT='~/geekhubds/HW09'):
#         ''' Path to data - csv files'''
#         # File operations
#         file_quant = len(glob(path_to_csv_files))
#         file_names = [file for file in glob(path_to_csv_files)]
#         file_length = len(file_names)
#         # Import csv files
#         user_data = [pd.read_csv(os.path.join(PATH_TO_PROJECT,file)) for file in file_names]
#         return user_data#, file_length
   
#     def vocabruary(user_data):
#         '''make vocabruary -> {site: (index,counts)}'''
    
#         nonlocal unique_site_all,index_list_all,site_freq_users_all,start_index

#         # Unique sites and them counts
#         unique_counts = user_data.groupby(['site'], sort=False).size()
#         unique_site = np.array(unique_counts.index)
#         #unique_counts = np.array(unique_counts)

#        # Intersect of unique sites (indexes) for each iteration 
#         common_site = intersect1d_set(unique_site, unique_site_all)
#         index_com_site = np.where(np.in1d(unique_site, common_site))[0]
    
#         # Difference of unique sites (indexes) for each iteration  
#         not_common_site = setdiff1d_modify(unique_site, unique_site_all)
#         index_not_common_site = np.where(np.in1d(unique_site, not_common_site))[0]
    
#         # Store to global
#         unique_site_all = unique_site_all + not_common_site.tolist()
#         lenght_unique = not_common_site.shape[0]
    
#         # Index of unique sites for each loop and store to global index value
#         index_list = np.arange(start_index + 1, start_index + lenght_unique + 1)
#         start_index = start_index + lenght_unique
#         index_list_all = index_list_all + index_list.tolist() 
    
#         # Unique elements for each iteration
#         count_index = map(lambda x: x, index_list)
#         counts_sites = map(lambda x: unique_counts[x], index_not_common_site)
#         freq_site =  zip(count_index, counts_sites)
    
#         # Store to dictionary
#         site_freq_users = dict(zip(unique_site[index_not_common_site].tolist(),freq_site))
#         site_freq_users_all.update(site_freq_users)

#         # Common unique elements for each iteration and store ti dictionary
#         count_comon_index = [site_freq_users_all.get(i)[0] for i in common_site]
#         counts_comon_sites = unique_counts[index_com_site].tolist()
#         freq_common_site = zip(count_comon_index, counts_comon_sites)
#         site_freq_common_users = dict(zip(common_site, freq_common_site))
#         site_freq_users_all.update(dict_modify(site_freq_users_all, site_freq_common_users))
   
#     # Main algorytm

#     # Import csv files 
#     user_data = path_to_csv(path_to_csv_files) 

#     # Dictionary algorytm
#     [*map(vocabruary,user_data)];
#     # Replacing index and raplace dictionary
#     replasment = dict(zip(unique_site_all, index_list_all))
            
#     # Replacing str sites for numbers
#     [*map(adding_algorytm, user_data)];

#     #Replacing site_id in column and delete NANs
#     for site in site_numeration:
#         resultData_all[site] = resultData_all[site].map(replasment.get)
#     resultData_all = resultData_all.fillna(0).astype(int)
    
#     # crs matrix algorytm
#     X_users, y_users = resultData_all.iloc[:, :-1].values, resultData_all.iloc[:, -1].values
    
#     data_users, indices_users, indptr_users = sparse_csr(X_users)
#     X_sparse_users = csr_matrix((data_users, indices_users, indptr_users), dtype=int)
    
#     return X_sparse_users, y_users


# In[ ]:





# In[ ]:


# # The 1 realisation of the function from the week 2 

# def prepare_train_set(path_to_csv_files, session_length=10, window_length=7):
 
#     # Str numeration for site
#     site_numeration = ['site' + str(index + 1) for index in range(session_length)]

#     # Inicnjdhtial dataframe
#     resultData_all = pd.DataFrame(columns=(site_numeration + ['user_id']))
#     # Initial unique site and index for them
#     #unique_site_all = []
#     index_list_all = []
#     #site_freq_users_all = dict()
#     all_time = [] 
#     start_index = 0

#     # Functions
    
#     def path_to_csv(path_to_csv_files, PATH_TO_PROJECT='~/geekhubds/HW09'):
#         ''' Path to data - csv files'''
#         # File operations
#         file_quant = len(glob(path_to_csv_files))
#         file_names = [file for file in glob(path_to_csv_files)]
#         file_length = len(file_names)
#         # Import csv files
#         user_data = [pd.read_csv(os.path.join(PATH_TO_PROJECT,file)) for file in file_names]
#         return user_data, file_length

#     def dict_modify(d1, d2):
#         '''Adding frequence count (second tuple element) from dict2 to dict1
#         d1 ->  site_freq_users_all
#         d2 ->  site_freq_common_users
#         '''
#         d1_keys = set(d1.keys())
#         d2_keys = set(d2.keys())
#         intersect_keys = d1_keys.intersection(d2_keys)
#         return {sites : (d1[sites][0], d1[sites][1] + d2[sites][1]) for sites in intersect_keys}

#     def intersect1d_set(A,B):
#         '''Intersection elements for lists'''
#         if not B:
#             result = []
#         else: result = list( set.intersection(set(B),set(A)) )    
#         return result

#     def setdiff1d_modify(A,B):
#         '''Differets elements of lists'''
#         if not B:
#             result = A
#         else: result = np.array( list( set(A) - set(B) ), dtype=object )
#         return result

#     def incomplete_condition(user_data_eval, session_length):
#         '''Condition for determining incomplete session'''
#         if not user_data_eval.shape[0] % session_length == 0:   
#             # Fill zeros for incomplete session
#             for index in range( user_data_eval.shape[0], (user_data_eval.shape[0] // session_length + 1)*session_length ):
#                 user_data_eval.loc[index] = [0, 0]

#     def window_session_items(session_length, window_length, array_size):
#         '''index of windowed session length'''
    
#         # first index
#         start_index = (session_length + 1) - (session_length - window_length)
#         tmp_start = start_index
#         indexs = [1]
#         indexs.append(start_index)
#         while tmp_start < array_size:   
#             tmp_start =  tmp_start +  window_length
#             indexs.append(tmp_start)
#         first_index = indexs[:-1]

#         # second index
#         step_index = session_length - start_index
#         second_index = [session_length]
#         tmp_end = [*map(lambda x: x + step_index, first_index[2:])]
#         tmp_end = [*map(lambda x: x if x < array_size else array_size, tmp_end)]
#         second_index = second_index + tmp_end
#         second_index.append(array_size)

#         return zip(first_index, second_index)

#     user_data, file_length = path_to_csv(path_to_csv_files)
           
#     for file in np.arange(file_length):
 
#         # Unique sites and them counts
#         unique_counts = user_data[file].groupby(['site'], sort=False).size()
#         unique_site = np.array(unique_counts.index)
#         unique_counts = np.array(unique_counts)

#         # Intersect of unique sites (indexes) for each iteration 
#         common_site = intersect1d_set(unique_site, unique_site_all)
#         index_com_site = np.where(np.in1d(unique_site, common_site))[0]

#         # Difference of unique sites (indexes) for each iteration  
#         not_common_site = setdiff1d_modify(unique_site, unique_site_all)
#         index_not_common_site = np.where( np.in1d( unique_site, not_common_site ))[0]

#         # Store to global
#         unique_site_all = unique_site_all + not_common_site.tolist()
#         lenght_unique = not_common_site.shape[0]

#         # Index of unique sites for each loop and store to global index value
#         index_list = np.arange(start_index + 1, start_index + lenght_unique + 1)
#         start_index = start_index + lenght_unique
#         index_list_all = index_list_all + index_list.tolist() 

#         # Unique elements for each iteration
#         count_index = map(lambda x: x, index_list)
#         counts_sites = map(lambda x: unique_counts[x], index_not_common_site)
#         freq_site = list( zip(count_index, counts_sites) )
#         # Store to dictionary
#         site_freq_users = dict(zip(unique_site[index_not_common_site].tolist(),freq_site))
#         site_freq_users_all.update(site_freq_users)

#         # Common unique elements for each iteration and store ti dictionary
#         count_comon_index = [site_freq_users_all.get(i)[0] for i in common_site]
#         counts_comon_sites = unique_counts[index_com_site].tolist()
#         freq_common_site = list( zip(count_comon_index, counts_comon_sites) )
#         site_freq_common_users = dict(zip(common_site, freq_common_site))
#         site_freq_users_all.update(dict_modify(site_freq_users_all, site_freq_common_users))

#         # Replacing index and raplace dictionary
#         replasment = dict(zip(unique_site_all,index_list_all))
    
#         # User id value
#         user_id = file + 1 
#         # Copy of daraframe for main algorytm
#         user_data_eval = user_data[file].copy() 
    
#         # Condition for determining incomplete session
#         incomplete_condition(user_data_eval, session_length)
    
#         # Main alhortm of replacing str sites for numbers
#         slice_index = [*map(lambda x: (x[0] - 1, x[1] - 1),\
#                             window_session_items(session_length, window_length, user_data_eval['site'].shape[0]))]
#         sessions_length = map(lambda x: x[1] - x[0] + 1, slice_index)
#         sessions_range = [*map(np.arange, sessions_length)]

#         all_sessions = [*map(lambda x: user_data_eval['site'].values[x[0]:x[1]+1], slice_index)]
#         sessions = len([*window_session_items(session_length, window_length, user_data_eval['site'].shape[0])])

#         resultData = pd.DataFrame(index=np.arange(sessions), columns=site_numeration)

#         for session in np.arange(sessions):
#             for index in sessions_range[session]:
#                 resultData.iat[session,index] = all_sessions[session].tolist()[index]
#         resultData = resultData.fillna(0)     

#         # resultData add user id
#         resultData['user_id'] = pd.Series(user_id, index=resultData.index)
    
#         # Store to the global value
#         resultData_all = resultData_all.append(resultData, ignore_index=True)
    
#     #Replacing site_id in column and delete NANs
#     for site in site_numeration:
#         resultData_all[site] = resultData_all[site].map(replasment.get)
#     resultData_all = resultData_all.fillna(0).astype(int)

#     return resultData_all, site_freq_users_all


# #     def vocabruary_index(user_data):

# #         nonlocal unique_site_all, index_list_all, site_freq_users_all, start_index
    
# #         # Unique sites and them counts
# #         unique_counts = user_data.groupby(['site'], sort=False).size()
# #         unique_site = np.array(unique_counts.index)

# #        # Intersect of unique sites (indexes) for each iteration 
# #         common_site = intersect1d_set(unique_site, unique_site_all)
# #         index_com_site = np.where(np.in1d(unique_site, common_site))[0]
    
# #         # Difference of unique sites (indexes) for each iteration  
# #         not_common_site = setdiff1d_modify(unique_site, unique_site_all)
# #         index_not_common_site = np.where(np.in1d(unique_site, not_common_site))[0]

    
# #         # Store to global
# #         unique_site_all = unique_site_all + not_common_site.tolist()
# #         lenght_unique = not_common_site.shape[0]
    
# #         index_list = np.arange(start_index + 1, start_index + lenght_unique + 1)
# #         start_index = start_index + lenght_unique
# #         index_list_all = index_list_all + index_list.tolist() 
    
# #         # Unique elements for each iteration
# #         count_index = map(lambda x: x, index_list)
    
# #         # Store to dictionary
# #         site_freq_users = dict(zip(unique_site[index_not_common_site].tolist(),count_index))
# #         site_freq_users_all.update(site_freq_users)  


# In[ ]:


# # The third realisation of the function from the week 1
# def prepare_train_set(path_to_csv_files, session_length=10, window_length=7):
 
#     # Str numeration for site
#     site_numeration = ['site' + str(index + 1) for index in range(session_length)]

#     # Inicnjdhtial dataframe
#     resultData_all = pd.DataFrame(columns=(site_numeration + ['user_id']))
#     # Initial unique site and index for them
#     unique_site_all = []
#     index_list_all = []
#     site_freq_users_all = dict()
#     all_time = [] 
#     start_index = 0

#     # Functions
    
#     def path_to_csv(path_to_csv_files, PATH_TO_PROJECT='~/geekhubds/HW09'):
#         ''' Path to data - csv files'''
#         # File operations
#         file_quant = len(glob(path_to_csv_files))
#         file_names = [file for file in glob(path_to_csv_files)]
#         file_length = len(file_names)
#         # Import csv files
#         user_data = [pd.read_csv(os.path.join(PATH_TO_PROJECT,file)) for file in file_names]
#         return user_data, file_length

#     def dict_modify(d1, d2):
#         '''Adding frequence count (second tuple element) from dict2 to dict1
#         d1 ->  site_freq_users_all
#         d2 ->  site_freq_common_users
#         '''
#         d1_keys = set(d1.keys())
#         d2_keys = set(d2.keys())
#         intersect_keys = d1_keys.intersection(d2_keys)
#         return {sites : (d1[sites][0], d1[sites][1] + d2[sites][1]) for sites in intersect_keys}

#     def intersect1d_set(A,B):
#         '''Intersection elements for lists'''
#         if not B:
#             result = []
#         else: result = list( set.intersection(set(B),set(A)) )    
#         return result

#     def setdiff1d_modify(A,B):
#         '''Differets elements of lists'''
#         if not B:
#             result = A
#         else: result = np.array( list( set(A) - set(B) ), dtype=object )
#         return result

#     def incomplete_condition(user_data_eval, session_length):
#         '''Condition for determining incomplete session'''
#         if not user_data_eval.shape[0] % session_length == 0:   
#             # Fill zeros for incomplete session
#             for index in range( user_data_eval.shape[0], (user_data_eval.shape[0] // session_length + 1)*session_length ):
#                 user_data_eval.loc[index] = [0, 0]

#     def window_session_items(session_length, window_length, array_size):
#         '''index of windowed session length'''
    
#         # first index
#         start_index = (session_length + 1) - (session_length - window_length)
#         tmp_start = start_index
#         indexs = [1]
#         indexs.append(start_index)
#         while tmp_start < array_size:   
#             tmp_start =  tmp_start +  window_length
#             indexs.append(tmp_start)
#         first_index = indexs[:-1]

#         # second index
#         step_index = session_length - start_index
#         second_index = [session_length]
#         tmp_end = [*map(lambda x: x + step_index, first_index[2:])]
#         tmp_end = [*map(lambda x: x if x < array_size else array_size, tmp_end)]
#         second_index = second_index + tmp_end
#         second_index.append(array_size)

#         return zip(first_index, second_index)

#     user_data, file_length = path_to_csv(path_to_csv_files)
           
#     for file in np.arange(file_length):
 
#         # Unique sites and them counts
#         unique_counts = user_data[file].groupby(['site'], sort=False).size()
#         unique_site = np.array(unique_counts.index)
#         unique_counts = np.array(unique_counts)

#         # Intersect of unique sites (indexes) for each iteration 
#         common_site = intersect1d_set(unique_site, unique_site_all)
#         index_com_site = np.where(np.in1d(unique_site, common_site))[0]

#         # Difference of unique sites (indexes) for each iteration  
#         not_common_site = setdiff1d_modify(unique_site, unique_site_all)
#         index_not_common_site = np.where( np.in1d( unique_site, not_common_site ))[0]

#         # Store to global
#         unique_site_all = unique_site_all + not_common_site.tolist()
#         lenght_unique = not_common_site.shape[0]

#         # Index of unique sites for each loop and store to global index value
#         index_list = np.arange(start_index + 1, start_index + lenght_unique + 1)
#         start_index = start_index + lenght_unique
#         index_list_all = index_list_all + index_list.tolist() 

#         # Unique elements for each iteration
#         count_index = map(lambda x: x, index_list)
#         counts_sites = map(lambda x: unique_counts[x], index_not_common_site)
#         freq_site = list( zip(count_index, counts_sites) )
#         # Store to dictionary
#         site_freq_users = dict(zip(unique_site[index_not_common_site].tolist(),freq_site))
#         site_freq_users_all.update(site_freq_users)

#         # Common unique elements for each iteration and store ti dictionary
#         count_comon_index = [site_freq_users_all.get(i)[0] for i in common_site]
#         counts_comon_sites = unique_counts[index_com_site].tolist()
#         freq_common_site = list( zip(count_comon_index, counts_comon_sites) )
#         site_freq_common_users = dict(zip(common_site, freq_common_site))
#         site_freq_users_all.update(dict_modify(site_freq_users_all, site_freq_common_users))

#         # Replacing index and raplace dictionary
#         replasment = dict(zip(unique_site_all,index_list_all))
    
#         # User id value
#         user_id = file + 1 
#         # Copy of daraframe for main algorytm
#         user_data_eval = user_data[file].copy() 
    
#         # Condition for determining incomplete session
#         incomplete_condition(user_data_eval, session_length)
    
#         # Main alhortm of replacing str sites for numbers
#         slice_index = [*map(lambda x: (x[0] - 1, x[1] - 1),\
#                             window_session_items(session_length, window_length, user_data_eval['site'].shape[0]))]
#         sessions_length = map(lambda x: x[1] - x[0] + 1, slice_index)
#         sessions_range = [*map(np.arange, sessions_length)]

#         all_sessions = [*map(lambda x: user_data_eval['site'].values[x[0]:x[1]+1], slice_index)]
#         sessions = len([*window_session_items(session_length, window_length, user_data_eval['site'].shape[0])])

#         resultData = pd.DataFrame(index=np.arange(sessions), columns=site_numeration)

#         for session in np.arange(sessions):
#             for index in sessions_range[session]:
#                 resultData.iat[session,index] = all_sessions[session].tolist()[index]
#         resultData = resultData.fillna(0)     

#         # resultData add user id
#         resultData['user_id'] = pd.Series(user_id, index=resultData.index)
    
#         # Store to the global value
#         resultData_all = resultData_all.append(resultData, ignore_index=True)
    
#     #Replacing site_id in column and delete NANs
#     for site in site_numeration:
#         resultData_all[site] = resultData_all[site].map(replasment.get)
#     resultData_all = resultData_all.fillna(0).astype(int)

#     return resultData_all, site_freq_users_all

