#!/usr/bin/env python
# coding: utf-8

# # <center> Capstone проект №1. Идентификация пользователей по посещенным веб-страницам
# 
# В этом проекте мы будем решать задачу идентификации пользователя по его поведению в сети Интернет. Это сложная и интересная задача на стыке анализа данных и поведенческой психологии. В качестве примера, компания Яндекс решает задачу идентификации взломщика почтового ящика по его поведению. В двух словах, взломщик будет себя вести не так, как владелец ящика: он может не удалять сообщения сразу по прочтении, как это делал хозяин, он будет по-другому ставить флажки сообщениям и даже по-своему двигать мышкой. Тогда такого злоумышленника можно идентифицировать и "выкинуть" из почтового ящика, предложив хозяину войти по SMS-коду. Этот пилотный проект описан в [статье](https://habrahabr.ru/company/yandex/blog/230583/) на Хабрахабре. Похожие вещи делаются, например, в Google Analytics и описываются в научных статьях, найти можно многое по фразам "Traversal Pattern Mining" и "Sequential Pattern Mining".
# 
# Мы будем решать похожую задачу: по последовательности из нескольких веб-сайтов, посещенных подряд один и тем же человеком, мы будем идентифицировать этого человека. Идея такая: пользователи Интернета по-разному переходят по ссылкам, и это может помогать их идентифицировать (кто-то сначала в почту, потом про футбол почитать, затем новости, контакт, потом наконец – работать, кто-то – сразу работать).
# 
# Будем использовать данные из [статьи](http://ceur-ws.org/Vol-1703/paper12.pdf) "A Tool for Classification of Sequential Data". И хотя мы не можем рекомендовать эту статью (описанные методы делеки от state-of-the-art, лучше обращаться к [книге](http://www.charuaggarwal.net/freqbook.pdf) "Frequent Pattern Mining" и последним статьям с ICDM), но данные там собраны аккуратно и представляют интерес.
# 
# Имеются данные с прокси-серверов Университета Блеза Паскаля, они имеют очень простой вид. Для каждого пользователя заведен csv-файл с названием user\*\*\*\*.csv (где вместо звездочек – 4 цифры, соответствующие ID пользователя), а в нем посещения сайтов записаны в следующем формате: <br>
# 
# <center>*timestamp, посещенный веб-сайт*</center>
# 
# Скачать исходные данные можно по ссылке в статье, там же описание.
# Для этого задания хватит данных не по всем 3000 пользователям, а по 10 и 150. [Ссылка](https://yadi.sk/d/3gscKIdN3BCASG) на архив *capstone_user_identification* (~7 Mb, в развернутом виде ~ 60 Mb). 
# 
# В финальном проекте уже придется столкнуться с тем, что не все операции можно выполнить за разумное время (скажем, перебрать с кросс-валидацией 100 комбинаций параметров случайного леса на этих данных Вы вряд ли сможете), поэтому мы будем использовать параллельно 2 выборки: по 10 пользователям и по 150. Для 10 пользователей будем писать и отлаживать код, для 150 – будет рабочая версия. 
# 
# Данные устроены следующем образом:
# 
#  - В каталоге 10users лежат 10 csv-файлов с названием вида "user[USER_ID].csv", где [USER_ID] – ID пользователя;
#  - Аналогично для каталога 150users – там 150 файлов;
#  - В каталоге 3users – игрушечный пример из 3 файлов, это для отладки кода предобработки, который Вы далее напишете.

# In[1]:


# %load_ext line_profiler
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p numpy,scipy,pandas,matplotlib,statsmodels,sklearn -g')


# In[1]:


from __future__ import division, print_function

#import pdb
import numpy as np
import pandas as pd
import os
import pickle
# import pixiedust
import timeit
import warnings

from glob import glob
from scipy.sparse import csr_matrix
from tqdm import tqdm_notebook

warnings.filterwarnings('ignore')


# **Посмотрим на один из файлов с данными о посещенных пользователем (номер 31) веб-страницах.**

# In[2]:


# Path to data
PATH_TO_DATA = '~/geekhubds/HW09/'
# path_to_csv_files = '3users\*'
# # File operations
# file_quant = len(glob(path_to_csv_files))
# file_names = [file for file in glob(path_to_csv_files)]
# file_length = len(file_names)

# # Import csv files
# user_data = [pd.read_csv(os.path.join(PATH_TO_DATA,file)) for file in file_names]

# print(file_names)


# **Поставим задачу классификации: идентифицировать пользователя по сессии из 10 подряд посещенных сайтов. Объектом в этой задаче будет сессия из 10 сайтов, последовательно посещенных одним и тем же пользователем, признаками – индексы этих 10 сайтов (чуть позже здесь появится "мешок" сайтов, подход Bag of Words). Целевым классом будет id пользователя.**

# ### <center>Пример для иллюстрации</center>
# **Пусть пользователя всего 2, длина сессии – 2 сайта.**
# 
# <center>user0001.csv</center>
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-yw4l{vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">timestamp</th>
#     <th class="tg-031e">site</th>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:01</td>
#     <td class="tg-031e">vk.com</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">00:00:11</td>
#     <td class="tg-yw4l">google.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:16</td>
#     <td class="tg-031e">vk.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:20</td>
#     <td class="tg-031e">yandex.ru</td>
#   </tr>
# </table>
# 
# <center>user0002.csv</center>
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-yw4l{vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">timestamp</th>
#     <th class="tg-031e">site</th>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:02</td>
#     <td class="tg-031e">yandex.ru</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">00:00:14</td>
#     <td class="tg-yw4l">google.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:17</td>
#     <td class="tg-031e">facebook.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:25</td>
#     <td class="tg-031e">yandex.ru</td>
#   </tr>
# </table>
# 
# Идем по 1 файлу, нумеруем сайты подряд: vk.com – 1, google.com – 2 и т.д. Далее по второму файлу. 
# 
# Отображение сайтов в их индесы должно получиться таким:
# 
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-yw4l{vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">site</th>
#     <th class="tg-yw4l">site_id</th>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">vk.com</td>
#     <td class="tg-yw4l">1</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">google.com</td>
#     <td class="tg-yw4l">2</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">yandex.ru</td>
#     <td class="tg-yw4l">3</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">facebook.com</td>
#     <td class="tg-yw4l">4</td>
#   </tr>
# </table>
# 
# Тогда обучающая выборка будет такой (целевой признак – user_id):
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-s6z2{text-align:center}
# .tg .tg-baqh{text-align:center;vertical-align:top}
# .tg .tg-hgcj{font-weight:bold;text-align:center}
# .tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-hgcj">session_id</th>
#     <th class="tg-hgcj">site1</th>
#     <th class="tg-hgcj">site2</th>
#     <th class="tg-amwm">user_id</th>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">1</td>
#     <td class="tg-s6z2">1</td>
#     <td class="tg-s6z2">2</td>
#     <td class="tg-baqh">1</td>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">2</td>
#     <td class="tg-s6z2">1</td>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-baqh">1</td>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-s6z2">2</td>
#     <td class="tg-baqh">2</td>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">4</td>
#     <td class="tg-s6z2">4</td>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-baqh">2</td>
#   </tr>
# </table>
# 
# Здесь 1 объект – это сессия из 2 посещенных сайтов 1-ым пользователем (target=1). Это сайты vk.com и google.com (номер 1 и 2). И так далее, всего 4 сессии. Пока сессии у нас не пересекаются по сайтам, то есть посещение каждого отдельного сайта относится только к одной сессии.

# ## Часть 1. Подготовка обучающей выборки
# Реализуйте функцию *prepare_train_set*, которая принимает на вход путь к каталогу с csv-файлами *path_to_csv_files* и параметр *session_length* – длину сессии, а возвращает 2 объекта:
# - DataFrame, в котором строки соответствуют уникальным сессиям из *session_length* сайтов, *session_length* столбцов – индексам этих *session_length* сайтов и последний столбец – ID пользователя
# - частотный словарь сайтов вида {'site_string': [site_id, site_freq]}, например для недавнего игрушечного примера это будет {'vk.com': (1, 2), 'google.com': (2, 2), 'yandex.ru': (3, 3), 'facebook.com': (4, 1)}
# 
# Детали:
# - Смотрите чуть ниже пример вывода, что должна возвращать функция
# - Используйте `glob` (или аналоги) для обхода файлов в каталоге. Для определенности, отсортируйте список файлов лексикографически. Удобно использовать `tqdm_notebook` (или просто `tqdm` в случае python-скрипта) для отслеживания числа выполненных итераций цикла
# - Создайте частотный словарь уникальных сайтов (вида {'site_string': (site_id, site_freq)}) и заполняйте его по ходу чтения файлов. Начните с 1
# - Рекомендуется меньшие индексы давать более часто попадающимся сайтам (приницип наименьшего описания)
# - Не делайте entity recognition, считайте *google.com*, *http://www.google.com* и *www.google.com* разными сайтами (подключить entity recognition можно уже в рамках индивидуальной работы над проектом)
# - Скорее всего в файле число записей не кратно числу *session_length*. Тогда последняя сессия будет короче. Остаток заполняйте нулями. То есть если в файле 24 записи и сессии длины 10, то 3 сессия будет состоять из 4 сайтов, и ей мы сопоставим вектор [*site1_id*, *site2_id*, *site3_id*, *site4_id*, 0, 0, 0, 0, 0, 0, *user_id*] 
# - В итоге некоторые сессии могут повторяться – оставьте как есть, не удаляйте дубликаты. Если в двух сессиях все сайты одинаковы, но сессии принадлежат разным пользователям, то тоже оставляйте как есть, это естественная неопределенность в данных
# - Не оставляйте в частотном словаре сайт 0 (уже в конце, когда функция возвращает этот словарь)
# - 150 файлов из *capstone_websites_data/150users/* у меня обработались за 1.7 секунды, но многое, конечно, зависит от реализации функции и от используемого железа. И вообще, первая реализация скорее всего будет не самой эффективной, дальше можно заняться профилированием (особенно если планируете запускать этот код для 3000 пользователей). Также эффективная реализация этой функции поможет нам на следующей неделе.

# In[40]:


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

def setdiff1d_modify(A,B):
    '''Differets elements of lists'''
    if not B:
        result = A
    else: result = np.array( list( set(A) - set(B) ), dtype=object )
    return result


def prepare_train_set(path_to_csv_files, session_length=10):
    
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
        
        session_items = int( user_data_eval.shape[0] / session_length )
        all_sessions = user_data_eval['site'].values.reshape(session_items,-1)
        
        resultData = pd.DataFrame(data=all_sessions, index=np.arange(session_items), columns=site_numeration)
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
        return user_data#, file_length
   
    def vocabruary(user_data):
        '''make vocabruary -> {site: (index,counts)}'''
    
        nonlocal unique_site_all,index_list_all,site_freq_users_all,start_index

        # Unique sites and them counts
        unique_counts = user_data.groupby(['site'], sort=False).size()
        unique_site = np.array(unique_counts.index)
        #unique_counts = np.array(unique_counts)

        # Intersect of unique sites (indexes) for each iteration 
        common_site = intersect1d_set(unique_site, unique_site_all)
        index_com_site = np.where(np.in1d(unique_site, common_site))[0]
    
        # Difference of unique sites (indexes) for each iteration  
        not_common_site = setdiff1d_modify(unique_site, unique_site_all)
        index_not_common_site = np.where(np.in1d(unique_site, not_common_site))[0]
    
        # Store to global
        unique_site_all = unique_site_all + not_common_site.tolist()
        lenght_unique = not_common_site.shape[0]
    
        # Index of unique sites for each loop and store to global index value
        index_list = np.arange(start_index + 1, start_index + lenght_unique + 1)
        start_index = start_index + lenght_unique
        index_list_all = index_list_all + index_list.tolist() 
    
        # Unique elements for each iteration
        count_index = map(lambda x: x, index_list)
        counts_sites = map(lambda x: unique_counts[x], index_not_common_site)
        freq_site =  zip(count_index, counts_sites)
    
        # Store to dictionary
        site_freq_users = dict(zip(unique_site[index_not_common_site].tolist(),freq_site))
        site_freq_users_all.update(site_freq_users)

        # Common unique elements for each iteration and store ti dictionary
        count_comon_index = [site_freq_users_all.get(i)[0] for i in common_site]
        counts_comon_sites = unique_counts[index_com_site].tolist()
        freq_common_site = zip(count_comon_index, counts_comon_sites)
        site_freq_common_users = dict(zip(common_site, freq_common_site))
        site_freq_users_all.update(dict_modify(site_freq_users_all, site_freq_common_users))
    
           
    # Import csv files 
    user_data = path_to_csv(path_to_csv_files) 

    # Dictionary algorytm
    #[vocabruary(data) for data in user_data]
    [*map(vocabruary,user_data)];
    # Replacing index and raplace dictionary
    replasment = dict(zip(unique_site_all, index_list_all))
            
    # Main alhortm of replacing str sites for numbers
    [*map(adding_algorytm, user_data)];

    #Replacing site_id in column and delete NANs
    for site in site_numeration:
        resultData_all[site] = resultData_all[site].map(replasment.get)
    resultData_all = resultData_all.fillna(0).astype(int)
    
    return resultData_all, site_freq_users_all


# **Примените полученную функцию к игрушечному примеру, убедитесь, что все работает как надо.**

# In[41]:


path_to_csv_files = '3users\*'
train_data_toy, site_freq_3users = prepare_train_set(path_to_csv_files, session_length=10)
train_data_toy


# Частоты сайтов (второй элемент кортежа) точно должны быть такими, нумерация может быть любой (первые элементы кортежей могут отличаться).

# In[42]:


site_freq_3users


# Примените полученную функцию к данным по 10 пользователям.
# 
# **<font color='red'> Вопрос 1. </font> Сколько уникальных сессий из 10 сайтов в выборке с 10 пользователями?**

# In[43]:


get_ipython().run_cell_magic('time', '', "\npath_to_csv_files = '10users\\*'\ntrain_data_10users, site_freq_10users = prepare_train_set(path_to_csv_files, session_length=10)\n\nall_sessions_10users = train_data_10users.loc[:, 'site1':'site10']\nprint(f'All sessions: {train_data_10users.shape[0]}')\nprint(f'Unique sessions: {all_sessions_10users.drop_duplicates().shape[0]}')")


# **<font color='red'> Вопрос 2. </font> Сколько всего уникальных сайтов в выборке из 10 пользователей? **

# In[44]:


print(f'Unique sites: {len(site_freq_10users)}')


# Примените полученную функцию к данным по 150 пользователям.
# 
# **<font color='red'> Вопрос 3. </font> Сколько уникальных сессий из 10 сайтов в выборке с 150 пользователями?**

# In[67]:


get_ipython().run_cell_magic('time', '', "path_to_csv_files = '150users\\*'\ntrain_data_150users, site_freq_150users = prepare_train_set(path_to_csv_files, session_length=10)")


# In[68]:


all_sessions_150users = train_data_150users.loc[:, 'site1':'site10']
print(f'All sessions: {train_data_150users.shape[0]}')
print(f'Unique sessions: {all_sessions_150users.drop_duplicates().shape[0]}')


# **<font color='red'> Вопрос 4. </font> Сколько всего уникальных сайтов в выборке из 150 пользователей? **

# In[69]:


print(f'Unique sites: {len(site_freq_150users)}')


# **<font color='red'> Вопрос 5. </font> Какой из этих сайтов НЕ входит в топ-10 самых популярных сайтов среди посещенных 150 пользователями?**
# - www.google.fr
# - www.youtube.com
# - safebrowsing-cache.google.com
# - www.linkedin.com

# In[70]:


site_test_top = ['www.google.fr','www.youtube.com','safebrowsing-cache.google.com','www.linkedin.com']
# Dictionary without index (value [0])
site_freq_150users_index_less = {key: site_freq_150users[key][1] for key in site_freq_150users}
# Top10 and site visit count
top10_site_freq_150users =  sorted(site_freq_150users_index_less.items(), key=lambda v: v[1], reverse=True)[:10]
top10 = list(map(lambda x: x[0], top10_site_freq_150users))
# Test
print([site in top10 for site in site_test_top])
print(f'\n top 10: \n {top10}')
print(f'\n These sites: \n\n {site_test_top} \n\n are NOT in the top 10 most popular sites')


# **Для дальнейшего анализа запишем полученные объекты DataFrame в csv-файлы.**

# In[16]:


train_data_10users.to_csv(os.path.join(PATH_TO_DATA, 
                                       'train_data_10users.csv'), 
                        index_label='session_id', float_format='%d')
train_data_150users.to_csv(os.path.join(PATH_TO_DATA, 
                                        'train_data_150users.csv'), 
                         index_label='session_id', float_format='%d')


# ## Часть 2. Работа с разреженным форматом данных

# Если так подумать, то полученные признаки *site1*, ..., *site10* смысла не имеют как признаки в задаче классификации. А вот если воспользоваться идеей мешка слов из анализа текстов – это другое дело. Создадим новые матрицы, в которых строкам будут соответствовать сессии из 10 сайтов, а столбцам – индексы сайтов. На пересечении строки $i$ и столбца $j$ будет стоять число $n_{ij}$ – cколько раз сайт $j$ встретился в сессии номер $i$. Делать это будем с помощью разреженных матриц Scipy – [csr_matrix](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html). Прочитайте документацию, разберитесь, как использовать разреженные матрицы и создайте такие матрицы для наших данных. Сначала проверьте на игрушечном примере, затем примените для 10 и 150 пользователей. 
# 
# Обратите внимание, что в коротких сессиях, меньше 10 сайтов, у нас остались нули, так что первый признак (сколько раз попался 0) по смыслу отличен от остальных (сколько раз попался сайт с индексом $i$). Поэтому первый столбец разреженной матрицы надо будет удалить. 

# In[71]:


X_toy, y_toy = train_data_toy.iloc[:, :-1].values, train_data_toy.iloc[:, -1].values
X_toy


# In[148]:


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


def sparse_csr_0(df, dic_freq):
    '''
    make array for sparse matrix
    df - input dataframe
    dic_freq - dictionary of frequency entranceis
    output - pandas dataframe for csr_matrix
    '''
    col_ind = df.index
    row_ind = [v[0] for k, v in dic_freq.items()]

    dataFrame_ID_less = df.loc[:, :df.columns[-2]]
    #dataFrame_ID_less = df
    dataFrame_sites = pd.DataFrame(index=df.index, columns=row_ind)

    for i in col_ind:
        dataFrame_sites.iloc[i] = dataFrame_ID_less.iloc[i].value_counts()
    dataFrame_sites.fillna(0, inplace=True)
    dataFrame_sites = dataFrame_sites.astype(int)
    
    return dataFrame_sites.values

def sparse_csr_1(df):
    '''
    make data, indices, indptr for sparse matrix
    df - input dataframe
    '''
    indices = pd.Series()
    indptr = [0]
    df = df.loc[:, :df.columns[-2]]
    
    for index in df.index:
        indices = indices.append(df.iloc[index].replace(0, np.nan).value_counts().sort_index())
        indptr.append(len(indices))
    
    data = indices.values.tolist()
    indices = list( map(lambda x: x-1, indices.index) )
    #indptr = indptr[:len(rowptr) - 1]
    
    return data, indices, indptr

def sparse_csr_2(array2D):
    '''
    make data, indices, indptr for sparse matrix
    array2D - input array
    '''
    data = []
    indices = []
    indptr = [0]

    for array in array2D:
        unique, counts = np.unique(array[array != 0], return_counts=True)
        indices = indices + unique.tolist()
        data = data + counts.tolist()
        indptr.append(indptr[-1] + len(unique)) 

    indices = [*map(lambda x: x-1, indices)]
    
    return data, indices, indptr


# In[149]:


# First function -> very slow
X_sparse_toy0 = csr_matrix(sparse_csr_0(train_data_toy, site_freq_3users))

# Second functon -> slow
data1, indices1, indptr1 = sparse_csr_1(train_data_toy)
X_sparse_toy1 = csr_matrix((data1, indices1, indptr1), dtype=int)

# Third functon -> normal
data2, indices2, indptr2 = sparse_csr_2(X_toy)
X_sparse_toy = csr_matrix((data2, indices2, indptr2), dtype=int)

# Main function -> fast 
data, indices, indptr = sparse_csr(X_toy)
X_sparse_toy = csr_matrix((data, indices, indptr), dtype=int)
X_sparse_toy.todense()


# In[ ]:





# **Размерность разреженной матрицы должна получиться равной 11, поскольку в игрушечном примере 3 пользователя посетили 11 уникальных сайтов.**

# In[150]:


X_10users, y_10users = train_data_10users.iloc[:, :-1].values,                        train_data_10users.iloc[:, -1].values
X_150users, y_150users = train_data_150users.iloc[:, :-1].values,                          train_data_150users.iloc[:, -1].values


# In[55]:


data_10users, indices_10users, indptr_10users = sparse_csr(X_10users)
X_sparse_10users = csr_matrix((data_10users, indices_10users, indptr_10users), dtype=int)


# In[56]:


data_150users, indices_150users, indptr_150users = sparse_csr(X_150users)
X_sparse_150users = csr_matrix((data_150users, indices_150users, indptr_150users), dtype=int)


# **Сохраним эти разреженные матрицы с помощью [pickle](https://docs.python.org/2/library/pickle.html) (сериализация в Python), также сохраним вектора *y_10users, y_150users* – целевые значения (id пользователя)  в выборках из 10 и 150 пользователей. То что названия этих матриц начинаются с X и y, намекает на то, что на этих данных мы будем проверять первые модели классификации.
# Наконец, сохраним также и частотные словари сайтов для 3, 10 и 150 пользователей.**

# In[22]:


with open('X_sparse_10users.pkl', 'wb') as X10_pkl:
    pickle.dump(X_sparse_10users, X10_pkl, protocol=2)
with open('y_10users.pkl', 'wb') as y10_pkl:
    pickle.dump(y_10users, y10_pkl, protocol=2)

with open('X_sparse_150users.pkl', 'wb') as X150_pkl:
    pickle.dump(X_sparse_150users, X150_pkl, protocol=2)
with open('y_150users.pkl', 'wb') as y150_pkl:
    pickle.dump(y_150users, y150_pkl, protocol=2)

with open('site_freq_3users.pkl', 'wb') as site_freq_3users_pkl:
    pickle.dump(site_freq_3users, site_freq_3users_pkl, protocol=2)
with open('site_freq_10users.pkl', 'wb') as site_freq_10users_pkl:
    pickle.dump(site_freq_10users, site_freq_10users_pkl, protocol=2)
with open('site_freq_150users.pkl', 'wb') as site_freq_150users_pkl:
    pickle.dump(site_freq_150users, site_freq_150users_pkl, protocol=2)


# **Чисто для подстраховки проверим, что число столбцов в разреженных матрицах `X_sparse_10users` и `X_sparse_150users` равно ранее посчитанным числам уникальных сайтов для 10 и 150 пользователей соответственно.**

# In[23]:


assert X_sparse_10users.shape[1] == len(site_freq_10users)


# In[21]:


assert X_sparse_150users.shape[1] == len(site_freq_150users)


# In[ ]:




