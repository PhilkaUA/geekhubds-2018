#!/usr/bin/env python
# coding: utf-8

# # HW07 Task 1 - Pandas<br></center>

# **В задании предлагается с помощью Pandas ответить на несколько вопросов по данным репозитория UCI [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) (качать данные не надо – они уже есть в репозитории).**

# Уникальные значения признаков (больше информации по ссылке выше):
# - age: continuous.
# - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - fnlwgt: continuous.
# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# - education-num: continuous.
# - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - sex: Female, Male.
# - capital-gain: continuous.
# - capital-loss: continuous.
# - hours-per-week: continuous.
# - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.   
# - salary: >50K,<=50K

# In[80]:


import numpy as np
import pandas as pd
import timeit


# In[2]:


data = pd.read_csv('adult.data.csv')
#data.head()


# **1. Сколько мужчин и женщин (признак *sex*) представлено в этом наборе данных?**

# In[5]:


# method whith .size more faster
#sex_count = [print(sex,': ',data['sex'][data['sex'] == sex].value_counts(dropna=False)[0]) for sex in ["Male","Female"]]
sex_list = ["Male","Female"]
sex_count = [print(sex,': ',data['sex'][data['sex'] == sex].size) for sex in sex_list]


# **2. Каков средний возраст (признак *age*) женщин?**

# In[6]:


averege_age = data[['sex','age']][['age']].mean()[0]
print(f'Averge female age = {averege_age:.5}')


# **3. Какова доля граждан Германии (признак *native-country*)?**

# In[7]:


german_count = data['native-country'][data['native-country'] == 'Germany'].size
data_count = data['native-country'].size
print(f'German people: {(german_count/data_count)*100:.3} %')


# **4-5. Каковы средние значения и среднеквадратичные отклонения возраста тех, кто получает более 50K в год (признак *salary*) и тех, кто получает менее 50K в год? **

# In[10]:


salary = ["<=50K",">50K"]
mean_salary = [round(data[['age','salary']][data['salary'] == mean_sal]['age'].mean(),3) for mean_sal in salary]
std_salary = [round(data[['age','salary']][data['salary'] == std_sal]['age'].std(),3) for std_sal in salary]

# printing results
[print('Salary: ', salary,' maen:', mean_salary[index],' std:', std_salary[index]) for index, salary in enumerate(salary)];


# **6. Правда ли, что люди, которые получают больше 50k, имеют как минимум высшее образование? (признак *education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters* или *Doctorate*)**

# In[76]:


'''
hs_grad: target atributes
data_grad: switch 2-columns by atributes
hs_grad_status: check for target atributes in columns
'''
hs_grad = ['Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']
data_grad = data[['education','salary']][data['salary'] == '>50K']['education'].values
hs_grad_status = [(element in hs_grad) for element in data_grad]

print('Answer for question is: ', all(hs_grad_status))


# In[83]:


[print('\n sex:', sex_list[i],'\n', data_sex_age[i]) for i in range(2)];


# **7. Выведите статистику возраста для каждой расы (признак *race*) и каждого пола. Используйте *groupby* и *describe*. Найдите таким образом максимальный возраст мужчин расы *Amer-Indian-Eskimo*.**

# In[84]:


# Grouping by race and age
data_sex_age = [data[data['sex'] == sex].groupby(['race'])['age'].describe().loc['Amer-Indian-Eskimo'] for sex in sex_list]

# Statistic by sex
[print('\n sex:', sex_list[index],'\n', data_sex_age[index]) for index in range(2)];

# Max age by sex
[print('\n sex:', sex_list[index],'  Max age:', data_sex_age[index][-1]) for index in range(2)];


# **8. Среди кого больше доля зарабатывающих много (>50K): среди женатых или холостых мужчин (признак *marital-status*)? Женатыми считаем тех, у кого *marital-status* начинается с *Married* (Married-civ-spouse, Married-spouse-absent или Married-AF-spouse), остальных считаем холостыми.**

# In[9]:


data_merried = data[['marital-status','salary']][(data['marital-status']    .apply(lambda marital: marital[0:7] == 'Married')) & (data['salary'] == '>50K')].size

data_not_merried = data[['marital-status','salary']][(data['marital-status']    .apply(lambda marital: marital[0:7] != 'Married')) & (data['salary'] == '>50K')].size

print(f'merried: {data_merried}  not merried: {data_not_merried}')


# **9. Какое максимальное число часов человек работает в неделю (признак *hours-per-week*)? Сколько людей работают такое количество часов и каков среди них процент зарабатывающих много?**

# In[85]:


'''
max_work_time: max hour for a week
human_max_work_time: quantity of humans for max hour for a week
'''
max_work_time = data[['hours-per-week']].max().loc['hours-per-week']
human_max_work_time = data[['hours-per-week','salary']][(data['salary'] == '>50K') & (data['hours-per-week'] == max_work_time)]

print(f'Max work hours: {max_work_time} for {human_max_work_time.size} humans')


# **10. Посчитайте среднее время работы (*hours-per-week*) зарабатывающих мало и много (*salary*) для каждой страны (*native-country*).**

# In[100]:


print(data.pivot_table(columns=['salary'], index=['native-country'], values='hours-per-week', aggfunc='mean'))

# #alternative code 1
# salary = ["<=50K",">50K"]
# avarge_work_time_country = [data[['native-country','hours-per-week']][data['salary'] == salary_value]\
#     .pivot_table(['hours-per-week'], ['native-country'],aggfunc='mean') for salary_value in salary]
# # printing results
# [print('\n', time_value) for time_value in avarge_work_time_country];

# #alternative code 2
# print('Average hours per week for people this small salary <=50K')
# print(data[(data['salary']=='<=50K')].groupby(['native-country'])[ 'hours-per-week'].mean())
# print()
# print('Average hours per week for people this high salary >50K')
# print(data[(data['salary']=='>50K')].groupby(['native-country'])[ 'hours-per-week'].mean())


# In[ ]:





# In[12]:


# Convert to PY
# get_ipython().system('jupyter nbconvert --to script topic2_habr_pandas.ipynb')


# In[13]:


# Time estimate

# start_time = timeit.default_timer()
# # code
# elapsed = timeit.default_timer() - start_time
# print(elapsed)

