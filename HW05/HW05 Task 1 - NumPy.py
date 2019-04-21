#!/usr/bin/env python
# coding: utf-8

# # HW05 Task 1 - NumPy
# 1. Извлечь колонку &#39;species&#39;
# 2. Преобразовать первые 4 колонки в 2D массив
# 3. Посчитать mean, median, standard deviation по 1-й колонке
# 4. Вставить 20 значений np.nan на случайные позиции в массиве
# 5. Найти позиции вставленных значений np.nan в 1-й колонке
# 6. Отфильтровать массив по условию значения в 3-й колонке &gt; 1.5 И значения
# в 1-й колонке &lt; 5.0
# 7. Заменить все значения np.nan на 0
# 8. Посчитать все уникальные значения в массиве и вывести их вместе с
# посчитанным количеством
# 9. Разбить массив по горизонтали на 2 массива
# 10. Отсортировать оба получившихся массива по 1-й колонке: 1-й по
# возрастанию, 2-й по убыванию
# 11. Склеить оба массива обратно
# 12. Найти наиболее часто повторяющееся значение в массиве
# 13. Написать функцию, которая бы умножала все значения в колонке, меньше
# среднего значения в этой колонке, на 2, и делила остальные значения на 4.
# применить к 3-й колонке

# In[1]:


import numpy as np

# import dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


# In[2]:


#1 Extract the column 'species'

# Index finding
idex_species = names.index('species')

# Column
column_species = iris[:,idex_species]
#print(column_species)


# In[20]:


#2 Convert the first 4 columns to a 2D array

# Slicing for 4 columns
sliced_array = iris[::, :4]

# Test for dismentions
print('this array is ',sliced_array.ndim,'D')

# reshaping for array (xxx,2) only for example
array2D = sliced_array.reshape(300,2)


# In[4]:


#3 Calculate mean, median, standard deviation on the 1st column

# First column
column_first = iris[:,0]

# Change type
column_tf_type = column_first.astype('str').astype('float')

# Mean, Median,  Square
print('\n Mean: ',np.mean(column_tf_type))
print(' Median: ',np.median(column_tf_type))
print(' St.Deviation: ',np.std(column_tf_type))


# In[5]:


#4 Insert 20 values of np.nan to random positions in the array

# Array copy
simple_array = iris.copy()

# Array length
column_length = simple_array.shape[0]
row_length = simple_array.shape[1]

# Sub of random indexs
for i in range(20):
    simple_array[np.random.randint(column_length),np.random.randint(row_length)] = np.nan
#print(simple_array)


# In[6]:


#5 Find the position of the inserted values np.nan in the 1st column

# First column
first_column = simple_array[:,0].astype(float)

# Indexies of NAN value
index_nan = np.argwhere(np.isnan(first_column)).flatten()
print(index_nan)


# In[7]:


#6 Filter an array by the condition of the value in the 3rd column > 1.5 And the values in the 1st column <5.0

#Array copy and columns
array_iris = iris.copy()

third_column = array_iris[:,2].astype('str').astype('float')
first_column = array_iris[:,0].astype('str').astype('float')

# Condition fo filter
index_first_column = np.where(first_column < 5.0)
index_third_column = np.where(third_column > 1.5)

# Join arrays
first_column_sorted = first_column.take(index_first_column)
third_column_sorted = third_column.take(index_third_column)

print(np.concatenate((first_column_sorted,third_column_sorted),axis=1))


# In[8]:


#7 Replace all np.nan values with 0

# Slice for 4 columns
nan_indexes = np.isnan(simple_array[::,:4].astype(float))
# Replace NAN elements for 0
simple_array[::,:4][nan_indexes] = 0
#print(simple_array)


# In[9]:


#8 Count all unique values in the array and list them together with the counted number

# Get the unique items and their counts
unique_elements, unique_counts = np.unique(iris, return_counts=True)

print("Unique elements : ", unique_elements)
print("Counts unique elements: ", unique_counts)


# In[10]:


#9 Split an array horizontally into 2 arrays

# Array copy
for_split_array = iris.copy()

# Array length
split_length = int(for_split_array.shape[1] - 1)
split_length_columns = for_split_array.shape[0]

# Splitting matrix
left_aray, right_aray = np.hsplit(for_split_array[::,:split_length], 2)

# Turning last columns in for to vstack (list for each element)
last_columns = np.array([[for_split_array[:,split_length][columns]] for columns in np.arange(split_length_columns)])

# Add last column
right_aray = np.concatenate((right_aray, last_columns),axis=1)

# print(' right column: \n',right_aray)
# print('\n left column: \n',left_aray)


# In[11]:


#10 Sort both resulting arrays by the 1st column: 1st ascending, 2nd descending

# Sorting in columns
sorted_left_aray = left_aray[left_aray[:, 0].argsort()]
sorted_right_aray = right_aray[right_aray[:, 0].argsort()[::-1]]

#11 Merge both arrays back

left_right_array = np.concatenate((sorted_left_aray, sorted_right_aray),axis=1)
#print(left_right_array)


# In[12]:


#12 Find the most frequently repeated value in the array

# Turning last columns in for to vstack (list for each element)

# # Forming 2D-array listed elements
# length_uniqie = unique_elements.shape[0]

# unique_counts_listed = np.array([[unique_counts[columns]] for columns in np.arange(length_uniqie)])
# unique_elements_listed = np.array([[unique_elements[columns]] for columns in np.arange(length_uniqie)])

# counts_elements = np.concatenate((unique_counts_listed, unique_elements_listed),axis=1)

# Forming 2D-array
counts_elements = np.vstack((unique_counts, unique_elements)).T

# Sorting inverse
sorted_counts_elements = counts_elements[counts_elements[:, 0].argsort()[::-1]]

# Most frequently repeated value
print('\n Most frequently repeated value: ',sorted_counts_elements[0,1], '  -->> times:',sorted_counts_elements[0,0],'\n')

for i in np.arange(5):
    if sorted_counts_elements[i,0] == sorted_counts_elements[i+1,0]:
        print(' Aloso frequently repeated value: ',sorted_counts_elements[i+1,1], '  -->> times:',sorted_counts_elements[i+1,0])


# In[13]:


#13 Write a function that would multiply all values in the column, less than the average value in this column, by 2,
#   and divide the remaining values by 4. apply to the 3rd column

# For Column
def dividing_function_for_column(column):
    '''
    dividing elevemnts of 1 column of array
    column_ : copy of column
    column_transform : transformin typy of column
    column_mean : mean for column elements
    '''
    column_ = column.copy()
    column_transform = column_.astype('float')
    column_mean = column_transform.mean()
    return np.where(column_transform < column_mean, column_transform/2, column_transform/4)

dividing_function_for_column(iris[:,3])

# For Array
# def dividing_function(array, column):
    
#     #Array copy
#     array_ = array.copy()
    
#     # Column
#     array_column = array_[:,column].astype('float')
    
#     #Mean
#     array_column_mean = array_column.mean()
    
#     return np.where(array_column < array_column_mean, array_column/2, array_column/4)

# dividing_function(iris, 3)


# In[ ]:




