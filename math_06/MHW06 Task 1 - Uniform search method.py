#!/usr/bin/env python
# coding: utf-8

# # MHW06 Task 1 - Uniform search method

# In[1]:


import numpy as np
import sympy as sp


# In[2]:


# Input function for : f(x) = x**2 - 12*x

# Input symbols and symbolyc funtion for X variable
x = sp.symbols('x')
symbol_function = x**2 - 12*x
  
# Input parameters
# Itarations
max_iterations = 100
# First aproximation
x_interval = np.array([0,25])


# In[13]:


# Functions

def x_few_intervals(x_interval, max_iterations):
    '''x_interval of X on intervals
    max_iterations - number of subintervals
    '''
    return x_interval[0] - range(1,max_iterations) * (x_interval[1] - x_interval[0]) / (max_iterations + 1)

def f_numerical(argument):
    '''
    substitute numerical argument in symbolic for function
    '''
    global symbol_function
    substitutions = [(x,argument)]
    return symbol_function.subs(substitutions)

# Vectorize for array
f_numerical_vector = np.vectorize(f_numerical)

# alternative
# def f_numerical(function, argument):
#     '''
#     substitute numerical argument in symbolic for function
#     '''
#     substitutions = [(x,argument)]
#     return function.subs(substitutions)


# In[59]:


# Main alhorytm

x_argument = x_few_intervals(x_interval, max_iterations)
f_argument = f_numerical_vector(x_argument)


# unnpatern alternative
#f_argument = [f_numerical(symbol_function, i) for i in x_argument]

# Answer
print(f'\n Algorithm convergence point = {2/(max_iterations + 1):.5f}')
print(f'\n minimum f(x) = {min(f_argument):.5f} \n x = {x_argument[f_argument.tolist().index(min(f_argument))]:.5f}')


# In[ ]:




