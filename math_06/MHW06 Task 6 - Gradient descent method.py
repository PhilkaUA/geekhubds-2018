#!/usr/bin/env python
# coding: utf-8

# # MHW06 Task 6 - Gradient descent method
# 
# #### Method of gradient descent with a constant step.

# In[1]:


import numpy as np
import sympy as sp


# In[ ]:





# In[2]:


# Input function : f(x) = x1**3-x1*x2+x2**2-2*x1+3*x2-4

# Input symbols and symbolyc funtion for 2 variables
x1, x2 = sp.symbols('x1, x2')
symbol_function = x1**3 - x1*x2 + x2**2 -  2*x1 + 3*x2 - 4
  
# Input parameters
# Accuracy
e1 = 0.1
e2 = 0.15
# Itarations
max_iterations = 10
# First aproximations
x = np.array([[0,0],[0,0]]) # duplicate first element need for iterations
# Step for alhorytm
step = 0.5


# In[3]:


# Functions

def f_numerical(function, argument1, argument2):
    '''
    substitute numerical argument in symbolic for function
    '''
    substitutions = [(x1,argument1),(x2,argument2)]
    return function.subs(substitutions)

def f_numerical_array(array, argument1, argument2):
    '''
    substitute numerical argument in symbolic for functions in array
    substitutions - substitutions for symbolic arguments
    first_element, second_element - elements of numerical function 
    '''
    substitutions = [(x1,argument1),(x2,argument2)]
    first_element = array[0].subs(substitutions)
    second_element = array[1].subs(substitutions)
    return np.array([first_element,second_element])

def diff_function(function, argument):
    '''Differentiation for symbolic functions'''
    return sp.diff(function, argument)

def grad_function(function, argument1, argument2):
    '''Gradient for symbolic functions'''
    fst_element = diff_function(symbol_function, argument1)
    scd_element = diff_function(symbol_function, argument2)
    return np.array([fst_element, scd_element])

def array_norm(array):
    '''Euclid norm for array'''
    return np.linalg.norm(array.astype(float))

def x_new(array, step):
    '''New value for X argument by the step
    array[-1] - last value of X argument
    numerical_gradient - numerical value of gradient in X argument
    step - step for algrytm
    '''
    numerical_gradient = f_numerical_array(grad_function(symbol_function,x1,x2),array[-1][0],array[-1][1])
    new_value = array[-1] - step * numerical_gradient
    return np.vstack((array,new_value)).astype(float)

def test_accuracy_1(array, accuracy):
    '''
    Testing accuracy by the norm of numerical gradient 
    
    array - values of argument X 
    accuracy - accuracy value
    numerical_gradient - numerical value of gradient in X argument (last)
    result: [0] is boolen, [1] is value of Euclid norm of numerical gradient
    '''    
    numerical_gradient = f_numerical_array(grad_function(symbol_function,x1,x2),array[-1][0],array[-1][1])
    norm_numerical_gradient = np.linalg.norm(numerical_gradient.astype(float))
    
    return norm_numerical_gradient < accuracy, norm_numerical_gradient

def test_accuracy_2(array, accuracy):
    '''
    Testing accuracy by the difference 
    between the two values of the argument and functions
    
    array - of argument X
    accuracy - accuracy value
    argument_norm - Euclid norm of argument X values
    new_function_value, old_function_value - function value for argument X values
       
    result: 
    [0] is boolen test accuracy
    [1] is value of Euclid norm of ||x(i+1) - x(i)||
    [2] is a function difference |f(x(i+1)) - f(x(i))|
    '''
    argument_norm = np.linalg.norm(array[-2:])
    new_function_value = f_numerical(symbol_function, array[-1][0], array[-1][1])
    old_function_value = f_numerical(symbol_function, array[-2][0], array[-2][1])
    function_difference = abs(new_function_value - old_function_value)
    test_accuracy = (argument_norm < accuracy) and (function_difference < accuracy) 
    return test_accuracy, argument_norm, function_difference 

def step_iteration(step, array):
    '''New value for step = step / 2 if accuracy is low'''
    next_step = step
    if not test_accuracy_2(x_new(array, next_step), e2)[0]:
        new_step = next_step/2
    return next_step


# In[4]:


# Main alhorytm

x_evaluate = x 
step_evaluate = step
for i in range(max_iterations): 
    if not test_accuracy_1(x_evaluate, e1)[0]:
        x_evaluate = x_new(x_evaluate, step_evaluate)
        step_evaluate = step_iteration(step_evaluate, x_evaluate)

        #print resukt  for each iteration       
#         print('\n\n iteration = ', i)
#         print(' ||grad(x)|| = ', test_accuracy_1(x_evaluate, e1)[1])
#         print(' ||x(i+1) - x(i)|| = ', test_accuracy_2(x_evaluate, e2)[1])
#         print(' |f(x(i+1)) - f(x(i))| = ', test_accuracy_2(x_evaluate, e2)[2])
#         print(' step = ', step_evaluate)
#         print('\n x1 = ', x_evaluate[-1][0],'\n x2 = ', x_evaluate[-1][1])
    
    else: break
else: print('Make more iterations')
        
print(' \n Answer:',' \n x1 = ',x_evaluate[-1][0],'\n x2 = ',x_evaluate[-1][1],
      ' \n accuracy = ', test_accuracy_1(x_evaluate, e1)[1],' \n iterations = ', i)


# In[ ]:




