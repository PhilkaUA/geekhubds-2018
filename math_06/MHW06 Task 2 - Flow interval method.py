#!/usr/bin/env python
# coding: utf-8

# # MHW06 Task 2 - Flow interval method

# In[3]:


import numpy as np
import sympy as sp


# In[ ]:





# In[38]:


# Input function for : f(x) = x**2 - 12*x

# Input symbols and symbolyc funtion for X variable
x = sp.symbols('x')
symbol_function = x**2 - 12*x
  
# Input parameters

# Accuracy of method
accuracy = 0.05
max_iterations = 1000


# In[39]:


# functions
# [-1] need to search last elements of iterations tracking alhorytm

def x_pnt(left_limit, right_limit, initial):
    '''
    half of interval for X values
    left_limit - left limit of interval
    right_limit - right limit of interval
    initial - initial value of half interval, also store all evaluated values
    '''
    half_interval = (left_limit[-1] + right_limit[-1])/2
    return np.vstack((initial, half_interval)).astype(float)

def y_pnt(left_limit, right_limit, initial):
    '''
    left interval evaluating
    left_limit - left limit of interval
    right_limit - right limit of interval
    initial - initial value of left limit value, also store all evaluated values
    '''
    interval_len = (right_limit[-1] - left_limit[-1])
    return np.vstack((initial, left_limit[-1] + 0.25 * abs(interval_len))).astype(float)

def z_pnt(left_limit, right_limit, initial):
    '''
    right interval evaluating
    left_limit - left limit of interval
    right_limit - right limit of interval
    initial - initial value of left limit value, also store all evaluated values
    '''
    interval_len = (right_limit[-1] - left_limit[-1])
    return np.vstack((initial, right_limit[-1] - 0.25 * abs(interval_len))).astype(float)

def f_num(argument):
    '''
    substitute numerical argument in symbolic for function
    '''
    global symbol_function
    substitutions = [(x,argument)]
    return symbol_function.subs(substitutions)

def ab_pnt(value, initial):
    '''
    initial - stacks values of left/right limit
    '''
    return np.vstack((value, initial)).astype(float)


def test_accuracy(a_value, b_value, accuracy):
    '''
    Testing accuracy by the difference 
    between the two values of arguments
    
    a_value, b_value - values of argument X interval limit
    accuracy - accuracy value
          
    result: 
    [0] is boolen test accuracy
    [1] is a arguments difference |a - b|
    '''
    difference = abs(a_value - b_value)
    return (difference < accuracy), difference 


# In[43]:


# Main alhorytm

# Initial values
# X value interval
a_init = np.array([0]) 
b_init = np.array([25])
# Inital values
x_init  = np.array([0])
y_init = np.array([0])
z_init = np.array([0])


for iterations in range(max_iterations): 
    
    # this need to white all stages of iterations
    x_init = x_pnt(a_init, b_init, x_init)
    y_init = y_pnt(a_init, b_init, y_init)
    z_init = z_pnt(a_init, b_init, z_init)
    
    # Accuracy test
    if not test_accuracy(a_init[-1], b_init[-1], accuracy)[0]:      

        #Interval selection conditions for the algorithm
        # left interval
        if f_num(y_pnt(a_init, b_init, y_init)[-1]) < f_num(x_pnt(a_init, b_init, x_init)[-1]):
            b_init = ab_pnt(b_init, x_pnt(a_init, b_init, x_init)[-1])
            a_init = ab_pnt(a_init, a_init[-1])
            
        elif f_num(y_pnt(a_init, b_init, y_init)[-1]) >= f_num(x_pnt(a_init, b_init, x_init)[-1]):
            
            # Right intervas
            if f_num(y_pnt(a_init, b_init, y_init)[-1]) < f_num(x_pnt(a_init, b_init, x_init)[-1]):
                a_init = ab_pnt(a_init, x_pnt(a_init, b_init, x_init)[-1])
                b_init = ab_pnt(b_init, b_init[-1])
            
            # Middle interval           
            elif f_num(z_pnt(a_init, b_init, z_init)[-1]) >= f_num(x_pnt(a_init, b_init, x_init)[-1]):
                a_init = ab_pnt(a_init, y_pnt(a_init, b_init, y_init)[-1])
                b_init = ab_pnt(b_init, z_pnt(a_init, b_init, z_init)[-1])

        # output in iterations
#         print(f'\n iter: {iterations}') 
#         print(f' acc: {test_accuracy(a_init[-1], b_init[-1], accuracy)[1][0]:.3f}')        
#         print(f' a =  {a_init[-1][0]:.3f}')
#         print(f' b =  {b_init[-1][0]:.3f}')
    else: break

else: print('Make more iterations')
        
print(f' \n Answer:\n \n x = {x_init[-1][0]:.3f} \n \n a = {a_init[-1][0]:.3f} \n b = {b_init[-1][0]:.3f}        \n accuracy = {test_accuracy(a_init[-1], b_init[-1], accuracy)[1][0]:.3f}        \n Alg. convergence = {1/2**(iterations/2):.3f} \n iterations = {iterations}')


# In[ ]:




