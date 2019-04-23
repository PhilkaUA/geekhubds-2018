#!/usr/bin/env python
# coding: utf-8

# # MHW06 Task 3 - Nelder-Mead method

# In[1]:


import numpy as np
import sympy as sp


# In[ ]:





# In[2]:


# Input function for : f(x) = x**2 - 12*x

# Input symbols and symbolyc funtion for X variable
x1, x2 = sp.symbols('x1, x2')
symbol_function = 4*(x1 - 5)**2 + (x2 - 6)**2
  
# Input parameters

# Triangle points
x_values = np.array([[8,9],[10,11],[8,11]])
#x = np.array([[8,9],[10,11],[8,11]])

# Trianglechange options
alpha = 1
betta = 0.5
gamma = 2

# Accuracy of method
accuracy = 0.1
max_iterations = 100


# In[3]:


# functions

def f_num_s(argument):
    '''substitute 1 numerical arguments in symbolic for functions'''
    global symbol_function
    return np.array(symbol_function.subs([(x1,argument[0]),(x2,argument[1])]), dtype=float)

def f_num(arguments):
    '''
    its for array
    substitute 3 numerical arguments in symbolic for functions
    elements - elements of numerical function 
    '''
    global symbol_function
    elements = [symbol_function.subs([(x1,arguments[index][0]),(x2,arguments[index][1])]) for index in np.arange(3)]
    return np.array([[elements[0]], [elements[1]], [elements[2]]], dtype=float)

def x_f_num_sort(values):
    '''
    sorting 3 numerical fucntion values by decending
    values - 3 triangle points
    add_function_value - concatenating fuction values to arguments array
    sort_function - sorting by function value
    result
    [0] - sorted arguments
    [1] - arument x(low)
    [2] - arument x(middle)
    [3] - arument x(high)  
    [4] - index of sorted elements in input array - values
    '''
    add_function = np.concatenate((values,f_num(values)),axis=1)
    index_sorting = add_function[:, 2].argsort()
    sort_function = add_function[index_sorting]
    
    return values, sort_function[:,:2][0], sort_function[:,:2][1], sort_function[:,:2][2], index_sorting

def weight_factor(x_f_num_sort):
    '''Weight factor of triangles points except the worst'''
    return 0.5 * (x_f_num_sort[1] + x_f_num_sort[2])

def test_accuracy(x_f_num_sort, weight_factor, accuracy):
    '''
    x_f_num_sort - sorted X-value
    weight_factor - weight factor by sorted X-value
    
    accuracy - accuracy of algorytm
    quadratic_form - quadratic form like (x[i] - x(mean))**2
    sum_form - sum elements of quadratic form 
    f_num - sum by array
    f_num_s - sum by 1 point of traingle
    
    result
    [0] is boolen test accuracy
    [1] is accuracy
    '''
    sum_func = (f_num(x_f_num_sort[0]) - f_num_s(weight_factor))**2
    result = np.sqrt(1/3 * np.sum(sum_func))
    return result < accuracy, result


# In[4]:


# Operations with triangle 

def reflection(x_f_num_sort, weight_factor):
    '''Operation of reflecting the worst triangle point through the center'''
    global alpha
    return weight_factor + alpha * (weight_factor - x_f_num_sort[3])
    
def stretching(reflection, weight_factor):
    '''Stretching operation'''
    global gamma
    return weight_factor + gamma * (reflection - weight_factor)

def compression(x_f_num_sort, weight_factor):
    '''Сompression peration'''
    global betta
    return weight_factor + betta * (x_f_num_sort[3] - weight_factor)

def reducion(f_num_sort):
    '''Reducion operation - new triangle formed with halved sides and points'''
    return f_num_sort[1] + 0.5 * (f_num_sort[0] - f_num_sort[1])


# In[38]:


# Main alhorytm

# Variable to tracking answer on itteration
x_values = np.array([[8,9],[10,11],[8,11]])
x = x_values

print('\n input points \n',x_values, '\n')


for iteration in range(10):
    
    print('\n Iteration: ',iteration + 1)

    # Accuracy test
    if not test_accuracy(x_f_num_sort(x), weight_factor(x_f_num_sort(x)), accuracy)[0]:
        
        print(f' Accuracy: {test_accuracy(x_f_num_sort(x), weight_factor(x_f_num_sort(x)), accuracy)[1]:.3f}')        
               
        # Reflection    
        x_n3 = reflection(x_f_num_sort(x), weight_factor(x_f_num_sort(x)))
    
        if f_num_s(x_n3) < f_num_s(x_f_num_sort(x)[1]):
        
            # Stretching
            x_n4 = stretching(reflection(x_f_num_sort(x), weight_factor(x_f_num_sort(x))), weight_factor(x_f_num_sort(x)))
        
            if f_num_s(x_n4) < f_num_s(x_f_num_sort(x)[1]):
                x[x_f_num_sort(x)[4][2]] = x_n4 # changing input array
                print(' Stretching: x_n4 ->> x(high)\n',x)
            elif f_num_s(x_n4) >= f_num_s(x_f_num_sort(x)[1]):
                x[x_f_num_sort(x)[4][2]] = x_n3
                print(' Stretching: x_n3 ->> x(high)\n',x)
    
        elif f_num_s(x_f_num_sort(x)[2]) < f_num_s(x_n3) <= f_num_s(x_f_num_sort(x)[3]):
        
            # Compression
            x_n5 = compression(x_f_num_sort(x), weight_factor(x_f_num_sort(x)))
            x[x_f_num_sort(x)[4][2]] = x_n5
            print(' Compression: x_n5 ->> x(high)\n',x)
    
    
        elif f_num_s(x_f_num_sort(x)[1]) < f_num_s(x_n3) <= f_num_s(x_f_num_sort(x)[2]):
        
            # Dublicating
            print(' Nothing done \n',x)    

        elif f_num_s(x_n3) >= f_num_s(x_f_num_sort(x)[3]):
        
            # Reducion
            x = reducion(x_f_num_sort(x))
            print(' Reducion \n',x)
    
    else: break
            
else: print('Make more iterations') 

# Answer
x_answer = x_f_num_sort(x)[1]
f_answer = f_num_s(x_f_num_sort(x)[1])


print(f'\n\n Answer:\n\n x1 = {x_answer[0]} \n x2 = {x_answer[1]}, \n f(x) = {f_answer}    \n Accuracy = {test_accuracy(x_f_num_sort(x), weight_factor(x_f_num_sort(x)), accuracy)[1]:.3f}    \n Iteration: {iteration + 1}')      


# In[ ]:




