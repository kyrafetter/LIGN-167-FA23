## LIGN 167 PSET 1
## Kyra Fetter, A16819058
## Lina Battikha, A16852105
## Oishani Bandopadhyay, A1600827

#  GPT 3.5 

# 1

# 2 

# 3

# 4

# 5

import numpy as np

def f(xi, yi, a, b):
    return (yi - a * xi - b) ** 2

def dL_db(x_vals, y_vals, a, b):
    n = len(x_vals)
    partial_sum = 0
    
    for i in range(n):
        xi = x_vals[i]
        yi = y_vals[i]
        partial_sum += -2 * (yi - a * xi - b)
    
    return (1 / n) * partial_sum

# Example usage:
x_vals = np.array([1, 2, 3, 4, 5])
y_vals = np.array([2, 4, 6, 8, 10])
a = 2
b = 1

partial_derivative = dL_db(x_vals, y_vals, a, b)
print("Partial derivative ∂L/∂b:", partial_derivative)

# Link to chatGPT 3.5 conversation: https://chat.openai.com/share/c10bfe80-f1c5-48e3-9e7f-64b5508b8a84
'''
Correction to #5 - chatGPT 3.5 creating a new function for f but it was never
used. Also, although not wrong, the code it wrote is longer than necessary. 
'''


# 6

# 7 

# 8 

# 9 

# 10

# 11

# GPT 4

# 1

# 2 

# 3

# 4

# 5

def dL_db_gpt_4(x_vals, y_vals, a, b):
    """
    Compute the partial derivative of L with respect to b.
    
    Parameters:
    - x_vals: 1-D numpy array of x values.
    - y_vals: 1-D numpy array of y values.
    - a, b: Parameters of the function f.
    
    Returns:
    - dLdb: Partial derivative of L with respect to b.
    """
    n = len(x_vals)
    # Compute the derivative for each pair (x_i, y_i)
    df_db = 2 * (y_vals - a * x_vals - b) * (-1)
    # Sum all the derivatives and divide by n
    dLdb = np.sum(df_db) / n
    return dLdb

# Example usage:
# x_vals = np.array([1, 2, 3, 4])
# y_vals = np.array([2, 4, 5, 4])
# a = 1
# b = 2
# print(dL_db(x_vals, y_vals, a, b))

# Link to chat conversation: https://chat.openai.com/share/c78cfcef-ed0c-4b8e-abea-cf794fa4f763


# 6

# 7

# 8 

# 9 

# 10 

# 11
