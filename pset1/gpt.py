## LIGN 167 PSET 1
## Kyra Fetter, A16819058
## Lina Battikha, A16852105
## Oishani Bandopadhyay, A1600827

#  GPT 3.5 

# 1

# 2 

# 3

# 4
"""
Link: https://chat.openai.com/share/94b7b159-d1a4-4ef5-8c99-ec7ce4460323
GPT-3.5 makes a few less desireable descisions here. First, it writes a function
f(xi, yi, a, b) which does not seem to have any utility in the scope of Problem
4 (the following function dL_da does not call this first function). Also, even
though GPT-3.5 imports numpy, it does not actually use functions from this
package which results in the need for a for loop, creating unnecessarily long
code. However, there are no significant mistakes.
"""
import numpy as np

def f(xi, yi, a, b):
    return (yi - a * xi - b)**2

def dL_da(x_vals, y_vals, a, b):
    n = len(x_vals)
    partial_sum = 0
    
    for i in range(n):
        partial_sum += -2 * x_vals[i] * (y_vals[i] - a * x_vals[i] - b)
    
    return (1/n) * partial_sum

# Example usage:
x_vals = np.array([1, 2, 3, 4, 5])
y_vals = np.array([2, 4, 5, 4, 5])
a = 1.0
b = 0.0

partial_derivative = dL_da(x_vals, y_vals, a, b)
print(partial_derivative)

#7
"""
Link: https://chat.openai.com/share/34448b18-ffb9-4b4e-8448-ab90ba9e8b79
GPT-3.5 almost provides a correct solution. One issue with the code it generates
is the inclusion of the learning_rate parameter in the function header. GPT-3.5
did not figure out that this parameter is initialized a value in the header of
gradient_descent_step and therefore it thought it needed to include a 
learning_rate parameter in the header of gradient_descent. However, this is not
needed. Other than this, however, the function is correct. 
"""
def gradient_descent(x_vals, y_vals, a_0=0.0, b_0=0.0, k=100, learning_rate=0.01):
    """
    Perform k gradient descent steps.

    Args:
    x_vals: List of x values
    y_vals: List of corresponding y values
    a_0: Initial value of parameter 'a' (default=0.0)
    b_0: Initial value of parameter 'b' (default=0.0)
    k: Number of gradient descent steps to perform (default=100)
    learning_rate: Learning rate for gradient descent (default=0.01)

    Returns:
    Tuple (ak, bk): Final parameter values 'a' and 'b'
    """
    a_current = a_0
    b_current = b_0
    
    for step in range(k):
        # Perform a single step of gradient descent using the provided function
        a_current, b_current = gradient_descent_step(x_vals, y_vals, a_current, b_current, learning_rate)
    
    return a_current, b_current



# GPT 4

# 1

# 2 

# 3

# 4
"""
Link: https://chat.openai.com/share/f7c022a2-7abc-44d5-af1d-e6670714208d
GPT-4 appears not to make any mistakes. The code is nicely commented, and GPT-4
makes use of the numpy package to write concise node. One difference is that
GPT-4 does not use numpy to calculate the size of the numpy array and instead
uses the python len function, but this is not necessarily a mistake.
"""
import numpy as np

def dL_da(x_vals, y_vals, a, b):
    """
    Compute the partial derivative of L with respect to a.

    Parameters:
    - x_vals (numpy array): 1D array containing x values.
    - y_vals (numpy array): 1D array containing y values.
    - a (float): The coefficient for x in the regression equation.
    - b (float): The constant term in the regression equation.

    Returns:
    - float: The partial derivative of L with respect to a.
    """
    n = len(x_vals)
    derivative = -2 * np.sum(x_vals * (y_vals - a * x_vals - b)) / n
    return derivative

# 7
"""
Link: https://chat.openai.com/share/f7bf4988-4bf4-4b35-89e6-8c053d8c3870
GPT-4 does not appear to make any mistakes; it provides clean, concise, and
well-commented code which makes correct use of the gradient_descent_step
function implemented in the previous problem. 
"""

def gradient_descent(x_vals, y_vals, a_0=0, b_0=0, k=1000):
    """
    Performs k steps of the gradient descent optimization algorithm.
    
    Parameters:
    - x_vals: List of x values
    - y_vals: List of y values
    - a_0: Initial coefficient a of the linear model
    - b_0: Initial coefficient b of the linear model
    - k: Number of gradient descent steps to perform
    
    Returns:
    Tuple of final coefficients (a_k, b_k).
    """
    a, b = a_0, b_0
    for _ in range(k):
        a, b = gradient_descent_step(x_vals, y_vals, a, b)
    return a, b
