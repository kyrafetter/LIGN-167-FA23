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
Link: https://chat.openai.com/share/7ae8c1db-18ca-4d0d-bdd4-95eb1653b6ef
GPT-3.5 makes a few less desireable descisions here. First, it writes a function
f(xi, yi, a, b) which does not seem to have any utility in the scope of Problem
4 (the following function dL_da does not call this first function). Also, even
though GPT-3.5 imports numpy, it does not actually use functions from this
package which results in the need for a for loop, creating unnecessarily long
code. When asked to correct for these mistakes, GPT-3.5 successfully writes
a more concise and correct function (see the Revised version).
"""
# Initial
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

# Revised
import numpy as np

def dL_da(x_vals, y_vals, a, b):
    n = len(x_vals)
    
    # Calculate the partial derivative ∂L/∂a
    gradient = (-2/n) * np.sum((y_vals - a * x_vals - b) * x_vals)
    
    return gradient

# Example usage:
x_vals = np.array([1, 2, 3, 4, 5])
y_vals = np.array([2, 3, 4, 5, 6])
a = 1.0
b = 0.5

partial_derivative = dL_da(x_vals, y_vals, a, b)
print("Partial derivative ∂L/∂a:", partial_derivative)



# 5
'''
Correction to #5 - chatGPT 3.5 creating a new function for f but it was never
used. Also, although not wrong, the code it wrote is longer than necessary which leads 
to un desirable steps that are taking place. Instead of utilizing array multiplication, 
it uses for a loop to go through all the values, which leads to excess uncessary code. 

Link to chatGPT 3.5 conversation: https://chat.openai.com/share/c10bfe80-f1c5-48e3-9e7f-64b5508b8a84
'''
def f(xi, yi, a, b):
    return (yi - a * xi - b) ** 2

def dL_db_3_5(x_vals, y_vals, a, b):
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


#7
"""
https://chat.openai.com/share/6b8bf5c6-0855-44a5-85ae-02c520e529d4
GPT-3.5 provides a correct solution after some help. Intially, I made sure that
the gradient_descent_step function it wrote was correct and had to ask GPT-3.5
to assume that the (1/n) factor was accounted for in the output of the dL_da
and dL_db functions respectively. After making this correction, GPT-3.5 wrote
a correct gradient_descent function; I did request that GPT-3.5 change the
k=100 parameter to k=1000 as per the function header provided to us. 
"""
# Assuming you have already defined the gradient_descent_step function
def gradient_descent(x_vals, y_vals, a_0=0.0, b_0=0.0, k=1000):
    """
    Perform gradient descent to optimize parameters 'a' and 'b'.

    Args:
    x_vals (list): List of x values.
    y_vals (list): List of corresponding y values.
    a_0 (float, optional): Initial value of parameter 'a'. Default is 0.0.
    b_0 (float, optional): Initial value of parameter 'b'. Default is 0.0.
    k (int, optional): Number of gradient descent steps to perform. Default is 1000.

    Returns:
    tuple: A tuple containing the final parameter values 'ak' and 'bk'.
    """
    a_current, b_current = a_0, b_0
    
    for _ in range(k):
        # Perform a single step of gradient descent
        a_current, b_current = gradient_descent_step(x_vals, y_vals, a_current, b_current)
    
    return a_current, b_current


#10
"""
Link: https://chat.openai.com/share/3a924609-6bf6-4201-a3d9-1be5d576d2ce
GPT-3.5 provides a correct solution to this problem, but does write some extra
tests which are thorough but technically not needed under the assumptions
of the problem.
"""
import numpy as np

def batch_matrix_vector_multiplication(A, B):
    # Ensure that the dimensions of A and B are compatible
    assert A.shape[0] == B.shape[0]  # Check if b dimensions match
    assert A.shape[2] == B.shape[1]  # Check if m dimensions match

    # Perform batch matrix-vector multiplication using np.einsum
    C = np.einsum('ijk,ik->ij', A, B)

    return C

# Example usage:
# Create a sample 3-D array A and a 2-D array B
A = np.array([[[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]],
              [[13, 14, 15, 16],
               [17, 18, 19, 20],
               [21, 22, 23, 24]]])

B = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

# Call the function to perform batch matrix-vector multiplication
C = batch_matrix_vector_multiplication(A, B)

# Output the result
print(C)

# 6

# 7 

# 8 
"""
Link: https://chat.openai.com/share/3f04c113-22e6-4903-9646-6f0a9f119453
There were no issues when with this problem for chatGPT 3.5. Please note that 
when I prompted chatGPT 3.5 with this question, I did included the instructions that 
np.einsum should be used. 
"""
import numpy as np

def multiply_matrices_3_5(A, B):
    # Check if the input arrays have the same shape
    if A.shape != B.shape:
        raise ValueError("Input arrays A and B must have the same shape")

    # Use np.einsum to multiply the matrices element-wise
    C = np.einsum('ij,ij->ij', A, B)

    return C

# Example usage:
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 0, 1], [0, 1, 0]])

C = multiply_matrices(A, B)
print(C)


# 9 

# 10

# 11
"""
Link: https://chat.openai.com/share/ab51ccd4-35de-431b-9ff4-ae6b51de08dc
Initially, when first prompting chatGPT 3.5, I didn't mention that it should np.einsum(), which lead 
to an unnecessary amount of code being included. It was using a for-loop, np.zeros, and np.dot to 
complete this problem. However, when I included additional instruction to use np.einsum() to complete 
the problem, it did so without any additional code or errors. It did include an additional edge case
test to account for any invalid inputs. 
"""
def batch_matmul_3_5(A, B):
    """
    Perform batch matrix multiplication between A and B using np.einsum.

    Args:
    A: 3-D NumPy array with shape (b, n, m)
    B: 3-D NumPy array with shape (b, m, p)

    Returns:
    C: 3-D NumPy array with shape (b, n, p)
    """
    if A.shape[2] != B.shape[1]:
        raise ValueError("Matrix dimensions do not align for multiplication")

    # Use np.einsum to perform batch matrix multiplication
    C = np.einsum('bik, bkq -> biq', A, B)

    return C

# Example usage:
# Create sample 3-D arrays A and B
A = np.random.rand(2, 4, 3)  # A has shape (2, 4, 3)
B = np.random.rand(2, 3, 5)  # B has shape (2, 3, 5)

# Perform batch matrix multiplication
C = batch_matmul(A, B)  # C will have shape (2, 4, 5)



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

# 10 
"""
Link: https://chat.openai.com/share/5490482e-392a-4245-8ac4-4e52dee1d1b4
GPT-4 provides a correct solution to this problem, but does write some extra
tests (ie. test if A and B are numpy arrays and ensuring that dimensions are
compatible) which are thorough but technically not needed under the assumptions
of the problem.
"""
import numpy as np

def batch_matvec_mult(A, B):
    """
    Perform batch matrix-vector multiplication.
    
    Parameters:
        A (np.array): A 3-D NumPy array with shape (b, n, m).
        B (np.array): A 2-D NumPy array with shape (b, m).
        
    Returns:
        C (np.array): A 2-D NumPy array with shape (b, n).
    """
    # Ensure the inputs are numpy arrays
    A = np.array(A)
    B = np.array(B)

    # Check the shapes of A and B for compatibility
    assert A.shape[2] == B.shape[1], "Incompatible shapes for matrix-vector multiplication."
    
    # Use Einstein summation convention to perform the batch matrix-vector multiplication
    C = np.einsum('ijk,ik->ij', A, B)
    
    return C

# 5
"""
Link: https://chat.openai.com/share/c78cfcef-ed0c-4b8e-abea-cf794fa4f763
There were no errors in the solution. The only difference that did occur
was including the negative multiplication which lead to a different ordering 
from what I did -- but essentially the same thing. 
"""
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

Example usage:
x_vals = np.array([1, 2, 3, 4])
y_vals = np.array([2, 4, 5, 4])
a = 1
b = 2
print(dL_db(x_vals, y_vals, a, b))



# 6

# 7

# 8 
"""
Link: https://chat.openai.com/share/f1892baa-dbe7-439b-95e0-3739d8c7ba32
The code that chatGPT 4 gave me to solve this problem was correct. However, as I was 
trying to test my function, I prompted it to give me doctests, which were wrong. I asked it to fix / double 
check this mistake, but it still got it wrong. 
"""
def elementwise_product_4(A, B):
    """
    Compute the element-wise product of two matrices A and B using Einstein summation.
    
    Parameters:
    A, B: 2D NumPy arrays of the same size (n x m).
    
    Returns:
    C: 2D NumPy array (n x m) where C[i,j] = A[i,j] * B[i,j].
    """
    # Check if A and B have the same shape
    if A.shape != B.shape:
        raise ValueError("Input matrices A and B must have the same shape")
    
    # Use np.einsum to compute the element-wise product
    # 'ij,ij->ij' indicates that we take element i,j from A and multiply it with element i,j from B, 
    # and place the result in position i,j in the output array
    C = np.einsum('ij,ij->ij', A, B)
    
    return C

# 9 

# 10 

# 11
"""
Link: https://chat.openai.com/share/3c234bd6-2b16-42a0-b30a-0236a80984ef
When I initially input my question, without specificfying that it needed to use 
np.einsum, it wrote the solution using np.matmul. However, after further 
prompting it to use np.einsum(), it wrote correct code and included 
code for edge cases. 
"""

import numpy as np

def batch_matmul_einsum_4(A, B):
    """
    Perform batch matrix-matrix multiplication of A and B using np.einsum.

    Parameters:
    - A (numpy.ndarray): a 3D numpy array with shape (b, n, m)
    - B (numpy.ndarray): a 3D numpy array with shape (b, m, p)

    Returns:
    - C (numpy.ndarray): the resulting 3D numpy array with shape (b, n, p)
    """
    # Check if A and B are numpy arrays
    if not (isinstance(A, np.ndarray) and isinstance(B, np.ndarray)):
        raise ValueError("Input should be numpy ndarrays")

    # Check the shapes of A and B
    b, n, m = A.shape
    b_, m_, p = B.shape
    if b != b_ or m != m_:
        raise ValueError("Mismatched shapes: A is {}, B is {}".format(A.shape, B.shape))
    
    # Perform batch matrix multiplication using np.einsum
    C = np.einsum('ijk,ikq->ijq', A, B)
    return C

# Example usage:
# Create random arrays A and B with shapes (b, n, m) and (b, m, p) respectively.
b, n, m, p = 10, 4, 5, 3
A = np.random.rand(b, n, m)
B = np.random.rand(b, m, p)

# Get the batch matrix multiplication result
C = batch_matmul_einsum(A, B)

