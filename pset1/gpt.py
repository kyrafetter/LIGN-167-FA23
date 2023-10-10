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

# Link to chatGPT 3.5 conversation: https://chat.openai.com/share/c10bfe80-f1c5-48e3-9e7f-64b5508b8a84
'''
Correction to #5 - chatGPT 3.5 creating a new function for f but it was never
used. Also, although not wrong, the code it wrote is longer than necessary. 
'''


# 6

# 7 

# 8 
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

# https://chat.openai.com/share/3f04c113-22e6-4903-9646-6f0a9f119453


# 9 

# 10

# 11
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

# link to conversation: https://chat.openai.com/share/ab51ccd4-35de-431b-9ff4-ae6b51de08dc


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

# Link to chat conversation: https://chat.openai.com/share/f1892baa-dbe7-439b-95e0-3739d8c7ba32


# 9 

# 10 

# 11

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

# Link to chat conversation: https://chat.openai.com/share/3c234bd6-2b16-42a0-b30a-0236a80984ef






