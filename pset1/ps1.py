'''
LIGN 167 PSET 1
Kyra Fetter, A16819058
Lina Battikha, A16852105
Oishani Bandopadhyay, A1600827

Group Member Contributions:
Kyra: Problems 4, 7, 10 (python code, GPT-3.5, GPT-4, testing)
Lina: Problems 5, 8, 11 (python code, GPT-3.5, GPT-4, testing)
Oishani: Problems 1, 2, 3, 6, 9 (python code, GPT-3.5, GPT-4, testing)
Everyone: GitHub version control, formatting the documents, debugging
'''

import numpy as np
from numpy.random import randn

# Problem 1 - Oishani
def compute_slope_estimator(x_vals,y_vals):
	n = len(x_vals)
	x_bar = np.mean(x_vals)
	y_bar = np.mean(y_vals)
	numerator = np.sum(x_vals*y_vals) - (n*x_bar*y_bar)
	denominator = np.sum(x_vals**2) - (n*(x_bar**2))
	a = float(numerator)/float(denominator)
	return a
	


# Problem 2 - Oishani
def compute_intercept_estimator(x_vals,y_vals):
	x_bar = float((np.sum(x_vals))/n)
	y_bar = float((np.sum(y_vals))/n)
	a = compute_slope_estimator(x_vals,y_vals)
	b = y_bar - (a*x_bar)
	return b



# Problem 3 - Oishani
def train_model(x_vals,y_vals):
	a = compute_slope_estimator(x_vals,y_vals)
	b = compute_intercept_estimator(x_vals,y_vals)
	return (a,b)



# Problem 4 - Kyra
def dL_da(x_vals,y_vals,a,b):
	dl_da = 2 * (np.sum((a * np.power(x_vals, 2)) - (x_vals * y_vals) + (b * x_vals)))
	return (1 / float(x_vals.size)) * dl_da



# Problem 5 - Lina 
def dL_db(x_vals,y_vals,a,b):
	"""
	>>> dL_db(np.array([1, 2, 3, 4]), np.array([2, 4, 5, 4]), 1, 2)
	1.5
	>>> dL_db(np.array([2, 4]), np.array([4, 7]), 0.5, 3)
	-2.0
	>>> dL_db(np.array([1, 3]), np.array([4, 6]), 1, 1)
	-4.0
	"""
	ax = a*x_vals
	temp = (ax - y_vals) + b
	total_sum = np.sum(temp)
	final_partial_b = 2 * total_sum
	final_partial_b = float(final_partial_b)/float(len(x_vals))
	return final_partial_b 



# Problem 6 - Oishani
def gradient_descent_step(x_vals,y_vals,a,b,k=0.01):
  
	dLda_over_n = dL_da(x_vals,y_vals,a,b)
	a_updated = a - (k*dLda_over_n)
	dLdb_over_n = dL_db(x_vals,y_vals,a,b)
	b_updated = b - (k*dLdb_over_n)
	return (a_updated, b_updated)



# Problem 7  - Kyra
def gradient_descent(x_vals,y_vals,a_0=0,b_0=0,k=1000):
	a_b = (a_0, b_0)
	for i in range(k):
		a_b = gradient_descent_step(x_vals, y_vals, a_b[0], a_b[1])
	return a_b 



# Problem 8 - Lina
def einsum_1(A, B):
	"""
	>>> A = np.array([[1, 2, 3], [4, 5, 6]])
	>>> B = np.array([[1, 0, 1], [0, 1, 0]])
	>>> einsum_1(A, B)
	array([[1,0,3], [0,5,0]])
	"""
	return np.einsum('ij, ij->ij', A, B)



# Problem 9 - Oishani
def einsum_2(A, B):
	return np.einsum('ij,j ->ij', A, B)



# Problem 10 - Kyra
def einsum_3(A, B):
	return np.einsum('ijk,ik->ij', A, B)



# Problem 11 - Lina 
def einsum_4(A, B):
	"""
	>>> A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> B = np.array([[[1, 3], [2, 4]], [[5, 7], [6, 8]]])
    >>> einsum_4(A, B)
    array([[[  5,  11],
            [ 11,  25]],
    <BLANKLINE>
           [[ 61,  83],
            [ 83, 113]]])
	"""
	return np.einsum('ijk, ikq -> ijq', A, B)