## LIGN 167 PSET 1
## Kyra Fetter, A16819058
## Lina Battikha, A16852105
## Oishani Bandopadhyay, A1600827

import numpy as np
from numpy.random import randn

#Problem 1 - Oishani
#hello
def compute_slope_estimator(x_vals,y_vals):
	n = len(x_vals)
	for i in range(0, n):
		x_bar = np.average(np.arange(, x_vals[n]))
		
	

#Problem 2 - Oishani
def compute_intercept_estimator(x_vals,y_vals):
	y_mean = np.mean()

#Problem 3 - Oishani
def train_model(x_vals,y_vals):
	#your code here
	return (a,b)

#Problem 4 - Kyra
def dL_da(x_vals,y_vals,a,b):
	n = x_vals.size
	x_sqrd = np.power(x_vals, 2)
	ax_sqrd = a * x_sqrd
	xy = x_vals * y_vals 
	bx = b * y_vals
	total 
	## hello, this is a test comment
		

#Problem 5 - Lina - doctest that chat gpt was wrong. 

def dL_db(x_vals,y_vals,a,b):
	"""
	>>> dL_db(np.array([1, 2, 3, 4]), np.array([2, 4, 5, 4]), 1, 2)
	1.5
	# >>> dL_db(np.array([2, 4]), np.array([4, 7]), 0.5, 3)
	# -3.5
	# >>> dL_db(np.array([1, 3]), np.array([4, 6]), 1, 1)
	# -3.0
	"""
	ax = a*x_vals
	temp = (ax - y_vals) + b
	total_sum = np.sum(temp)
	final_partial_b = 2 * total_sum
	final_partial_b = float(final_partial_b)/float(len(x_vals))
	return final_partial_b 



#Problem 6 - Oishani
def gradient_descent_step(x_vals,y_vals,a,b,k=0.01):
	

#Problem 7  - Kyra
def gradient_descent(x_vals,y_vals,a_0=0,b_0=0,k=1000):
	


# Problem 8  Lina -- fix doctest, right now it's not working 
def einsum_1(A, B):
	"""
	>>> A = np.array([[1, 2, 3], [4, 5, 6]])
	>>> B = np.array([[1, 0, 1], [0, 1, 0]])
	>>> einsum_1(A, B)
	array([[1, 0, 3], [0, 5, 0]])
	"""
	return np.einsum('ij, ij->ij', A, B)

# Problem 9 - Oishani
def einsum_2(A, B):
	return np.einsum...

# Problem 10 - Kyra
def einsum_3(A, B):
	return np.einsum...

# Problem 11 - Lina 
def einsum_4(A, B):
	return np.einsum('ijk, ikq -> ijq', A, B)
