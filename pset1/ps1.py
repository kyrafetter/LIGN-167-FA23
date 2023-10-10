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
	x_bar = (np.sum(x_vals))/n
	y_bar = (np.sum(y_vals))/n
	numerator_sum = 0
	for i in range(0,n):
		prod = x_vals[i]*y_vals[i]
		numerator_sum += prod
	numerator = (numerator_sum - (n*x_bar*y_bar))
	denom_sum = 0
	for i in range(0,n):
		prod = x_vals[i]*x_vals[i]
		denom_sum += prod
	denominator = (denom_sum - (n*x_bar*x_bar)
	a = numerator/denominator
	return a		
	

#Problem 2 - Oishani
def compute_intercept_estimator(x_vals,y_vals):
	x_bar = (np.sum(x_vals))/n
	y_bar = (np.sum(y_vals))/n
	a = compute_slope_estimator(x_vals,y_vals)
	b = y_bar - (a*x_bar)
	return b


#Problem 3 - Oishani
def train_model(x_vals,y_vals):
	a = compute_slope_estimator(x_vals,y_vals)
	b = compute_intercept_estimator(x_vals,y_vals)
	return (a,b)


#Problem 4 - Kyra
def dL_da(x_vals,y_vals,a,b):
	n = x_vals.size
	x_sqrd = np.power(x_vals, 2)
	ax_sqrd = a * x_sqrd
	xy = x_vals * y_vals 
	bx = b * y_vals
	total 
	

#Problem 5 - Lina -- COME BACK AND TEST THIS 
def dL_db(x_vals,y_vals,a,b):
	ax = a*x_vals
	total_sum = 0 
	temp = ax - y_vals
	temp = temp + b
	total_sum += np.sum(temp)
	final_partial_b = 2 * total_sum
	final_partial_b = final_partial_b/x_vals.size
	return final_partial_b 

#Problem 6 - Oishani
def gradient_descent_step(x_vals,y_vals,a,b,k=0.01):

#Problem 7  - Kyra
def gradient_descent(x_vals,y_vals,a_0=0,b_0=0,k=1000):


# Problem 8  Lina
def einsum_1(A, B):
	return np.einsum...

# Problem 9 - Oishani
def einsum_2(A, B):
	return np.einsum...

# Problem 10 - Kyra
def einsum_3(A, B):
	return np.einsum...

# Problem 11 - Lina 
def einsum_4(A, B):
	return np.einsum...
