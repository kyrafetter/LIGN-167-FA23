'''
LIGN 167 PSET 2
Kyra Fetter, A16819058
Lina Battikha, A16852105
Oishani Bandopadhyay, A16800427

Group Member Contributions:
Kyra: 
Lina: 
Oishani: 
Everyone: 
'''

################################ BEGIN GPT-3.5 RESPONSES #################################################

# PROBLEM 1
"""
ChatGPT 3.5 provided the right answer.
Link to Conversation: https://chat.openai.com/share/90741672-a643-4266-8172-8e238b9dcb9f
"""
import numpy as np 

def sigmoid(x):
	#Numerically stable sigmoid function.
	#Taken from: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
	if x >= 0:
		z = np.exp(-x)
		return 1 / (1 + z)
	else:
		# if x is less than zero then z will be small, denom can't be
		# zero because it's 1+z.
		z = np.exp(x)
		return z / (1 + z)



# PROBLEM 2
"""
The solution that chatgpt3.5 provided is correct. When I asked it to write some doctests, it gave me wrong answers. 
Link to Conversation: https://chat.openai.com/share/9651f192-c6e5-4387-a930-bc58f50482f8
"""
import numpy as np

def logistic_derivative_per_datapoint(y_i, x_i, a, j):
    # Calculate the sigmoid value
    sigmoid_value = 1 / (1 + np.exp(-np.dot(x_i, a)))

    # Compute the derivative
    derivative = -(y_i - sigmoid_value) * x_i[j]

    return derivative



# PROBLEM 3
"""
The code that chatgpt3.5 provided for this problem was correct.
When I asked it to write some doctests, it gave me wrong answers. 
Link to Conversation: https://chat.openai.com/share/a7a76bf6-4c45-4e5d-a6f8-2173ffd6b3e8 
"""
import numpy as np

# Sigmoid function
def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

# Compute the derivative ∂`i` / ∂`aj` for a single data point
def logistic_derivative_per_datapoint(yi, xi, a, j):
    return -(yi - sigmoid(np.dot(xi, a))) * xi[j]

# Compute the partial derivative ∂L / ∂aj using logistic_derivative_per_datapoint for all data points
def logistic_partial_derivative(y, x, a, j):
    n = len(y)
    derivative_sum = 0.0
    
    for i in range(n):
        derivative_sum += logistic_derivative_per_datapoint(y[i], x[i], a, j)
    
    return derivative_sum / n



# PROBLEM 4
"""
ChatGPt 3.5 messed up on this problem. The function is used intially for partial 
derivative is wrong. The partial dervivative function that it wrote was wrong. This 
may be because I didn't provide the previous solution to it. Once I added it, I did
provide the additional information (the previous function), it didn't compute it correctly. 
It used a nested for-loop unnecessarily and then used the previous function which I 
provided incorrectly. To sum this all up, it over complicated the problem. 

"""
import numpy as np

def logistic_partial_derivative(y, x, a, j):
    # Calculate the predicted probability for this data point
    predicted_prob = sigmoid(np.dot(x, a))
    
    # Compute the partial derivative with respect to the j-th element of a
    derivative = (y - predicted_prob) * x[j]
    
    return derivative

def compute_logistic_gradient(a, y, x):
    # Initialize the gradient vector with zeros.
    gradient = np.zeros_like(a)
    
    # Calculate the number of data points.
    n = len(y)
    
    # Loop through each data point to compute the gradient.
    for i in range(n):
        # Compute the partial derivative for this data point.
        for j in range(len(a)):
            gradient[j] += logistic_partial_derivative(y[i], x[i], a, j)
    
    # Normalize the gradient by dividing by the number of data points.
    gradient /= n
    
    return gradient

# Define the sigmoid function
def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)



# PROBLEM 5


# PROBLEM 6


# PROBLEM 7


# PROBLEM 8
 

# PROBLEM 9


# PROBLEM 10
 

# PROBLEM 11


