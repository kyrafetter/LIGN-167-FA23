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



# PROBLEM 5 - Kyra
'''
Link: https://chat.openai.com/share/de29e4da-04da-465e-8f8a-da0659450d30
The solution given by GPT-3.5 is correct. It is functionally identical to my solution.
I also asked GPT-4 to check this code and GPT-4 reported it to be correct.
'''
import numpy as np

def gradient_update(a, lr, gradient):
    """
    Update the parameter vector 'a' using gradient descent.

    Args:
        a (numpy.ndarray): Current parameter vector of length k.
        lr (float): Learning rate (positive real number).
        gradient (numpy.ndarray): Gradient vector of length k.

    Returns:
        numpy.ndarray: Updated parameter vector 'a'.
    """
    updated_a = a - lr * gradient
    return updated_a


# PROBLEM 6 - Kyra
'''
Link: https://chat.openai.com/share/1e2c8c68-1de6-417d-af37-f5bc4cc3e41b
The solution given by GPT-3.5 is functionally correct; however, it doesn't 
necessarily need to make a copy of initial_a when assigning a value to a. If 
initial_a is large, this may be an inefficient operation. 
'''
def gradient_descent_logistic(initial_a, lr, num_iterations, y, x):
    a = initial_a.copy()
    for _ in range(num_iterations):
        gradient = compute_logistic_gradient(a, y, x)
        a = gradient_update(a, lr, gradient)
    return a


# PROBLEM 7 - Kyra
'''
Link: https://chat.openai.com/share/ce1ad669-f212-4b32-bd4e-e27d9eb8bfd8
There are a couple instances of semantic issues with the solution GPT-3.5 provides.
First, it claims that the weights of the object are initialized as "nn.Parameter
objects with zero values". Technically there is only one nn.Parameter object here;
it is a tensor of length num_features. GPT-3.5 also does not mention the num_features
parameter when explaining this line of __init__, which may have added clarity.
Finally, in the final sentence of its response, it describes self.weights as a "vector"
which is technically accurate, but it would have been more specific if GPT-3.5 
called self.weights a tensor. Nevertheless, GPT-3.5 correctly identifies the line
at which __init__ is called.

******** GPT-3.5 Solution: ********

The __init__ function in the TorchLogisticClassifier class is the constructor method for
initializing an instance of the class. In this specific context, it does the following:

It calls the constructor of the parent class nn.Module using super().__init__(), which is
necessary when creating custom PyTorch models.

It initializes the weights of the logistic regression model as nn.Parameter objects with
zero values. These weights are learned during training.

Here is the __init__ function in your code:

def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(num_features))

The line in the starter code that calls the __init__ function is the following line:

model = TorchLogisticClassifier(num_features)

In this line, a new instance of the TorchLogisticClassifier class is created, and the
__init__ function is called to initialize the model. The num_features argument specifies
the number of features in the input data, which determines the size of the weight
vector self.weights.
'''


# PROBLEM 8 - Kyra
'''
Link: https://chat.openai.com/share/cb2a9021-9e27-48d5-9a59-12001de71e03
GPT-3.5 provides a correct response to this problem. It correctly identifies the function
to be a logistic regression sigmoid function, and it also correctly identifies the line
at which the forward function will be called. 

******** GPT-3.5 Solution: ********

The forward method of the TorchLogisticClassifier class computes the probability of the
positive class (class 1) for a given input feature vector x_vector using logistic
regression. This is done by taking the dot product of the model's weights (self.weights)
and the input feature vector x_vector, and then applying the sigmoid function to the result.

Here is the mathematical function computed by the forward method:

P(y=1∣x,a)=σ(a⋅x)

Where:
    - P(y=1∣x,a) is the probability of the positive class (class 1).
    - σ(a⋅x) is the sigmoid function applied to the dot product of the model's weights (a)
a) and the input feature vector (x).

The line of code in the starter code that calls the forward method is within the
nonbatched_gradient_descent function:

prediction = model(d_x)

Here, d_x is the input feature vector for a particular data point, and model(d_x)
computes the predicted probability of the positive class using the logistic regression
model.
'''
 

# PROBLEM 9


# PROBLEM 10
 

# PROBLEM 11


