'''
LIGN 167 PSET 3
Kyra Fetter, A16819058
Lina Battikha, A16852105
Oishani Bandopadhyay, A16800427

Group Member Contributions:
Kyra: 1, 2, 3, 4
Lina: 9, 10, 11, 12
Oishani: 5, 6, 7, 8 
Everyone: Reviewing code and debugging
'''

# Problem 1 - Kyra
'''
Link: https://chat.openai.com/share/f76525f6-84aa-4511-9c30-c1babd64803a
GPT-4 did not provide a new solution. It tested it's former solution and deemed
it to be correct. It is correct in making this conclusion. 
'''

# Let's redefine the function and make sure it's correct.

def d_loss_d_ypredicted(variable_dict, y_observed):
    # Retrieve the predicted value from the variable_dict
    y_predicted = variable_dict['y_predicted']
    
    # Compute the derivative of the loss with respect to the predicted value
    # This is the gradient of the squared loss function: 2 * (ypredicted - yobserved)
    derivative = 2 * (y_predicted - y_observed)
    
    return derivative

# There doesn't seem to be an obvious mistake, but let's run a test case to ensure it's working correctly.

# Define a test case
variable_dict_test = {'y_predicted': 10}  # Suppose the predicted value is 10
y_observed_test = 8  # Suppose the actual observed value is 8

# Run the function with the test case
d_loss_d_ypredicted(variable_dict_test, y_observed_test)
RESULT
4

# Problem 2 - Kyra
'''
Link: https://chat.openai.com/share/b0406c6d-8ae7-4fe8-87f8-c4dcaeb9c39b
GPT-4 did provide a new response, as pasted below. In this chat, GPT-4 thought that
it's initial function did not return a numpy array even though it did since it
simply multiplied h1 (a numpy array) by a scalar. However, it deemed this to be a
"mistake" and re-wrote the function to explicitly return a numpy array with the 
correct components. Both this new solution and the old solution are correct.
'''

import numpy as np

def d_loss_d_W2(variable_dict, y_observed):
    # Call the function from Problem 1 to get the partial derivative of the loss with respect to y_predicted
    d_loss_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Retrieve the network's value for the layer h1 from variable_dict
    h1 = variable_dict['h1']
    
    # Calculate the partial derivatives of the loss with respect to each weight in W2
    # Ensure the result is a 1x3 NumPy array
    d_loss_W2 = np.array([d_loss_ypred * h1[0], d_loss_ypred * h1[1], d_loss_ypred * h1[2]])
    
    return d_loss_W2

# Problem 3 - Kyra
'''
Link: https://chat.openai.com/share/167490f1-a2c8-4883-a37b-192a2c80a5a6
GPT-4 provides an identical response to it's initial response, both of which are
functionally correct. Please note that in this chat, GPT-4 provides this concise code
right off the bat in it's first response unlike the code I got in the gpt_solution_1.py
file.
'''

def d_loss_d_h1(variable_dict, W2, y_observed):
    # Calculate ∂L/∂y_pred
    dL_dypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # The gradient of the loss with respect to h1 is the weight W2, because y_pred = W2 * h1
    # So we need to multiply dL_dypred with each weight of W2 to get the gradient w.r.t to each h1 element.
    dL_dh1 = dL_dypred * W2
    
    return dL_dh1

# Problem 4 - Kyra
'''
Link: https://chat.openai.com/share/95da3c2d-1f01-4a5d-baf6-41f6abcbede4
GPT-4 provides the same response, which is a correct response. It did think again
about how ReLU is defined at 0 (it should return 0); however it's initial solution
implements this correctly so there was no change to the code.
'''

# Correct the derivative of the ReLU function
def relu_derivative(x):
    # The derivative is 1 if x > 0, else it is 0 (also 0 when x is exactly 0)
    return 1 if x > 0 else 0

# Test the relu_derivative function with a positive, negative, and zero value again
test_values = [10, -5, 0]
derivatives_corrected = [relu_derivative(x) for x in test_values]
derivatives_corrected
RESULT
[1, 0, 0]

# Problem 5

# Problem 6

# Problem 7

# Problem 8

# Problem 9

# Problem 10

# Problem 11

# Problem 12
