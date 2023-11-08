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
Link: https://chat.openai.com/share/863fb0b6-24bc-4276-b7fe-ebfad3faa4c7
GPT=4 provides a correct response. 
'''

# Here we define the function d_loss_d_ypredicted according to the problem statement.
def d_loss_d_ypredicted(variable_dict, y_observed):
    # Retrieve the network's predicted value ypred from variable_dict
    y_predicted = variable_dict['y_predicted']
    # Compute the partial derivative of the loss with respect to y_predicted
    # According to the given loss function `i = (ypred - yi)^2,
    # the derivative is d`i/dypred = 2 * (ypred - yi)
    return 2 * (y_predicted - y_observed)

# Test the function with dummy data
test_variable_dict = {'y_predicted': 4}  # Example where the network predicted a value of 4
test_y_observed = 3  # Example where the observed value is 3

# Call the function with the test data
d_loss_d_ypredicted_result = d_loss_d_ypredicted(test_variable_dict, test_y_observed)
d_loss_d_ypredicted_result
RESULT
2

# Problem 2 - Kyra
'''
Link: https://chat.openai.com/share/df62c37a-68ce-4373-861e-13bc0186436a
GPT-4 provides a correct response.
'''

import numpy as np

def d_loss_d_W2(variable_dict, y_observed):
    # Call d_loss_d_ypredicted from the previous problem
    d_loss_dy_pred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Retrieve the network's value for the layer h1 from variable_dict
    h1 = variable_dict['h1']
    
    # Calculate the partial derivatives of the loss with respect to W2
    d_loss_d_W2 = d_loss_dy_pred * h1
    
    return d_loss_d_W2

# Problem 3 - Kyra
'''
Link: https://chat.openai.com/share/bffbc793-afc4-464a-ae5e-b36554e0af58
GPT-4 provides a correct response. However, it is not the most concise code in regard
to syntax. There is no need for a for loop. This computation can be done simply in
one line by multiplying d_loss_d_ypred by W2 since W2 is a vector. 
'''

def d_loss_d_h1(variable_dict, W2, y_observed):
    # Calculate the derivative of the loss with respect to y_predicted
    d_loss_d_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Initialize the partial derivatives array
    d_loss_d_h1 = np.zeros_like(variable_dict['h1'])
    
    # Compute the partial derivatives of the loss with respect to h1
    for i in range(len(W2)):
        # Apply the chain rule: d_loss/d_h1 = d_loss/d_ypred * d_ypred/d_h1
        d_loss_d_h1[i] = d_loss_d_ypred * W2[i]
    
    return d_loss_d_h1

# Sample variable dictionary, weights W2 and observed y for testing the function
variable_dict_sample = {
    'h1': np.array([1.0, 2.0, 3.0]),
    'y_predicted': 10.0
}
W2_sample = np.array([1, 3, -3])
y_observed_sample = 12.0

# Testing the function with the sample data
d_loss_d_h1_result = d_loss_d_h1(variable_dict_sample, W2_sample, y_observed_sample)
d_loss_d_h1_result
RESULT
array([ -4., -12.,  12.])

# Problem 4 - Kyra
'''
Link: https://chat.openai.com/share/a34a1330-6528-4134-880b-b0d3ae5f1199
GPT-4 provides a correct repsonse.
'''

def relu_derivative(x):
    """
    Compute the derivative of ReLU function.

    The derivative of ReLU function is:
    - 0 if x <= 0
    - 1 if x > 0

    Parameters:
    x (float): The input value to the ReLU function.

    Returns:
    float: The derivative of ReLU at x.
    """
    return 1 if x > 0 else 0

# Example usage:
relu_derivative_example = relu_derivative(5), relu_derivative(-3), relu_derivative(0)
relu_derivative_example
RESULT
(1, 0, 0)

# Problem 5

# Problem 6

# Problem 7

# Problem 8

# Problem 9

# Problem 10

# Problem 11

# Problem 12
