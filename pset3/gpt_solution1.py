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

# Problem 1

# Problem 2

# Problem 3

# Problem 4

# Problem 5

# Problem 6

# Problem 7

# Problem 8

# Problem 9
"""
Link to conversation: https://chat.openai.com/share/a7f99cd5-82fe-466b-8be1-c23772249d75
Although I provided the functions that it can use, rewrote some of the functions 
unnecessarily. It didn't correctly access "ypredicted" from the dictionary, instead 
used "ypred". Also, it grabbed h0, but never used it. Also, the line that uses np.dot is unecesarily complicated and may not 
lead to the correct results. Instead of doing the dot product, it should be multiplying 
the partial derivatives together. 
"""
import numpy as np

def d_loss_d_W0(variable_dict, W1, W2, y_observed):
    # Retrieve necessary variables from the variable_dict
    x = variable_dict['x']
    h0 = variable_dict['h0']
    r1 = variable_dict['r1']
    
    # Calculate the derivative of the loss with respect to the prediction
    d_loss_d_ypred = 2 * (variable_dict['ypred'] - y_observed)
    
    # Calculate the derivative of the prediction with respect to h1
    d_ypred_d_h1 = W2
    
    # Calculate the derivative of h1 with respect to r1 (ReLU derivative)
    d_h1_d_r1 = relu_derivative(r1)
    
    # Calculate the derivative of r1 with respect to h0
    d_r1_d_h0 = W1
    
    # Calculate the derivative of h0 with respect to r0 (ReLU derivative)
    d_h0_d_r0 = relu_derivative(variable_dict['r0'])
    
    # Calculate the derivative of r0 with respect to W0 (which is simply x)
    d_r0_d_W0 = x
    
    # Chain rule to get the derivative of the loss with respect to W0
    d_loss_d_W0 = d_loss_d_ypred * np.dot(d_ypred_d_h1, d_h1_d_r1) * np.dot(d_r1_d_h0, d_h0_d_r0) * d_r0_d_W0
    
    return d_loss_d_W0.reshape(1, 3)  # Reshape to match the required 1x3 NumPy array format


# Problem 10
"""
Link to conversation: https://chat.openai.com/share/2dc6cd26-9de4-4673-899b-abdd5763b85b
The code that was provided here is correct. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchMLP(nn.Module):
    def __init__(self):
        super(TorchMLP, self).__init__()
        # Initialize weights for W0, W1, W2
        self.W0 = nn.Parameter(torch.randn(3, 1))  # Assuming W0 has a shape of 3x1 as per the problem
        self.W1 = nn.Parameter(torch.randn(3, 3))  # Assuming W1 has a shape of 3x3
        self.W2 = nn.Parameter(torch.randn(1, 3))  # Assuming W2 has a shape of 1x3

    def forward(self, x):
        # Forward pass through the network
        r0 = torch.matmul(self.W0, x)
        h0 = F.relu(r0)
        
        r1 = torch.matmul(self.W1, h0)
        h1 = F.relu(r1)
        
        y_pred = torch.matmul(self.W2, h1)
        
        return y_pred

    def loss(self, y_pred, y_true):
        # Mean Squared Error loss
        return ((y_pred - y_true) ** 2).mean()

# Usage
# Instantiate the model
model = TorchMLP()

# Input tensor 'x' and target tensor 'y_true'
x = torch.tensor([1.0], requires_grad=True)
y_true = torch.tensor([2.0])

# Forward pass
y_pred = model(x)

# Compute loss
loss_value = model.loss(y_pred, y_true)

# Compute gradients
loss_value.backward()

# Gradients are automatically computed and stored in the '.grad' attribute of the parameters
print(model.W0.grad, model.W1.grad, model.W2.grad)


# Problem 11
"""
Link to conversation: https://chat.openai.com/share/152aa8e0-b825-4a7d-a528-ea54a422aabb
The code that was provided is correct. 
"""
import torch

def torch_loss(y_predicted, y_observed):
    # Assuming y_predicted and y_observed are PyTorch tensors of length 1
    loss = (y_predicted - y_observed) ** 2
    return loss

# Problem 12
"""
Link to conversation: https://chat.openai.com/share/1bcf0c46-5b7d-4c6a-a105-3896c985af34
The code that was provided is correct. 
"""
import torch

def torch_compute_gradient(x, y_observed, model):
    # Ensure the model is in training mode
    model.train()
    
    # Zero the gradients
    model.zero_grad()
    
    # Forward pass to get the predicted output
    y_pred = model(x)
    
    # Compute the loss using MSE
    loss = torch.nn.functional.mse_loss(y_pred, y_observed)
    
    # Backward pass to compute the gradient
    loss.backward()
    
    # Return the model (with updated gradients)
    return model

