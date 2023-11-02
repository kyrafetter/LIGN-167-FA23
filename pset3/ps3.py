import numpy as np
import torch
from torch import nn


######################################## BEGIN STARTER CODE ########################################

def relu(x):
	if x<0:
		return 0
	else:
		return x

def loss(y_predicted, y_observed):
	return (y_predicted - y_observed)**2


def mlp(x,W0,W1,W2):
	

	r0_0 = x*W0[0]
	r0_1 = x*W0[1]
	r0_2 = x*W0[2]
	r0 = np.array([r0_0,r0_1,r0_2])

	h0_0 = relu(r0_0)
	h0_1 = relu(r0_1)
	h0_2 = relu(r0_2)
	h0 = np.array([h0_0,h0_1,h0_2])

	

	r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2]
	r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2]
	r1_2 = h0_0*W1[2,0] + h0_1*W1[2,1]+ h0_2*W1[2,2]
	r1 = np.array([r1_0,r1_1,r1_2])

	h1_0 = relu(r1_0)
	h1_1 = relu(r1_1)
	h1_2 = relu(r1_2)
	h1 = np.array([h1_0,h1_1,h1_2])

	y_predicted = h1_0*W2[0] + h1_1*W2[1]+ h1_2*W2[2]

	variable_dict = {}
	variable_dict['x'] = x
	variable_dict['r0'] = r0
	variable_dict['h0'] = h0
	variable_dict['r1'] = r1
	variable_dict['h1'] = h1
	variable_dict['y_predicted'] = y_predicted

	return variable_dict


# x = 10
# W0 = np.array([1,2,3])
# W1 = np.array([[3,4,5],[-5,4,3],[3,4,1]])
# W2 = np.array([1,3,-3])






######################################## END STARTER CODE ########################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES



#PROBLEM 1
def d_loss_d_ypredicted(variable_dict,y_observed):
##YOUR CODE HERE##


#PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
##YOUR CODE HERE##


#PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
##YOUR CODE HERE##


#PROBLEM 4
def relu_derivative(x):
##YOUR CODE HERE##


#PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
##YOUR CODE HERE##


#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
##YOUR CODE HERE##


#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
##YOUR CODE HERE##



#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
##YOUR CODE HERE##


#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
##YOUR CODE HERE##


#PROBLEM 10
class TorchMLP(nn.Module):
	def __init__(self):
	    super().__init__()
	    ##YOUR CODE HERE##

	def forward(self, x):
		##YOUR CODE HERE##


#PROBLEM 11
def torch_loss(y_predicted, y_observed):
	##YOUR CODE HERE##


#PROBLEM 12
def torch_compute_gradient(x,y_observed,model):
	##YOUR CODE HERE##
	return model

