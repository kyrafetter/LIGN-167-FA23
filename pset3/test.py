import numpy as np
import torch
from torch import nn
#from torch import F


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

#PROBLEM 1 - Kyra
def d_loss_d_ypredicted(variable_dict,y_observed):
	ypred = variable_dict['y_predicted']
	return 2 * (ypred - y_observed)


#PROBLEM 2 - Kyra
def d_loss_d_W2(variable_dict,y_observed):
	dl_dypred = d_loss_d_ypredicted(variable_dict,y_observed)
	dl_dW2_0 = dl_dypred * variable_dict['h1'][0]
	dl_dW2_1 = dl_dypred * variable_dict['h1'][1]
	dl_dW2_2 = dl_dypred * variable_dict['h1'][2]
	return np.array([dl_dW2_0, dl_dW2_1, dl_dW2_2])


#PROBLEM 3 - Kyra
def d_loss_d_h1(variable_dict,W2,y_observed):
	dl_dypred = d_loss_d_ypredicted(variable_dict,y_observed)
	dl_dh1_0 = dl_dypred * W2[0]
	dl_dh1_1 = dl_dypred * W2[1]
	dl_dh1_2 = dl_dypred * W2[2]
	return np.array([dl_dh1_0, dl_dh1_1, dl_dh1_2])
	

#PROBLEM 4 - Kyra
def relu_derivative(x):
	if x > 0:
		return 1
	return 0


#PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
	d1_dh1 = d_loss_d_h1(variable_dict, W2, y_observed)
	r1 = variable.dict['r1']
	relu_grads = np.array([relu_derivative(r1_j) for r1_j in r1])
	dl_dr1 = d1_dh1*relu_grads
	return dl_dr1

#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
    # Calculate the gradient of the loss with respect to r1
    dl_dr1 = d_loss_d_r1(variable_dict, W2, y_observed)
    
    # Retrieve the activation of h0 from variable_dict
    h0 = variable_dict['h0']
    
    # Use einsum to calculate the outer product for each row in dl_dr1 and h0
    # This effectively calculates the gradient for each weight in W1
    dl_dW1 = np.einsum('i,j->ij', dl_dr1, h0)
    
    return dl_dW1


#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
	dl_dr1 = d_loss_d_r1(variable_dict, W2, y_observed)
	relu_grads_r1 = np.array([relu_derivative(ri) for ri in variable_dict['r1']])
	dl_dh1 = dl_dr1 * relu_grads_r1
	dl_dh0 = np.dot(W1.T, dl_dh1)
	return dl_dh0


#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
	d1_dh0 = d_loss_d_h0(variable_dict, W2, y_observed)
	r0 = variable.dict['r0']
	relu_grads = np.array([relu_derivative(r0_j) for r0_j in r0])
	dl_dr0 = d1_dh0*relu_grads
	return dl_dr0

#PROBLEM 9 - Lina
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
	d_loss_d_ro = d_loss_d_r0(variable_dict, W1, W2, y_observed)
	x = variable_dict["x"]
	return x * d_loss_d_ro

class TorchMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.first_layer = nn.Parameter(torch.rand(3,1)) 
		self.second_layer = nn.Parameter(torch.rand(3,3)) 
		self.third_layer = nn.Parameter(torch.rand(1,3))

	def forward(self, x):
		#first layer
		r0 = torch.matmul(self.first_layer, x)
		h0 = torch.relu(r0)
		#h0 = F.relu(self.first_layer(x))

		#second layer 
		r1 = torch.matmul(self.second_layer, h0)
		h1 = torch.relu(r1)
		#h1 = F.relu(self.second_layer(h0))

		# compute final output
		y_pred = torch.matmul(self.third_layer, h1)
		#y_pred = self.third_layer(h1)

		return y_pred


def torch_loss(y_predicted, y_observed):
	return (y_predicted - y_observed)**2


def torch_compute_gradient(x,y_observed,model):
	model.zero_grad() # zero out the gradients from the previous steps
	y_pred = model(x) # run forward pass
	loss_val = torch_loss(y_pred, y_observed) # use the loss function to get a value
	loss_val.backward() # performs the backward pass
	return model


x = torch.as_tensor([50.0])
y_observed = torch.as_tensor([30.0])
model = TorchMLP()

torch_compute_gradient(x, y_observed, model)

print("TORCH ANSWERS")
print("Gradients of the first layer:", model.first_layer.grad)
print("Gradients of the second layer:", model.second_layer.grad)
print("Gradients of the third layer:", model.third_layer.grad)

print("OUR ANSWERS")
W0 = model.first_layer.detach().numpy()
W1 = model.second_layer.detach().numpy()
W2 = model.third_layer.detach().numpy()
print(d_loss_d_W0(mlp(x,W0,W1,W2),W1,W2,30.0))
