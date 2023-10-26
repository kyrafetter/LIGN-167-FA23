'''
LIGN 167 PSET 2
Kyra Fetter, A16819058
Lina Battikha, A16852105
Oishani Bandopadhyay, A16800427

Group Member Contributions:
Kyra: 
Lina: 
Oishani: 9, 10, 11
Everyone: 
'''

import numpy as np
import torch
from torch import nn
import torch.optim as optim


################################ BEGIN NUMPY STARTER CODE #################################################
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


def sample_logistic_distribution(x,a):
	#np.random.seed(1)
	num_samples = len(x)
	y = np.empty(num_samples)
	for i in range(num_samples):
		y[i] = np.random.binomial(1,logistic_positive_prob(x[i],a))
	return y

def create_input_values(dim,num_samples):
	#np.random.seed(100)
	x_inputs = []
	for i in range(num_samples):
		x = 10*np.random.rand(dim)-5
		x_inputs.append(x)
	return x_inputs


def create_dataset():
	x= create_input_values(2,100)
	a=np.array([12,12])
	y=sample_logistic_distribution(x,a)

	return x,y

################################ END NUMPY STARTER CODE ####################################################



################################ BEGIN PYTORCH STARTER CODE ################################################

class TorchLogisticClassifier(nn.Module):

  def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(num_features))

  def forward(self, x_vector):
    logit = torch.dot(self.weights, x_vector)
    prob = torch.sigmoid(logit)

    return prob


def loss_fn(y_predicted, y_observed):
  return -1 * y_observed * torch.log(y_predicted) - (
    1 - y_observed) * torch.log(1 - y_predicted)


def nonbatched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01):
  first_example = dataset[0]
  # first_example is a pair (x,y), where x is a vector of features and y is 0 or 1
  # note that both x and y are torch tensors
  first_example_x = first_example[0]
  first_example_y = first_example[1]
  num_features = first_example_x.size(0)

  model = TorchLogisticClassifier(num_features)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  for i in range(num_epochs):

    for d in dataset:
      d_x = d[0]
      d_y = d[1]

      optimizer.zero_grad()

      prediction = model(d_x)
      loss = loss_fn(prediction, d_y)

      loss.backward()
      optimizer.step()


def generate_nonbatched_data(num_features=3, num_examples=100):
  x_vectors = [torch.randn(num_features) for _ in range(num_examples)]
  prob_val = 0.5 * torch.ones(1)
  y_vectors = [torch.bernoulli(prob_val) for _ in range(num_examples)]

  dataset = list(zip(x_vectors, y_vectors))

  return dataset


def main():
  nonbatched_dataset = generate_nonbatched_data()
  nonbatched_gradient_descent(nonbatched_dataset)

################################ END PYTORCH STARTER CODE ###################################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES


# PROBLEM 1
def logistic_positive_prob(x,a):
    # YOUR CODE HERE

# PROBLEM 2
def logistic_derivative_per_datapoint(y_i,x_i,a,j):
    # YOUR CODE HERE

# PROBLEM 3
def logistic_partial_derivative(y,x,a,j):
    # YOUR CODE HERE

# PROBLEM 4
def compute_logistic_gradient(a,y,x):
    # YOUR CODE HERE

# PROBLEM 5
def gradient_update(a,lr,gradient):
    # YOUR CODE HERE

# PROBLEM 6
def gradient_descent_logistic(initial_a,lr,num_iterations,y,x):
    # YOUR CODE HERE

# PROBLEM 7
# Free Response Answer Here: 

# PROBLEM 8
# Free Response Answer Here: 

# PROBLEM 9
# Free Response Answer Here:
'''
Explaining the inner loop line-by-line:
    for d in dataset:
    # This loops through each element d of the dataset in this epoch
      d_x = d[0]
      # Here, since each d is a pair (x,y), d_x gives the value of the
      feature vector for the element d
      d_y = d[1]
      # Similarly, d_y gives the value of the label for the element d

      optimizer.zero_grad()
      # Explained in next problem

      prediction = model(d_x)
      # This step passes d_x through the Logistic Classifier model as defined
      # in the model function above to get the predicted output between 0 and 1
      loss = loss_fn(prediction, d_y)
      # This step calculates the loss using the loss function defined above
      # between the prediction made by the model in the preceding step and the
      # actual d_y value from the dataset element d

      loss.backward()
      # This is the backpropagation step calculates the gradient of the loss
      # with respect to the weights (and other parameters such as biases) using
      # chain rule.
      optimizer.step()
      # This step uses the gradients and previous computations to change the model 
      # parameters to try to reduce error.

These are the same steps we covered for stochastic gradient descent.
'''

# PROBLEM 10
# Free Response Answer Here:
'''
optimizer.zero_grad() sets the gradients to zero so that they do not
accumulate and update the paramters with wrong gradient calculations.
In the non-batched gradient descent, this step happens before every 
prediction is calculated, so that the predictions are updated to the latest
data point. In batched gradient descent, this step occurs before the 
predictions are made for the entire batch, the loss is aggregated within the 
batch, and then averaged over the batch size. 
Removing this from either of the functions would result in the gradient values
being accumulated and causing incorrect paramter updates.
'''

# PROBLEM 11
# Note: I used GPT-4 to check how the innermost loop should work and to explain the 
# code differences, given the code for non-batched gradient descent
def batched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    # YOUR CODE HERE
    first_example = dataset[0]
    first_example_x = first_example[0]
    first_example_y = first_example[1]
    num_features = first_example_x.size(0)
    
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = batch_start + batch_size
            batch = dataset[batch_start:batch_end]

            optimizer.zero_grad()  
		
            sum_loss = 0
            for d in batch:
                d_x = d[0]
                d_y = d[1]
                prediction = model(d_x)
                loss = loss_fn(prediction, d_y)
		    
                sum_loss += loss 

            (sum_loss/batch_size).backward() 
            optimizer.step() 

    return model
	
