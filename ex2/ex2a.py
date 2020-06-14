# Name: Ofir Cohen
# ID: 312255847
# Date: 27/5/2020

import numpy as np
import matplotlib.pylab as plt

#   A very simple neural network to do exclusive or.

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_(x):
	return x * (1 - x)


def execute_xor_neural_network(X, Y, weights, epochs=1, learning_rate=0.1):
	for i in range(epochs):
		# First layer
		L1 = np.dot(X, weights['Wh'])
		print(L1.shape == (4,3))
		
		# Sigmoid first layer results
		H = sigmoid(L1)
		print(H.shape == (4,3))

		# Second layer
		L2 = np.dot(H, weights['Wz'])
		print(L2.shape == (4,1))

		# Sigmoid second layer results
		Z = sigmoid(L2)
		print(Z.shape == (4,1))

		# Error
		E = Y - Z
		print(E.shape == (4,1))

		# Gradient Z
		dZ = E * sigmoid_(Z)
		print(dZ.shape == (4,1))

		# Gradient H
		dH = dZ.dot(weights['Wz'].T) * sigmoid_(H)
		print(dH.shape == (4,3))
		
		# Update output layer weights
		weights['Wz'] += H.T.dot(dZ) * learning_rate
		print(weights['Wz'].shape == (3,1))

		# Update first layer weights
		weights['Wh'] += X.T.dot(dH) * learning_rate
		print(weights['Wh'].shape == (3,3))

		print(Z[0] < 0.05)  # what have we learnt?
		print(Z[1] > 0.95)  # what have we learnt?
		print(Z[2] > 0.95)  # what have we learnt?
		print(Z[3] < 0.05)  # what have we learnt?



def main():
	inputLayerSize, hiddenLayerSize, outputLayerSize = 3, 3, 1

	# prepare the dataset
	X = np.array([[0, 0, 1], [0, 1, 1], [1, 0,1], [1, 1, 1]])
	Y = np.array([[0], [1], [1], [0]])
	

	# Test your sigmoid and sigmoid_ implementation	
	print(sigmoid(-10)-0.9999 < 6e-4)
	print(sigmoid(10)-0.9999 < 6e-4)
	print(sigmoid(0)==0.5)
	print(sigmoid_(0)==0.25)

	"""# Train the network (Forward + backword)"""

	X = np.array([[0, 0, 1], [0, 1, 1], [1, 0,1], [1, 1, 1]])
	Y = np.array([[0], [1], [1], [0]])
	Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
	Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
	learning_rate = 1
	execute_xor_neural_network(X=X, Y=Y, weights={ 'Wh': Wh, 'Wz': Wz }, epochs=2000, learning_rate=learning_rate)



if __name__ == "__main__":
	main()