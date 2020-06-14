# Name: Ofir Cohen
# ID: 312255847
# Date: 24/4/2020
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


class Perceptron(object):
	
	def __init__(self, size, num_steps=1000, learning_rate=1):
		self.weights = np.zeros((1, size))
		self.errors = []
		self.learning_rate = learning_rate
		self.num_steps = num_steps
		self.classifications = { 0: 0, 1: 0 }


	def z_function(self, xi):
		'''
			This function return dot product between row in X and model weights
		'''
		z = np.dot(self.weights, xi.T)
		return z


	def fit(self, X, y):
		for step in range(self.num_steps):
			error = 0
			for xi, yi in zip(X, y):
				y_pred = self.z_function(xi)
				output = 1 if y_pred > 0 else 0
				delta = (yi - output)
				
				if delta:
					error += 1
					self.weights += delta * xi * self.learning_rate
			
			self.errors.append(error)

	def predict(self, X, y):
		y_vector = np.zeros_like(y)
		for i in range(X.shape[0]):
			y_pred = self.z_function(X[i])
			output = 1 if y_pred > 0 else 0
			delta = (y[i] - output)
			
			if delta == 0:
				if y[i] == 1:
					self.classifications[1] += 1
				elif y[i] == 0:
					self.classifications[0] += 1
			y_vector[i] = output
		
		return y_vector


def read_data(path):
	'''
		Input: path to the data file
		Output: dataset splitted to X_train, X_test, y_train, y_test
	'''
	df = pd.read_csv(path, names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'output'])
	b = np.ones((len(df),1), dtype=np.int64)
	df.insert(loc=0, column='x0(b)', value=b)
	df.loc[df['output'] == 'positive', 'output'] = 1
	df.loc[df['output'] == 'negative', 'output'] = 0
	df = shuffle(df)
	return split_train_test(df)


def split_train_test(df, train_percent=0.8):
	'''
		Input: dataset, train percent size from the dataset
		Output: dataset splitted to X_train, X_test, y_train, y_test
	'''
	train_size = int(len(df) * train_percent)
	df_train = df[:train_size]
	df_test = df[train_size:]
	
	X_train = np.asarray(df_train.drop('output', 1))
	y_train = np.asarray(df_train['output'])
	
	X_test = np.asarray(df_test.drop('output', 1))
	y_test = np.asarray(df_test['output'])
	
	return X_train, X_test, y_train, y_test



def get_test_classification_count(y):
	'''
		Input: y_test vector
		Output: how many rows in the y_test vector are in class positive and in class negative
	'''
	unique, count = np.unique(y, return_counts=True)
	classifications = dict(zip(unique, count))
	return classifications


def main():
	X_train, X_test, y_train, y_test = read_data('./data1.txt')
	classifications = get_test_classification_count(y_test)
	num_steps=5000
	learning_rate = 0.01
	perceptron = Perceptron(X_train.shape[1], num_steps=num_steps, learning_rate=learning_rate)

	perceptron.fit(X_train, y_train)
	y_pred = perceptron.predict(X_test, y_train)

	print("Number of rows that been classified as positive: {}.".format(perceptron.classifications[1]))
	print("Number of rows that are positive: {}".format(classifications[1]))
	print("Number of rows that been classified as negative: {}.".format(perceptron.classifications[0]))
	print("Number of rows that are negative: {}".format(classifications[0]))
	accuracy = (perceptron.classifications[0] + perceptron.classifications[1]) / (classifications[0] + classifications[1])
	print("Accuracy of perceptron: {}".format(accuracy))
		
	epoch = np.arange(1, num_steps+1)
	plt.figure(figsize=(20,20))
	plt.plot(epoch, perceptron.errors)
	plt.xlabel('iterations')
	plt.ylabel('error')
	plt.show()


if __name__ == "__main__":
	main()