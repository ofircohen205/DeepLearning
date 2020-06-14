# Name: Ofir Cohen
# ID: 312255847
# Date: 22/4/2020

import numpy as np
import matplotlib.pylab as plt


def sigmoid(w, x):
	z = np.dot(x, w)
	return 1 / (1 + np.exp(-z))

def log_likelihod(x, y, w):
	z = np.dot(x, w)
	return np.sum(y * z - np.log(1 + np.exp(z)))


def logistic_regression(x, y, num_steps, learning_rate, add_interxept=False):
	if(add_interxept):
		intercept = np.ones((x.shape[0], 1))
		x = np.hstack((intercept, x))
	
	w = np.zeros(x.shape[1])
	
	for step in range(num_steps):
		wx = np.dot(w, x.T)
		est_y = sigmoid(w, x)
		err = y - est_y
		gradient = np.dot(x.T, err)
		w += learning_rate * gradient
		if step % 10000 == 0:
			print(log_likelihod(x, y, w))
		
	return w


def main():
	np.random.seed(12)
	num_observation = 5000
	means = [[1,4], [1,3], [1,2], [1,1]]

	for mean in means:
		x1 = np.random.multivariate_normal([0, 0], [[1 , 0.75], [0.75, 1]], num_observation)
		x2 = np.random.multivariate_normal(mean, [[1 ,0.75], [0.75, 1]], num_observation)
		x = np.vstack((x1, x2)).astype(np.float32)
		y = np.hstack((np.zeros(num_observation), np.ones(num_observation)))
		plt.figure(figsize=(12, 4))
		plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.4)
		plt.title("mean: {}".format(mean))
		plt.show()
		
		weights = logistic_regression(x, y, num_steps=100000, learning_rate=5e-5, add_interxept=True)
		print(weights)
		
		data_with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
		preds = np.round(sigmoid(weights, data_with_intercept))
		
		print("Accuracy from scratch {0}".format((preds == y).sum().astype(float) / len(preds)))
		
		plt.figure(figsize=(12, 8))
		plt.scatter(x[:, 0], x[:, 1], c=(preds == y)-1, alpha=.8, s=50)
		plt.show()


if __name__ == "__main__":
	main()