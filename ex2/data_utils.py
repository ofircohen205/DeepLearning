# Name: Ofir Cohen
# ID: 312255847
# Date: 27/5/2020

from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os

import platform

cifar10_dir = 'cifar-10-batches-py'

def load_pickle(f):
	version = platform.python_version_tuple()
	if version[0] == '2':
		return  pickle.load(f)
	elif version[0] == '3':
		return  pickle.load(f, encoding='latin1')
	raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
	""" load single batch of cifar """
	with open(filename, 'rb') as f:
		datadict = load_pickle(f)
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
		Y = np.array(Y)
		return X, Y

def load_CIFAR10(ROOT = 'cifar-10-batches-py/'):
	""" load all of cifar """
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
		X, Y = load_CIFAR_batch(f)
		xs.append(X)
		ys.append(Y)    
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X, Y
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
	return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
	"""
	Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
	it for classifiers. These are the same steps as we used for the SVM, but
	condensed to a single function.
	"""
	# Load the raw CIFAR-10 data

	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
		
	# Subsample the data
	mask = list(range(num_training, num_training + num_validation))
	X_val = X_train[mask]
	y_val = y_train[mask]
	mask = list(range(num_training))
	X_train = X_train[mask]
	y_train = y_train[mask]
	mask = list(range(num_test))
	X_test = X_test[mask]
	y_test = y_test[mask]

	# Normalize the data: subtract the mean image
	mean_image = np.mean(X_train, axis=0)
	X_train -= mean_image
	X_val -= mean_image
	X_test -= mean_image

	# Reshape data to rows
	X_train = X_train.reshape(num_training, -1)
	X_val = X_val.reshape(num_validation, -1)
	X_test = X_test.reshape(num_test, -1)

	return X_train, y_train, X_val, y_val, X_test, y_test


def get_CIFAR10_data_SVM_softmax(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
	"""
	Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
	it for the linear classifier. These are the same steps as we used for the
	SVM, but condensed to a single function.
	"""
	# Load the raw CIFAR-10 data

	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	# subsample the data
	mask = list(range(num_training, num_training + num_validation))
	X_val = X_train[mask]
	y_val = y_train[mask]
	mask = list(range(num_training))
	X_train = X_train[mask]
	y_train = y_train[mask]
	mask = list(range(num_test))
	X_test = X_test[mask]
	y_test = y_test[mask]
	mask = np.random.choice(num_training, num_dev, replace=False)
	X_dev = X_train[mask]
	y_dev = y_train[mask]

	# Preprocessing: reshape the image data into rows
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	X_val = np.reshape(X_val, (X_val.shape[0], -1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1))
	X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

	# Normalize the data: subtract the mean image
	mean_image = np.mean(X_train, axis=0)
	X_train -= mean_image
	X_val -= mean_image
	X_test -= mean_image
	X_dev -= mean_image

	# add bias dimension and transform into columns
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
	X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

	return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev
