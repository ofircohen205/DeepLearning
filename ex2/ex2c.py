# Name: Ofir Cohen
# ID: 312255847
# Date: 27/5/2020

import numpy as np
import matplotlib.pylab as plt


## Ex0 (a) find the minimums of x^2  using Gradient Descent

# square of x
def x2(x):
	return (x*x)

# derivative of x2
def x2_(x):
	#your code here:
	return 2*x

# starting point 
X = 10

#your code here:
lr = 10**-4

num_of_steps = 10**5
for i in range(num_of_steps):
	#your code here:
	dx = x2_(X)
	X +=  - lr * dx

"""## Ex0 (b) find the minimums of x^4 using Gradient Descent"""

# x to the power of 4
def x4(x):
	return (x*x*x*x)

# derivative of x4
def x4_(x):
	#your code here:
	return 4*x*x*x

# starting point 
X = 10

#your code here:
lr = 10**-4

num_of_steps = 10**5
for i in range(num_of_steps):
	#your code here:
	dx = x4_(X)
	X +=  - lr * dx

"""## Ex1 - find the minimums of x^2 and x^4 using the Momentum methos and compare it to Gradient Descent"""

# starting point for the Gradient Descent
X2 = 10
X4 = 10

# starting point for the Momentum methos
X2m = 10
X4m = 10

#your code here:
lr = 10**-4

# your code here (Find the appropriate learning rate, one that works for both functions)
lrm = 10**-4
mu = 0.9
v2m, v4m = 0, 0

#your code here:
num_of_steps = 10**5
for i in range(num_of_steps):
	#your code here:
	print("X2:{} \t X4:{} \t X2m:{} \t X4m:{}".format(X2,X4,X2m,X4m))
	dx2 = x2_(X2)
	X2 +=  - lr * dx2
	
	dx4 = x4_(X4)
	X4 +=  - lr * dx4
	
	dx2m = x2_(X2m)
	v2m = mu * v2m - lrm * dx2m
	X2m += v2m
	
	dx4m = x4_(X4m)
	v4m = mu * v4m - lrm * dx4m
	X4m += v4m