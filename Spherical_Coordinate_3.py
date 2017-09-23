import numpy as np
import math 
from scipy.optimize import minimize
from scipy.optimize import fmin
import numpy.matlib
import time
from numpy import linalg as LA


def sin(angle):
	return math.sin(angle)

def cos(angle):
	return math.cos(angle)


def compute_norms(x, k, n):
    norms = np.zeros(n)
    for i in range(n):
        norms[i] = LA.norm(x[:,i])
    return norms
	
def thomposon_function(x, k, n):
    x = np.reshape(x, (k, n), order='F')
    fx = 0
    for i in range(n):
        for j in range(i):
            if i != j:
                fx = fx + (1.0/LA.norm(x[:,i] - x[:,j])**2)
    return fx	
	
def compute_cost(X, n, k):
	X = np.array(X)
	X = np.transpose(X.reshape(n, k))
	fx = 0.0
	for i in range(n):
		repeated_col = np.matlib.repmat((np.asmatrix(X[:,i].reshape(k, 1))), 1, n)
		diff_mat = repeated_col - X
		vec_norm_sq = np.power(np.linalg.norm(diff_mat, axis = 0), 2)
		temp = 0
		for j in range(i):
			temp += 1/vec_norm_sq[j]		
		fx = fx + temp
	return fx


def f_x(X, n, k):
	X = np.array(X)
	X = np.transpose(X.reshape(n, 2))
	fx = 0.0
	for i in range (n):
		phi_i = X[0, i]
		theta_i = X[1, i]
		for j in  range(i):
			phi_j = X[0, j]
			theta_j = X[1, j]
			temp = 2 *(sin(phi_i) * sin(phi_j) * cos(theta_i - theta_j) + cos(phi_i) * cos(phi_j) -1)
			fx += 1/ temp
	return -1*fx


def spherical_coordinates_3(k, n):
	X0 = np.random.rand(2* n) * 2 * math.pi
	X0 = X0.tolist()
	res = minimize(f_x, X0, args=(n, k), method='L-BFGS-B', options={'gtol': 1e-4, 'disp': True, 'maxiter': 1000}) #100000			
	angles = res.x
	sol = []
	i = 0
	while i < angles.shape[0]:
		phi = angles[i]
		i += 1
		theta = angles[i]
		i += 1
		# print phi, theta
		sol.append([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)])
	x_sol = np.transpose(np.array(sol))
	return x_sol

def evaluate_spherical_coordinates_time(tol, k, n, max_iterations, random_iterations_number):
    time_array = np.zeros(random_iterations_number)
    f_x = np.zeros(random_iterations_number)
    xs_norm = np.zeros(random_iterations_number)
    xs = np.zeros(k*n)
    for i in range(random_iterations_number):
        xs = np.random.rand(k* n)
        xs = xs.tolist()
        #xs = [0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 5.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5, 1.25, 0.25]
        start_time = time.time()
        x = spherical_coordinates_3(k, n)
        end_time = time.time()
        time_array[i] = (end_time - start_time)
        f_x[i] = thomposon_function(x, k, n)
        xshaped = np.reshape(x, (k, n), order='F')
        xs_norm[i] = sum(abs(compute_norms(xshaped, k, n) - np.ones(n)))
        xs = x
    return time_array, f_x, xs, xs_norm
		
	
	