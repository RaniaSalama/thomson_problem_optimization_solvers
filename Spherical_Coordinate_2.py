import numpy as np
import math 
from scipy.optimize import minimize
from scipy.optimize import fmin
import numpy.matlib
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA

def sin(angle):
    return math.sin(angle)

def cos(angle):
    return math.cos(angle)

def compute_cost(X, n, k):
    X = np.array(X)
    X = np.transpose(X.reshape(n, k))
    fx = 0.0
    for i in range(n):
        repeated_col = np.matlib.repmat((np.asmatrix(X[:,i].reshape(k, 1))), 1, n)
        diff_mat = repeated_col - X
        vec_norm_sq = np.power(np.linalg.norm(diff_mat, axis = 0), 2)
        for j in range(i):
            fx += 1/vec_norm_sq[j]
    return fx

def f_x(X, n, k):
    X = np.array(X)
    fx = 0.0
    for i in range (n):
        theta_i = X[i]
        for j in  range(i):
            theta_j = X[j]
            temp = 2 *(cos(theta_i - theta_j) + -1)
            fx += 1/ temp
    return -1*fx

def spherical_coordinate(X0, tol, max_iterations, n, k):
    res = minimize(f_x, X0, args=(n, k), method='bfgs', options={'gtol': tol, 'disp': True, 'maxiter': max_iterations})
    angles = res.x
    sol = []
    i = 0
    while i < angles.shape[0]:
        theta = angles[i]
        i += 1
        sol.append([cos(theta), sin(theta)])
    x_sol = np.transpose(np.array(sol))
    return x_sol
	
def compute_eta(x, k, n):
    xshaped = np.reshape(x, (k, n), order='F')
    eta = 0
    for i in range(n):
        eta = eta + abs(LA.norm(xshaped[:,i]) - 1)
    return eta
	
def evaluate_spherical_coordinate_time(tol, k, n, max_iterations, random_iterations_number):
    time_array = np.zeros(random_iterations_number)
    fx = np.zeros(random_iterations_number)
    xs_norm = np.zeros(random_iterations_number)
    xs = np.zeros(k*n)
    for i in range(random_iterations_number):
        print(i)
        xs = np.random.rand(n) * 2 * math.pi
        xs = xs.tolist()
        #xs = [0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 5.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5, 1.25, 0.25]
        start_time = time.time()
        x = spherical_coordinate(xs, tol, max_iterations, n, k)
        end_time = time.time()
        time_array[i] = (end_time - start_time)
        fx[i] = compute_cost(x, n, k)
        xs_norm[i] = compute_eta(x, k, n)
        xs = x
    return time_array, fx, xs, xs_norm
