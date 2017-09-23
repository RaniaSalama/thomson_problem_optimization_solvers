# Augmented Lagrangian Multiplier
import numpy as np
from numpy import random
from numpy import linalg as LA
from math import sqrt
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

def central_difference(function, x, h, mu, l, k, n):
    gx = np.zeros(k*n)
    index = 0
    for j in range(n):
        for i in range(k):
            z = np.zeros(k*n)
            z[index] = z[index] + h
            gx[index] = (function(x+z, mu, l, k, n) - function(x-z, mu, l, k, n))/(2*h)
            index = index + 1
    return gx

def check_gradient(function, gradient, iterations_number, h, mu, l, tol, n, k, min_random_number, max_random_number):
    for i in range(iterations_number):
        x = [np.random.uniform(min_random_number, max_random_number) for i in range(k*n)]
        gx = gradient(x, mu, l, k, n)
        agx = central_difference(function, x, h, mu, l, k, n)
        if abs(LA.norm(gx - agx)) >= tol:
            print("Error: Gradient is not Correct!")
            return
    print("Gradient is Correct :)")

def compute_norms(x, k, n):
    norms = np.zeros(n)
    eta = 0
    for i in range(n):
        norms[i] = LA.norm(x[:,i])
    return norms
	
def thompson_problem_alm_function(x, mu, l, k, n):
    x = np.reshape(x, (k, n), order='F')
    fx = 0
    norms = compute_norms(x, k, n)
    for i in range(n):
        for j in range(i):
            if i != j:
                fx = fx + (1/(norms[i]**2 + norms[j]**2 - 2*np.dot(x[:,i], x[:,j])))
        fx = fx + mu/2 * (norms[i]-1)**2 - l[i] * (norms[i]-1)
    return fx

def thompson_problem_alm_drivative(x, mu, l, m, norms):
    # The drivative with respect to x[w, m] for Thompson
    # problem augmented lagrangian method (alm) function.
    (k, n) = x.shape
    gx = np.zeros(k)
    for j in range(n):
            if m != j:
                gx = gx - (2.0*(x[:,m] - x[:,j]))/(norms[m]**2 + norms[j]**2 - 2*np.dot(x[:,m], x[:,j]))**2
    normx = norms[m]
    gx = gx + (mu * (normx - 1) - l[m]) * (x[:, m]/normx)
    return gx
	
def thompson_problem_alm_gradient(x, mu, l, k, n):
    # The gradient
    x = np.reshape(x, (k, n), order='F')
    norms = compute_norms(x, k, n)
    gx = np.zeros(k*n)
    index = 0
    for m in range(n):
            gx[m*k:(m+1)*k] = thompson_problem_alm_drivative(x, mu, l, m, norms)
            index = index + 1
    return gx	

def central_difference(function, x, h, mu, l, k, n):
    fx = function(x, mu, l, k, n)
    gx = np.zeros(k*n)
    index = 0
    for j in range(n):
        for i in range(k):
            z = np.zeros(k*n)
            z[index] = z[index] + h
            gx[index] = (function(x+z, mu, l, k, n) - function(x-z, mu, l, k, n))/(2*h)
            index = index + 1
    return gx
	
def check_gradient(function, gradient, iterations_number, h, mu, l, tol, n, k, min_random_number, max_random_number):
    for i in range(iterations_number):
        x = [np.random.uniform(min_random_number, max_random_number) for i in range(k*n)]
        #x = [1, 2, 3, 4]
        gx = gradient(x, mu, l, k, n)
        agx = central_difference(function, x, h, mu, l, k, n)
        if abs(LA.norm(gx - agx)) >= tol:
            print("Error: Gradient is not Correct!")
            return    
    print("Gradient is Correct :)")
	
def augmented_lagrangian_multiplier(mu, xs, l, tol, mu_step, k, n, max_iterations):
    prevx = xs
    eta = 10**(-8)
    for i in range(max_iterations):
        result = minimize(thompson_problem_alm_function, prevx,  args=(mu, l, k, n), jac=thompson_problem_alm_gradient,
                          method='L-BFGS-B', options={'gtol': eta, 'maxiter' : 10.0})
        currx = result.x
        xshaped = np.reshape(currx, (k, n), order='F')
        norms = compute_norms(xshaped, k, n)
        constrains_diff = norms - np.ones(n)
        abs_constrains_diff = abs(constrains_diff)
        if all(abs_constrains_diff <= tol):
            return currx
        eta = sum(abs_constrains_diff)
        l = l - mu * (constrains_diff)
        mu = mu + mu_step
        prevx = currx
    return prevx
	
def thomposon_function(x, k, n):
    x = np.reshape(x, (k, n), order='F')
    fx = 0
    for i in range(n):
        for j in range(i):
            if i != j:
                fx = fx + (1.0/LA.norm(x[:,i] - x[:,j])**2)
    return fx
	
def evaluate_alm_time(mu, l, tol, mu_step, k, n, max_iterations, random_iterations_number):
    time_array = np.zeros(random_iterations_number)
    f_x = np.zeros(random_iterations_number)
    xs_norm = np.zeros(random_iterations_number)
    xs = np.zeros(k*n)
    for i in range(random_iterations_number):
        print(i)
        xs = np.random.rand(k* n)
        xs = xs.tolist()
        #xs = [0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 5.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5, 1.25, 0.25]
        start_time = time.time()
        x = augmented_lagrangian_multiplier(mu, xs, l, tol, mu_step, k, n, max_iterations)
        end_time = time.time()
        time_array[i] = (end_time - start_time)
        f_x[i] = thomposon_function(x, k, n)
        xshaped = np.reshape(x, (k, n), order='F')
        xs_norm[i] = sum(abs(compute_norms(xshaped, k, n) - np.ones(n)))
        xs = x
    return time_array, f_x, xs, xs_norm

if __name__ == '__main__':

	# Check Gradient First!
	iterations_number = 100
	h = 10**(-5)
	mu = 2
	l = [10, 1, 2]
	tol = 10**(-4)
	n = 2
	k = 2
	min_random_number = 0
	max_random_number = 100
	check_gradient(thompson_problem_alm_function, thompson_problem_alm_gradient, iterations_number, h, mu, l, tol, n, k,
				   min_random_number, max_random_number)

	# Run 
	mu = 1.0
	tol = 10**(-8)
	mu_step = 100.0
	k = 2
	n = 20
	l = 10.0*np.ones(n)
	max_iterations = 100000
	random_iterations_number = 10
	time_n_k, f_x, xs, xs_norms = evaluate_alm_time(mu, l, tol, mu_step, k, n, max_iterations, random_iterations_number)
	zstar = 1.96 #Confidence Interval 95% two sided.
	time_mean = np.mean(time_n_k)
	time_std = np.std(time_n_k)
	time_ci_lower = time_mean - (zstar * time_std/(np.sqrt(random_iterations_number)))
	time_ci_upper = time_mean + (zstar * time_std/(np.sqrt(random_iterations_number)))
	print(time_mean)
	print(time_ci_lower)
	print(time_ci_upper)

	fx_mean = np.mean(f_x)
	fx_std = np.std(f_x)
	fx_ci_lower = fx_mean - (zstar * fx_std/(np.sqrt(random_iterations_number)))
	fx_ci_upper = fx_mean + (zstar * fx_std/(np.sqrt(random_iterations_number)))
	print(fx_mean)
	print(fx_ci_lower)
	print(fx_ci_upper)

	xnorms_mean = np.mean(xs_norms)
	xnorms_std = np.std(xs_norms)
	xnorms_ci_lower = xnorms_mean - (zstar * xnorms_std/(np.sqrt(random_iterations_number)))
	xnorms_ci_upper = xnorms_mean + (zstar * xnorms_std/(np.sqrt(random_iterations_number)))
	print(xnorms_mean)
	print(xnorms_ci_lower)
	print(xnorms_ci_upper)

	xsshaped = np.reshape(xs, (k, n), order='F')
	print(xsshaped)
	plt.scatter(xsshaped[0,:], xsshaped[1,:])
	plt.show()
