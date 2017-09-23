import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin
import numpy.matlib
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time


def calculate_norms(x, k, n):
	xshaped = np.reshape(x, (k, n), order='F')
	norms = np.zeros(n)
	for i in range(n):
	    norms[i] = LA.norm(xshaped[:,i])
	return norms  

def f_x (X, n, k, alpha):
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
	norm_term = sum(np.power(np.linalg.norm(X, axis = 0) - np.ones(n), 2))
	fx = fx + (alpha /2) * norm_term
	return fx	


def penlaty_method(x0, n, k, alpha, alpha_step, tol, num_iterations):
    list_val = {}
    for i in range(num_iterations):
        # nelder-mead
        # bfgs
        #L-BFGS-B
        res = minimize(f_x, x0, args=(n, k, alpha), method='Nelder-Mead', options={'gtol': 1e-4, 'disp': False, 'maxiter': 10})
        list_val[res.fun] = (alpha, res.fun, res.x)
        x0 = res.x
        cur_f = res.fun
        res.clear()
        alpha += alpha_step
        norms = calculate_norms(x0, k, n)
        if all(abs(norms - np.ones(n)) <= tol):
            #print("Norms conditions are met!")
            break
    return x0
	
def compute_eta(x, k, n):
    xshaped = np.reshape(x, (k, n), order='F')
    eta = 0
    for i in range(n):
        eta = eta + abs(LA.norm(xshaped[:,i]) - 1)
    return eta
	
def evaluate_penalty_time(alpha, tol, alpha_step, k, n, max_iterations, random_iterations_number):
    time_array = np.zeros(random_iterations_number)
    fx = np.zeros(random_iterations_number)
    xs_norm = np.zeros(random_iterations_number)
    xs = np.zeros(k*n)
    for i in range(random_iterations_number):
        print(i)
        xs = np.random.rand(k* n)
        xs = xs.tolist()
        #xs = [0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 5.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5, 1.25, 0.25]
        start_time = time.time()
        x = penlaty_method(xs, n, k, alpha, alpha_step, tol, max_iterations)
        end_time = time.time()
        time_array[i] = (end_time - start_time)
        fx[i] = f_x(x, n, k, 0)
        xs_norm[i] = compute_eta(x, k, n)
        xs = x
    return time_array, fx, xs, xs_norm


if __name__ == '__main__':

	n = 10
	k = 2
	alpha = 1
	alpha_step = 100
	num_iterations = 1000
	x0 = np.random.rand(k* n)
	x0 = x0.tolist()
	tol = 10**(-4)
	random_iterations_number = 10
	time_n_k, fx, xs, xs_norms = evaluate_penalty_time(alpha, tol, alpha_step, k, n, num_iterations, random_iterations_number)

	zstar = 1.96 #Confidence Interval 95% two sided.
	time_mean = np.mean(time_n_k)
	time_std = np.std(time_n_k)
	time_ci_lower = time_mean - (zstar * time_std/(np.sqrt(random_iterations_number)))
	time_ci_upper = time_mean + (zstar * time_std/(np.sqrt(random_iterations_number)))
	print(time_mean, '\t', time_ci_lower, '\t', time_ci_upper)

	fx_mean = np.mean(fx)
	fx_std = np.std(fx)
	fx_ci_lower = fx_mean - (zstar * fx_std/(np.sqrt(random_iterations_number)))
	fx_ci_upper = fx_mean + (zstar * fx_std/(np.sqrt(random_iterations_number)))
	print(fx_mean, '\t', fx_ci_lower, '\t', fx_ci_upper)

	xnorms_mean = np.mean(xs_norms)
	xnorms_std = np.std(xs_norms)
	xnorms_ci_lower = xnorms_mean - (zstar * xnorms_std/(np.sqrt(random_iterations_number)))
	xnorms_ci_upper = xnorms_mean + (zstar * xnorms_std/(np.sqrt(random_iterations_number)))
	print(xnorms_mean, '\t', xnorms_ci_lower, '\t', xnorms_ci_upper)

	xsshaped = np.reshape(xs, (k, n), order='F')
	print(xsshaped)
	plt.scatter(xsshaped[0,:], xsshaped[1,:])
	plt.show()
