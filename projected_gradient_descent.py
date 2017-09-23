import numpy as np
from numpy import random
from numpy import linalg as LA
from math import sqrt
from scipy.optimize import optimize
import time
import matplotlib.pyplot as plt

def central_difference(function, x, h, k, n):
    gx = np.zeros(k*n)
    index = 0
    for j in range(n):
        for i in range(k):
            z = np.zeros(k*n)
            z[index] = z[index] + h
            gx[index] = (function(x+z, k, n) - function(x-z, k, n))/(2*h)
            index = index + 1
    return gx

def check_gradient(function, gradient, iterations_number, h, tol, n, k, min_random_number, max_random_number):
    for i in range(iterations_number):
        x = [np.random.uniform(min_random_number, max_random_number) for i in range(k*n)]
        gx = gradient(x, k, n)
        agx = central_difference(function, x, h, k, n)
        if abs(LA.norm(gx - agx)) >= tol:
            print("Error: Gradient is not Correct!")
            return
    print("Gradient is Correct :)")

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
	
def thompson_problem_drivative(x, m, w, norms):
    gx = 0
    (k, n) = x.shape
    for j in range(n):
            if m != j:
                dem = (norms[m]**2 + norms[j]**2 - 2*np.dot(x[:,m], x[:,j]))**2
                if(dem > 10**(-8)):
                    gx = gx - (2.0*(x[w,m] - x[w,j]))/dem
    return gx
	
def thompson_problem_gradient(x, k, n):
    # The gradient
    x = np.reshape(x, (k, n), order='F')
    norms = compute_norms(x, k, n)
    gx = np.zeros(k*n)
    index = 0
    for m in range(n):
        for w in range(k): 
            gx[index] = thompson_problem_drivative(x, m, w, norms)
            index = index + 1
    return gx
	
def central_difference(function, x, h, k, n):
    fx = function(x, k, n)
    gx = np.zeros(k*n)
    index = 0
    for j in range(n):
        for i in range(k):
            z = np.zeros(k*n)
            z[index] = z[index] + h
            gx[index] = (function(x+z, k, n) - function(x-z, k, n))/(2*h)
            index = index + 1
    return gx
	
def check_gradient(function, gradient, iterations_number, h, tol, n, k, min_random_number, max_random_number):
    for i in range(iterations_number):
        x = [np.random.uniform(min_random_number, max_random_number) for i in range(k*n)]
        #x = [1, 2, 3, 4]
        gx = gradient(x, k, n)
        agx = central_difference(function, x, h, k, n)
        if abs(LA.norm(gx - agx)) >= tol:
            print("Error: Gradient is not Correct!")
            return    
    print("Gradient is Correct :)")
	
def projected_gradient_descent(max_iter, xs, k, n, tol):
    currx = xs
    currgradient = thompson_problem_gradient(currx, k, n)
    for iter in range(max_iter):
        res = optimize.line_search(thomposon_function,thompson_problem_gradient, args=(k, n), xk=currx, pk=currgradient)
        currx = currx - res[0] * currgradient
        # Project currx by normalizing it!
        currgradient = thompson_problem_gradient(currx, k, n)
        if all(abs(currgradient) <= tol):
            break  
    x = np.reshape(currx, (k, n), order='F')
    norms = compute_norms(x, k, n)
    index = 0
    for i in range(n):
        for j in range(k):
            currx[index] = currx[index] / norms[i]
            index = index + 1    
    return currx
	
def evaluate_pgs_time(tol, k, n, max_iterations, random_iterations_number):
    time_array = np.zeros(random_iterations_number)
    f_x = np.zeros(random_iterations_number)
    xs_norm = np.zeros(random_iterations_number)
    xs = np.zeros(k*n)
    for i in range(random_iterations_number):
        xs = np.random.rand(k* n)
        xs = xs.tolist()
        #xs = [0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 5.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5, 1.25, 0.25]
        start_time = time.time()
        x = projected_gradient_descent(max_iterations, xs, k, n, tol)
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
	tol = 10**(-4)
	n = 2
	k = 2
	min_random_number = 0
	max_random_number = 100
	check_gradient(thomposon_function, thompson_problem_gradient, iterations_number, h, tol, n, k,
				   min_random_number, max_random_number)

	prefix = ''
	n_vals = [2]
	n_vals.extend(range(10, 110, 10))
	tol = 10**(-8)
	k = 2
	max_iterations = 10000    
	time_mean_list = []
	time_ci_lower_list = []
	time_ci_upper_list = []

	fx_mean_list = []
	fx_ci_lower_list = []
	fx_ci_upper_list = []

	xnorms_mean_list = []
	xnorms_ci_lower_list = []
	xnorms_ci_upper_list = []

	random_iterations_number = 10
	for n in n_vals:
		time_n_k, f_x, xs, xs_norms = evaluate_pgs_time(tol, k, n, max_iterations, random_iterations_number)
		zstar = 1.96 #Confidence Interval 95% two sided.
		time_mean = np.mean(time_n_k)
		time_std = np.std(time_n_k)
		time_ci_lower = time_mean - (zstar * time_std/(np.sqrt(random_iterations_number)))
		time_ci_upper = time_mean + (zstar * time_std/(np.sqrt(random_iterations_number)))
		# print(time_mean)
		# print(time_ci_lower)
		# print(time_ci_upper)
		time_mean_list.append(time_mean)
		time_ci_lower_list.append(time_ci_lower)
		time_ci_upper_list.append(time_ci_upper)

		fx_mean = np.mean(f_x)
		fx_std = np.std(f_x)
		fx_ci_lower = fx_mean - (zstar * fx_std/(np.sqrt(random_iterations_number)))
		fx_ci_upper = fx_mean + (zstar * fx_std/(np.sqrt(random_iterations_number)))
		# print(fx_mean)
		# print(fx_ci_lower)
		# print(fx_ci_upper)
		fx_mean_list.append(fx_mean)
		fx_ci_lower_list.append(fx_ci_lower)
		fx_ci_upper_list.append(fx_ci_upper)

		xnorms_mean = np.mean(xs_norms)
		xnorms_std = np.std(xs_norms)
		xnorms_ci_lower = xnorms_mean - (zstar * xnorms_std/(np.sqrt(random_iterations_number)))
		xnorms_ci_upper = xnorms_mean + (zstar * xnorms_std/(np.sqrt(random_iterations_number)))
		# print(xnorms_mean)
		# print(xnorms_ci_lower)
		# print(xnorms_ci_upper)
		xnorms_mean_list.append(xnorms_mean)
		xnorms_ci_lower_list.append(xnorms_ci_lower)
		xnorms_ci_upper_list.append(xnorms_ci_upper)

		xsshaped = np.reshape(xs, (k, n), order='F')
		plt.scatter(xsshaped[0,:], xsshaped[1,:])
		plt.savefig("projected_n_%d_k_%d.png" % (prefix,n, k))
		plt.clf()
	plt.errorbar(n_vals, fx_mean_list, yerr= fx_ci_upper)   
	plt.xlabel('n')
	plt.ylabel('fx') 
	plt.savefig("projected_fx_k_%d.png" % (k))
	plt.clf()

	plt.errorbar(n_vals, time_mean_list, yerr= time_ci_upper)   
	plt.xlabel('n')
	plt.ylabel('time(sec)') 
	plt.savefig("projected_time_k_%d.png" % (k))
	plt.clf()

	plt.errorbar(n_vals, xnorms_mean_list, yerr= xnorms_ci_upper)
	plt.xlabel('n')
	plt.ylabel('norm')
	plt.savefig("projected_norms_k_%d.png" % (k))
	plt.clf()

	f = open("projected_k_%d.txt" % (k), 'w')
	f.write('fx\n')    
	for n in n_vals:
		f.write("%d\t" %(n))
	f.write('\n')    

	f.write('n\n')
	for f_mean in fx_mean_list:
		f.write("%f\t" %(f_mean))
	f.write('\n')    
	for f_ci_lower in fx_ci_lower_list:
		f.write("%f\t" %(f_ci_lower))
	f.write('\n')    
	for f_ci_upper in fx_ci_upper_list:
		f.write("%f\t" %(f_ci_upper))
	f.write('\n')    

	f.write('Norms\n')
	for norm_mean in xnorms_mean_list:
		f.write("%.17f\t" %(norm_mean))
	f.write('\n')    
	for norm_ci_lower in xnorms_ci_lower_list:
		f.write("%.17f\t" %(norm_ci_lower))
	f.write('\n')    
	for norm_ci_upper in xnorms_ci_upper_list:
		f.write("%.17f\t" %(norm_ci_upper))
	f.write('\n')    

	f.write('Time(sec)\n')
	for time_mean in time_mean_list:
		f.write("%f\t" %(time_mean))
	f.write('\n')    

	for time_ci_lower in time_ci_lower_list:
		f.write("%f\t" %(time_ci_lower))
	f.write('\n')    

	for time_ci_upper in time_ci_upper_list:
		f.write("%f\t" %(time_ci_upper))
	f.write('\n')        

	f.close()    