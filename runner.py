import numpy as np
from numpy import random
from numpy import linalg as LA
from math import sqrt
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

from alm import evaluate_alm_time
from Spherical_Coordinate_2 import evaluate_spherical_coordinate_time
from Spherical_Coordinate_3 import evaluate_spherical_coordinates_time
from penalty import evaluate_penalty_time
from penalty_nealder_mead import evaluate_penalty_time
from projected_gradient_descent import evaluate_pgs_time


from SPD_chelosky_s_greater_than_neg_1 import evaluate_SPD_chelosky_s_greater_than_neg_1
from SPD_chelosky_s_greater_than_zero import evaluate_SPD_chelosky_s_greater_than_zero
from SPD_svd_s_greater_than_neg_1 import evaluate_SPD_svd_s_greater_than_neg_1
from SPD_svd_s_greater_than_zero import evaluate_SPD_svd_s_greater_than_zero
from SPD_svd_s_greater_than_neg_1_and_restrict_sim_sum import evaluate_SPD_svd_s_greater_than_neg_1_and_restrict_sim_sum
from mpl_toolkits.mplot3d import Axes3D 
import sys

if __name__ == '__main__':

    
    prefix = 'SPD_svd_s_greater_than_neg_1_and_restrict_sim_sum'
    dir = os.getcwd()
    if not os.path.exists(prefix):
        os.makedirs(dis+"/"+prefix)
    n_vals = [2]
    n_vals.extend(range(10, 110, 10))
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

    # random_iterations_number = 5
    random_iterations_number = 1

    #For ALM
    mu = 1.0
    tol = 10**(-4)
    mu_step = 500.0
    k = 2
    # n = 20
    
    for n in n_vals:
        # print(n)
        l = 10.0*np.ones(n)
        # time_n_k, f_x, xs, xs_norms = evaluate_alm_time(tol, k, n, max_iterations, random_iterations_number)
        # time_n_k, f_x, xs, xs_norms = evaluate_alm_time(mu, l, tol, mu_step, k, n, max_iterations, random_iterations_number)

        # num_iterations = 1000
        # time_n_k, f_x, xs, xs_norms = evaluate_spherical_coordinate_time(tol, k, n, num_iterations, random_iterations_number)

        alpha = 1
        alpha_step = 100
        num_iterations = 200
        
        # time_n_k, f_x, xs, xs_norms = evaluate_penalty_time(alpha, tol, alpha_step, k, n, num_iterations, random_iterations_number)

        # time_n_k, f_x, xs, xs_norms = evaluate_SPD_chelosky_s_greater_than_neg_1(k, n)
        # time_n_k, f_x, xs, xs_norms = evaluate_SPD_chelosky_s_greater_than_zero(k, n)
        # time_n_k, f_x, xs, xs_norms = evaluate_SPD_svd_s_greater_than_neg_1(k, n)
        # time_n_k, f_x, xs, xs_norms = evaluate_SPD_svd_s_greater_than_zero(k, n)
        time_n_k, f_x, xs, xs_norms = evaluate_SPD_svd_s_greater_than_neg_1_and_restrict_sim_sum(k, n)

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
        plt.savefig("%s/projected_n_%d_k_%d.png" % (prefix,n, k))
        plt.clf()
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xsshaped[0,:], xsshaped[1,:], xsshaped[2,:], color = 'r', s=100)
        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        # x = np.cos(u)*np.sin(v)
        # y = np.sin(u)*np.sin(v)
        # z = np.cos(v)
        # ax.plot_wireframe(x, y, z, color="b")
        # plt.savefig("%s/projected_n_%d_k_%d.png" % (prefix,n, k))
        # plt.clf()

        print(n, time_mean, time_ci_lower, time_ci_upper, fx_mean, fx_ci_lower, fx_ci_upper, xnorms_mean, xnorms_ci_lower, xnorms_ci_upper)
        sys.stdout.flush()
    plt.errorbar(n_vals, fx_mean_list, yerr= fx_ci_upper)   
    plt.xlabel('n')
    plt.ylabel('fx') 
    plt.savefig("%s/projected_fx_k_%d.png" % (prefix, k))
    plt.clf()

    plt.errorbar(n_vals, time_mean_list, yerr= time_ci_upper)   
    plt.xlabel('n')
    plt.ylabel('time(sec)') 
    plt.savefig("%s/projected_time_k_%d.png" % (prefix, k))
    plt.clf()

    plt.errorbar(n_vals, xnorms_mean_list, yerr= xnorms_ci_upper)
    plt.xlabel('n')
    plt.ylabel('norm')
    plt.savefig("%s/projected_norms_k_%d.png" % (prefix, k))
    plt.clf()

    f = open("%s/projected_k_%d.txt" % (prefix, k), 'w')
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