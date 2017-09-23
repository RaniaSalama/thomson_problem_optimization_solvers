import numpy as np
from numpy import random
from numpy import linalg as LA
from math import sqrt
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
	n_vals = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
	for i in range(11):
		n = n_vals[i]
		xsshaped = np.loadtxt('plot_k'+str(n)+'.txt', dtype='f', delimiter=' ')
		xsshaped = np.transpose(xsshaped)
		deleted_count = 0;
		for i in range(n):
			if(xsshaped[0,i] <= -1.5 or xsshaped[1,i] <= -1.5 or xsshaped[2,i] <= -1.5
			or xsshaped[0,i] >= 1.5 or xsshaped[1,i] >= 1.5 or xsshaped[2,i] >= 1.5):
				deleted_count = deleted_count + 1
		(k, n) = xsshaped.shape
		xsshaped_new = np.zeros([k, n - deleted_count])
		index = 0
		for i in range(n):
			if(xsshaped[0,i] <= -1.5 or xsshaped[1,i] <= -1.5 or xsshaped[2,i] <= -1.5
			or xsshaped[0,i] >= 1.5 or xsshaped[1,i] >= 1.5 or xsshaped[2,i] >= 1.5):
				continue
			xsshaped_new[:, index] = xsshaped[:, i]
			index = index + 1
		print(xsshaped_new.shape)
		xsshaped = xsshaped_new
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		s = [100 for ni in range(n)]
		ax.scatter(xsshaped[0,:], xsshaped[1,:], xsshaped[2,:], color="r", s = s)
		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
		x = np.cos(u)*np.sin(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(v)
		
		ax.plot_wireframe(x, y, z, color="b")
		ax.set_xlim(-1, 1)
		ax.set_ylim(-1, 1)
		plt.savefig('plot_k'+str(n)+'.png')
		plt.clf()
