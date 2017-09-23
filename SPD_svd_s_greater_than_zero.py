# Problem data.
# s >= 0
# SVD
from util import *
def SPD_svd_s_greater_than_zero(k, n):
    np.random.seed(1)
    # Construct the problem.
    s = Variable(n*n)
    objective = Minimize(f_s(s, n))
    A = np.zeros((n, n*n))
    for i in range(n):
        A[i,i*n+i] = 1

    b = np.ones((n, 1))

    c = np.zeros((n, n*n))
    j = 0
    for i in range(n):
        while (j < n*(i+1)):
            c[i,j] = 1
            j = j + 1
    d =  0*np.ones((n, 1))

    constraints = [A*s == b, s >= 0, s <= 1]
    prob = Problem(objective, constraints)

    print("Optimal value", prob.solve())
    smatrix = np.reshape(s.value,(n, n))
    u, sigma, v = np.linalg.svd(smatrix)
    xsshaped = np.diag((sigma**(0.5)))[0:k,0:k]*v[0:k,:]
    for i in range(n):
        xsshaped[:,i] = xsshaped[:,i]/LA.norm(xsshaped[:,i])
    plt.scatter(xsshaped[0,:], xsshaped[1,:])
    def Circle(x,y):
        return (x*x+y*y)

    xx=np.linspace(-1,1,400)
    yy=np.linspace(-1,1,400)
    [X,Y]=np.meshgrid(xx,yy)

    Z=Circle(X,Y)
    plt.contour(X,Y,Z,1)
    plt.savefig('SPD_svd_s_greater_than_zero.png')
    return xsshaped

def evaluate_SPD_svd_s_greater_than_zero(k, n):
    random_iterations_number = 1
    time_array = np.zeros(random_iterations_number)
    fx = np.zeros(random_iterations_number)
    xs_norm = np.zeros(random_iterations_number)
    xs = np.zeros(k*n)
    for i in range(random_iterations_number):
        # print "\t", i
        xs = np.random.rand(k* n)
        xs = xs.tolist()
        #xs = [0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 5.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5, 1.25, 0.25]
        start_time = time.time()
        x = SPD_svd_s_greater_than_zero(k, n)
        end_time = time.time()
        time_array[i] = (end_time - start_time)
        fx[i] = f_x(x, k, n)
        xs_norm[i] = sum(np.abs(compute_norms(x, k, n) - np.ones(n)))
        xs = x
    return time_array, fx, xs, xs_norm

if __name__ == '__main__':
    n = 2
    k = 3
    SPD_svd_s_greater_than_zero(k, n)         