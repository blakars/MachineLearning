# V2A1_LinearRegression.py 
# Programmgeruest zu Versuch 2, Aufgabe 1
import numpy as np 
import matplotlib.pyplot as plt

def fun_true(X):                              # compute 1-dim. parable function; X must be Nx1 data matrix
    w2,w1,w0 = 3.0,-1.0,2.0                   # true parameters of parable y(x)=w0+w1*x+w2*x*x
    return w0+w1*X+w2*np.multiply(X,X)        # return function values (same size as X)

def generateDataSet(N,xmin,xmax,sd_noise):    # generate data matrix X and target values T
    X=xmin+np.random.rand(N,1)*(xmax-xmin)    # get random x values uniformly in [xmin;xmax)
    T=fun_true(X);                            # target values without noise
    if(sd_noise>0):
        T=T+np.random.normal(0,sd_noise,X.shape) # add noise 
    return X,T

def getDataError(Y,T):                        # compute data error (least squares) between prediction Y and true target values T
    D=np.multiply(Y-T,Y-T);                   # squared differences between Y and T
    return 0.5*sum(sum(D));                   # return least-squares data error function E_D

def phi_polynomial(x,deg=1):                            # compute polynomial basis function vector phi(x) for data x 
    assert(np.shape(x)==(1,)), "currently only 1dim data supported"
    return np.array([x[0]**i for i in range(deg+1)]).T; # returns feature vector phi(x)=[1 x x**2 x**3 ... x**deg]

# (I) generate data 
np.random.seed(10)                            # set seed of random generator (to be able to regenerate data)
N=10                                          # number of data samples
xmin,xmax=-5.0,5.0                            # x limits
sd_noise=10                                   # standard deviation of Guassian noise
X,T           = generateDataSet(N, xmin,xmax, sd_noise)             # generate training data
X_test,T_test = generateDataSet(N, xmin,xmax, sd_noise)             # generate test data
print("X=",X, "T=",T)

# (II) generate linear least squares model for regression
lmbda=0                                                           # no regression
deg=5                                                             # degree of polynomial basis functions
N,D = np.shape(X)                                                 # shape of data matrix X
N,K = np.shape(T)                                                 # shape of target value matrix T
PHI = np.array([phi_polynomial(X[i],deg).T for i in range(N)])    # generate design matrix
N,M = np.shape(PHI)                                               # shape of design matrix
print("PHI=", PHI)
W_LSR = np.zeros((M,1))                                           # REPLACE THIS BY REGULARIZED LEAST SQUARES WEIGHTS!  
print("W_LSR=",W_LSR)

# (III) make predictions for training and test data
Y_train = np.zeros((N,1))  # REPLACE THIS BY PROGNOSIS FOR TRAINING DATA X! (result should be N x 1 matrix, i.e., one prognosis per row)
Y_test = np.zeros((N,1))   # REPLACE THIS BY PROGNOSIS FOR TEST DATA X_test! (result should be N x 1 matrix, i.e., one prognosis per row)
print("Y_test=",Y_test)
print("T_test=",T_test)
print("training data error = ", getDataError(Y_train,T))
print("test data error = ", getDataError(Y_test,T_test))
print("W_LSR=",W_LSR)
print("mean weight = ", np.mean(np.mean(np.abs(W_LSR))))

# (IV) plot data
ymin,ymax = -50.0,150.0                     # interval of y data
x_=np.arange(xmin,xmax,0.01)                # densely sampled x values
Y_LSR = np.array([np.dot(W_LSR.T,np.array([phi_polynomial([x],deg)]).T)[0] for x in x_]);   # least squares prediction
Y_true = fun_true(x_).flat

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X.flat,T.flat,c='g',marker='x',s=100)             # plot learning data points (green x)
ax.scatter(X_test.flat,T_test.flat,c='g',marker='.',s=100)   # plot test data points (green .)
ax.plot(x_,Y_LSR.flat, c='r')         # plot LSR regression curve (red)
ax.plot(x_,Y_true, c='g')             # plot true function curve (green)
ax.set_xlabel('x')                    # label on x-axis
ax.set_ylabel('y')                    # label on y-axis
ax.grid()                             # draw a grid
plt.ylim((ymin,ymax))                 # set y-limits
plt.show()                            # show plot on screen

