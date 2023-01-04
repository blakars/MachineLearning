#!/usr/bin/env python
# V2A2_Regression.py
# Programmgeruest zu Versuch 2, Aufgabe 2

import numpy as np
import scipy.spatial
from random import randint

# ----------------------------------------------------------------------------------------- 
# base class for regressifiers
# ----------------------------------------------------------------------------------------- 
class Regressifier:
    """
    Abstract base class for regressifiers
    Inherit from this class to implement a concrete regression algorithm
    """

    def fit(self,X,T):        # train/compute regression with lists of feature vectors X and class labels T
        """
        Train regressifier by training data X, T, should be overwritten by any derived class
        :param X: Data matrix of size NxD, contains in each row a data vector of size D
        :param T: Target vector matrix of size NxK, contains in each row a target vector of size K
        :returns: -
        """
        pass

    def predict(self,x):      # predict a target vector given the data vector x 
        """
        Implementation of the regression algorithm; should be overwritten by any derived class 
        :param x: test data vector of size D
        :returns: predicted target vector
        """
        return None           

    def crossvalidate(self,S,X,T,dist=lambda t: np.linalg.norm(t)):  # do a S-fold cross validation 
        """
        Do a S-fold cross validation
        :param S: Number of parts the data set is divided into
        :param X: Data matrix (one data vector per row)
        :param T: Matrix of target vectors; T[n] is target vector of X[n]
        :param dist: a fuction dist(t) returning the length of vector t (default=Euklidean)
        :returns (E_dist,sd_dist,E_min,E_max) : mean, standard deviation, minimum, and maximum of absolute error 
        :returns (Erel_dist,sdrel_dist,Erel_min,Erel_max) : mean, standard deviation, minimum, and maximum of relative error 
        """
        X,T=np.array(X),np.array(T)                         # ensure array type
        N=len(X)                                            # N=number of data vectors
        perm = np.random.permutation(N)                     # do a random permutation of X and T...
        X1,T1=[X[i] for i in perm], [T[i] for i in perm]    # ... to get random partitions of the data set
        idxS = [range(i*N//S,(i+1)*N//S) for i in range(S)] # divide data set into S parts:
        E_dist,E_dist2,E_max,E_min=0,0,-1,-1                # initialize first two moments of (absolute) regression error as well as max/min error 
        Erel_dist,Erel_dist2,Erel_max,Erel_min=0,0,-1,-1    # initialize first two moments of relative regression error as well as max/min error 
        for idxTest in idxS:                                # loop over all possible test data sets
            # (i) generate training and testing data sets and train classifier        
            idxLearn = [i for i in range(N) if i not in idxTest]                      # remaining indices (not in idxTest) are learning data
            if(S<=1): idxLearn=idxTest                                                # if S==1 use entire data set for learning and testing
            X_learn, T_learn = np.array([X1[i] for i in idxLearn]), np.array([T1[i] for i in idxLearn]) # learn data 
            X_test , T_test  = np.array([X1[i] for i in idxTest ]), np.array([T1[i] for i in idxTest ]) # test data 
            self.fit(X_learn,T_learn)                       # train regressifier
            # (ii) test regressifier
            for i in range(len(X_test)):  # loop over all data vectors to be tested
                # (ii.a) regress for i-th test vector
                xn_test = X_test[i].T                           # data vector for testing
                t_test = self.predict(xn_test)                  # predict target value for given test vector 
                # (ii.b) check for regression errors
                t_true = T_test[i].T                            # true target value
                d=dist(t_test-t_true)                           # (Euklidean) distance between t_test and t_true 
                dttrue=dist(t_true)                             # length of t_true
                E_dist  = E_dist+d                              # sum up distances (for first moment)
                E_dist2 = E_dist2+d*d                           # sum up squared distances (for second moment)
                if(E_max<0)or(d>E_max): E_max=d                 # collect maximal error
                if(E_min<0)or(d<E_min): E_min=d                 # collect minimal error
                drel=d/dttrue
                Erel_dist  = Erel_dist+drel                     # sum up relative distances (for first moment)
                Erel_dist2 = Erel_dist2+(drel*drel)             # sum up squared relative distances (for second moment)
                if(Erel_max<0)or(drel>Erel_max): Erel_max=drel  # collect maximal relative error
                if(Erel_min<0)or(drel<Erel_min): Erel_min=drel  # collect minimal relative error
        E_dist      = E_dist/float(N)                           # estimate of first moment (expected error)
        E_dist2     = E_dist2/float(N)                          # estimate of second moment (expected squared error)
        Var_dist    = E_dist2-E_dist*E_dist                     # variance of error
        sd_dist     = np.sqrt(Var_dist)                         # standard deviation of error
        Erel_dist   = Erel_dist/float(N)                        # estimate of first moment (expected error)
        Erel_dist2  = Erel_dist2/float(N)                       # estimate of second moment (expected squared error)
        Varrel_dist = Erel_dist2-Erel_dist*Erel_dist            # variance of error
        sdrel_dist  = np.sqrt(Varrel_dist)                      # standard deviation of error
        return (E_dist,sd_dist,E_min,E_max), (Erel_dist,sdrel_dist,Erel_min,Erel_max) # return mean, standard deviation, minimum, 
                                                                # and maximum error (for absolute and relative distances)  


# -------------------------------------------------------------------------------------------- 
# DataScaler: scale data to standardize data distribution (for mean=0, standard deviation =1)  
# -------------------------------------------------------------------------------------------- 
class DataScaler: 
    """
    Class for standardizing data vectors 
    Some regression methods require standardizing of data before training to avoid numerical instabilities!!
    """

    def __init__(self,X):               # X is data matrix, where rows are data vectors
        """
        Constructor: Set parameters (mean, std,...) to standardize data matrix X
        :param X: Data matrix of size NxD the standardization parameters (mean, std, ...) should be computed for 
        :returns: object of class DataScaler
        """
        self.meanX = np.mean(X,0)       # mean values for each feature column
        self.stdX  = np.std(X,0)        # standard deviation for each feature column 
        if isinstance(self.stdX,(list,tuple,np.ndarray)): 
            self.stdX[self.stdX==0]=1.0 # do not scale data with zero std (that is, constant features)
        else:
            if(self.stdX==0): self.stdX=1.0   # in case stdX is a scalar
        self.stdXinv = 1.0/self.stdX    # inverse standard deviation


    def scale(self,x):                  # scales data vector x to mean=0 and std=1
        """
        scale data vector (or data matrix) x to mean=0 and s.d.=1 
        :param x: data vector or data matrix  
        :returns: scaled (standardized) data vector or data matrix 
        """
        return np.multiply(x-self.meanX,self.stdXinv)

    def unscale(self,x):                # unscale data vector x to original distribution
        """
        unscale data vector (or data matrix) x to original data ranges  
        :param x: standardized data vector or data matrix  
        :returns: unscaled data vector or data matrix 
        """
        return np.multiply(x,self.stdX)+self.meanX

    def printState(self):
        """
        print standardization parameters (mean value, standard deviation (std), and inverse of std)  
        """
        print("mean=",self.meanX, " std=",self.stdX, " std_inv=",self.stdXinv)


# ----------------------------------------------------------------------------------------- 
# function to compute polynomial basis functions 
# ----------------------------------------------------------------------------------------- 
def phi_polynomial(x,deg=1):           # x should be list or np.array or 1xD matrix; returns an 1xM matrix 
    """
    polynomial basis function vector; may be used to transform a data vector x into a feature vector phi(x) having polynomial basis function components
    :param x: data vector to be transformed into a feature vector
    :param deg: degree of polynomial
    :returns phi: feature vector 
    Example: phi_polynomial(x,3) returns for one-dimensional x the vector [1, x, x*x, x*x*x]
    """
    x=np.array(np.mat(x))[0]           # ensure that x is a 1D array (first row of x)
    D=len(x)
    assert (D==1) or ((D>1) and (deg<=3)), "phi_polynomial(x,deg) not implemented for D="+str(D)+" and deg="+str(deg)    # MODIFY CODE HERE FOR deg>3 !!!!
    if(D==1):
        phi = np.array([x[0]**i for i in range(deg+1)])
    else:
        phi = np.array([])
        if(deg>=0):
            phi = np.concatenate((phi,[1]))      # include degree 0 terms
            if(deg>=1): 
                phi = np.concatenate((phi,x))    # includes degree 1 terms
                if(deg>=2):
                    for i in range(D):
                        phi = np.concatenate(( phi, [x[i]*x[j] for j in range(i+1)] ))    # include degree 2 terms
                    if(deg>=3):
                        for i in range(D):
                            for j in range(i+1):
                                phi = np.concatenate(( phi, [x[i]*x[j]*x[k] for k in range(j+1)] ))   # include degree 3 terms
                        # EXTEND CODE HERE FOR deg>3!!!!
    return phi.T  # return basis function vector (=feature vector corresponding to data vector x)


# -----------------------------------------------------------------------------------------
# Least Squares (ML) linear regression with sum of squares Regularization,
# -----------------------------------------------------------------------------------------
class LSRRegressifier(Regressifier):
    """
    Class for Least Squares (or Maximum Likelihood) Linear Regressifier with sum of squares regularization 
    """

    def __init__(self,lmbda=0,phi=lambda x: phi_polynomial(x,1),flagSTD=0,eps=1e-6):
        """
        Constructor of class LSRegressifier
        :param lmbda: Regularization coefficient lambda
        :param phi: Basis-functions used by the linear model (default linear polynomial)
        :param flagSTD: If >0 then standardize data X and target values T (to mean 0 and s.d. 1)
        :param eps: maximal residual value to tolerate (instead of zero) for numerically good conditioned problems
        :returns: -
        """
        self.lmbda=lmbda       # set regression parameter (default 0)
        self.phi=phi           # set basis functions used for linear regression (default: degree 1 polynomials)
        self.flagSTD=flagSTD;  # if flag >0 then data will be standardized, i.e., scaled for mean 0 and s.d. 1
        self.eps=eps;          # maximal residual value to tolerate (instead of zero) for numerically good conditioned problems


    def fit(self,X,T,lmbda=None,phi=None,flagSTD=None): # train/compute LS regression with data matrix X and target value matrix T
        """
        Train regressifier (see lecture manuscript, theorem 3.11, p33) 
        :param X: Data matrix of size NxD, contains in each row a data vector of size D
        :param T: Target vector matrix of size NxK, contains in each row a target vector of size K
        :param lmbda: Regularization coefficient lambda
        :param phi: Basis-functions used by the linear model (default linear polynomial)
        :param flagSTD: If >0 then standardize data X and target values T (to mean 0 and s.d. 1)
        :returns: flagOK: if >0 then all is ok, otherwise matrix inversion was bad conditioned (and results should not be trusted!!!) 
        """
        # (i) set parameters
        if lmbda==None: lmbda=self.lmbda       # reset regularization coefficient?
        if phi==None: phi=self.phi             # reset basis functions?
        if flagSTD==None: flagSTD=self.flagSTD # standardize data vectors?
        # (ii) scale data for mean=0 and s.d.=0 ?
        if flagSTD>0:                          # if yes, then...
            self.datascalerX=DataScaler(X)     # create datascaler for data matrix X
            self.datascalerT=DataScaler(T)     # create datascaler for target matrix T
            X=self.datascalerX.scale(X)        # scale all features (=columns) of data matrix X to mean=0 and s.d.=1
            T=self.datascalerT.scale(T)        # ditto for target matrix T
        # (iii) compute weight matrix and check numerical condition
        flagOK,maxZ=1,0;                       # if <1 then matrix inversion is numerically infeasible
        try:
            self.N,self.D = X.shape            # data matrix X has size N x D (N is number of data vectors, D is dimension of a vector)
            self.M = self.phi(self.D*[0]).size # get number of basis functions  
            self.K = T.shape[1]                # DELTE dummy code (just required for dummy code in predict(.): number of output dimensions
            PHI = None                         # REPLACE dummy code: compute design matrix
            PHIT_PHI_lmbdaI = None             # REPLACE dummy code: compute PHI_T*PHI+lambda*I
            PHIT_PHI_lmbdaI_inv = None         # REPLACE dummy code: compute inverse matrix (may be bad conditioned and fail)
            self.W_LSR = None                  # REPLACE dummy code: compute regularized least squares weights 
            # (iv) check numerical condition
            Z=None                             # REPLACE dummy code: Compute Z:=PHIT_PHI_lmbdaI*PHIT_PHI_lmbdaI_inv-I which should become the zero matrix if good conditioned!
            maxZ = 0                           # REPLACE dummy code: Compute maximum (absolute) componente of matrix Z (should be <eps for good conditioned problem)
            assert maxZ<=self.eps              # maxZ should be <eps for good conditioned problems (otherwise the result cannot be trusted!!!)
        except: 
            flagOK=0;
            print("EXCEPTION DUE TO BAD CONDITION:flagOK=", flagOK, " maxZ=", maxZ)
            raise
        return flagOK 

    def predict(self,x,flagSTD=None):      # predict a target value given data vector x 
        """
        predicts the target value y(x) for a test vector x
        :param x: test data vector of size D
        :param flagSTD: If >0 then standardize data X and target values T (to mean 0 and s.d. 1)
        :returns: predicted target vector y of size K
        """
        if flagSTD==None: flagSTD=self.flagSTD      # standardization?
        if flagSTD>0: x=self.datascalerX.scale(x)   # if yes, then scale x before computing the prediction!
        phi_of_x = self.phi(x)                      # compute feature vector phi_of_x for data vector x
        y=np.zeros((1,self.K)).T                    # REPLACE dummy code:  compute prediction y for data vector x 
        if flagSTD>0: y=self.datascalerT.unscale(y) # scale prediction back to original range?
        return y                  # return prediction y for data vector x



# -----------------------------------------------------------------------------------------
# KNN regression 
# -----------------------------------------------------------------------------------------
class KNNRegressifier(Regressifier): 
    """
    Class for fast K-Nearest-Neighbor-Regression using KD-trees 
    """

    def __init__(self,K,flagKLinReg=0):
        """
        Constructor of class KNNRegressifier
        :param K: number of nearest neighbors that are used to compute prediction 
        :flagKLinReg: if >0 then the do a linear (least squares) regression on the the K nearest neighbors and their target values
                      otherwise just take the mean of the K nearest neighbors target vectors
        :returns: -
        """
        self.K = K                                 # K is number of nearest-neighbors used for majority decision
        self.X, self.T = [],[]                     # initially no data is stored
        self.flagKLinReg=flagKLinReg               # if flag is set then do a linear regression of the KNN (otherwise just return mean T of the KNN)

    def fit(self,X,T): # train/compute regression with lists of data vectors X and target values T
        """
        Train regressifier by stroing X and T and by creating a KD-Tree based on X   
        :param X: Data matrix of size NxD, contains in each row a data vector of size D
        :param T: Target vector matrix of size NxK, contains in each row a target vector of size K
        :returns: -
        """
        self.X, self.T = np.array(X),np.array(T)   # just store feature vectors X and corresponding class labels T
        self.N, self.D = self.X.shape              # store data number N and dimension D
        self.kdtree = scipy.spatial.KDTree(self.X) # do an indexing of the feature vectors

    def predict(self,x,K=None,flagKLinReg=None):   # predict a target value given data vector x 
        """
        predicts the target value y(x) for a test vector x
        :param x: test data vector of size D
        :param K: number of nearest neighbors that are used to compute prediction 
        :flagKLinReg: if >0 then the do a linear (least squares) regression on the the K nearest neighbors and their target values
                      otherwise just take the mean of the K nearest neighbors target vectors
        :returns: predicted target vector of size K
        """
        if(K==None): K=self.K                      # do a K-NN search...
        if(flagKLinReg==None): flagKLinReg=self.flagKLinReg # if flag >0 then do a regression on the K Nearest Neighbors to get the prediction
        nn = self.kdtree.query(x,K)                # get indexes of K nearest neighbors of x
        if K==1: idxNN=[nn[1]]                     # cast nearest neighbor indexes nn as a list idxNN
        else: idxNN=nn[1]
        t_out=0
        if(self.flagKLinReg==0):
            # just take mean value of KNNs
            t_out=np.mean([self.T[i] for i in idxNN])
        else:
            # do a linear regression of the KNNs
            lsr=LSRegressifier(lmbda=0.0001,phi=lambda x:phi_polynomial(x,1),flagSTD=1)
            lsr.fit(self.X[idxNN],self.T[idxNN])
            t_out=lsr.predict(x)
        return t_out


# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    print("\n-----------------------------------------")
    print("Example: 1D-linear regression problem")
    print("-----------------------------------------")
    # (i) generate data
    N=100
    w0,w1=4,2                 # parameters of line
    X=np.zeros((N,1))         # x data: allocate Nx1 matrix as numpy ndarray
    X[:,0]=np.arange(0,50.0,50.0/N)  # equidistant sampling of the interval [0,50)
    T=np.zeros((N,1))         # target values: allocate Nx1 matrix as numpy ndarray
    sd_noise = 1.0            # noise power (=standard deviation)
    T=T+w1*X+w0 + np.random.normal(0,sd_noise,T.shape)  # generate noisy target values on line y=w0+w1*x
    par_lambda = 0            # regularization parameter
    print("X=",X)
    print("T=",T)

    # (ii) define basis functions (phi should return list of basis functions; x should be a list)
    deg=2;                               # degree of polynomial
    phi=lambda x: phi_polynomial(x,2)    # define phi by polynomial basis-functions up to degree deg 
    print("phi(4)=", phi([4]))           # print basis function vector [1, x, x*x ...] for x=4
    print("phi([1,2])=", phi([1,2]))     # print basis function vector for two-dim. inputs (yields many output components) 

    # (iii) compute LSR regression
    print("\n-----------------------------------------")
    print("Do a Least-Squares-Regression")
    print("-----------------------------------------")
    lmbda=0;
    lsr = LSRRegressifier(lmbda,phi)
    lsr.fit(X,T)
    print("lsr.W_LSR=",lsr.W_LSR)        # weight vector (should be approximately [w0,w1]=[4,2])
    x=np.array([3.1415]).T
    print("prediction of x=",x,"is y=",lsr.predict(x))

    # do S-fold crossvalidation
    S=3
    err_abs,err_rel = lsr.crossvalidate(S,X,T)
    print("LSRRegression cross-validation: absolute errors (E,sd,min,max)=", err_abs, "  relative errors (E,sd,min,max)=", err_rel)

    # (iv) compute KNN-regression
    print("\n-----------------------------------------")
    print("Do a KNN-Regression")
    print("-----------------------------------------")
    K=5;
    knnr = KNNRegressifier(K)
    knnr.fit(X,T)
    print("prediction of x=",x,"is y=",knnr.predict(x))

    # do S-fold crossvalidation
    err_abs,err_rel = knnr.crossvalidate(S,X,T)
    print("KNNRegression cross-validation: absolute errors (E,sd,min,max)=", err_abs, "  relative errors (E,sd,min,max)=", err_rel)


