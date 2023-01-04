#!/usr/bin/env python
# Programmgeruest zu IAS, Versuch 3, Aufgabe 3

import numpy as np
import matplotlib.pyplot as plt
from V1A2_Classifier import *
from V3A2_MLP_Backprop import *

# ----------------------------------------------------------------------------------------------------- 
# Multi-Layer-Perceptron with 3 neuron layers for classification using the crossentropy error function
# ----------------------------------------------------------------------------------------------------- 
class MLP3Classifier(Classifier): 
    """
    Multi-Layer-Perceptron with 3 neuron layers for classification using the crossentropy error function
    """

    def __init__(self,M=3,flagBiasUnits=1,lmbda=0,eta0=0.01,eta_fade=0,maxEpochs=100,nTrials=1,eps=0.01,debug=0):
        """
        Constructor of class MultiLayerPerceptron  
        :param M: number of hidden units 
        :param flagBiasUnits: if flagBiasUnits>0 then add a bias unit to the input and hidden layers 
        :param lmbda: regularization coefficient
        :param eta0: initial learning rate (at the beginning of learning)
        :param eta_fade: defines decrease of learning rate during learning for "simulated annealing": eta = eta0/(1+eta_fade*epoch)
        :param maxEpochs: maximal number of learning epochs (1 epoch correponds to presentation of all all training data)
        :param nTrials: number of learning trials (in each trial weights are initialized at random and learned by backprop until convergence or maxEpochs has been reached; the best weights over all trials are kept)
        :param eps: test for convergence: stop learning if error function becomes smaller than eps 
        :param debug: if >0 then debug mode: 1 = print Error, mean weight; 2=additionally check gradients; 3=additionally print weights
        """
        self.W1,self.W2 = None,None                                                      # weights of the two layers not yet allocated
        self.X,self.T,self.T_onehot,self.N = None,None,None,None                         # training data not yet defined
        self.D,self.M,self.M_total,self.K = None,None,None,None                          # network sizes not yet defined
        self.configure(M,flagBiasUnits,lmbda,eta0,eta_fade,maxEpochs,nTrials,eps,debug) # set MLP parameters

    def configure(self,M=None,flagBiasUnits=None,lmbda=None,eta0=None,eta_fade=None,maxEpochs=None,nTrials=None,eps=None,debug=None):
        """
        set one or more parameters of MLP object
        if one of the parameter is None then it is kept unchanged   
        :param M: number of hidden units 
        :param flagBiasUnits: if flagBiasUnits>0 then add a bias unit to the input and hidden layers
        :param lmbda: regularization coefficient
        :param eta0: initial learning rate (at the beginning of learning)
        :param eta_fade: defines decrease of learning rate during learning for "simulated annealing": eta = eta0/(1+eta_fade*epoch)
        :param maxEpochs: maximal number of learning epochs (1 epoch correponds to presentation of all all training data)
        :param nTrials: number of learning trials (in each trial weights are initialized at random and learned by backprop until convergence or maxEpochs has been reached; the best weights over all trials are kept)
        :param eps: test for convergence: stop learning if error function becomes smaller than eps 
        :param debug: if >0 then debug mode: 1 = print Error, mean weight; 2=additionally check gradients; 3=additionally print weights
        """
        # (i) set new parameters
        if(M             !=None): self.M              = M              # M=number of neuerons in the hidden layer  
        if(flagBiasUnits !=None): self.flagBiasUnits = flagBiasUnits   # if flagBiasUnits>0 then add a bias unit to the input and hidden layer 
        if(lmbda         !=None): self.lmbda          = lmbda          # regularization to avoid overfitting?   
        if(eta0          !=None): self.eta0           = eta0           # initial learning rate 
        if(eta_fade      !=None): self.eta_fade       = eta_fade       # temporal fading of learning: eta = eta0/(1+eta_fade*epoch)
        if(maxEpochs     !=None): self.maxEpochs      = maxEpochs      # max. number of learning epochs per trial (MLP may not converge!)
        if(nTrials       !=None): self.nTrials        = nTrials        # number of learning trials (to select from the best weights) 
        if(eps           !=None): self.eps            = eps            # stop learning if error function becomes smaller than eps  
        if(debug         !=None): self.debug          = debug          # if >0 then debug mode: 1 = print Error, mean weight; 2=additionally check gradients; 3=additionally print weights
        # (ii) reset further object attributes if necessary
        self.M_total=M
        if(flagBiasUnits>0): self.M_total=M+1                      # add bias unit to hidden layer?

    def printState(self,flagData=0,flagWeights=0):
        """
        print state of MLP for debugging 
        :param flagData: if >0 then print also training data X,T 
        :param flagWeights: if >0 then print also weight matrices W1,W2
        """
        print("MLP state variables:")
        print("MLP size D,M,M_total,K=",self.D,self.M,self.M_total,self.K)
        print("MLP parameters: flagBiasUnits=",self.flagBiasUnits," lmbda=",self.lmbda," eta0=",self.eta0," eta_fade=",self.eta_fade," maxEpochs=",self.maxEpochs," nTrials=",self.nTrials," eps=",self.eps," debug=",self.debug)
        print("N=",self.N)
        if(flagData>0):
            print("X=",self.X)
            print("T=",self.T)
            print("T_onehot=",self.T_onehot)
        if(flagWeights>0):
            print("W1=",self.W1)
            print("W2=",self.W2)
        if(not self.W1 is None)and(not self.W2 is None):
            print("Mean absolute weight of W1 = ", np.mean(np.array(np.sqrt(np.multiply(self.W1,self.W1))),axis=(0,1)))
            print("Mean absolute weight of W2 = ", np.mean(np.array(np.sqrt(np.multiply(self.W2,self.W2))),axis=(0,1)))
            print("Error E=", self.getError())

    def setTrainingData(self,X,T):
        """
        set training data 
        :param X: data matrix (one data vector per row) of size NxD
        :param T: target vector of class labels (should be coded as integers)  (one target vector per row; should be "one-hot" 1-of-K coding of size NxK
        """
        # (i) set data matrix X
        self.X = np.array(X)                    # set NxD data matrix
        self.N,self.D = X.shape                 # N is number of data vectors; D is dimensionality
        if self.flagBiasUnits>0:                # extend data vectors by bias unit 
            self.X=np.concatenate((np.ones((self.N,1)),self.X),1)  # X is extended by a column vector with ones (for bias weight w_j0)
            self.D=self.D+1
        # (ii) set target matrix T and T_onehot
        assert len(T)==self.N,"X and T should have same length! But NX="+str(self.N)+" whereas NT="+str(len(T))+ "!!"
        self.T = np.array(T)                      # set Nx1 target vector (will be translated into one-hot form)
        self.K = np.max(self.T)+1                 # number of different classes (=number of output units)
        self.T_onehot = np.zeros((self.N,self.K)) # allocate zero matrix for one-hot-coding
        for n in range(self.N):
            self.T_onehot[n,self.T[n]]=1.0     # set in n-th row the class-component to 1    

    def setRandomWeights(self):
        """
        initialize weight matrixes W1,W2 with random values between -0.5 and 0.5  
        """
        self.W1=np.random.rand(self.M,self.D)-0.5        # initialize weights of layer 1 randomly 
        self.W2=np.random.rand(self.K,self.M_total)-0.5  # initialize weights of layer 2 randomly


    def getError(self):
        """
        Return value of cross entropy error function including with regularization   
        """
        return getError(self.W1,self.W2,self.X,self.T_onehot,self.lmbda,self.flagBiasUnits)

    def doLearningEpoch(self,eta=None):  # do one learning epoch (over whole data set)
        """
        Do one learning epoch (over whole data set) with a fixed learning rate 
        :param eta    : learning rate to be used for the current epoch 
        """
        perm = np.random.permutation(self.N)      # get a random permutation of data set (to present data vectors in random order...) 
        for p in perm:
            n=perm[p]      # present n-th data vector
            self.W1,self.W2 = doLearningStep(self.W1,self.W2,self.X[n,:],self.T_onehot[n,:],eta,self.lmbda/self.N,self.flagBiasUnits) # do weight update by the backpropatation algorithm 

    def doLearningTrial(self, flagInitWeights=1): # do one learning trial 
        """
        Do one learning trail (seek convergence to local minimum of error function)  
        :param flagInitWeight: if >0 then initialize with random weights  
        :returns E: Final normalized crossentropy error value (per vector)  
        """
        if flagInitWeights>0: self.setRandomWeights() # random initialization of synaptic weights
        for epoch in range(self.maxEpochs):           # loop over epochs
            eta = self.eta0/(1.0+self.eta_fade*epoch) # learning rate for this epoch
            self.doLearningEpoch(eta)                 # do learning epoch
            E=self.getError()/self.N                  # normalized error      
            if(self.debug>0):    # output debug info?
                mw=(self.D*self.M*np.mean(np.array(np.sqrt(np.multiply(self.W1,self.W1))),axis=(0,1))+self.M_total*self.K*np.mean(np.array(np.sqrt(np.multiply(self.W2,self.W2))),axis=(0,1)))/(self.D*self.M+self.M_total*self.K)
                print("after learning epoch=",epoch, ", normalized error E/N=",E, "  eta=",eta, " mean_weight=",mw)
                if(self.debug>1): self.printState()
            if(E<self.eps):break                      # if error is smaller than eps then break learning trial early
        self.epoch=epoch
        return E;                          # return final error after learning trial

    def fit(self,X,T, flagInitWeights=1):
        """ 
        Train classifier by training data X, T
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        # (i) do training procedure
        self.setTrainingData(X,T)                           # set training data
        if flagInitWeights>0: self.setRandomWeights()       # random initialization of synaptic weights
        W1_opt,W2_opt=np.array(self.W1),np.array(self.W2)   # remember copies of initial weights
        E_opt=self.getError()                               # remember initial value of error function
        for trial in range(self.nTrials):# loop over learning trials
            E=self.doLearningTrial(flagInitWeights)         # do one learning trial (returns error)
            if(E<E_opt):                                              # better result?
                E_opt         = E                                     # ... then keep new error value
                W1_opt,W2_opt = np.array(self.W1),np.array(self.W2)  # ... and keep copies of new weights
        # (ii) finally set optimal weights (with minimal errors)
        self.W1,self.W2 = W1_opt, W2_opt # set best weights

    def predict(self,x):
        """ 
        Implementation of classification algorithm
        :param x: test data vector
        :returns: prediction: label of most likely class that test vector x belongs to 
        :returns: pClassPosteriori: A-Posteriori probabilities, pClassPosteriori[i] is probability that x belongs to class i
        """
        if self.flagBiasUnits>0:         # append 1-component ?
            x=np.concatenate(([1.0],x))  # yes...
        z_1,z_2 = forwardPropagateActivity(x,self.W1,self.W2,self.flagBiasUnits)   # propagate activity from input x through network
        return np.argmax(z_2),z_2,None  


# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    # (i) create training data
    flagDataset1=1          # set this flag to switch between dataset 1 (flag=1) and dataset 2 (flag=2)
    if flagDataset1>0:
        # data set 1
        X1 = np.array([[-2.,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # set 1: class 1 data
        X2 = np.array([[-1.,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # set 1: class 2 data
    else:
        # data set 2
        X1 = np.array([[-1.,1], [1,1], [-1,-1], [1,-1], [0,1.5], [1.5,0], [0,-1.5], [-1.5,0]])  # set 1: class 1 data
        X2 = np.array([[0,2.], [2,0], [-2,0], [0,-2], [2,2], [2,-2], [-2,2], [-2,-2]])          # set 2: class 2 data
    N1,D1 = X1.shape
    N2,D2 = X2.shape
    T1,T2 = N1*[0],N2*[1]              # corresponding class labels (1 versus -1) 
    X = np.concatenate((X1,X2))        # entire data set
    T = np.concatenate((T1,T2))        # entire label set
    N,D = X.shape
    print("X=",X)
    print("T=",T)
    
    # (ii) Define and train MLP
    print("\n(ii) Training MLP:")
    if flagDataset1>0:
        M=3                           # number of hidden units
        eta0=0.6                       # initial learning rate
        eta_fade=1.0/30                 # fading factor for decreasing learning rate (e.g., 1/50 means after 50 epochs is learning rate half the initial value...)
        maxEpochs=75                  # number of learning epochs
    else:
        M=8                            # number of hidden units
        eta0=0.6                       # initial learning rate
        eta_fade=1.0/30                 # fading factor for decreasing learning rate (e.g., 1/50 means after 50 epochs is learning rate half the initial value...)
        maxEpochs=300                  # number of learning epochs
    flagBiasUnits=1                    # bias units in input layer and hidden layers?
    lmbda=0                            # regularization coefficient
    nTrials = 1                        # number of learning trials
    eps = 0.01                         # stop learning if error function becomes smaller than eps
    debug = 1                            # if >0 then debug mode: 1 = print Error, mean weight; 2=additionally check gradients; 3=additionally print weights
    mlp = MLP3Classifier(M,flagBiasUnits,lmbda,eta0,eta_fade,maxEpochs,nTrials,eps,debug)
    mlp.fit(X,T) 
    print("State of MLP after learning:")
    mlp.printState(1,1)            # print state of MLP

    # (iii) test MLP with training data
    print("\n(iii) Test MLP with training data:")
    errc = 0                       # initialize classification errors with zero
    for n in range(N):             # loop over all training data        
        xn=X[n]                    # n-th data vector
        tn=T[n]                    # n-th target value
        y_hat=mlp.predict(xn)[0]   # prediction for training vector xn
        if(tn!=y_hat): errc=errc+1 # count classification error
    E=mlp.getError()
    print("After Learning: Error function E=", E, " Number of classification errors errc=", errc)

    # (iv) make a new prediction
    print("\n(iv) Make a new prediction:")
    x=[-1,0]
    y_hat,p_posteriori,dummy=mlp.predict(x)
    print("New prediction for x=",x," : y_hat=",y_hat, "; A Posteriori Class Distribution: p_posteriori=",p_posteriori)

    # (v) do plots 
    contlevels=[-1,0,1]                # plot contour levels (of log-odds-ratio)
    gridX,gridY = np.meshgrid(np.arange(-3,5,0.1),np.arange(-3,3,0.1))  # mesh grid for plot
    fig,ax = plotDecisionSurface(mlp.W1,mlp.W2,gridX,gridY,X1,X2,contlevels,mlp.epoch,mlp.flagBiasUnits)
    ax.scatter(x[0],x[1], c='b', marker='o', s=200)
plt.show()

