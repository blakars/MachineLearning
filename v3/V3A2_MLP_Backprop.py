#!/usr/bin/env python
# Programmgeruest zu IAS, Versuch3, Aufgabe 2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------- 
# Simple Multi-Layer-Perceptron with 3 Layers for Classification with K classes
# -------------------------------------------------------------------------------------------------- 

def softmax(a):  
    """
    Compute Softmax function for potential vector a
    :param a: Vector of dendritic potentials of the softmax neuron population  
    :returns softmax(a) which is a vector of same size as a  
    """
    e_a = np.exp(a - np.max(a))  # subtract maximum potential such that maximal exponent is 1 (for numerical stability)
    return e_a / e_a.sum()       # return softmax function value

def forwardPropagateActivity(x,W1,W2,flagBiasUnit=1): 
    """
    Propagate neuronale activity through the network in forward direction from input layer to output layer   
    :param x: Input vector (may be extended with a bias unit) 
    :param W1: Weight matrix for synaptic layer 1 (connecting the input layer to the hidden layer)
    :param W2: Weight matrix for synaptic layer 2 (connecting the hidden layer to the output layer)
    :param flagBiasUnit: If >0 then add a bias unit to the hidden layer
    :returns z_1: Firing rates of the neurons in the hidden layer (=layer 1)
    :returns z_2: Firing rates of the neurons in the output layer (=layer 2); it is z2=y:=output activity 
    """    
    a_1 = np.dot(W1,x);                         # compute dendritic potentials of hidden layer a_1
    z_1 = np.tanh(np.array(a_1,'float'));       # compute activity z_1 of hidden layer 1
    if flagBiasUnit>0: z_1=np.append(z_1,[1.0]); # add bias unit (with constant activity 1) to hidden layer ?
    a_2 = np.dot(W2,z_1);                        # compute dendritic potentials of output layer a_2
    z_2 = softmax(a_2);                          # compute softmax activations for output layer
    return z_1, z_2;                             # return activities in layers 1 and 2; z_2 corresponds to outputs y

def backPropagateErrors(z_1,z_2,t,W1,W2,flagBiasUnit=1): # backpropagate error signals delta_L
    """
    Backpropagate error signals through the network in backward direction from output layer back to input layer
    :param z_1: Firing rate activity in (hidden) layer 1 (as obtained from forwardPropagateActivity(.) )
    :param z_2: Firing rate activity in (output) layer 2 (as obtained from forwardPropagateActivity(.) )
    :param t: true target vector (=output label) for input vector x as obtained from training data
    :param W1: Weight matrix for synaptic layer 1 (connecting the input layer to the hidden layer)
    :param W2: Weight matrix for synaptic layer 2 (connecting the hidden layer to the output layer)
    :param flagBiasUnit: If >0 then add a bias unit to the hidden layer (i.e., remove bias unit from hidden layer z_1)
    :returns delta_1: Error signals for (hidden) layer 1
    :returns delta_2: Error signals for (output) layer 2
    """
    y=z_2                                     # layer 2 is output layer
    delta_2=y-t;                              # Initializing error signals in output layer 2
    alpha_1 = np.dot(W2.T, delta_2);          # compute error potentials in hidden layer 1 by backpropagating errors delta_2
    h_prime=(1.0-np.multiply(z_1,z_1))        # factor (1-z_1.*z_1) is h'(a) for tanh sigmoid function
    delta_1=np.multiply(h_prime,alpha_1)      # compute error signals in hidden layer 1
    if flagBiasUnit>0: delta_1 = delta_1[:-1] # remove last error signal corresponding to the bias unit ?
    return delta_1, delta_2                   # return error signals for each layer

def doLearningStep(W1,W2,xn,tn,eta,lmbda_by_N=0,flagBiasUnit=1): # do one backpropagation learning step...
    """
    Do one backpropagation learning step for one input vector xn with corresponding traget vector tn 
    :param W1: Weight matrix for synaptic layer 1 (connecting the input layer to the hidden layer)
    :param W2: Weight matrix for synaptic layer 2 (connecting the hidden layer to the output layer)
    :param xn: Input vector (may be extended with a bias unit) 
    :param tn: true target vector (=output label) for input vector xn as obtained from training data
    :param eta: learning rate (determines how far to go into neg-gradient direction)
    :param lmbda_by_N: regularization coefficient lambda divided by N (=lambda/N)
    :param flagBiasUnit: If >0 then add a bias unit to the hidden layer 
    :returns W1,W2: updated weight matrices 
    """
    z_1    ,z_2    =forwardPropagateActivity(xn,W1,W2,flagBiasUnit);           # forward propagation of activity according to input vector xn
    delta_1,delta_2=backPropagateErrors(z_1,z_2,tn,W1,W2,flagBiasUnit);        # get error signals by backpropagation
    nablaED_1 = np.outer(delta_1, xn)                                          #gradient of data error function for first layer
    nablaED_2 = np.outer(delta_2, z_1)                                         #gradient of data error function for second layer
    W1=W1*(1.0-lmbda_by_N*eta)-eta*nablaED_1                                   # update weights for first layer with "weight decay" regularization
    W2=W2*(1.0-lmbda_by_N*eta)-eta*nablaED_2                                   # update weights for second layer with "weight decay" regularization
    return W1,W2                                                               # return new weights

def getError(W1,W2,X,T,lmbda=0,flagBiasUnit=1): # compute total crossentropy error function over whole data set (X,T) for MLP with weight layers W1,W2
    """
    Compute total crossentropy error function over whole data set (X,T) for MLP with weight layers W1,W2
    :param W1: Weight matrix for synaptic layer 1 (connecting the input layer to the hidden layer)
    :param W2: Weight matrix for synaptic layer 2 (connecting the hidden layer to the output layer)
    :param X: Data matrix (in each row of the matrix is one data vector)
    :param T: Target matrix (in each row of the matrix is one target vector, typically with "one-hot" coding)
    :param lmbda: regularization coefficient
    :param flagBiasUnit: If >0 then add a bias unit to the hidden layer 
    :returns E: Crossentropy error value W1,W2: updated weight matrices 
    """
    N,D = X.shape                               # get size of data set
    E=0;                                        # initialize error with 0
    for n in range(N):                          # test all data vectors
        y=forwardPropagateActivity(X[n,:],W1,W2,flagBiasUnit)[1]; # get output values y for input X[n]
        t=T[n,:];                               # get actual target vector (should be "one hot" coded)
        e=[-t[i]*np.log(y[i]) for i in range(len(t)) if t[i]>0 and y[i]>0]   # error contributions of each componente of y and t
        E=E+np.sum(e)                           # add sum of component errors to total error 
    if lmbda>0:                                 # regularization?
        EW1=np.sum(np.sum(np.multiply(W1,W1)))  # sum of squared weights for layer 1
        EW2=np.sum(np.sum(np.multiply(W2,W2)))  # sum of squared weights for layer 2
        E=E+0.5*lmbda*(EW1+EW2)                 # add weight error
    return E;                                   # return final error

def plotDecisionSurface(W1,W2,gridX,gridY,dataX1,dataX2,contlevels,epoch,flagBiasUnit=1): # plot decision surface (only for K=2 and D=2)
    """
    Compute decision surface for K=2 classes and D=2 dimensional data
    :param W1: Weight matrix for synaptic layer 1 (connecting the input layer to the hidden layer)
    :param W2: Weight matrix for synaptic layer 2 (connecting the hidden layer to the output layer)
    :param gridX,gridY: defines x-y grid for contour plot (where the log-odds-ratio will be valuated)
    :param dataX1,dataX2: data matrices for class1 and class2 
    :param contlevels: contour levels of the log odds ratio that will be plotted
    :param epoch: learning epoch (just for printing on the title line of the plots)
    :param flagBiasUnit: If >0 then add a bias unit to the hidden layer 
    """
    m,n=gridX.shape
    gridZ=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            yn=forwardPropagateActivity([1,gridX[i,j],gridY[i,j]],W1,W2,flagBiasUnit)[1]   # activity for input xn
            gridZ[i,j]=np.log(yn[0]/yn[1])                                                 # plot contours of log-odds-ratio 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.array(dataX1)[:,0],np.array(dataX1)[:,1], c='r', marker='x', s=200)
    ax.scatter(np.array(dataX2)[:,0],np.array(dataX2)[:,1], c='g', marker='*', s=200)
    CS=ax.contour(gridX, gridY, gridZ,levels=contlevels)
    ax.clabel(CS,CS.levels,inline=True)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Log-Odds-Contours after learning epoch '+str(epoch))
    return fig,ax
    #plt.savefig('testfig.png');


# *******************************************************
# Main program 
# *******************************************************
if __name__ == '__main__':

    # (i) Create training data
    X1 = np.array([[-2,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # class 1 data
    N1,D1 = X1.shape
    T1 = np.array(N1*[[1.,0]])     # corresponding class labels with one-hot coding: [1,0]=class 1;
    X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # class 2 data
    N2,D2 = X2.shape
    T2 = np.array(N2*[[0,1.]])     # corresponding class labels with one-hot coding: [0,1]=class 2 
    X = np.concatenate((X1,X2))    # entire data set
    T = np.concatenate((T1,T2))    # entire label set
    N,D = X.shape
    X=np.concatenate((np.ones((N,1)),X),1)  # X is extended by a column vector with ones (for bias weights w_j0)
    N,D = X.shape                      # update size parameters
    N,K = T.shape                      # update size parameters
    print("X=",X)
    print("T=",T)

    # (ii) Train MLP
    M=3                                # number of hidden units
    flagBiasUnit=1                     # add an extra bias unit to hidden units?
    M_total=M                          # total number of hidden units...
    if flagBiasUnit>0: M_total=M+1     # ... including the bias unit?
    b=1.0
    eta=0.4                            # learning rate
    lmbda=0                            # regularization coefficient
    nEpochs=10                         # number of learning epochs
    contlevels=[-1,0,1]                # plot contour levels (of log-odds-ratio)
    epochs4plot=[-1,0,5,10,50,100,nEpochs-1] # learning epochs for which a plot will be made
    gridX,gridY = np.meshgrid(np.arange(-3,5,0.1),np.arange(-3,3,0.1))  # mesh grid for plot
    W1=1.0*(np.random.rand(M,D)-0.5)       # initialize weights of layer 1 randomly 
    W2=1.0*(np.random.rand(K,M_total)-0.5) # initialize weights of layer 2 randomly
    E=getError(W1,W2,X,T,lmbda)
    print("initial error E=",E)
    if -1 in epochs4plot: plotDecisionSurface(W1,W2,gridX,gridY,X1,X2,contlevels,-1,flagBiasUnit)
    for epoch in range(nEpochs):       # loop over learning epochs
        errc = 0                       # initialize classification errors with zero
        for n in range(N):             # loop over all training data        
            xn=X[n,:]                  # n-th data vector
            tn=T[n,:]                  # n-th target value
            yn=forwardPropagateActivity(xn,W1,W2)[1]  # test training vector xn
            yhat,that=2,2              # initialize class labels 
            if(tn[0]>=tn[1]): that=1   # actual class label
            if(yn[0]>=yn[1]): yhat=1   # predicted class by MLP 
            if(yhat!=that): errc=errc+1                                   # count classification error
            W1,W2=doLearningStep(W1,W2,xn,tn,eta,lmbda/N) # do one backprop learning update of weights
        E=getError(W1,W2,X,T)
        print("after epoch ", epoch, " error function E=",E, " and classification errors = ", errc)
        if epoch in epochs4plot: plotDecisionSurface(W1,W2,gridX,gridY,X1,X2,contlevels,epoch,flagBiasUnit)
plt.show()

