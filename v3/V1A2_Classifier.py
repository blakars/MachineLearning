#!/usr/bin/env python
# Python Module for Classification Algorithms
# Programmgeruest zu Versuch 1, Aufgabe 2
import numpy as np
import scipy.spatial
from random import randint

# ----------------------------------------------------------------------------------------- 
# Base class for classifiers
# ----------------------------------------------------------------------------------------- 
class Classifier:
    """
    Abstract base class for a classifier.
    Inherit from this class to implement a concrete classification algorithm
    """

    def __init__(self,C=2): 
        """
        Constructor of class Classifier
        Should be called by the constructors of derived classes
        :param C: Number of different classes
        """
        self.C = C            # set C=number of different classes 

    def fit(self,X,T):    
        """ 
        Train classier by training data X, T, should be overwritten by any derived class
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        shapeX,shapeT=X.shape,T.shape  # X must be a N x D matrix; T must be a N x 1 matrix; N is number of data vectors; D is dimensionality
        assert len(shapeX)==2, "Classifier.fit(self,X,T): X must be two-dimensional array!"
        assert len(shapeT)==1, "Classifier.fit(self,X,T): T must be one-dimensional array!"
        assert shapeX[0]==shapeT[0], "Classifier.fit(self,X,T): Data matrix X and class labels T must have same length!"
        self.C=max(T)+1;       # number of different integer-type class labels (assuming that T(i) is in {0,1,...,C-1})

    def predict(self,x):
        """ 
        Implementation of classification algorithm, should be overwritten in any derived class
        :param x: test data vector
        :returns: label of most likely class that test vector x belongs to (and possibly additional information)
        """
        return -1,None,None

    def crossvalidate(self,S,X,T):    # do a S-fold cross validation 
        """
        Do a S-fold cross validation
        :param S: Number of parts the data set is divided into
        :param X: Data matrix (one data vector per row)
        :param T: Vector of class labels; T[n] is label of X[n]
        :returns pClassError: probability of a classification error (=1-Accuracy)
        :returns pConfErrors: confusion matrix, pConfErrors[i,j] is the probability that a vector from true class j will be mis-classified as class i
        """
        N=len(X)                                            # N=number of data vectors
        perm = np.random.permutation(N)                     # do a random permutation of X and T...
        Xp,Tp=[X[i] for i in perm], [T[i] for i in perm]    # ... to get a random partition of the data set
        idxS = [range(i*N//S,(i+1)*N//S) for i in range(S)] # divide data set into S parts:
        C=max(T)+1;                                         # number of different class labels (assuming that t is in {0,1,...,C-1})
        nC          = np.zeros(C)                           # initialize class probabilities: nC[i]:=N*pr[xn is of class i]
        pConfErrors = np.zeros((C,C))                       # initialize confusion error probabilities pr[class i|class j]
        pClassError = 0                                     # initialize probability of a classification error
        for idxTest in idxS:                                # loop over all possible test data sets
            # (i) generate training and testing data sets and train classifier        
            idxLearn = [i for i in range(N) if i not in idxTest]                      # remaining indices (not in idxTest) are learning data
            if(S<=1): idxLearn=idxTest                                                # if S==1 use entire data set for learning and testing
            X_learn, T_learn = [Xp[i] for i in idxLearn], [Tp[i] for i in idxLearn]   # learning data for training the classifier
            X_test , T_test  = [Xp[i] for i in idxTest] , [Tp[i] for i in idxTest]    # test data 
            self.fit(np.array(X_learn),np.array(T_learn))                             # train classifier
            # (ii) test classifier
            for i in range(len(X_test)):  # loop over all data vectors to be tested
                # (ii.a) classify i-th test vector
                t_test = self.predict(X_test[i])[0]             # classify test vector
                # (ii.b) check for classification errors
                t_true = T_test[i]                              # true class label
                nC[t_true]=nC[t_true]+1                         # count occurrences of individual classes
                pConfErrors[t_test][t_true]=pConfErrors[t_test][t_true]+1  # count conditional class errors
                if(t_test!=t_true): pClassError=pClassError+1              # count total number of errors
        pClassError=float(pClassError)/float(N)         # probability of a classification error
        for i in range(C): 
            for j in range(C): 
                pConfErrors[i,j]=float(pConfErrors[i,j])/float(nC[j])   # finally compute confusion error probabilities
        self.pClassError,self.pConfErrors=pClassError,pConfErrors       # store error probabilities as object fields
        return pClassError, pConfErrors                 # return error probabilities


# ----------------------------------------------------------------------------------------- 
# (Naive) k-nearest-neighbor classifier based on simple look-up-table and exhaustive search
# ----------------------------------------------------------------------------------------- 
class KNNClassifier(Classifier):
    """
    (Naive) k-nearest-neighbor classifier based on simple look-up-table and exhaustive search
    Derived from base class Classifier
    """

    def __init__(self,C=2,k=1):
        """
        Constructor of the KNN-Classifier
        :param C: Number of different classes
        :param k: Number of nearest neighbors that classification is based on
        """
        Classifier.__init__(self,C) # call constructor of base class  
        self.k = k                  # k is number of nearest-neighbors used for majority decision
        self.X, self.T = [],[]      # initially no data is stored

    def fit(self,X,T):
        """
        Train classifier; for naive KNN Classifier this just means to store data matrix X and label vector T
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        Classifier.fit(self,X,T);   # call to base class to check for matrix dimensions etc.
        self.X, self.T = X,T        # just store the N x D data matrix and the N x 1 label matrix (N is number and D dimensionality of data vectors) 
        
    def getKNearestNeighbors(self, x, k=None, X=None):
        """
        compute the k nearest neighbors for a query vector x given a data matrix X
        :param x: the query vector x
        :param X: the N x D data matrix (in each row there is data vector) as a numpy array
        :param k: number of nearest-neighbors to be returned
        :return: list of k line indexes referring to the k nearest neighbors of x in X
        """
        if(k==None): k=self.k                      # per default use stored k 
        if(X==None): X=self.X                      # per default use stored X
        return k*[0]                               # REPLACE: Insert/adapt your code from V1A1_KNearestNeighborSearch.py

    def predict(self,x,k=None):
        """ 
        Implementation of classification algorithm, should be overwritten in any derived classes
        :param x: test data vector
        :param k: search k nearest neighbors (default self.k)
        :returns prediction: label of most likely class that test vector x belongs to
                             if there are two or more classes with maximum probability then one class is chosen randomly
        :returns pClassPosteriori: A-Posteriori probabilities, pClassPosteriori[i] is probability that x belongs to class i
        :returns idxKNN: indexes of the k nearest neighbors (ordered w.r.t. ascending distance) 
        """
        if k==None: k=self.k                       # use default parameter k?
        idxKNN = self.getKNearestNeighbors(x,k)    # get indexes of k nearest neighbors of x
        prediction=0                               # REPLACE DUMMY CODE BY YOUR OWN CODE!
        pClassPosteriori=self.C*[1.0/self.C]       # REPLACE DUMMY CODE BY YOUR OWN CODE!
        return prediction, pClassPosteriori, idxKNN  # return predicted class, a-posteriori-distribution, and indexes of nearest neighbors


# ----------------------------------------------------------------------------------------- 
# Fast k-nearest-neighbor classifier based on scipy KD trees
# ----------------------------------------------------------------------------------------- 
class FastKNNClassifier(KNNClassifier):
    """
    Fast k-nearest-neighbor classifier based on kd-trees 
    Inherits from class KNNClassifier
    """

    def __init__(self,C=2,k=1):
        """
        Constructor of the KNN-Classifier
        :param C: Number of different classes
        :param k: Number of nearest neighbors that classification is based on
        """
        KNNClassifier.__init__(self,C,k)     # call to parent class constructor  

    def fit(self,X,T):
        """
        Train classifier by creating a kd-tree 
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        KNNClassifier.fit(self,X,T)                # call to parent class method (just store X and T)
        self.kdtree = None                         # REPLACE DUMMY CODE BY YOUR OWN CODE! Do an indexing of the feature vectors by constructing a kd-tree
        
    def getKNearestNeighbors(self, x, k=None):  # realizes fast K-nearest-neighbor-search of x in data set X
        """
        fast computation of the k nearest neighbors for a query vector x given a data matrix X by using the KD-tree
        :param x: the query vector x
        :param k: number of nearest-neighbors to be returned
        :return idxNN: return list of k line indexes referring to the k nearest neighbors of x in X
        """
        if(k==None): k=self.k                      # do a K-NN search...
        idxNN = k*[0]                              # REPLACE DUMMY CODE BY YOUR OWN CODE! Compute nearest neighbors using the KD-Tree
        return idxNN                               # return indexes of k nearest neighbors



# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    # (i) Generate dummy data 
    X = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]);      # data matrix X: list of data vectors (=database) of dimension D=3
    T = np.array([0,1,0,1]);                              # target values aka class labels
    x = np.array([1.5,3.6,5.7]);                          # a test data vector
    print("Data matrix X=\n",X)
    print("Class labels T=\n",T)
    print("Test vector x=",x)
    print("Euklidean distances d=",[])                     # REPLACE DUMMY CODE (IF YOU WANT) ...

    # (ii) Train simple KNN-Classifier
    knnc = KNNClassifier()         # construct kNN Classifier
    knnc.fit(X,T)                  # train with given data

    # (iii) Classify test vector x
    k=3
    c,pc,idx_knn=knnc.predict(x,k)
    print("\nClassification with the naive KNN-classifier:")
    print("Test vector is most likely from class ",c)
    print("A-Posteriori Class Distribution: prob(x is from class i)=",pc)
    print("Indexes of the k=",k," nearest neighbors: idx_knn=",idx_knn)

    # (iv) Repeat steps (ii) and (iii) for the FastKNNClassifier (based on KD-Trees)
    # INSERT YOUR CODE
