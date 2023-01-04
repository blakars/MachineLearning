#!/usr/bin/env python
# coding: utf-8

# # 2a)

# Klassen & Zweck:
# * Abstrakte Klasse "Classifier" als Basis-Klasse für unterschiedliche Klassifikations-Verfahren/Algorithmen
# * Klasse "KNNClassifier": Naive Umsetzung einer KNN-Klassifikation
# * Klasse "FastKNNClassifier": Effizientere Umsetzung einer KNN-Klassifikation mittels KD-Trees
# 
# Methoden der Basis-Klasse "Classifier":
# 
# * **_ _ init _ _(self,C)**: Konstruktor der Basis-Klasse, C=Anzahl der unterschiedlichen Klassen, abgeleitete Klassen rufen diesen Konstruktor dann auf
# * **fit(self,X,T)**: Abstrakte Methode zum "Traineren" der Klassifikation. Für die naive KNN-Klassifikation bedeutet "Traineren" lediglich "Abspeichern", für die KD-Tree Version von KNN erfolgt hier das Erstellen des KDTree
# * **predict(self,x)**: Abstrakte Methode, die in der Umsetzung dann den eigentlichen Klassifikations-Algorithmus enthält, d.h. die Vorhersage, dass ein gegebener Vektor x zu einer bestimmten Klasse gehört
# * **crossvalidate(self,S,X,T)**: Methode zur Kreuz-Validierung eines in S Teile aufgeteilten Daten-Sets, benötigt S Trainings-Durchläufe, liefert Generalisierungsfehler 
# 
# 
# für Details siehe auch V1A2_Classifier.html (erstellt mit pydoc)

# # 2b)

# - Für einen kNN-Klassifikator, bedeutet "Lernen" in dem Fall einfach nur, dass er die Zuordnung von Daten-Vektoren (aus Matrix X) zu unterschiedlichen Klassen-Labels (Vektor T) abspeichert. Außerdem ruft die Funktion fit(self,X,T) der "KNNClassifier"-Klasse noch die fit-Methode der abstrakten Klasse "Classifier" auf, in der geprüft wird ob die Dimensionen von X und T auch (zueinander) passen und die Anzahl unterschiedlicher Klassen C auf Basis des Übergebenen Vektors T aktualisiert wird.
# - Implementierung s.u.
#     

# # Implementierung 2b-d)

# In[12]:


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
        return np.argsort([np.linalg.norm(x-a) for a in X])[0:k]   # analog V1A1

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
        labelsKNN=[self.T[i] for i in idxKNN]      # list of classes of k nearest neighbors
        pClassPosteriori=[labelsKNN.count(i)/float(k) for i in range(self.C)]    #calculate probabilities
        p_max=np.max(pClassPosteriori)            #yields highest class probability
        c_max=np.where(pClassPosteriori==p_max)[0]   #yields list of classes with maximum probability
        prediction=c_max[randint(0,len(c_max)-1)]   #Randomisierung wenn mehrere Klassen gleiche maximale Wahrscheinlichkeit
        '''
        for cl in range(self.C):                  #iterieren über Klassen
            for i in idxKNN:                      #iterieren über Indexe der k nearest neighbors
                if self.T[i]==cl:                 #wenn die Klasse des neighbors, der aktuell untersuchten Klasse entspricht,
                    pClassPosteriori[cl]+=1/k     #erhöhe die Wahrscheinlichkeit um 1/k        
        prediction=np.argmax(pClassPosteriori)      => hier aber keine randomisierte Auswahl wenn mehrere Klassen mit höchster Wahrscheinlichkeit daher auskommentiert!
        '''
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
        self.kdtree=None

    def fit(self,X,T):
        """
        Train classifier by creating a kd-tree 
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        KNNClassifier.fit(self,X,T)                # call to parent class method (just store X and T)
        self.kdtree = scipy.spatial.KDTree(X)   # Do an indexing of the feature vectors by constructing a kd-tree
        
    def getKNearestNeighbors(self, x, k=None):  # realizes fast K-nearest-neighbor-search of x in data set X
        """
        fast computation of the k nearest neighbors for a query vector x given a data matrix X by using the KD-tree
        :param x: the query vector x
        :param k: number of nearest-neighbors to be returned
        :return idxNN: return list of k line indexes referring to the k nearest neighbors of x in X
        """
        if(k==None): k=self.k                      # do a K-NN search...      
        dd,ii = self.kdtree.query(x,k)             # do a K-NN search on the generated KD-Tree using scipy.spacial.KDtree.query()
        if k==1:                                   # if k==1 the query function does not return lists, therefore cast necessary
            idxNN=[ii]
        else:
            idxNN = ii                             #Store indexes of k nearest neighbors in list idxNN
        return idxNN                               # return indexes of k nearest neighbors