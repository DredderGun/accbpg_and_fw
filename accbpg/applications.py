# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from .functions import *
from .utils import load_libsvm_file, random_point_in_l2_ball

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import pandas as pd

def D_opt_libsvm(filename):
    """
    Generate a D-Optimal Design problem from LIBSVM datasets
    """
    X, y = load_libsvm_file(filename)
    if X.shape[0] > X.shape[1]:
        H = X.T.toarray('C')
    else:
        H = X.toarray('C')
    n = H.shape[1]
     
    f = DOptimalObj(H)
    h = BurgEntropySimplex()
    L = 1.0
    x0 = (1.0/n)*np.ones(n)
    
    return f, h, L, x0
   

def D_opt_design(m, n, randseed=-1):
    """
    Generate a random instance of the D-Optimal Design problem
        m, n: size of design matrix H is m by n wiht m < n
    Return f, h, L, x0:
        f:  f(x) = - log(det(H*diag(x)*H'))
        h:  Burg Entrop with Simplex constraint
        L:  L = 1
        x0: initial point is center of simplex
    """

    if randseed > 0:
        np.random.seed(randseed)
    H = np.random.randn(m,n)

    f = DOptimalObj(H)
    h = BurgEntropySimplex()
    L = 1.0
    x0 = (1.0/n)*np.ones(n)
    
    return f, h, L, x0


def D_opt_KYinit(V):
    """
    Return a sparse initial point for MVE or D-optimal design problem
    proposed by Kuman and Yildirim (JOTA 126(1):1-21, 2005)

    """
    m, n = V.shape

    if n <= 2*m:
        return (1.0/n)*np.ones(n)
    
    I = []
    Q = np.zeros((m, m))
    # Using (unstable) Gram-Schmidt without calling QR repetitively
    for i in range(m):
        b = np.random.rand(m)
        q = np.copy(b)
        for j in range(i):
            Rij = np.dot(Q[:,j], b) 
            q = q - Rij * Q[:,j]
        qV = np.dot(q, V)
        kmax = np.argmax(qV)
        kmin = np.argmin(qV)
        I.append(kmax)
        I.append(kmin)
        v = V[:,kmin] - V[:,kmax]
        q = np.copy(v)
        for j in range(i):
            Rij = np.dot(Q[:,j], v) 
            q = q - Rij * Q[:,j]
        Q[:,i] = q / np.linalg.norm(q)
        
    x0 = np.zeros(n)
    x0[I] = np.ones(len(I)) / len(I)
    # in case there are repeated entries in I, scale to sum 1
    x0 /= x0.sum()  
    return x0
        

def Poisson_regrL1(m, n, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L1-regularized Poisson regression problem
            minimize_{x >= 0}  D_KL(b, Ax) + lamda * ||x||_1
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L1 regularization weight
        normalizeA: wether or not to normalize columns of A
    
    Return f, h, L, x0: 
        f: f(x) = D_KL(b, Ax)
        h: Burg entropy with L1 regularization 
        L: L = ||b||_1
        x0: initial point, scaled version of all-one vector
    """
    
    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m,n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n) / n
    xavg = x.sum() / x.size
    x = np.maximum(x - xavg, 0) * 10
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    # L1 regularization often not enough for convergence!
    h = BurgEntropyL1(lamda)
    L = b.sum()
    # Initial point should be far from 0 in order for ARDA to work well!
    x0 = (1.0/n)*np.ones(n) * 10

    return f, h, L, x0


def Poisson_regrL2(m, n, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L2-regularized Poisson regression problem
            minimize_{x >= 0}  D_KL(b, Ax) + (lamda/2) * ||x||_2^2
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A
    
    Return f, h, L, x0: 
        f: f(x) = D_KL(b, Ax)
        h: Burg entropy with L1 regularization 
        L: L = ||b||_1
        x0: initial point is center of simplex
    """

    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m,n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n) / n
    xavg = x.sum() / x.size
    x = np.maximum(x - xavg, 0) * 10
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    h = BurgEntropyL2(lamda)
    L = b.sum()
    # Initial point should be far from 0 in order for ARDA to work well!
    x0 = (1.0/n)*np.ones(n)

    return f, h, L, x0


def Poisson_regrL2_ball(m, n, radius=1, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L2-regularized Poisson regression problem
            minimize_{x \in B}  D_KL(b, Ax)
    where
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: whether to normalize columns of A

    Return f, h, L, x0:
        f: f(x) = D_KL(b, Ax)
        h: Burg entropy with L2 ball projection
        L: L = ||b||_2
        x0: initial point is center of simplex
    """

    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m, n)
    if normalizeA:
        A = A / A.sum(axis=0)  # scaling to make column sums equal to 1
    x = np.random.rand(n) / n
    xavg = x.sum() / x.size
    x = np.maximum(x - xavg, 0) * 10
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    h = BurgEntropyL2Ball(lamda, radius=radius)
    L = b.sum()
    # Initial point should be far from 0 in order for ARDA to work well!
    x0 = (1.0 / n) * np.ones(n)

    return f, h, L, x0


def Poisson_regr_diff_divs(m, n, radius=1, center=None, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
    if center is None:
        center = np.array([radius] * n)

    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m, n) * (radius / 2)
    if normalizeA:
        A = A / A.sum(axis=0)  # scaling to make column sums equal to 1
    x = random_point_in_l2_ball(center=center, radius=radius)
    xavg = x.sum() / x.size
    x = np.maximum(x - xavg, 0) + center
    assert np.linalg.norm(x - center) <= radius
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    [burg_h, sqL2_h, shannon] = BurgEntropy(), SquaredL2Norm(), ShannonEntropy()
    L = b.sum()
    x0 = random_point_in_l2_ball(center, radius)
    assert np.linalg.norm(x0 - center) <= radius

    return f, [burg_h, sqL2_h, shannon], L, x0, x


def KL_nonneg_regr(m, n, noise=0.01, lamdaL1=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L1-regularized KL regression problem
            minimize_{x >= 0}  D_KL(Ax, b) + lamda * ||x||_1
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A
    
    Return f, h, L, x0: 
        f: f(x) = D_KL(Ax, b)
        h: h(x) = Shannon entropy (with L1 regularization as Psi)
        L: L = max(sum(A, axis=0)), maximum column sum
        x0: initial point, scaled version of all-one vector
    """
    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m,n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n)
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = KLdivRegression(A, b)
    h = ShannonEntropyL1(lamdaL1)
    L = max( A.sum(axis=0) )    #L = 1.0 if columns of A are normalized
    x0 = 0.5*np.ones(n)
    #x0 = (1.0/n)*np.ones(n)

    return f, h, L, x0


def svm_alg_iris_ds(radius=1, center=None):
    iris = load_iris()
    X = iris.data
    Y = iris.target
    Y = (Y > 0).astype(int) * 2 - 1  # [0,1,2] --> [False,True,True] --> [0,1,1] --> [0,2,2] --> [-1,1,1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2024)

    f = SoftMarginLoss(0.5, X_train, Y_train)

    if center is None:
        center = np.array([radius + 1] * X_train.shape[1])

    h = BurgEntropyL2Ball(lamda=0.5, radius=radius, center=radius + 1)
    L = max(X_train.sum(axis=0))
    x0 = random_point_in_l2_ball(center, radius)
    assert np.linalg.norm(x0 - center) <= radius

    return f, h, L, x0

def svm_alg_banknote(radius=1, center=None):
    X = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')
    Y = X['0'].to_numpy()
    X = X.to_numpy()

    f = SoftMarginLoss(0.5, X, Y)

    if center is None:
        center = np.array([radius + 1] * X.shape[1])

    h = BurgEntropyL2Ball(lamda=0.5, radius=radius, center=radius + 1)
    L = max(X.sum(axis=0))
    x0 = random_point_in_l2_ball(center, radius)
    assert np.linalg.norm(x0 - center) <= radius

    return f, h, L, x0


if __name__ == "__main__":
    pass
