# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np

from .functions import *
from .utils import load_libsvm_file, random_point_on_simplex, random_point_in_l2_ball, \
    edge_point_on_simplex, generate_dataset_for_svm

from sklearn.datasets import load_digits


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
    x0 = 0.5 * np.ones(n)

    return f, h, L, x0


def Poisson_regr_simplex_acc(m, n, noise=0.01, normalizeA=True):
    x0 = random_point_on_simplex(n, center=False)
    solution = random_point_on_simplex(n, center=False)
    A = np.random.rand(m, n)
    if normalizeA:
        A = A / A.sum(axis=0)  # scaling to make column sums equal to 1

    b = np.dot(A, solution) + noise * (np.random.rand(m))
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    L = np.abs(b).sum()
    h = BurgEntropySimplex(eps=1e-7)
    h_euklid = SquaredL2Norm()

    return f, [h, h_euklid], L, x0


def Poisson_regr_simplex(m, n, noise=0.01, normalizeA=True):
    """
    Generate a random instance of the Poisson regression problem on the unit simplex
            minimize_{x \in simplex}  D_KL(b, Ax)
    where
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        normalizeA: whether to normalize columns of A

    Return f, h, L, x0:
        f: f(x) = D_KL(b, Ax)
        h: Divergence
        L: L = ||b||_2
        x0: initial point is center of simplex
    """
    key1 = 'x0_center_sol_center'
    key2 = 'x0_edge_sol_edge'
    key3 = 'x0_edge_sol_center'
    key4 = 'x0_center_sol_edge'

    def generate_problem(solution_and_x0):
        solution, x0 = solution_and_x0
        A = np.random.rand(m, n)
        if normalizeA:
            A = A / A.sum(axis=0)  # scaling to make column sums equal to 1

        b = np.dot(A, solution) + noise * (np.random.rand(m))
        assert b.min() > 0, "need b > 0 for nonnegative regression."

        f = PoissonRegression(A, b)
        L = b.sum()

        return f, L, solution, x0

    def generate_sol_and_x0(place):
        # Initial point should be far from 0 in order for ARDA to work well!

        if place == key1:
            x0 = random_point_on_simplex(n, center=True)
            solution = random_point_on_simplex(n)

            return solution, x0
        elif place == key2:
            x0 = edge_point_on_simplex(np.random.randint(n), n)
            solution = edge_point_on_simplex(np.random.randint(n), n)

            return solution, x0
        elif place == key3:
            x0 = edge_point_on_simplex(np.random.randint(n), n)
            solution = random_point_on_simplex(n, center=True)

            return solution, x0
        elif place == key4:
            x0 = random_point_on_simplex(n, center=True)
            solution = edge_point_on_simplex(np.random.randint(n), n)

            return solution, x0
        else:
            assert 0, 'Place had not been defined'

    points_positions = {key1: generate_problem(generate_sol_and_x0(key1)),
                        key2: generate_problem(generate_sol_and_x0(key2)),
                        key3: generate_problem(generate_sol_and_x0(key3)),
                        key4: generate_problem(generate_sol_and_x0(key4))}

    h = BurgEntropySimplex()

    return h, points_positions


def svm_digits_ds_divs_ball(center=None, lamda=0.5, real_ds=False):
    """
    SVM (binary classification) problem with l2 ball constraint, n=264
    where
        center:  ball constraint center
        lamda: lambda coef in SVM regularization
        real_ds: what will DS be used
    """

    if real_ds:
        X, Y = load_digits(n_class=2, return_X_y=True, as_frame=True)
        Y = (Y > 0).astype(int) * 2 - 1  # [0,1,2] --> [False,True,True] --> [0,1,1] --> [0,2,2] --> [-1,1,1]
        X = X.to_numpy()
        Y = Y.to_numpy()
    else:
        X, Y = generate_dataset_for_svm(700, 2000)

    f = SVM_fun(lamda, X, Y)

    n = X.shape[1]
    radius = min(n ** -1 * lamda ** -1 * np.sum(np.linalg.norm(X[:, :-1], axis=1)), (2 / lamda) ** 0.5)
    if center is None:
        center = np.zeros(n)

    [poly_h, sqL2_h] = PolyDiv(X, lamda=lamda, radius=radius), SquaredL2Norm()
    # we know an upper bound of L
    L = (poly_h.DS_mean + min((2*lamda)**0.5, poly_h.DS_mean_quad))*0.08
    x0 = random_point_in_l2_ball(center, radius, pos_dir=False)

    return f, [poly_h, sqL2_h], L, x0, radius


def FrobeniusSymLossEx(M, r, noise):
    X0 = np.random.rand(M.shape[0], r)
    upper_bound = 5
    assert np.all(X0 >= 0) >= 0, "X0 must be non-negative"
    f = FrobeniusSymLoss(M, X0)
    # h = SumOf2nd4thPowers(6, 2*np.linalg.norm(M, 2))
    h = SumOf2nd4thPowersPositiveOrthant(6, 2*np.linalg.norm(M, 2), upper_bound)
    h_dual = SumOf2nd4thPowersDualProxMap(6, 2*np.linalg.norm(M, 2))
    h_fw = SumOf2nd4thPowersWithFrankWolfe(6, 2*np.linalg.norm(M, 2), lmo_matrix_box(0, upper_bound))
    L = 1

    return f, [h, h_dual, h_fw], L, X0


def FrobeniusSymLossResMeasEx(M, r, noise=0.0):
    X0 = np.random.rand(M.shape[0], r)
    upper_bound = 5
    assert np.all(X0 >= 0) >= 0, "X0 must be non-negative"
    f = FrobeniusSymLoss(M, X0)
    # h = SumOf2nd4thPowers(6, 2*np.linalg.norm(M, 2))
    h = SumOf2nd4thPowersPositiveOrthant(6, 2*np.linalg.norm(M, 2), upper_bound=None)
    # h_fw = SumOf2nd4thPowersWithFrankWolfe(6, 2*np.linalg.norm(M, 2), lmo_matrix_box(0, upper_bound))
    h_euklid = SquaredL2Norm()
    L = 1

    return f, [h, h_euklid], L, X0


def FrobeniusSymLossExWithLinearCnstrnts(M, r, noise=0):
    X0 = np.random.rand(M.shape[0], r)
    # assert np.all(X0 >= 0), "X0 must be non-negative"
    f = FrobeniusSymLoss(M, X0)
    h = SumOf2nd4thPowersPositiveOrthant(6, 2*np.linalg.norm(M, 2))

    A = np.random.rand(r)
    b = np.random.rand(M.shape[0]) + 0.5
    g = AX_b(A, b)
    L = 1

    return f, g, h, L, X0
