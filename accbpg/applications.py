# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os


from .functions import *
from .utils import load_libsvm_file, generate_random_value_in_df, random_point_on_simplex, random_point_in_l2_ball, \
    edge_point_on_simplex

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


def Poisson_regr_simplex(m, n, radius=1, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
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

    x = random_point_on_simplex(n)

    b = np.dot(A, x) + noise * (np.random.rand(m))
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    [h1, h2, h3] = BurgEntropySimplex(), ShannonEntropy(), SquaredL2Norm()
    L = b.sum()
    # Initial point should be far from 0 in order for ARDA to work well!
    x0 = edge_point_on_simplex(80, n)

    return f, [h1, h2, h3], L, x0, x


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


def svm_digits_ds_divs_ball(center=None, lamda=0.5):
    """
    SVM problem with l2 ball constraint, n=264
    """
    X, Y = load_digits(n_class=2, return_X_y=True, as_frame=True)
    Y = (Y > 0).astype(int) * 2 - 1  # [0,1,2] --> [False,True,True] --> [0,1,1] --> [0,2,2] --> [-1,1,1]

    X = generate_random_value_in_df(X, Y, 200).to_numpy()
    Y = Y.to_numpy()

    f = SoftMarginLoss(lamda, X, Y)

    n = X.shape[1]
    radius = min(n ** -1 * lamda ** -1 * np.sum(np.linalg.norm(X[:, :-1], axis=1)), (2 / lamda) ** 0.5)
    if center is None:
        center = np.zeros(n)

    [poly_h, sqL2_h, kl] = PolyDiv(X, lamda=lamda), SquaredL2Norm(), PowerNeg1()
    L = poly_h.DS_mean + min((2*lamda)**0.5, poly_h.DS_mean_quad) - 0.01
    x0 = random_point_in_l2_ball(center, radius, pos_dir=False)
    # x0 = np.zeros(X.shape[1])
    # x0 += 1e-20
    # x0[4] = radius

    return f, [poly_h, sqL2_h, kl], L, x0, radius


def smv_digits_ds_divs_simplex(lamda=0.5):
    """
    SVM problem with simplex, n=264
    """
    X, Y = load_digits(n_class=2, return_X_y=True, as_frame=True)
    Y = (Y > 0).astype(int) * 2 - 1  # [0,1,2] --> [False,True,True] --> [0,1,1] --> [0,2,2] --> [-1,1,1]

    X = generate_random_value_in_df(X, Y, 200).to_numpy()
    Y = Y.to_numpy()

    f = SoftMarginLoss(lamda, X, Y)

    radius = 100
    [poly_h, sqL2_h, kl] = PolyDiv(X, lamda=lamda, B=radius), SquaredL2Norm(), ShannonEntropy()
    L = 0.1
    x0 = random_point_on_simplex(X.shape[1], radius)
    # x0 = np.zeros(X.shape[1])
    # x0 += 1e-20
    # x0[0] = radius

    # assert x0.sum() - radius <= 1e-12

    return f, [poly_h, sqL2_h, kl], L, x0, radius


def distributed_ridge_regression_problem(d, n, comp_nmbr=30, noise=0.1, lamda=0, randseed=-1):
    """
    \\todo
    """
    assert comp_nmbr > 0

    def generate_covariance_matrix(d):
        # Generate eigenvalues uniformly distributed in [1, 1000]
        eigenvalues = np.random.uniform(1, 1000, size=d)

        # Generate eigenvectors using QR decomposition of a random matrix
        random_matrix = np.random.randn(d, d)
        q, _ = np.linalg.qr(random_matrix)

        # Construct covariance matrix using eigenvalue decomposition
        covariance_matrix = np.sum([(eigenvalues[j] * np.outer(q[:, j], q[:, j])) for j in range(d)], axis=0)

        return covariance_matrix

    def generate_matrix_A(n, d, covariance_matrix):
        # Generate samples from a multivariate normal distribution
        mean = np.zeros(d)
        A = np.random.multivariate_normal(mean, covariance_matrix, size=n)
        return A

    def create_matrices(N, d):
        matrices = []
        for i in range(N):
            matrix = generate_covariance_matrix(d)
            matrices.append(matrix)
        return np.array(matrices)

    def save_matrices(matrices, filename):
        np.save(filename, matrices)

    def load_matrices(filename):
        return np.load(filename, allow_pickle=True)

    filename = "covariance_matrices.npy"
    if os.path.exists(filename):
        covariance_matrices = load_matrices(filename)
        print("Matrices loaded from file.")
    else:
        covariance_matrices = create_matrices(comp_nmbr, d)
        save_matrices(covariance_matrices, filename)
        print("Matrices created and saved to file.")

    comp_datas = []
    solution = np.random.normal(loc=5, scale=1, size=d)
    L = lamda
    for i in range(comp_nmbr):
        if randseed > 0:
            np.random.seed(randseed)

        covariance_matrix = covariance_matrices[i]
        A = generate_matrix_A(n, d, covariance_matrix)

        b = np.dot(A, solution) + noise * (np.random.rand(n) - 0.001)
        f = RidgeRegression(A, b, lamda)
        comp_datas.append(f)

        L += A.T.dot(A)

    f = DistributedRidgeRegression(np.array(comp_datas))
    similarity=math.sqrt((math.log(d/0.2)/1))
    [h1, h2] = DistributedRidgeRegressionDiv(comp_datas[0], similarity), SquaredL2Norm()
    L = np.linalg.norm(L, 'fro')/(comp_nmbr*n)
    radius = 100
    x0 = random_point_in_l2_ball(np.zeros(d), radius, pos_dir=False)

    return f, [h1, h2], L, x0, radius, solution


if __name__ == "__main__":
    pass
