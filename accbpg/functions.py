# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import cvxpy as cp


class RSmoothFunction:
    """
    Relatively-Smooth Function, can query f(x) and gradient
    """
    def __call__(self, x):
        assert 0, "RSmoothFunction: __call__(x) is not defined"
        
    def gradient(self, x):
        assert 0, "RSmoothFunction: gradient(x) is not defined"
 
    def func_grad(self, x, flag):
        """
        flag=0: function, flag=1: gradient, flag=2: function & gradient 
        """
        assert 0, "RSmoothFunction: func_grad(x, flag) is not defined"


class DOptimalObj(RSmoothFunction):
    """
    f(x) = - log(det(H*diag(x)*H')) where H is an m by n matrix, m < n
    """
    def __init__(self, H):
        self.H = H
        self.m = H.shape[0]
        self.n = H.shape[1]
        assert self.m < self.n, "DOptimalObj: need m < n"
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)
        
    def gradient(self, x):
        return self.func_grad(x, flag=1)
        
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "DOptimalObj: x.size not equal to n"
        assert x.min() >= 0,     "DOptimalObj: x needs to be nonnegative"
        HXHT = np.dot(self.H*x, self.H.T)
        
        if flag == 0:       # only return function value
            f = -np.log(np.linalg.det(HXHT))
            return f
        
        HXHTinvH = np.dot(np.linalg.inv(HXHT), self.H)
        g = - np.sum(self.H * HXHTinvH, axis=0)

        if flag == 1:       # only return gradient
            return g
        
        # return both function value and gradient
        f = -np.log(np.linalg.det(HXHT))
        return f, g

    def func_grad_slow(self, x, flag=2):
        assert x.size == self.n, "DOptimalObj: x.size not equal to n"
        assert x.min() >= 0,     "DOptimalObj: x needs to be nonnegative"
        sx = np.sqrt(x)
        Hsx = self.H*sx;    # using numpy array broadcast
        HXHT = np.dot(Hsx,Hsx.T)
        
        if flag == 0:       # only return function value
            f = -np.log(np.linalg.det(HXHT))
            return f
        
        Hsx = np.linalg.solve(HXHT, self.H)
        g = np.empty(self.n)
        for i in range(self.n):
            g[i] = - np.dot(self.H[:,i], Hsx[:,i])
            
        if flag == 1:       # only return gradient
            return g
        
        # return both function value and gradient
        f = -np.log(np.linalg.det(HXHT))
        return f, g


class PoissonRegression(RSmoothFunction):
    """
    f(x) = D_KL(b, Ax) for linear inverse problem A * x = b
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b sizes not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)
    
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "PoissonRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = sum( self.b * np.log(self.b / Ax) + Ax - self.b )
            return fx

        # use array broadcasting
        g = ((1-self.b/Ax).reshape(self.m, 1) * self.A).sum(axis=0)
        # line above is the same as the following code
        #g = np.zeros(x.shape)
        #for i in range(self.m):
        #    g += (1 - self.b[i]/np.dot(self.A[i,:], x)) * self.A[i,:]
        if flag == 1:
            return g
        
        # return both function value and gradient
        fx = sum( self.b * np.log(self.b / Ax) + Ax - self.b )
        return fx, g


class KLdivRegression(RSmoothFunction):
    """
    f(x) = D_KL(Ax, b) for linear inverse problem A * x = b
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b size not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)
    
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "NonnegRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = sum( Ax * np.log(Ax / self.b) - Ax + self.b )
            return fx

        # use array broadcasting
        g = (np.log(Ax/self.b).reshape(self.m, 1) * self.A).sum(axis=0)
        # line above is the same as the following code
        #g = np.zeros(x.shape)
        #for i in range(self.m):
        #    g += np.log(Ax[i]/self.b[i]) * self.A[i,:]
        if flag == 1:
            return g
        
        # return both function value and gradient
        fx = sum( Ax * np.log(Ax / self.b) - Ax + self.b )
        return fx, g
           

class SVM_fun(RSmoothFunction):
    def __init__(self, lamda, A, y):
        self.lamda = lamda
        self.A = A
        self.y = y

    def __call__(self, x):
        return self.F(x)

    def hinge_loss(self, x):
        return np.mean(np.maximum(0, 1 - self.y * np.dot(self.A, x)))

    def F(self, x):
        return self.hinge_loss(x) + self.lamda/2 * np.dot(x, x)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def subgradient_loss(self, x):
        indicator = (self.y * np.dot(self.A, x) < 1).astype(int)
        return np.mean(indicator[:, np.newaxis] * self.y[:, np.newaxis] * self.A, axis=0)

    def func_grad(self, x, flag=2):

        if flag == 0:
            fx = self.F(x)
            return fx

        g = self.lamda * x - self.subgradient_loss(x)
        if flag == 1:
            return g

        fx = self.F(x)
        return fx, g

#######################################################################


class LegendreFunction:
    """
    Function of Legendre type, used as the kernel of Bregman divergence for
    composite optimization 
         minimize_{x in C} f(x) + Psi(x) 
    where f is L-smooth relative to a Legendre function h(x),
          Psi(x) is an additional simple convex function.
    """
    def __call__(self, x):
        assert 0, "LegendreFunction: __call__(x) is not defined."
        
    def extra_Psi(self, x):
        return 0
        
    def gradient(self, x):
        assert 0, "LegendreFunction: gradient(x) is not defined."

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        assert 0, "LegendreFunction: divergence(x,y) is not defined."

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        assert 0, "LegendreFunction: prox_map(x, L) is not defined."

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } 
        default implementation by calling prox_map(g - L*g(y), L)
        """
        assert y.shape == g.shape, "Vectors y and g should have same size." 
        assert L > 0, "Relative smoothness constant L should be positive."
        return self.prox_map(g - L*self.gradient(y), L)


class BurgEntropy(LegendreFunction):
    """
    h(x) = - sum_{i=1}^n log(x[i]) for x > 0
    """
    def __call__(self, x):
        assert x.min()>0, "BurgEntropy only takes positive arguments."
        return -sum(np.log(x))
    
    def gradient(self, x):
        assert x.min()>0, "BurgEntropy only takes positive arguments."
        return -1/x
    
    def divergence(self, x, y):
        assert x.shape == y.shape, "Vectors x and y are of different sizes."
        assert x.min() > 0 and y.min() > 0, "Entries of x or y not positive."
        return sum(x/y - np.log(x/y) - 1)        

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { <g, x> + L * h(x) } 
        This function needs to be replaced with inheritance
        """
        assert L > 0, "BurgEntropy prox_map only takes positive L value."
        assert g.min() > 0, "BurgEntropy prox_map only takes positive value."
        return L / g
           
    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x > C} { <g, x> + L * D(x,y) }
        This is a general function that works for all derived classes
        """
        assert y.shape == g.shape, "Vectors y and g are of different sizes." 
        assert y.min() > 0 and L > 0, "Either y or L is not positive."
        return self.prox_map(g - L*self.gradient(y), L)


class BurgEntropyL1(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem 
            min_{x > 0} f(x) + lamda * ||x||_1 
    """
    def __init__(self, lamda=0, x_max=1e4):
        assert lamda >= 0, "BurgEntropyL1: lambda should be nonnegative."
        self.lamda = lamda
        self.x_max = x_max

    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { lambda * ||x||_1 + <g, x> + L h(x) }
        !!! This proximal mapping may have unbounded solution x->infty
        """
        assert L > 0, "BurgEntropyL1: prox_map only takes positive L."
        assert g.min() > -self.lamda, "Not getting positive solution."
        #g = np.maximum(g, -self.lamda + 1.0 / self.x_max)
        return L / (self.lamda + g)

       
class BurgEntropyL2(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem 
            min_{x > 0} f(x) + (lambda/2) ||x||_2^2 
    """
    def __init__(self, lamda=0):
        assert lamda >= 0, "BurgEntropyL2: lamda should be nonnegative."
        self.lamda = lamda

    def extra_Psi(self, x):
        """
        return (lamda/2) * ||x||_2^2
        """
        return (self.lamda / 2) * np.dot(x, x)

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { (lamda/2) * ||x||_2^2 + <g, x> + L * h(x) }
        """
        assert L > 0, "BurgEntropyL2: prox_map only takes positive L value."
        gg = g / L
        lamda_L = self.lamda / L
        return (np.sqrt(gg*gg + 4*lamda_L) - gg) / (2 * lamda_L)

       
class BurgEntropySimplex(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i])  used in the context of solving 
    min_{x \in C} f(x)  where C is the standard simplex, with  Psi(x) = 0
    """
    def __init__(self, eps=1e-8):
        # eps is precision for solving prox_map using Newton's method
        assert eps > 0, "BurgEntropySimplex: eps should be positive."
        self.eps = eps
     
    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { <g, x> + L h(x) } where C is unit simplex
        """
        assert L > 0, "BergEntropySimplex prox_map only takes positive L."
        gg = g / L
        cmin = -gg.min()    # choose cmin to ensure min(gg+c) >= 0
        # first use bisection to find c such that sum(1/(gg+c)) > 0
        c = cmin + 1        
        while sum(1/(gg+c))-1 < 0:
            c = (cmin + c) / 2.0
        # then use Newton's method to find optimal c
        fc = sum(1/(gg+c))-1
        while abs(fc) > self.eps:
            fpc = sum(-1.0/(gg+c)**2)
            if (c - (c - fc / fpc)) == 0:
                break
            c = c - fc / fpc
            fc = sum(1/(gg+c))-1
        x = 1.0/(gg+c)
        return x
       

class BurgEntropyL2Ball(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem
            min_{x in ||B||_2} f(x)
    The ball must lie on the positive side of the axes! (x > 0)
    """
    def __init__(self, lamda=0, radius=1, center=None, delta=1e-30):
        assert lamda >= 0, "BurgEntropyL2Projection: lamda should be nonnegative."
        self.lamda = lamda
        self.radius = radius
        self.center = center
        self.delta = delta

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in ||B||_inf, x > 0} { <g, x> + L * D(x,y) }
        B are with center in 0 + radius due to Burg entropy positiveness

        0 < x <= 2*radius
        """
        assert y.shape == g.shape, "Vectors y and g are of different sizes."
        assert L > 0, "L is not positive."
        x = L / (g - L * self.gradient(y))
        if self.center is None:
            center = np.zeros(x.shape)
        else:
            center = np.array([self.center] * x.shape[0])

        x -= center
        x /= max(self.radius, np.linalg.norm(x))
        x *= self.radius
        x += center
        x[x == 0] = self.delta

        assert np.linalg.norm(x - center) - self.radius <= 1e-12

        return x


class ShannonEntropy(LegendreFunction):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    """
    def __init__(self, delta=1e-20):
        self.delta = delta
        
    def __call__(self, x):
        assert x.min() >= 0, "ShannonEntropy takes nonnegative arguments."
        xx = np.maximum(x, self.delta)
        return sum( xx * np.log(xx) )

    def gradient(self, x):         
        assert x.min() >= 0, "ShannonEntropy takes nonnegative arguments."
        xx = np.maximum(x, self.delta)
        return 1.0 + np.log(xx)

    def divergence(self, x, y):
        assert x.shape == y.shape, "Vectors x and y are of different shapes."
        assert x.min() >= 0 and y.min() >= 0, "Some entries are negative."
        #for i in range(x.size):
        #    if x[i] > 0 and y[i] == 0:
        #        return np.inf 
        return sum(x*np.log((x+self.delta)/(y+self.delta))) + (sum(y)-sum(x))        
        
    def prox_map(self, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * h(x) }
        """
        assert L > 0, "ShannonEntropy prox_map require L > 0."
        return np.exp(-g/L - 1)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * D(x,y) }
        """
        assert y.shape == g.shape, "Vectors y and g are of different sizes." 
        assert y.min() >= 0 and L > 0, "Some entries of y are negavie."
        #gg = g/L - self.gradient(y)
        #return self.prox_map(gg, 1)
        return y * np.exp(-g/L)
   

class ShannonEntropyL1(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x >=0 } f(x) + lamda * ||x||_1
    """
    def __init__(self, lamda=0, delta=1e-20): 
        ShannonEntropy.__init__(self, delta)
        self.lamda = lamda
        
    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()
       
    def prox_map(self, g, L):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + L * h(x) }
        """
        return ShannonEntropy.prox_map(self, self.lamda + g, L)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + L * D(x,y) }
        """
        return ShannonEntropy.div_prox_map(self, y, self.lamda + g, L)
   
       
class ShannonEntropySimplex(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x in C } f(x) where C is standard simplex 
    """
    
    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { <g, x> + L * h(x) } where C is unit simplex
        """
        assert L > 0, "ShannonEntropy prox_map require L > 0."
        x = np.exp(-g/L - 1)
        return x / sum(x)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { <g, x> + L*d(x,y) } where C is unit simplex
        """
        assert y.shape == g.shape, "Vectors y and g are of different shapes."
        assert y.min() > 0 and L > 0, "prox_map needs positive arguments."
        x = y * np.exp(-g/L)
        return x / sum(x)
   

class SumOf2nd4thPowers(LegendreFunction):
    """
    h(x) = (sigma/2)||x||_2^2 + (alpha/4)||x||_2^4
    """
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, x):
        if isinstance(x, np.ndarray) and x.ndim > 1:
            norm = np.linalg.norm(x, 'fro')
        else:
            norm = np.linalg.norm(x, 2)

        return (self.alpha / 4) * norm ** 4 + self.sigma / 2 * norm ** 2

    def gradient(self, x):
        if isinstance(x, np.ndarray):
            norm = np.linalg.norm(x, 'fro')
        else:
            norm = np.linalg.norm(x, 2)

        return (self.sigma + self.alpha * norm**2) * x

    def divergence(self, x, y):
        assert x.shape == y.shape, "Bregman div: x and y not same shape."
        return self.__call__(x) - (self.__call__(y)
                                   + np.vdot(self.gradient(y), x-y))

    def solve_cubic(self, c, alpha):
        """
        Finds the unique real root of the equation:
            z^3 - alpha * z^2 = c, with c > 0.

        Parameters:
        c (float): Constant term (must be > 0)
        alpha (float): Coefficient of z^2

        Returns:
        float: The unique real root of the equation
        """
        z = alpha / 3.0
        alpha3 = alpha ** 3
        delta = c ** 2 + 4 * alpha3 * c / 27.0
        sq_delta = np.sqrt(delta)

        b = 0.5 * c + alpha3 / 27.0

        z += np.cbrt(b + 0.5 * sq_delta)
        z += np.cbrt(b - 0.5 * sq_delta)

        return z

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } 
        default implementation by calling prox_map(g - L*g(y), L)
        """
        z = self.alpha*np.linalg.norm(y, 'fro')**2 + self.sigma
        y = z*y - (1/L)*g
        
        z = self.solve_cubic(self.alpha*np.linalg.norm(y, 'fro')**2, self.sigma)
        return y / z


class SumOf2nd4thPowersPositiveOrthant(SumOf2nd4thPowers):
    """
    h(x) = (sigma/2)||x||_2^2 + (alpha/4)||x||_2^4
    """
    def __init__(self, alpha, sigma, upper_bound=None):
        self.alpha = alpha
        self.sigma = sigma
        self.upper_bound = upper_bound
    
    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } where C is positive orhtant
        default implementation by calling prox_map(g - L*g(y), L)
        """
        z = self.alpha*np.linalg.norm(y, 'fro')**2 + self.sigma
        y = z*y - (1/L)*g
        y = np.clip(y, 0, self.upper_bound)

        z = self.solve_cubic(self.alpha*np.linalg.norm(y, 'fro')**2, self.sigma)
        return y / z


class SumOf2nd4thPowersOnSimplex(SumOf2nd4thPowers):
    """
    h(x) = (sigma/2)||X||_2^2 + (alpha/4)||X||_2^4
    used in the context of  min_{X in C} f(x) where C is the matrix simplex
    """
    def __init__(self, alpha, sigma, raduis=1.0, eps=1e-4):
        super().__init__(alpha, sigma)
        self.raduis = raduis
        self.eps = eps

    def _project_onto_simplex(self, y):
        """
        Projects a matrix y onto the unit simplex (non-negative entries summing to 1),
        using Newton's method with bisection fallback.
        """
        y_flat = y.flatten()
        cmin = -np.min(y_flat)
        c = cmin + 1.0
        
        # Bisection to find a valid starting point
        while np.sum(1 / (y_flat + c)) - 1 < 0:
            c = 0.5 * (cmin + c)
        
        # Newton's method
        f_c = np.sum(1 / (y_flat + c)) - 1
        while abs(f_c) > self.eps:
            f_prime_c = np.sum(-1.0 / (y_flat + c) ** 2)
            delta = f_c / f_prime_c
            if delta == 0:
                break
            c -= delta
            f_c = np.sum(1 / (y_flat + c)) - 1
        
        x_flat = 1.0 / (y_flat + c)
        return x_flat.reshape(y.shape)


    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { <g, x> + L*d(x,y) } where C is the matrix simplex
        """
        assert y.shape == g.shape, "Vectors y and g are of different shapes."
        # assert y.min() >= 0 and L > 0, "prox_map needs positive arguments."

        # norm_y = np.linalg.norm(y, 'fro')
        # z = self.alpha * norm_y**2 + self.sigma
        # y = z * y - (1 / L) * g

        # # Projection onto simplex (non-negativity + sum = 1)
        # y = self._project_onto_simplex(y)

        # norm_y = np.linalg.norm(y, 'fro')
        # z = self.solve_cubic(self.alpha * norm_y**2, self.sigma)
        x_next = super().div_prox_map(y, g, L)
        result = self._project_onto_simplex(x_next)

        return result


class SumOf2nd4thPowersDualProxMap(SumOf2nd4thPowers):
    """
    h(x) = (sigma/2)||X||_2^2 + (alpha/4)||X||_2^4
    used in the context of  min_{X in C} f(x) where C is ...
    """
    def __init__(self, alpha, sigma):
        super().__init__(alpha, sigma)

    def project_X(self, X, radius=15.0):
        X_nonneg = np.maximum(X, 0)
        norm = np.linalg.norm(X_nonneg)
        return X if norm <= radius else (radius / norm) * X_nonneg

    def project_lambda(self, lam):
        return np.maximum(lam, 0.0)

    def div_prox_map(self, y, g, L, num_iters=7000, eta_base=1e-2, tolerance=1e-4, verbose=False):
        G = g + self.gradient(y)
        X = np.zeros_like(y)
        lambda_ = np.zeros_like(y)

        def f_func(X_val, lambda_val):
            norm_X_sq = np.sum(X_val**2)
            return (np.sum(G * (X_val - y))
                    - L * (self.alpha / 4 * norm_X_sq**2 + self.sigma / 2 * norm_X_sq)
                    - np.sum(lambda_val * X_val))

        for i in range(num_iters):
            # eta = eta_base / np.sqrt(i + 1)
            eta = eta_base / (i + 1)
            
            norm_X_sq = np.sum(X**2)
            grad_X = G - L * (self.alpha * norm_X_sq * X + self.sigma * X) - lambda_
            grad_lambda = -X

            X_mid = self.project_X(X + eta * grad_X)
            lambda_mid = self.project_lambda(lambda_ - eta * grad_lambda)
            
            norm_X_mid_sq = np.sum(X_mid**2)
            grad_X_mid = G - L * (self.alpha * norm_X_mid_sq * X_mid + self.sigma * X_mid) - lambda_mid
            grad_lambda_mid = -X_mid

            X = self.project_X(X + eta * grad_X_mid)
            lambda_ = self.project_lambda(lambda_ - eta * grad_lambda_mid)
            
            X_candidate = self.project_X(X + eta * grad_X)
            lambda_candidate = self.project_lambda(lambda_ - eta * grad_lambda)
            f_primal = f_func(X_candidate, lambda_)
            f_dual = f_func(X, lambda_candidate)
            gap = f_primal - f_dual
            
            if verbose and i % 100 == 0:
                print(f"[Iter {i:4d}] Gap: {gap:.6f}, f(X,λ): {f_func(X, lambda_):.6f}")
            
            if i > 0 and gap < tolerance:
                if verbose:
                    print(f"Stopping at iteration {i} as gap {gap:.6f} < tolerance {tolerance:.6f}")
                break

        return X


class SumOf2nd4thPowersWithFrankWolfe(LegendreFunction):
    """
    h(x) = (sigma/2)||x||_2^2 + (alpha/4)||x||_2^4
    """
    def __init__(self, alpha, sigma, lmo):
        self.alpha = alpha
        self.sigma = sigma
        self.lmo = lmo

    def __call__(self, x):
        if isinstance(x, np.ndarray) and x.ndim > 1:
            normsq = np.linalg.norm(x, 'fro')
        else:
            normsq = np.linalg.norm(x, 2)

        return (self.alpha / 4) * normsq ** 4 + self.sigma / 2 * normsq ** 2

    def gradient(self, x):
        if isinstance(x, np.ndarray):
            normsq = np.linalg.norm(x, 'fro')
        else:
            normsq = np.linalg.norm(x, 2)

        return (self.sigma + self.alpha * normsq**2) * x

    def divergence(self, x, y):
        assert x.shape == y.shape, "Bregman div: x and y not same shape."
        return self.__call__(x) - (self.__call__(y)
                                   + np.vdot(self.gradient(y), x-y))

    def div_prox_map(self, y, g, L):
        """
        Return LMO(g) where LMO is the linear minimization oracle from FW algorithm
        """
        return self.lmo(g)
    

class SquaredL2Norm(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2
    """       
    def __call__(self, x):
        return 0.5*np.vdot(x, x)

    def gradient(self, x):         
        return x

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return 0.5*np.vdot(xy, xy)

    def prox_map(self, g, L):
        assert L > 0, "SquaredL2Norm: L should be positive."
        return -(1/L)*g
        
    def div_prox_map(self, y, g, L):
        assert y.shape == g.shape and L > 0, "Vectors y and g not same shape."
        return y - (1/L)*g


class PowerNeg1(LegendreFunction):
    """
    h(x) = 1/x  for x>0
    """       
    def __call__(self, x):
        return 1/x

    def gradient(self, x):         
        return -1/(x*x)

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return np.sum(xy*xy/(x*y*y))

    def prox_map(self, g, L):
        assert L > 0, "SquaredL2Norm: L should be positive."
        return np.sqrt(L/g)
        

class L2L1Linf(LegendreFunction):
    """
    usng h(x) = (1/2)||x||_2^2 in solving problems of the form
    
        minimize    f(x) + lamda * ||x||_1
        subject to  ||x||_inf <= B
        
    """       
    def __init__(self, lamda=0, B=1): 
        self.lamda = lamda
        self.B = B
        
    def __call__(self, x):
        return 0.5*np.dot(x, x)

    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * np.sum(abs(x))

    def gradient(self, x):         
        """
        gradient of h(x) = (1/2)||x||_2^2
        """
        return x

    def divergence(self, x, y):
        """
        Bregman divergence D(x, y) = (1/2)||x-y||_2^2
        """
        assert x.shape == y.shape, "L2L1Linf: x and y not same shape."
        xy = x - y
        return 0.5*np.dot(xy, xy)

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        assert L > 0, "L2L1Linf: L should be positive."
        x = -(1.0/L) * g
        threshold = self.lamda / L
        x[abs(x) <= threshold] = 0
        x[x >  threshold] -= threshold
        x[x < -threshold] += threshold
        np.clip(x, -self.B, self.B, out=x)
        return x
        
    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } 
        """
        assert y.shape == g.shape and L > 0, "Vectors y and g not same shape."
        return self.prox_map(g - L*y, L)


class PolyDiv(LegendreFunction):
    """
    A divegrence with reference function from https://arxiv.org/pdf/1710.04718.pdf (27) with constraint on l2 ball
    """

    def __init__(self, DS, lamda=0, radius=1):
        self.lamda = lamda
        self.radius = radius
        self.DS = DS

        self.DS_mean = np.mean(np.linalg.norm(DS, axis=1))
        self.DS_mean_quad = np.mean(np.linalg.norm(DS, axis=1) ** 2)

    def __call__(self, x):
        """
        https://arxiv.org/pdf/1710.04718.pdf (27)
        """
        return self.h(x)

    def h(self, x):
        norm4 = 1/4 * np.linalg.norm(x)**4
        norm3 = 1/3 * np.linalg.norm(x)**3
        norm2 = 1/2 * np.linalg.norm(x)**2

        return self.lamda**2 * norm4 + 2*self.lamda*self.DS_mean*norm3 + self.DS_mean_quad*norm2

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        x = cp.Variable(g.shape[0])
        h_x = (self.lamda ** 2 * 1 / 4 * cp.norm(x) ** 4 + self.lamda * 2 / 3 * self.DS_mean * cp.norm(x) ** 3 +
               self.DS_mean_quad * 1 / 2 * cp.norm(x) ** 2)

        g_norm = np.linalg.norm(g)
        if g_norm == 0.0:
            g_norm = 1e-8
        g = (g / g_norm) * self.radius

        prob = cp.Problem(cp.Minimize(L*h_x + g@x), [cp.norm(x) <= self.radius])
        prob.solve(solver=cp.SCS, verbose=True)

        return x.value

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  }
        default implementation by calling prox_map(g - L*g(y), L)
        """
        grad_h_y = self.gradient(y)
        return self.prox_map(g - L * grad_h_y, L)


    def extra_Psi(self, x):
        return 0

    def gradient(self, x):
        """
        gradient of h(x)
        """
        return (self.lamda**4 * np.linalg.norm(x)**2 + 2*self.lamda*self.DS_mean + self.DS_mean_quad) * x

    def divergence(self, x, y):
        """
        Bregman divergence D(x, y) = h(x) - h(y) - \nabla h(y) (x - y)
        """
        assert x.shape == y.shape, "PolyDivBall: x and y not same shape."
        return self.h(x) - self.h(y) - np.dot(self.gradient(y), x - y)


class FrobeniusSymLoss(RSmoothFunction):
    """
    f(X) = 0.5 * ||M - X @ X.T||^2
    """

    def __init__(self, M: np.ndarray, X_init: np.ndarray, noise_level=None):
        assert np.allclose(M, M.T), "Matrix M must be symmetric."
        self.M = M
        self.M_norm = np.linalg.norm(M)
        self.XM = np.zeros_like(X_init) # preallocated X @ alpha
        self.G = np.zeros_like(X_init) # preallocated gradient
        self.noise_level = noise_level

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def frobenius_sym_loss(self, X: np.ndarray) -> float:
        """
        Efficient computation of the SymNMF loss function:
        0.5 * ||M - A @ A.T||^2
        with preallocated X @ A.
        """
        # assert np.all(X >= 0.0), "X must be non-negative"
        M = self.M
        XM = self.XM
        t1 = 0.5 * (self.M_norm ** 2 + np.linalg.norm(X.T @ X) ** 2)  # X.T @ X is small r x r
        np.dot(M, X, out=XM)
        return t1 - np.dot(X.ravel(), XM.ravel())

    def grad_snmf_inplace(self, X: np.ndarray, is_normed=False) -> np.ndarray:
        """
        Computes the gradient of the SNMF objective function
            0.5 * ||M - X @ X.T||^2
        with preallocated M @ X.
        """
        M = self.M
        np.dot(M, X, out=self.XM)
        np.dot(X, X.T @ X, out=self.G)
        np.subtract(2 * self.G, 2 * self.XM, out=self.G)
        if is_normed:
            G_norm = np.linalg.norm(self.G)
            if G_norm > 1e-9:
                self.G /= G_norm
        return self.G

    def func_grad(self, X, flag=2):
        noise_vector = 0
        if self.noise_level is not None:
            # noise_vector = (np.random.randn(*self.M.shape) - 0.5)*self.noise_level
            noise_vector = (np.random.randn(*X.shape) - 0.5)*self.noise_level

        if flag == 0:  # only return function value
            return self.frobenius_sym_loss(X)

        g = self.grad_snmf_inplace(X)

        if flag == 1:  # only return gradient
            return g + noise_vector

        # return both function value and gradient
        f = self.frobenius_sym_loss(X)
        return f, g + noise_vector

    def grad_snmf(self, M: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the SNMF objective function
            0.5 * ||M - A @ A.T||^2
        """
        MA = M @ A
        AAtA = A @ (A.T @ A)  # Order is important for efficiency
        return 2 * (AAtA - MA)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * D(x,y) }
        """
        X = cp.Variable(g.shape, nonneg=True)  # Enforce nonnegativity constraint
        
        # Scale down large values
        scale_factor = max(1.0, max(
            np.abs(g).max(),
            np.abs(self.gradient(y)).max()
        )) * 1e-6
        
        # Scale the inputs
        scaled_g = g / scale_factor
        scaled_grad_y = self.gradient(y) / scale_factor
        scaled_L = L / scale_factor
        
        # Define the objective function components
        linear_term = cp.sum(cp.multiply(scaled_g - scaled_L*scaled_grad_y, X))
        
        # For numerical stability, split into smaller terms
        normsq = cp.norm(X, 'fro')**2 if X.ndim > 1 else cp.sum_squares(X)
        quartic_term = (scaled_L * self.alpha / 4) * cp.power(normsq, 2)
        quadratic_term = (scaled_L * self.sigma / 2) * normsq
        
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * normsq
        
        # The full objective
        objective = cp.Minimize(linear_term + quartic_term + quadratic_term + reg_term)
        
        # Solve the problem with multiple solvers and parameters
        problem = cp.Problem(objective)
        try:
            # Try SCS first with adjusted parameters
            problem.solve(solver=cp.SCS, verbose=False,
                         eps=1e-8, 
                         alpha=1.8,  # Over-relaxation parameter
                         scale=1,    # Auto-scaling
                         max_iters=10000)
        except:
            pass
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            try:
                # Try MOSEK with adjusted parameters
                problem.solve(solver=cp.MOSEK, verbose=False,
                            mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8})
            except:
                # Final attempt with ECOS with adjusted parameters
                problem.solve(solver=cp.ECOS, verbose=False,
                             abstol=1e-8,
                             reltol=1e-8,
                             feastol=1e-8,
                             max_iters=1000)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            # If all solvers fail, return a feasible point
            return np.maximum(np.zeros_like(g), y)
        
        return np.maximum(X.value, 0)  # Ensure output is nonnegative


class AX_b(RSmoothFunction):
    """
    f(X) = AX - b

    X can be matrix or vector. A can be matrix or vector.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

    def __call__(self, x):
        return self.func_grad(x, flag=0)
        
    def gradient(self, x):
        return self.func_grad(x, flag=1)
        
    def func_grad(self, X, flag=2):
        if flag == 0:
            f = X.dot(self.A) - self.b
            return f
        
        g = self.A

        if flag == 1:
            return g
        
        f = self.A.dot(X) - self.b
        return f, g
    

def calculate_tse_constant(
    h: LegendreFunction,
    x_next: np.ndarray,
    y: np.ndarray,
    z_next: np.ndarray,
    z: np.ndarray,
    alpha: float
) -> float:
    """
    Calculate the maximal triangle scaling exponent (TSE) for an accelerated gradient method.

    Parameters
    ----------
    h : LegendreFunction
        A divergence-generating function.
    x_next : np.ndarray
        A point on the left-hand side of the inequality.
    y : np.ndarray
        A reference point on the left-hand side.
    z_next : np.ndarray
        A point on the right-hand side of the inequality.
    z : np.ndarray
        A reference point on the right-hand side.
    alpha : float
        Step size, must be in the interval (0, 1).

    Returns
    -------
    float
        The maximal triangle scaling exponent γ satisfying the inequality.
    """
    if not (0 < alpha < 1):
        raise ValueError("Step size alpha must be in the interval (0, 1)")

    lhs = h.divergence(x_next, y)
    rhs_base = h.divergence(z_next, z)

    gamma = 2.5
    # max_iters = 100000

    while True:
        if lhs <= alpha ** gamma * rhs_base or gamma < 1.01:
            break
        gamma -= 1e-5

    return gamma


#######################################################################

def lmo_nuclear_norm_ball():
    """
    The Frank-Wolfe lmo for matrix completion problem.
    """

    def f(g):
        U, S, Vh = np.linalg.svd(g, full_matrices=False)
        return np.outer(U[:, 0], Vh[0])

    return lambda g: f(g)

def lmo_l2_ball(radius, center=None):
    """
    The Frank-Wolfe lmo function for the l2 ball:
    x \in {x : ||x - center||_2 <= radius}
    
    Parameters:
    -----------
    radius : float
        Radius of the l2 ball
    center : array-like or None
        Center of the ball. If None, center is set to zero vector
        
    Returns:
    --------
    function
        Linear minimization oracle that projects onto the l2 ball
    """

    def f(g):
        if center is None:
            center_p = np.zeros_like(g)
        else:
            center_p = np.broadcast_to(center, g.shape)
            
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-10:
            return center_p
            
        s = center_p - radius * g/g_norm
        
        assert abs(np.linalg.norm(s - center_p) - radius) <= 1e-12, \
            "Solution does not lie on ball boundary"
            
        return s

    return lambda g: f(g)


def lmo_l2_ball_positive_orthant(radius, center=None):
    """
    The Frank-Wolfe lmo function for the l2 ball with constraints:
    1. x >= 0 (positive orthant)
    2. ||x - center||_2 <= radius
    
    Parameters:
    -----------
    radius : float
        Radius of the l2 ball
    center : array-like or None
        Center of the ball. If None, center is set to zero vector
        
    Returns:
    --------
    function
        Linear minimization oracle that projects onto the constrained l2 ball
    """
    def f(g):
        if center is None:
            center_p = np.zeros(g.shape[0])
        else:
            center_p = np.array([center] * g.shape[0])
            
        # Project gradient onto positive orthant
        g_proj = g.copy()
        g_proj[g_proj < 0] = 0
        
        # Handle zero gradient case
        g_norm = np.linalg.norm(g_proj)
        if g_norm < 1e-10:
            # Return point on sphere boundary that satisfies positivity
            s = np.zeros_like(g)
            s[0] = radius  # Put all weight in first coordinate
            return s + center_p
            
        # Calculate point on sphere that minimizes linear objective
        s = center_p - radius * g_proj/g_norm
        
        # Project onto positive orthant
        s = np.maximum(s, 1e-20)
        
        # Note: After projection, the point might be inside the ball, which is valid
        return s

    return lambda g: f(g)


def lmo_linf_ball(radius, center=None):
    """
    The Frank-Wolfe linear minimization oracle (LMO) for the l_∞ ball:
    x ∈ {x : ||x - center||_∞ <= radius}
    
    Parameters:
    -----------
    radius : float
        Radius of the l_∞ ball
    center : array-like or None
        Center of the ball. If None, center is set to zero vector
        
    Returns:
    --------
    function
        Linear minimization oracle that returns a vertex of the l_∞ ball
    """
    def f(g):
        if center is None:
            center_p = np.zeros_like(g)
        else:
            center_p = np.array(center)
            
        # For l_∞ ball, the solution is a vertex where we move radius units 
        # in the opposite direction of the gradient's sign
        s = center_p - radius * np.sign(g)
        return s

    return lambda g: f(g)


def lmo_simplex(radius=1):
    """
    The Frank-Wolfe Linear Minimization Oracle (LMO) for the simplex constraint.

    Parameters:
    -----------
    radius : float, optional
        The radius of the simplex. Default is 1.

    Returns:
    --------
    function
        A function that takes a gradient vector `g` and returns the vertex of the simplex
        that minimizes the linear approximation of the objective function.
    """

    def f(g):
        s = np.zeros(g.shape)
        s += 1e-60
        min_indices = np.where(g == np.min(g))[0]
        s[min_indices[0]] = radius
        return s

    return f


def lmo_matrix_simplex(radius=1.0):
    """
    Frank-Wolfe LMO for the matrix simplex constraint:
    { X in R^{n x m} | X >= 0, sum(X) = radius }

    Parameters:
    -----------
    radius : float, optional
        The total sum of the elements in the matrix (i.e., the 'radius' of the simplex).

    Returns:
    --------
    function
        A function that takes a gradient matrix `G` and returns a matrix `S` in the
        matrix simplex that minimizes <G, S>.
    """
    
    def f(G):
        S = np.zeros_like(G)
        S += 1e-60
        min_index = np.unravel_index(np.argmin(G), G.shape)
        S[min_index] = radius
        return S

    return f


def lmo_matrix_box(lower, upper):
    """
    Frank-Wolfe LMO for general box constraints on a matrix:
    { X in R^{n x m} | lower[i][j] <= X[i][j] <= upper[i][j] }

    Parameters:
    -----------
    lower : ndarray
        Lower bounds matrix of shape (n, m)
    upper : ndarray
        Upper bounds matrix of shape (n, m)

    Returns:
    --------
    function
        A function that takes a gradient matrix `G` and returns a matrix `S` in the
        box constraint that minimizes <G, S>.
    """
    def f(G):
        S = np.where(G < 0, upper, lower)
        return S

    return f
