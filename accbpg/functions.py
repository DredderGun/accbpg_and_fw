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


class RidgeRegression(RSmoothFunction):
    """
    \\ todo
    """

    def __init__(self, A, b, lamda):
        assert A.shape[0] == b.shape[0], "A and b sizes not matching"
        self.A = A
        self.b = b
        self.lamda = lamda
        self.n = A.shape[0]
        self.d = A.shape[1]

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def get_cvxpy_objective(self, x):
        lambd = cp.Parameter(nonneg=True, value=self.lamda)
        return cp.pnorm(self.A @ x - self.b, p=2) ** 2 + lambd * cp.pnorm(x, p=2) ** 2

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        assert x.size == self.d, "RidgeRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = (1 / (2*self.n)) * np.linalg.norm(Ax - self.b)**2 + self.lamda * np.linalg.norm(x)**2
            return fx

        g = (1 / self.n) * np.dot(self.A.T, (Ax - self.b)) + 2 * self.lamda * x
        if flag == 1:
            return g

        # return both function value and gradient
        fx = (1 / (2*self.n)) * sum((Ax - self.b)**2) + self.lamda * np.sum(x**2)
        return fx, g

    def get_mx_prod_itself(self, similarity):
        """
        Function computes 2\n A^\top A
        """

        return (2 / self.d)*np.dot(self.A.T, self.A) + np.eye(self.d) * similarity

    def get_mx_prod_b(self):
        """
        Function computes 2\n A^\top b
        """
        return (2 / self.d)*np.dot(self.A.T, self.b)

class DistributedRidgeRegression(RSmoothFunction):
    """
    \\ todo
    """

    def __init__(self, Nodes):
        self.Nodes = Nodes
        self.m = Nodes.shape[0]

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def get_cvxpy_solution(self):
        x = cp.Variable(self.Nodes[0].A.shape[1])
        def objective_fn(x):
            return sum([node.get_cvxpy_objective(x) for node in self.Nodes])
        problem = cp.Problem(cp.Minimize(objective_fn(x)))
        problem.solve()

        return x.value

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        if flag == 0:
            return 1 / self.m * sum(node() for node in self.Nodes)

        # use array broadcasting
        g = 1 / self.m * sum(node.gradient(x) for node in self.Nodes)
        if flag == 1:
            return g

        # return both function value and gradient
        fx = 1 / self.m * sum(node(x) for node in self.Nodes)
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
    h(x) = (1/2)||x||_2^2 + (M/4)||x||_2^4
    """       
    def __init__(self, M):
        self.M = M
    
    def __call__(self, x):
        normsq = np.dot(x, x)
        return 0.5 * normsq + (self.M / 4) * normsq**2

    def gradient(self, x):
        normsq = np.dot(x, x)         
        return (1 + self.M * normsq) * x

    def divergence(self, x, y):
        assert x.shape == y.shape, "Bregman div: x and y not same shape."
        return self.__call__(x) - (self.__call__(y) 
                                   + np.dot(self.gradient(y), x-y))

class SquaredL2Norm(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2
    """       
    def __call__(self, x):
        return 0.5*np.dot(x, x)

    def gradient(self, x):         
        return x

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return 0.5*np.dot(xy, xy)

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

    def __init__(self, DS, lamda=0, B=1):
        self.lamda = lamda
        self.DS = DS
        self.B = B

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

        regularizator = 10**3 # to make a solver works correct
        prob = cp.Problem(cp.Minimize((L / regularizator)*h_x + (g / regularizator)@x))
        solution = prob.solve(verbose=False, abstol=1e-7)

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


class DistributedRidgeRegressionDiv(LegendreFunction):
    """
    \\ todo
    """

    def __init__(self, f, similarity):
        assert similarity > 0, "BurgEntropyL1: lambda should be nonnegative."
        self.similarity = similarity
        self.f = f

    def __call__(self, x):
        return self.f(x) + 0.5 * self.similarity * np.linalg.norm(x) ** 2

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        return self(x) - self(y) - np.dot(self.gradient(y), x - y)

    def gradient(self, x):
        return self.f.gradient(x) + self.similarity*x

#######################################################################


def lmo_l2_ball(radius, center=None):
    """
    The Frank-Wolfe lmo function for the l2 ball on x > 0 and
    x \in ||radius - center||_2 <= radius
    """

    def f(g):
        if center is None:
            center_p = np.zeros(g.shape[0])
        else:
            center_p = np.array([center] * g.shape[0])
        s = center_p - radius * g/np.linalg.norm(g)
        s[s == 0] = 1e-20

        assert abs(np.linalg.norm(s - center_p) - radius) <= 1e-12

        return s

    return lambda g: f(g)


def lmo_linf_ball(radius, center=None):
    """
    The Frank-Wolfe lmo function for the l_\infty ball on x > 0 and
    x \in ||radius - center||_2 <= radius
    """
    assert center is None, "center are not implemented yet"

    def f(g):
        return np.ones(g.shape[0]) * radius * (-1) * np.sign(g)

    return lambda g: f(g)


def lmo_simplex(radius=1):
    """
    The Frank-Wolfe lmo function for the simplex
    """

    def f(g):
        s = np.zeros(g.shape)
        s += 1e-60
        s[np.argmin(g)] = radius  # for example see LMO for simplex e.g.: https://arxiv.org/abs/2106.10261v1 page 9

        return s

    return lambda g: f(g)
