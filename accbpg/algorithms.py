# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import time

from accbpg.utils import get_random_float, get_random_vector


def BPG(f, h, L, x0, maxitrs, epsilon=1e-14, linesearch=True, ls_ratio=1.2,
        verbose=True, verbskip=1):
    """
    Bregman Proximal Gradient (BGP) method for min_{x in C} f(x) + Psi(x): 
        
    x(k+1) = argmin_{x in C} { Psi(x) + <f'(x(k)), x> + L(k) * D_h(x,x(k))}
 
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        epsilon:  stop if F(x[k])-F(x[k-1]) < epsilon, where F(x)=f(x)+Psi(x)
        linesearch:  whether or not perform line search (True or False)
        ls_ratio: backtracking line search parameter >= 1
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        F:  array storing F(x[k]) for all k
        Ls: array storing local Lipschitz constants obtained by line search
        T:  array storing time used up to iteration k
    """

    if verbose:
        print("\nBPG_LS method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)         Lk       time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        
        if linesearch:
            L = L / ls_ratio
            x1 = h.div_prox_map(x, g, L)
            while f(x1) > fx + np.dot(g, x1-x) + L*h.divergence(x1, x):
                L = L * ls_ratio
                x1 = h.div_prox_map(x, g, L)
            x = x1
        else:
            x = h.div_prox_map(x, g, L)

        # store and display computational progress
        Ls[k] = L
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:6.1f}".format(k, F[k], L, T[k]))
            
        # stopping criteria
        if k > 0 and abs(F[k]-F[k-1]) < epsilon:
            break;

    F = F[0:k+1]
    Ls = Ls[0:k+1]
    T = T[0:k+1]
    return x, F, Ls, T


def solve_theta(theta, gamma, gainratio=1):
    """
    solve theta_k1 from the equation
    (1-theta_k1)/theta_k1^gamma = gainratio * 1/theta_k^gamma
    using Newton's method, starting from theta
    
    """
    ckg = theta**gamma / gainratio
    cta = theta
    eps = 1e-6 * theta
    phi = cta**gamma - ckg*(1-cta)
    while abs(phi) > eps:
        drv = gamma * cta**(gamma-1) + ckg
        cta = cta - phi / drv
        phi = cta**gamma - ckg*(1-cta)
        
    return cta
      

def ABPG(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, theta_eq=False, 
         restart=False, restart_rule='g', verbose=True, verbskip=1):
    """
    Accelerated Bregman Proximal Gradient (ABPG) method for solving 
            minimize_{x in C} f(x) + Psi(x): 

    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma:    triangle scaling exponent (TSE) for Bregman div D_h(x,y)
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        theta_eq: calculate theta_k by solving equality using Newton's method
        restart:  restart the algorithm when overshooting (True or False)
        restart_rule: 'f' for function increasing or 'g' for gradient angle
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x: the last iterate of BPG
        F: array storing F(x[k]) for all k
        G: triangle scaling gains D(xk,yk) / D(zk,zk_1) / theta_k^gamma
        T: array storing time used up to iteration k
    """

    if verbose:
        print("\nABPG method for minimize_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)     time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    z = np.copy(x0)
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x     # only required for restart mode
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x + theta*z_1
        g = f.gradient(y)
        z = h.div_prox_map(z_1, g, theta**(gamma-1) * L)
        x = (1-theta)*x + theta*z

        # compute triangle scaling quantities
        dxy = h.divergence(x, y)
        dzz = h.divergence(z, z_1)
        Gdr = dxy / dzz / theta**gamma

        # store and display computational progress
        G[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, F[k], theta, Gdr, dxy, dzz, T[k]))

        # restart if gradient predicts objective increase
        kk += 1
        if restart and k > 0:
            #if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:
            if (restart_rule == 'f' and F[k] > F[k-1]) or (restart_rule == 'g' and np.dot(g, x-x_1) > 0):
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    G = G[0:k+1]
    T = T[0:k+1]
    return x, F, G, T


def ABPG_expo(f, h, L, x0, gamma0, maxitrs, epsilon=1e-14, delta=0.2, 
              theta_eq=True, checkdiv=False, Gmargin=10, restart=False, 
              restart_rule='g', verbose=True, verbskip=1):
    """
    Accelerated Bregman Proximal Gradient method with exponent adaption for
            minimize_{x in C} f(x) + Psi(x) 
 
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma0:   initial triangle scaling exponent(TSE) for D_h(x,y) (>2)
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        delta:    amount to decrease TSE for exponent adaption
        theta_eq: calculate theta_k by solving equality using Newton's method
        checkdiv: check triangle scaling inequality for adaption (True/False)
        Gmargin:  extra gain margin allowed for checking TSI
        restart:  restart the algorithm when overshooting (True or False)
        restart_rule: 'f' for function increasing or 'g' for gradient angle
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        F:  array storing F(x[k]) for all k
        Gamma: gamma_k obtained at each iteration
        G:  triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma_k
        T:  array storing time used up to iteration k
    """
    
    if verbose:
        print("\nABPG_expo method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta       gamma" +
              "        TSG       D(x+,y)     D(z+,z)     time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    Gamma = np.ones(maxitrs) * gamma0
    T = np.zeros(maxitrs)
    
    gamma = gamma0
    x = np.copy(x0)
    z = np.copy(x0)
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x_1 + theta*z_1
        #g = f.gradient(y)
        fy, g = f.func_grad(y)
        
        condition = True
        while condition:    # always execute at least once per iteration 
            z = h.div_prox_map(z_1, g, theta**(gamma-1) * L)
            x = (1-theta)*x_1 + theta*z

            # compute triangle scaling quantities
            dxy = h.divergence(x, y)
            dzz = h.divergence(z, z_1)
            Gdr = dxy / dzz / theta**gamma

            if checkdiv:
                condition = (dxy > Gmargin * (theta**gamma) * dzz )
            else:
                condition = (f(x) > fy + np.dot(g, x-y) + theta**gamma*L*dzz)
                
            if condition and gamma > 1:
                gamma = max(gamma - delta, 1)
            else:
                condition = False
               
        # store and display computational progress
        G[k] = Gdr
        Gamma[k] = gamma
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:10.3e}  {7:6.1f}".format(
                    k, F[k], theta, gamma, Gdr, dxy, dzz, T[k]))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            #if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:
            if (restart_rule == 'f' and F[k] > F[k-1]) or (restart_rule == 'g' and np.dot(g, x-x_1) > 0):
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    Gamma = Gamma[0:k+1]
    G = G[0:k+1]
    T = T[0:k+1]
    return x, F, Gamma, G, T


def ABPG_gain(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, G0=1, 
              ls_inc=1.2, ls_dec=1.2, theta_eq=True, checkdiv=False, 
              restart=False, restart_rule='g', verbose=True, verbskip=1):
    """
    Accelerated Bregman Proximal Gradient (ABPG) method with gain adaption for 
            minimize_{x in C} f(x) + Psi(x): 
    
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma:    triangle scaling exponent(TSE) for Bregman distance D_h(x,y)
        G0:       initial value for triangle scaling gain
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        ls_inc:   factor of increasing gain (>=1)
        ls_dec:   factor of decreasing gain (>=1)
        theta_eq: calculate theta_k by solving equality using Newton's method
        checkdiv: check triangle scaling inequality for adaption (True/False)
        restart:  restart the algorithm when overshooting (True/False)
        restart_rule: 'f' for function increasing or 'g' for gradient angle
        verbose:  display computational progress (True/False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        F:  array storing F(x[k]) for all k
        Gain: triangle scaling gains G_k obtained by LS at each iteration
        Gdiv: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma_k
        Gavg: geometric mean of G_k at all steps up to iteration k
        T:  array storing time used up to iteration k
    """
    if verbose:
        print("\nABPG_gain method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta         Gk" + 
              "         TSG       D(x+,y)     D(z+,z)      Gavg       time")

    start_time = time.time()    
    F = np.zeros(maxitrs)
    Gain = np.ones(maxitrs) * G0
    Gdiv = np.zeros(maxitrs)
    Gavg = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    z = np.copy(x0)
    G = G0
    # logGavg = (gamma*log(G0) + log(G_1) + ... + log(Gk)) / (k+gamma)
    sumlogG = gamma * np.log(G) 
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        # adaptive option: always try a smaller Gain first before line search
        G_1 = G
        theta_1 = theta
        
        G = G / ls_dec
        
        condition = True
        while condition:
            if kk > 0:
                if theta_eq:
                    theta = solve_theta(theta_1, gamma, G / G_1)
                else:
                    alpha = G / G_1
                    theta = theta_1*((1+alpha*(gamma-1))/(gamma*alpha+theta_1))

            y = (1-theta)*x_1 + theta*z_1
            #g = f.gradient(y)
            fy, g = f.func_grad(y)
        
            z = h.div_prox_map(z_1, g, theta**(gamma-1) * G * L)
            x = (1-theta)*x_1 + theta*z

            # compute triangle scaling quantities
            dxy = h.divergence(x, y)
            dzz = h.divergence(z, z_1)
            if dzz < epsilon:
                break
            
            Gdr = dxy / dzz / theta**gamma

            if checkdiv:
                condition = (Gdr > G )
            else:
                condition = (f(x) > fy + np.dot(g,x-y) + theta**gamma*G*L*dzz)
                
            if condition:
                G = G * ls_inc
               
        # store and display computational progress
        Gain[k] = G
        Gdiv[k] = Gdr
        sumlogG += np.log(G)
        Gavg[k] = np.exp(sumlogG / (gamma + k)) 
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:10.3e}  {7:10.3e}  {8:6.1f}".format(
                    k, F[k], theta, G, Gdr, dxy, dzz, Gavg[k], T[k]))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            #if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:
            if (restart_rule == 'f' and F[k] > F[k-1]) or (restart_rule == 'g' and np.dot(g, x-x_1) > 0):
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    Gain = Gain[0:k+1]
    Gdiv = Gdiv[0:k+1]
    Gavg = Gavg[0:k+1]
    T = T[0:k+1]
    return x, F, Gain, Gdiv, Gavg, T


def ABDA(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, theta_eq=True,
           verbose=True, verbskip=1):
    """
    Accelerated Bregman Dual Averaging (ABDA) method for solving
            minimize_{x in C} f(x) + Psi(x) 
    
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma:    triangle scaling exponent (TSE) for Bregman distance D_h(x,y)
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        theta_eq: calculate theta_k by solving equality using Newton's method
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x: the last iterate of BPG
        F: array storing F(x[k]) for all k
        G: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma
        T: array storing time used up to iteration k
    """
    # Simple restart schemes for dual averaging method do not work!
    restart = False
    
    if verbose:
        print("\nABDA method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)     time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    z = np.copy(x0)
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    gavg = np.zeros(x.size)
    csum = 0
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x_1 + theta*z_1
        g = f.gradient(y)
        gavg = gavg + theta**(1-gamma) * g
        csum = csum + theta**(1-gamma)
        z = h.prox_map(gavg/csum, L/csum)
        x = (1-theta)*x_1 + theta*z

        # compute triangle scaling quantities
        dxy = h.divergence(x, y)
        dzz = h.divergence(z, z_1)
        Gdr = dxy / dzz / theta**gamma

        # store and display computational progress
        G[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, F[k], theta, Gdr, dxy, dzz, T[k]))

        kk += 1
        # restart does not work for ABDA (restart = False)
        if restart:
            if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:   # this does not work for dual averaging
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y
                gavg = np.zeros(x.size) # this is why restart does not work
                csum = 0

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    G = G[0:k+1]
    T = T[0:k+1]
    return x, F, G, T


def FW_alg_div_step(f, h, L, x0, maxitrs, gamma, lmo, epsilon=1e-14, linesearch=True, ls_ratio=2,
                    verbose=True, verbskip=1):
    """
    Frank-Wolfe's algorithm with the Bregman divergence

    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        gamma:    triangle scaling exponent (TSE) for Bregman distance D_h(x,y)
        lmo:      linear minimization oracle
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        linesearch:  whether or not perform line search (True or False)
        ls_ratio: backtracking line search parameter >= 1
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of the algorithm
        F:  array storing F(x[k]) for all k
        Ls: array storing local Lipschitz constants obtained by line search
        T:  array storing time used up to iteration k
    """
    if verbose:
        print("\nFW adaptive algorithm")
        print("     k      F(x)         Lk       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)
    delta = epsilon

    x = np.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        s_k = lmo(g)
        d_k = s_k - x
        div = h.divergence(s_k, x)
        if div == 0:
            div = delta

        grad_d_prod = np.dot(g, d_k)
        if 0 < grad_d_prod <= delta:
            grad_d_prod = 0
        assert grad_d_prod <= 0, "np.dot(g, d_k) must be negative."

        if linesearch:
            L = L / ls_ratio
        while True:
            alpha_k = min((-grad_d_prod / (2 * L * div)) ** (1 / (gamma - 1)), 1)
            x1 = x + alpha_k * d_k
            if not linesearch:
                break
            if f.func_grad(x1, flag=0) <= fx + alpha_k * grad_d_prod + alpha_k ** gamma * L * div:
                break
            L = L * ls_ratio
        x = x1
        x[x == 0] = delta

        Ls[k] = L
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:6.1f}".format(k, F[k], L, T[k]))

        # stopping criteria
        if k > 0 and abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[0:k + 1]
    Ls = Ls[0:k + 1]
    T = T[0:k + 1]
    return x, F, Ls, T


class SwitchingGradientDescent:
    def __init__(self, target_f, solution_f=0, M=1, cnstrnt_nmbr=0, lamda=0,
                 name="Switching Gradient Descent with polyak step", constraints=None):
        self.name = name
        self.cnstrnt_nmbr = cnstrnt_nmbr
        self.target_f = target_f
        self.solution_f = solution_f
        self.M = M
        self.constraints = constraints
        self.lamda = lamda

    def check_constraints(self, x):
        """
        All constraint functions of the problem
        :return: a failed constrained function and its gradient function
        """
        for i, g_func in enumerate(self.constraints):
            if g_func(x) > self.cnstrnt_nmbr:
                return g_func
        return None

    def get_max_constraint(self, x):
        """
        All constraint functions of the problem
        :return: a failed constrained function and its gradient function
        """
        return np.max(np.array([g(x) for g in self.constraints]))

    def solve(self, x0, epsilon=1e-9, maxitrs=20000, verbose=True, verbskip=1):
        f = self.target_f
        k_unprod = 0

        if verbose:
            print(f'{self.name}')
            print("k     F(x)    time")

        start_time = time.time()
        F = np.zeros(maxitrs)
        T = np.zeros(maxitrs)

        x = np.copy(x0)
        F[0] = f.func_grad(x, flag=0)
        T[0] = time.time() - start_time
        k = 1
        while True:
            failed_constraint_f = self.check_constraints(x)
            if failed_constraint_f is None:
                fx, grad = f.func_grad(x)
                alpha = max(epsilon, (fx - self.solution_f) / self.M * np.linalg.norm(grad))
            else:
                grad = failed_constraint_f.gradient(x)
                alpha = failed_constraint_f(x) / np.linalg.norm(grad)**2
                k_unprod += 1
            # projection
            x_prev, x = x, np.clip(x - alpha * grad, 1e-9, None)

            F[k] = f.func_grad(x, flag=0)
            T[k] = time.time() - start_time

            if verbose and k % verbskip == 0:
                print("{0:6d}  {1:10.3e}  {2:10.3e}".format(k, F[k], T[k]))

            if abs(F[k] - F[k-1]) < epsilon or k >= maxitrs - 1:
                break
            k += 1

        print(f'Unprod steps amount: {k_unprod}, method: {self.name}')

        F = F[0:k + 1]
        T = T[0:k + 1]
        return x, F, T


def AIBM(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, verbose=True, noise=0, verbskip=1):
    if verbose:
        print("\nAIBM method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       L       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    p = 2

    x = z = np.ones(x0.shape[0]) * h.prox_map(np.zeros(x0.shape[0]), 1)

    delta = get_random_float(noise)
    delta_grad = get_random_vector(x.shape[0], noise)
    fx, g = f.func_grad(x, flag=2)
    fx += delta
    g += delta_grad
    while True:
        alpha = 1 / L
        y = h.prox_map(g, 1)
        if f(y) <= fx + np.dot(g, y - x) + L * h.divergence(y, x) + epsilon:
            break
        L = L * 2

    B = A = alpha
    xi_grad = alpha * f.gradient(x)

    F[0] = fx + h.extra_Psi(x)
    G[0] = L
    T[0] = time.time() - start_time

    for k in range(1, maxitrs):
        L /= 2
        delta = get_random_float(noise)
        while True:
            alpha = (1 / L) * (1 + k / (2 * p)) ** ((p - 1) * (gamma - 1))
            B = (L * alpha ** gamma) ** (1 / (gamma - 1))
            x = (alpha / B) * z + (1 - alpha / B) * y
            delta_grad = get_random_vector(x.shape[0], noise)
            grad_x = f.gradient(x) + delta_grad
            xi_grad += alpha * grad_x
            z_k = h.prox_map(xi_grad, 1)
            w = alpha / B * z_k + (1 - alpha / B) * y
            fx = f(x) + delta
            if f(w) <= fx + np.dot(grad_x, w - x) + L * h.divergence(w, x) + epsilon:
                break
            xi_grad -= alpha * grad_x
            L = L * 2

        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time

        # Update
        A += alpha
        y = B / A * w + (1 - B / A) * y
        z = z_k

        # store and display computational progress
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:6.1f}".format(k, F[k], L, T[k]))

        # stopping criteria
        if abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[0:k + 1]
    G = G[0:k + 1]
    T = T[0:k + 1]
    return x, F, G, T


class GradientDescentDualityCriteria:
    def __init__(self, StepSizeChoice, stopping_criteria, return_history=True, name=None):
        self.stopping_criteria_f = stopping_criteria
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.history = []

    def __call__(self, x0, f, gradf, N):
        pass

    def stop_criteria(self):
        pass

    def solve(self, x0, f, gradf, tol=1e-3, max_iter=1000):
        self.history = [(f.f(x0), time.time())]
        x = x0.copy()
        k = 0
        x_prev = None
        while x_prev is None or abs(self.stopping_criteria_f(x) - f.f(x)) > tol:
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x_prev, x = x, x + alpha * h
            if self.return_history:
                self.history.append((f.f(x), time.time()))
            if k >= max_iter:
                break
            k += 1
        return x
