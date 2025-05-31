# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import math

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
    fx, g = f.func_grad(x, flag=2)
    while True:
        alpha = 1 / L
        y = h.prox_map(g, 1)
        if f(y) <= fx + np.dot(g, y - x) + L * h.divergence(y, x) + epsilon + delta:
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
            grad_x = f.gradient(x)
            xi_grad += alpha * grad_x
            z_k = h.prox_map(xi_grad, 1)
            w = alpha / B * z_k + (1 - alpha / B) * y
            fx = f(x)
            if f(w) <= fx + np.dot(grad_x, w - x) + L * h.divergence(w, x) + delta:
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
        if abs(F[k] - F[k - 1]) < 1e-9:
            break

    F = F[0:k + 1]
    G = G[0:k + 1]
    T = T[0:k + 1]
    return x, F, G, T


def AdaptFGM(f, h, L, x0, maxitrs, epsilon=1e-14, verbose=True, noise=0, verbskip=1):
    if verbose:
        print("\nAdaptFGM method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       L       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)

    x_k = y = u_k = np.ones(x0.shape)
    A_k = alpha_k = 0

    fx, g = f.func_grad(x_k, flag=2)

    F[0] = fx + h.extra_Psi(x_k)
    G[0] = L
    T[0] = time.time() - start_time

    for k in range(1, maxitrs):
        L /= 2
        delta = get_random_float(noise)
        while True:
            alpha = (1 + math.sqrt(1 + 4*L*A_k)) / (2*L)
            A = L * alpha**2
            y = (alpha*u_k + A_k*x_k) / A
            g_y = f.gradient(y)
            u = h.div_prox_map(u_k, g_y*alpha, 1)
            x = (alpha*u + A_k*x_k) / A

            fx = f(x_k)
            if f(x) <= fx + np.sum(g_y * (x - y)) + L * h.divergence(x, y) + delta:
                A_k = A
                u_k = u
                x_k = x
                break
            L = L * 2

        F[k] = f(x_k) + h.extra_Psi(x_k)
        G[k] = L
        T[k] = time.time() - start_time

        # store and display computational progress
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:6.1f}".format(k, F[k], L, T[k]))

        # stopping criteria
        if abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[0:k + 1]
    G = G[0:k + 1]
    T = T[0:k + 1]
    return x_k, F, G, T


def UniversalGM(f, h, L, x0, maxitrs, epsilon=1e-14, verbose=True, noise_level=0, verbskip=1):
    if verbose:
        print("\nUniversalGM method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       L       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)

    x = np.copy(x0)
    x_k = np.copy(x0)
    fx, g = f.func_grad(x, flag=2)
    F[0] = fx + h.extra_Psi(x)
    G[0] = L
    T[0] = time.time() - start_time

    A_k = 0
    u_k = np.ones(x0.shape)

    for k in range(1, maxitrs):
        noise = np.random.rand() * noise_level if noise_level > 0 else 0

        L /= 2
        while True:
            alpha = (1 + math.sqrt(1 + 4*L*A_k)) / (2*L)
            A = L * alpha**2
            y = (alpha*u_k + A_k*x_k) / A
            g_y = f.gradient(y)
            g_y += noise
            u = h.div_prox_map(u_k, g_y*alpha, 1)
            x = (alpha*u + A_k*x_k) / A
            
            fy = f(y)
            fy += noise
            # if f(x) <= fy + np.sum(g_y * (x - y)) + L * h.divergence(x, y) + np.random.rand() * noise_level:
            if f(x) <= fy + np.sum(g_y * (x - y)) + L * h.divergence(x, y):
                A_k = A
                u_k = u
                x_k = x
                break
            L = L * 2
            if L is None or math.isinf(L):
                raise ValueError("L cannot be None or infinity")

        F[k] = f(x_k) + h.extra_Psi(x_k)
        G[k] = L
        T[k] = time.time() - start_time

        # store and display computational progress
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:6.1f}".format(k, F[k], L, T[k]))

        # stopping criteria
        if abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[0:k + 1]
    G = G[0:k + 1]  
    T = T[0:k + 1]
    return x_k, F, G, T


def PrimalDualSwitchingGradientMethod(f, h, L_init, cnstrnt_fun, x0, maxitrs, epsilon=1e-14,
                                      linesearch=True, verbose=True, verbskip=100):
    """
    Solve:
        minimize_{x in C} f(x) + Psi(x) 
        s.t. cnstrnt_fun(x) <= ε

    Inputs:
        f:       RSmoothFunction (f is L-smooth relative to h)
        h:       LegendreFunction (defines the kernel and Psi)
        L_init:  initial guess for the relative smoothness constant
        cnstrnt_fun: constraint function (must satisfy cnstrnt_fun(x) <= ε)
                     cnstrnt_fun implements RSmoothFunction interface 
                     (i.e. supports __call__ and .gradient(x))
        x0:      initial point
        maxitrs: maximum number of iterations
        epsilon: tolerance for the constraint violation
        linesearch: if True, performs a line search on productive steps.
        verbose: print progress if True.
        verbskip: print every verbskip iterations.
    
    Returns:
        F:             array of [f(x)+Psi(x)] values at productive steps.
        duality_gaps:  array of estimated real duality gaps.
        Ls:            array storing the step constants (from productive steps).
    """
    
    def is_nan_or_inf(x):
        return np.any(np.isnan(x)) or np.any(np.isinf(x))
    
    def compute_dual_value(z_scalar, f, cnstrnt_fun, y0, max_inner=100, tol_inner=1e-9, alpha=1e-3):
        """
        Solve the inner maximization:
            max_{y in Q} { -f(y) - z^T cnstrnt_fun(y) }
        using gradient ascent.
        
        Parameters:
            z_scalar : current Lagrange multiplier estimate (a scalar)
            f        : the RSmoothFunction instance (must support __call__ and gradient)
            cnstrnt_fun: constraint function (must support __call__ and gradient)
            y0       : starting point (typically the current iterate x)
            max_inner: maximum iterations for the inner loop
            tol_inner: convergence tolerance for the inner loop
            alpha    : step size for gradient ascent
            
        Returns:
            dual_val : approximate maximum value, i.e. 
                       d(z) ≈ max_{y in Q} { -f(y) - z_vec^T cnstrnt_fun(y) }
            y        : approximate maximizer.
        """
        y = np.copy(y0)
        for it in range(max_inner):
            g_val = cnstrnt_fun(y)
            if np.ndim(g_val) == 0:
                dual_val = -f(y) - z_scalar * g_val
                grad_dual = -f.gradient(y) - z_scalar * cnstrnt_fun.gradient(y)
            else:
                z_vec = z_scalar * np.ones_like(g_val)
                dual_val = -f(y) - np.dot(z_vec, g_val)
                g_grad = cnstrnt_fun.gradient(y)  # expected shape: (m, n)
                grad_dual = -f.gradient(y) - np.sum(z_vec[:, None] * g_grad, axis=0)
            y_next = y + alpha * grad_dual  # gradient ascent step
            y_next = np.maximum(y_next, 1e-6)
            
            if is_nan_or_inf(y_next):
                raise ValueError("compute_dual_value encountered NaN or infinite value in y_next")
            if np.linalg.norm(y_next - y) < tol_inner:
                y = y_next
                break
            y = y_next
        # Final evaluation
        g_val = cnstrnt_fun(y)
        if np.ndim(g_val) == 0:
            dual_val = -f(y) - z_scalar * g_val
        else:
            z_vec = z_scalar * np.ones_like(g_val)
            dual_val = -f(y) - np.dot(z_vec, g_val)
        return dual_val, y

    def productive_line_search(x, fx, grad_fx, L_current):
        """
        Performs backtracking line search for the productive step.
        Updates the candidate iterate using the proximal mapping induced by h.
        """
        max_line_search_iters = 1000  # safety bound
        iter_ls = 0
        while iter_ls < max_line_search_iters:
            if L_current is None or math.isinf(L_current) or L_current <= 0:
                raise ValueError("L_current became invalid during line search")

            # Use h.div_prox_map which performs a proximal step using the divergence induced by h.
            x_new = h.div_prox_map(x, grad_fx, L_current)
            if x_new is None or is_nan_or_inf(x_new):
                L_current *= 2
                iter_ls += 1
                continue

            # Descent condition for relative smoothness.
            if f(x_new) <= fx + np.sum(grad_fx * (x_new - x)) + L_current * h.divergence(x_new, x):
                return x_new, L_current
            L_current *= 2
            iter_ls += 1
        raise RuntimeError("Line search did not converge within maximum iterations")
    
    if verbose:
        print("\n Primal-Dual method")
        print("     k       F(x)         L_k       duality_gap     time")
    
    F = []
    duality_gaps = np.zeros(maxitrs)
    Ls = np.zeros(maxitrs)

    x = np.copy(x0)
    if is_nan_or_inf(x):
        raise ValueError("Initial x0 contains NaN or infinite values")

    L_prod = L_init
    step_size_prod_sum = 0.0
    step_size_unprod_sum = 0.0

    productive_count = 0
    start_time = time.time()
    
    for k in range(maxitrs):
        fx, grad_fx = f.func_grad(x, flag=2)

        constraint_val = cnstrnt_fun(x)
        if np.any(np.isnan(constraint_val)) or np.any(np.isinf(constraint_val)):
            raise ValueError("Constraint function returned NaN or Inf.")

        if np.all(constraint_val <= epsilon):
            # PRODUCTIVE STEP: use target function gradient.
            F.append(fx + h.extra_Psi(x))

            if linesearch:
                try:
                    x_new, L_candidate = productive_line_search(x, fx, grad_fx, L_prod/2)
                except RuntimeError as e:
                    print("Line search failed at iteration", k)
                    raise e
                L_prod = L_candidate
            else:
                x_new = h.div_prox_map(x, grad_fx, L_prod)
            
            productive_count += 1
            prod_step_size = L_prod**-1
            step_size_prod_sum += prod_step_size
            x = x_new
            Ls[k] = prod_step_size
        else:
            # UNPRODUCTIVE STEP: use the gradient of the constraint function.
            grad_g = cnstrnt_fun.gradient(x)
            # Here, we take an ordinary gradient step using the constraint gradient.
            # A projection (here, using np.maximum) keeps x well-behaved.
            unprod_step = 1e-1 * np.linalg.norm(grad_g)**-2
            x = x - grad_g * unprod_step
            x = np.maximum(x, 1e-6)
            step_size_unprod_sum += unprod_step
            Ls[k] = 0.0
        
        # Estimate the Lagrange multiplier.
        if step_size_prod_sum > 0 and step_size_unprod_sum > 0:
            lagrange_multiplier = step_size_unprod_sum / step_size_prod_sum
        else:
            lagrange_multiplier = 1.0
        
        # Compute the real duality gap:
        # d(z) = max_{y in Q} { -f(y) - z_vec^T cnstrnt_fun(y) }
        # Then gap(x) = f(x) + d(z)
        dual_val, _ = compute_dual_value(lagrange_multiplier, f, cnstrnt_fun, x)
        duality_gaps[k] = fx + dual_val
        
        if verbose and k % verbskip == 0 and productive_count > 0:
            elapsed = time.time() - start_time
            print(f"{k:6d}  {F[-1]:.3e}  {Ls[k]:.3e}  {duality_gaps[k]:.3e}  {elapsed:.2f}s")
        
        # Stopping criterion: stop if the real duality gap becomes sufficiently small.
        if k > 0 and abs(duality_gaps[k]) < 1e-6:
            if verbose:
                print("Stopping criterion reached at iteration", k)
            F = F[:k+1]
            duality_gaps = duality_gaps[:k+1]
            Ls = Ls[:k+1]
            return F, duality_gaps, Ls
        
    print(f"Unprod steps is {(k - productive_count):n}  and prod is {productive_count:n}")
    Fs = np.array(F)
    duality_gaps = duality_gaps[:maxitrs]
    return Fs, duality_gaps, Ls[:maxitrs]
 