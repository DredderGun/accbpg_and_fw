import numpy as np
import time


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

    Returns:
        x: Final solution vector
        F: Array of function values at each iteration
        Ls: Array of L values at each iteration 
        T: Array of cumulative computation times
    """
    if ls_ratio < 1:
        raise ValueError("ls_ratio must be >= 1")
    if L <= 0:
        raise ValueError("Initial L must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if verbose:
        print("\nFW adaptive algorithm")
        print("     k      F(x)         Lk       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)
    delta = 1e-6

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

        grad_d_prod = np.dot(g.ravel(), d_k.ravel())
        if 0 < grad_d_prod <= delta:
            grad_d_prod = 0
        if grad_d_prod > 0:
            raise ValueError("grad_d_prod must be non-positive")

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
            print(f"{k:6d}  {F[k]:10.3e}  {L:10.3e}  {T[k]:6.1f}")

        if k > 0 and abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[0:k + 1]
    Ls = Ls[0:k + 1]
    T = T[0:k + 1]
    return x, F, Ls, T


def FW_alg_descent_step(f, h, x0, maxitrs, lmo, epsilon=1e-14, verbose=True, verbskip=1):
    if verbose:
        print("\nFW descent step size algorithm")
        print("     k      F(x)         alpha_k       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)

    x = np.copy(x0)

    fx, g = f.func_grad(x)
    F[0] = fx + h.extra_Psi(x)
    T[0] = time.time() - start_time

    for k in range(1, maxitrs):
        s_k = lmo(g)
        d_k = s_k - x
        alpha_k = 2 / (k + 2)

        x = x + alpha_k * d_k

        fx, g = f.func_grad(x)
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time

        if verbose and (k % verbskip == 0 or k == 1):
            print(f"{k:6d}  {F[k]:10.3e}  {alpha_k:10.3e}  {T[k]:6.1f}")

        if abs(F[k] - F[k - 1]) < epsilon or np.linalg.norm(g) < epsilon:
            break
    
    F = F[:k + 1]
    T = T[:k + 1]
    G = G[:k + 1]

    return x, F, T, G


def FW_alg_div_step_adapt(f, h, L, x0, maxitrs, gamma, lmo, ls_ratio, 
                          divisor_for_tse, change_tse_each_n=2, epsilon=1e-14, linesearch=True, 
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

    Returns:
        x: Final solution vector
        F: Array of function values at each iteration
        Ls: Array of L values at each iteration 
        T: Array of cumulative computation times
    """
    if ls_ratio < 1:
        raise ValueError("ls_ratio must be >= 1")
    if L <= 0:
        raise ValueError("Initial L must be positive")
    if gamma <= 1:
        raise ValueError("gamma must be greater than 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if verbose:
        print("\nFW full adaptive algorithm")
        print("     k      F(x)         Lk       time       gamma")

    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)
    Gammas = np.zeros(maxitrs)
    delta = epsilon
    gamma_min = 1  # minimum value for gamma to avoid division by zero

    x = np.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        s_k = lmo(g)
        d_k = s_k - x
        div = h.divergence(s_k, x)
        if abs(div) <= 1: # ATTENTION. Value on the RHS may vary depend on the task.
            div = delta

        assert div >= 0, 'Divergence must be more or equals 0'

        grad_d_prod = np.vdot(g, d_k)
        if 0 < grad_d_prod <= delta:
            grad_d_prod = 0
        if grad_d_prod > 0:
            raise ValueError("grad_d_prod must be non-positive")

        if linesearch:
            L = L / ls_ratio
            gamma = min(gamma + divisor_for_tse * (gamma - 1), 2.0)
            
        adapt_iter = 1
        while True:
            assert gamma > 1, "gamma must be greater than 1 for the algorithm to work"
            assert L > 0, "L must be positive for the algorithm to work"

            alpha_k = min((-grad_d_prod / (2 * L * div)) ** (1 / (gamma - 1)), 1)
            x1 = x + alpha_k * d_k
            
            if not linesearch:
                break
                
            if f.func_grad(x1, flag=0) <= fx + alpha_k * grad_d_prod + alpha_k ** gamma * L * div:
                break
            elif adapt_iter % change_tse_each_n != 0:
                L = L * ls_ratio
            elif gamma == gamma_min:
                raise ValueError("gamma has reached its minimum value, cannot continue line search")
            else:
                gamma = max(1 + (gamma - 1) / divisor_for_tse, gamma_min)
            
            adapt_iter += 1

        x = x1
        x[x == 0] = delta

        Ls[k] = L
        Gammas[k] = gamma
        if verbose and k % verbskip == 0:
            print(f"{k:6d}  {F[k]:10.3e}  {L:10.3e}  {T[k]:6.1f}  {gamma:10.3e}")

        if k > 0 and abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[0:k + 1]
    Ls = Ls[0:k + 1]
    Gammas = Gammas[0:k + 1]
    T = T[0:k + 1]
    return x, F, Ls, T, Gammas

