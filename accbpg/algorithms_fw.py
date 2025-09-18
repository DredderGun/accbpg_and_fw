import numpy as np
import time
import math


import jax
import jax.numpy as jnp
import time


def FW_alg_div_step(
    f, h, L, x0, maxitrs, gamma, lmo,
    epsilon=1e-14, linesearch=True, ls_ratio=2,
    verbose=True, verbskip=1
):
    r"""
    Frank-Wolfe's algorithm with the Bregman divergence (JAX version)
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
    F = jnp.zeros(maxitrs, dtype=jnp.float32)
    Ls = jnp.ones(maxitrs, dtype=jnp.float32)
    T = jnp.zeros(maxitrs, dtype=jnp.float32)
    delta = 1e-6

    x = jnp.copy(x0)

    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F = F.at[k].set(fx + h.extra_Psi(x))
        T = T.at[k].set(time.time() - start_time)

        s_k = lmo(g)
        d_k = s_k - x
        div = h.divergence(s_k, x)
        div = jnp.where(div == 0, delta, div)

        grad_d_prod = jnp.dot(g.ravel(), d_k.ravel())
        grad_d_prod = jnp.where((0 < grad_d_prod) & (grad_d_prod <= delta), 0.0, grad_d_prod)

        if grad_d_prod > 0:
            raise ValueError("grad_d_prod must be non-positive")

        if linesearch:
            L = L / ls_ratio

        while True:
            alpha_k = jnp.minimum(
                (-grad_d_prod / (2 * L * div)) ** (1 / (gamma - 1)),
                1.0
            )
            x1 = x + alpha_k * d_k

            if not linesearch:
                break

            assert math.isinf(L) == False, "L is infinite"

            if f.func_grad(x1, flag=0) <= fx + alpha_k * grad_d_prod + alpha_k ** gamma * L * div:
                break

            L = L * ls_ratio

        x = x1
        Ls = Ls.at[k].set(L)

        if verbose and k % verbskip == 0:
            print(f"{k:6d}  {float(F[k]):10.3e}  {float(L):10.3e}  {float(T[k]):6.1f}")

        if k > 0 and jnp.abs(F[k] - F[k - 1]) < epsilon:
            break

    return x, F[:k+1], Ls[:k+1], T[:k+1]


def FW_alg_L0_L1_shortest_step(
    f, h, L0, L1, x0, maxitrs, gamma, lmo, epsilon=1e-14,
    linesearch=True, ls_ratio=2, verbose=True, verbskip=1
):
    r"""
    Frank-Wolfe algorithm for (L0, L1)-smooth functions with shortest-step rule.

    This routine minimizes a composite objective of the form

        F(x) = f(x) + h.extra_Psi(x),

    where `f` is (L0, L1)-smooth. The step-size (this is quite similar to the classic shortest-step rule):

        alpha_k = min( [ -<∇f(x), d> / (a_k * D(s, x) * e) ]^(1 / (γ - 1)), 1 ),

    where `d = s - x`, `s` is returned by the Linear Minimization Oracle (LMO),
    `D(s, x)` is a divergence measure provided by `h`, `a_k = L0 + L1 * ||∇f(x)||`,
    and `γ > 1` is a curvature parameter.

    Parameters
    ----------
    f : object
        Target function.
    h : object with a convex penalty term.
    L0 : float
        Initial zero-order smoothness parameter (must be nonnegative).
    L1 : float
        Initial first-order smoothness parameter (must be nonnegative).
    x0 : ndarray
        Initial feasible point.
    maxitrs : int
        Maximum number of iterations.
    gamma : float
        Exponent in the shortest-step rule, must satisfy `γ > 1`.
    lmo : callable
        Linear Minimization Oracle (LMO).
    epsilon : float, optional (default=1e-14)
        Stopping tolerance on successive objective values.
    linesearch : bool, optional (default=True)
        Whether to use line search or not.
    ls_ratio : float, optional (default=2)
        Factor by which `(L0, L1)` are scaled during line search. Must be >= 1.
    verbose : bool, optional (default=True)
        If True, prints progress information.
    verbskip : int, optional (default=1)
        Frequency (in iterations) of verbose output.

    Returns
    -------
    x : ndarray
        Final iterate.
    F : ndarray
        Array of objective values across iterations.
    Ls : ndarray
        Sequence of smoothness constants `a_k = L0 + L1 * ||∇f(x)||`.
    T : ndarray
        Cumulative runtime (in seconds) at each iteration.
    """
    if ls_ratio < 1:
        raise ValueError("ls_ratio must be >= 1")
    if L0 < 0 or L1 < 0:
        raise ValueError("Initial L must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if verbose:
        print("\nFW L0,L1 smooth algorithm")
        print("     k      F(x)         L           L0              L1     step-size     time")

    start_time = time.time()
    F = jnp.zeros(maxitrs, dtype=jnp.float64)
    Ls = jnp.ones(maxitrs, dtype=jnp.float64) * 1.0
    T = jnp.zeros(maxitrs, dtype=jnp.float64)
    delta = 1e-6
    toggle = 0

    x = jnp.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F = F.at[k].set(fx + h.extra_Psi(x))
        T = T.at[k].set(time.time() - start_time)
        s_k = lmo(g)
        d_k = s_k - x
        div = h.divergence(s_k, x)
        if div == 0:
            div = delta

        grad_d_prod = jnp.dot(g.ravel(), d_k.ravel())
        if 0 < grad_d_prod <= delta:
            grad_d_prod = 0
        if grad_d_prod > 0:
            raise ValueError("grad_d_prod must be non-positive")

        if linesearch:
            L0 = L0 / ls_ratio
            L1 = L1 / ls_ratio

        g_norm = jnp.linalg.norm(g)
        while True:
            a_k = L0 + L1 * g_norm
            # attention: in case of euklidian divergence, we have div = 0.5 ||s-x||^2
            # so we already have 2 power in the denominator
            alpha_k = min((-grad_d_prod / (a_k * div * math.e)) ** (1 / (gamma - 1)), 1) 
            x1 = x + alpha_k * d_k

            if not linesearch:
                break
                
            if f.func_grad(x1, flag=0) <= fx + alpha_k * grad_d_prod + alpha_k ** gamma * (a_k / 2) * math.e * div:
                break

            if toggle == 0:
                L0 = L0 * ls_ratio
                toggle = 1
            else:
                L1 = L1 * ls_ratio
                toggle = 0

        x = x1

        Ls = Ls.at[k].set(a_k)
        if verbose and k % verbskip == 0:
            print(f"{k:6d}   {F[k]:10.3e}   {Ls[k]:10.3e}    {L0:10.3e}      {L1:10.3e}       {alpha_k:10.3e}     {T[k]:6.1f}")

        if k > 0 and abs(F[k] - F[k - 1]) < epsilon:
            break

    return x, F[0:k + 1], Ls[0:k + 1], T[0:k + 1]


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


def FW_alg_div_step_adapt(f, h, L, x0, maxitrs, gamma, gamma_max, lmo, ls_ratio, 
                          divisor_for_tse, change_tse_each_n=2, epsilon=1e-14, linesearch=True, 
                          verbose=True, verbskip=1):
    r"""
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
        gamma_max: maximum value for gamma

    Returns:
        x: Final solution vector
        F: Array of function values at each iteration
        Ls: Array of L values at each iteration 
        T: Array of cumulative computation times
        Gammas: Array of gamma values at each iteration
    """
    if ls_ratio < 1:
        raise ValueError("ls_ratio must be >= 1")
    if L <= 0:
        raise ValueError("Initial L must be positive")
    if gamma <= 1:
        raise ValueError("gamma must be greater than 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if gamma_max <= 1:
        raise ValueError("gamma_max must be greater than 1")

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
            gamma = min(gamma + divisor_for_tse * (gamma - 1), gamma_max)
            
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
                gamma = min(gamma, gamma_max)
            
            adapt_iter += 1

        x = x1
        # x[x == 0] = delta

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


def FW_l0l1_log_and_linear_step(
    f, h, L0, L1, x0, maxitrs, lmo, ls_ratio, epsilon=1e-14, 
    L0_max=None, L1_max=None, linesearch=True, verbose=True, 
    verbskip=50,
):
    r"""
    The step-size rule is logarithmic when `L1 * ||d_k|| \geq ln2` is large, and
      $L_1 (- \nabla f(x_k)^\top d_k) / (L_0 + L_1 \| \nabla f(x_k) \|) \| d_k \|$ otherwise.

    This routine minimizes a composite objective of the form

        F(x) = f(x) + h.extra_Psi(x),

    where `f` is (L0, L1)-smooth (relative smoothness model).

    Parameters
    ----------
    f : object
        Target function.
    h : object
        Regularizer with method `extra_Psi(x)` returning a convex penalty term.
    L0 : float
        Initial zero-order smoothness parameter (must be positive).
    L1 : float
        Initial first-order smoothness parameter (must be positive).
    x0 : ndarray
        Initial feasible point.
    maxitrs : int
        Maximum number of iterations.
    lmo : callable
        Linear Minimization Oracle (LMO).
    ls_ratio : float
        Factor by which (L0, L1) are increased during line search. Must be >= 1.
    epsilon : float, optional (default=1e-14)
        Stopping tolerance on successive objective values.
    L0_max : float, optional
        Upper bound for L0 during line search. If None, no upper bound is enforced.
    L1_max : float, optional
        Upper bound for L1 during line search. If None, no upper bound is enforced.
    linesearch : bool, optional (default=True)
        Whether to use line search to adjust smoothness parameters.
    verbose : bool, optional (default=True)
        If True, prints progress information.
    verbskip : int, optional (default=50)
        Frequency (in iterations) of verbose output.

    Returns
    -------
    x : ndarray
        Final iterate.
    F : ndarray
        Array of objective values across iterations.
    Ls : ndarray
        Sequence of smoothness constants `a_k = L0 + L1 * ||∇f(x)||`.
    LOG_STEPS : ndarray
        Count of logarithmic step-size updates per iteration.
    T : ndarray
        Cumulative runtime (in seconds) at each iteration.
    """
    if ls_ratio < 1:
        raise ValueError("ls_ratio must be >= 1")
    if L0 <= 0 or L1 <= 0:
        raise ValueError("Initial L0 and L1 must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if verbose:
        print("\nFW L0,L1 smooth logarithmic algorithm")
        print("     k      F(x)         L         L0         L1     log step count       time")

    start_time = time.time()

    F = jnp.zeros(maxitrs, dtype=jnp.float64)
    Ls = jnp.ones(maxitrs, dtype=jnp.float64)
    T = jnp.zeros(maxitrs, dtype=jnp.float64)
    LOG_STEPS = jnp.zeros(maxitrs, dtype=int)

    delta = 1e-8
    x = jnp.copy(x0)

    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        gx_norm = jnp.linalg.norm(g)

        F = F.at[k].set(fx + h.extra_Psi(x))
        T = T.at[k].set(time.time() - start_time)

        s_k = lmo(g)
        d_k = s_k - x
        d_norm = jnp.linalg.norm(d_k)

        grad_d_prod = jnp.vdot(g, d_k)
        if 0 < grad_d_prod <= delta:
            grad_d_prod = 0
        if grad_d_prod > 0:
            raise ValueError("grad_d_prod must be non-positive (we minimize)")

        if linesearch:
            L0 /= ls_ratio
            L1 /= ls_ratio

        while True:
            assert L0 >= 0 and L1 >= 0, "Smoothness parameters must stay positive"

            a_k = L0 + L1 * gx_norm

            if L1 * d_norm >= jnp.log(2):
                alpha_k = (1 / (L1 * d_norm)) * jnp.log(1 - (L1 * grad_d_prod) / (a_k * d_norm))
                if k > 0:
                    LOG_STEPS = LOG_STEPS.at[k].set(LOG_STEPS[k - 1] + 1)
                else:
                    LOG_STEPS = LOG_STEPS.at[k].set(1)
            else:
                alpha_k = L1 * (-grad_d_prod) / (a_k * d_norm)
                LOG_STEPS = LOG_STEPS.at[k].set(LOG_STEPS[k - 1])

            x1 = x + alpha_k * d_k

            if not linesearch:
                break

            fx1 = f.func_grad(x1, flag=0)

            z = L1 * alpha_k * d_norm
            if z < 50:  # Safe zone for accurate exponential evaluation
                exp_term = jnp.expm1(z) - z
            else:
                # Use a safe upper bound approximation (e.g., quadratic growth)
                exp_term = 0.5 * z**2  # upper bound: exp(z) - z - 1 <= 0.5 * z^2 for large z

            rhs = fx + alpha_k * grad_d_prod + (a_k / L1**2) * exp_term
            if fx1 <= rhs:
                break
            else:
                L0 = min(L0 * ls_ratio, L0_max) if L0_max else L0 * ls_ratio
                L1 = min(L1 * ls_ratio, L1_max) if L1_max else L1 * ls_ratio
                a_k = L0 + L1 * gx_norm

        x = x1
        Ls = Ls.at[k].set(a_k)

        if verbose and k % verbskip == 0:
            print(f"{k:6d}   {F[k]:10.3e}   {Ls[k]:10.3e}   {L0:10.3e}   {L1:10.3e}   {LOG_STEPS[k]:6d}      {T[k]:6.1f}")

        if k > 0 and abs(F[k] - F[k - 1]) < epsilon:
            break

    return x, F[: k + 1], Ls[: k + 1], LOG_STEPS[: k + 1], T[: k + 1]


def FW_l0l1_log_only(
    f, h, L0, L1, x0, maxitrs, lmo, ls_ratio, epsilon=1e-14, 
    L0_max=None, L1_max=None, linesearch=True, verbose=True, 
    verbskip=50,
):
    r"""
    Frank-Wolfe algorithm for (L0, L1)-smooth functions with log only step size.
    Here `L1` is adaptively set to satisfy `L1 >= log(2) / ||d_k||` to ensure log only step size
    at each iteratoin.

    This routine minimizes a composite objective of the form

        F(x) = f(x) + h.extra_Psi(x),

    where `f` is (L0, L1)-smooth (relative smoothness model).

    Parameters
    ----------
    f : object
        Target function.
    h : object
        Regularizer with method `extra_Psi(x)` returning a convex penalty term.
    L0 : float
        Initial zero-order smoothness parameter (must be positive).
    L1 : float
        Initial first-order smoothness parameter (must be positive).
        This value is dynamically adjusted but never drops below
        `log(2) / ||d_k||` at each iteration.
    x0 : ndarray
        Initial feasible point.
    maxitrs : int
        Maximum number of iterations.
    lmo : callable
        Linear Minimization Oracle (LMO).
    ls_ratio : float
        Factor by which (L0, L1) are increased during line search. Must be >= 1.
    epsilon : float, optional (default=1e-14)
        Stopping tolerance on successive objective values.
    L0_max : float, optional
        Upper bound for L0 during line search. If None, no upper bound is enforced.
    L1_max : float, optional
        Upper bound for L1 during line search. If None, no upper bound is enforced.
    linesearch : bool, optional (default=True)
        Whether to use line search to adjust smoothness parameters.
        Line search alternates between updating L0 and L1 when backtracking fails.
    verbose : bool, optional (default=True)
        If True, prints progress information.
    verbskip : int, optional (default=50)
        Frequency (in iterations) of verbose output.

    Returns
    -------
    x : ndarray
        Final iterate.
    F : ndarray
        Array of objective values across iterations.
    Ls : ndarray
        Sequence of smoothness constants `a_k = L0 + L1 * ||∇f(x)||`.
    LOG_STEPS : ndarray
        Count of logarithmic step-size updates per iteration.
    T : ndarray
        Cumulative runtime (in seconds) at each iteration.
    """
    if ls_ratio < 1:
        raise ValueError("ls_ratio must be >= 1")
    if L0 <= 0 or L1 <= 0:
        raise ValueError("Initial L0 and L1 must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if verbose:
        print("\nFW L0,L1 smooth algorithm with fixed L1")
        print("     k      F(x)         L         L0         L1     log step count       time")

    start_time = time.time()

    F = np.zeros(maxitrs)
    Ls = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    LOG_STEPS = np.zeros(maxitrs, dtype=int)

    delta = 1e-8
    toggle = 0
    x = np.copy(x0)

    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        gx_norm = np.linalg.norm(g)

        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time

        s_k = lmo(g)
        d_k = s_k - x
        d_norm = np.linalg.norm(d_k)

        grad_d_prod = np.vdot(g, d_k)
        if 0 < grad_d_prod <= delta:
            grad_d_prod = 0
        if grad_d_prod > 0:
            raise ValueError("grad_d_prod must be non-positive (we minimize)")

        if linesearch:
            L0 /= ls_ratio
            L1 /= ls_ratio

        L1 = max(math.log(2) / d_norm, L1)

        while True:
            assert L0 >= 0 and L1 >= 0, "Smoothness parameters must stay positive"

            a_k = L0 + L1 * gx_norm

            z = L1 * d_norm
            if z >= math.log(2) - 1e-5:
                alpha_k = (1 / (L1 * d_norm)) * math.log(1 - (L1 * grad_d_prod) / (a_k * d_norm))
                if k > 0:
                    LOG_STEPS[k] = LOG_STEPS[k - 1] + 1
                else:
                    LOG_STEPS[k] = 1
            else:
                assert False, "No use for the second step!"

            x1 = x + alpha_k * d_k

            if not linesearch:
                break

            fx1 = f.func_grad(x1, flag=0)

            z = L1 * alpha_k * d_norm
            if z < 50:  # Safe zone for accurate exponential evaluation
                exp_term = np.expm1(z) - z
            else:
                # Use a safe upper bound approximation (e.g., quadratic growth)
                exp_term = 0.5 * z**2  # upper bound: exp(z) - z - 1 <= 0.5 * z^2 for large z

            rhs = fx + alpha_k * grad_d_prod + (a_k / L1**2) * exp_term
            if fx1 <= rhs:
                break
            else:
                if toggle == 0:
                    L0 = min(L0 * ls_ratio, L0_max) if L0_max else L0 * ls_ratio
                    toggle = 1
                else:
                    L1 = min(L1 * ls_ratio, L1_max) if L1_max else L1 * ls_ratio
                    toggle = 0
                a_k = L0 + L1 * gx_norm

        x = x1
        Ls[k] = a_k

        if verbose and k % verbskip == 0:
            print(f"{k:6d}   {F[k]:10.3e}   {Ls[k]:10.3e}   {L0:10.3e}   {L1:10.3e}   {LOG_STEPS[k]:6d}      {T[k]:6.1f}")

        if k > 0 and abs(F[k] - F[k - 1]) < epsilon:
            break

    F = F[: k + 1]
    Ls = Ls[: k + 1]
    LOG_STEPS = LOG_STEPS[: k + 1]
    T = T[: k + 1]

    return x, F, Ls, LOG_STEPS, T
