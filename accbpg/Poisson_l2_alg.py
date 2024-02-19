import numpy as np
import time


def lmo_notnegative_ball(radius, is_shifted_pos_ball=False):
    """
    The Frank-Wolfe lmo function for the l2 ball on x > 0 and x \in ||radius||_2
        is_shifted_pos_ball: Ball moves to the positive quartile
    """

    def f(g):
        if is_shifted_pos_ball:
            center = radius
        else:
            center = 0
        s = np.zeros(g.shape)
        argmin = np.argmin(g)
        s[argmin] = radius*(-1*np.sign(g[argmin]))
        s += center
        s += 1e-60
        return s

    return lambda g: f(g)


def FW_alg_div_step(f, h, L, x0, maxitrs, gamma, lmo, epsilon=1e-14, linesearch=True, ls_ratio=2,
                              verbose=True, verbskip=1):
    if verbose:
        print("\nFW RS adaptive method")
        print("     k      F(x)         Lk       time")

    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)

    x = np.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F[k] = fx
        T[k] = time.time() - start_time
        s_k = lmo(g)
        d_k = s_k - x
        div = h.divergence(s_k, x)
        if div == 0:
            div = 1e-60

        grad_d_prod = np.dot(g, d_k)
        if 0 < grad_d_prod <= 1e-18:
            grad_d_prod = 0
        assert grad_d_prod <= 0, "np.dot(g, d_k) must be negative."

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


def main():
    pass
