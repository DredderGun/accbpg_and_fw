import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 12, 'font.family': 'serif'})

import accbpg


def d_opt_design_tests():
    # Generate a random instance of the D-optimal design problem of size m by n
    m = 80
    n = 200
    f, h, L, x0 = accbpg.D_opt_libsvm('data/housing.txt')
    # x0KY = accbpg.D_opt_KYinit(f.H)
    # x0 = (1 - 1e-3) * x0KY + 1e-3 * x0Kh
    # x0 = accbpg.random_point_on_simplex(n, 1)

    # Solve the problem using BPG and ABPG with different values of gamma (TSE)
    x00, F00, G00, T00 = accbpg.BPG(f, h, L, x0, maxitrs=1000, linesearch=True, ls_ratio=2, verbskip=100)
    x20, F20, G20, T20 = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=1000, theta_eq=True, verbskip=100)
    x00_fw, F00_fw, G00_fw, T00_fw, alphas = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_simplex(),
                                                            maxitrs=1000, gamma=2.0, ls_ratio=2, verbskip=100)
    x2e, F2e, Gamma2e, G2e, T2e = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=5000, theta_eq=True, Gmargin=100,
                                                   verbskip=1000)
    x2g, F2g, G2g, Gdiv2g, Gavg2g, T2g = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=5000, G0=0.1, theta_eq=True,
                                                          verbskip=1000)

    # Plot the objective gap and estimated gains for triangle scaling
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"BPG", r"ABPG$", r"Frank-Wolfe$", r"ABPG_expo", r"ABPG_gain"]
    styles = ['k:', 'g-', 'b-.', 'k-', 'r--']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    ax1 = plt.subplot(1, 2, 1)
    y_vals = [F00, F20, F00_fw, F2e, F2g]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[-10, 1000],
                            ylim=[1e-5, 25],
                            xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    ax2 = plt.subplot(1, 2, 2)
    y_vals = [T00, T20, T00_fw, T2e, T2g]
    accbpg.plot_comparisons(ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="log", xlim=[-10, 1000],
                            ylim=[1e-3, 5],
                            xlabel=r"Iteration number $k$", ylabel=r'$\hat{G}_k$', legendloc="lower right",
                            linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.show()


if __name__ == "__main__":
    d_opt_design_tests()
