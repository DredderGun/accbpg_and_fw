import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

import accbpg


def start():
    """
    Adaptive Intermediate Bregman Method (AIBM) VS other accelerated Bregman gradient methods.
    See the paper Accelerated Bregman gradient methods for relatively smooth and
    relatively Lipschitz continuous minimization problems
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    # np.random.seed(1)
    N = 1000
    m = 2000
    n = 1000
    epsilon = 1e-12

    f, h, L, x0 = accbpg.Poisson_regr_simplex_acc(m, n, noise=0.001)
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 4)
    for gamma, ax in zip([2.0, 1.4, 1.1], [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]):
        x_AIBM, F_AIBM, G_AIBM, T_AIBM = accbpg.AIBM(f, h, L, x0, gamma=gamma, maxitrs=N, verbskip=250, epsilon=epsilon,
                                                     noise=1e-6)
        xLS_, FLS_, GLS_, TLS_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=True, ls_ratio=1.5, verbskip=250,
                                            epsilon=epsilon)
        x20_, F20_, G20_, T20_ = accbpg.ABPG(f, h, L, x0, gamma=gamma, maxitrs=N, theta_eq=False, verbskip=250, epsilon=epsilon)
        x2e_, F2e_, _, G2e_, T2e_ = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=N, theta_eq=False, Gmargin=1,
                                                     verbskip=250, epsilon=epsilon)

        labels = [r"AIBM", r"BPG-Adapt", r"AccBPGM-2", r"AccBPGM-1"]
        styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

        y_vals = [F_AIBM, FLS_, F20_, F2e_]
        ax.set_title(f'$\gamma$ = {gamma}')
        accbpg.plot_comparisons(ax, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[0, N],
                                ylim=[1e-6, 200],
                                xlabel=r"$k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start()
