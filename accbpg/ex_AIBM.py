import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import accbpg


def start():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    np.random.seed(2024)
    N = 1000
    m = 2000
    n = 1000
    radius = 1

    f, h, L, x0 = accbpg.Poisson_regr_simplex_acc(m, n, noise=0.001)

    fig, axs = plt.subplots(2, 2, figsize=(11, 4))
    axs = axs.flatten()

    for i, gamma in enumerate([2.0, 1.7, 1.4, 1.1]):
        x_AIBM, F_AIBM, G_AIBM, T_AIBM = accbpg.AIBM(f, h, L, x0, gamma=gamma, maxitrs=N, verbskip=250, epsilon=1e-9)
        x00_, F00_, G00_, T00_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=False, verbskip=250, epsilon=1e-9)
        xLS_, FLS_, GLS_, TLS_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=True, ls_ratio=1.5, verbskip=250,
                                            epsilon=1e-9)
        x20_, F20_, G20_, T20_ = accbpg.ABPG(f, h, L, x0, gamma=gamma, maxitrs=N, theta_eq=False, verbskip=250, epsilon=1e-9)
        x2e_, F2e_, _, G2e_, T2e_ = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=N, theta_eq=False, Gmargin=1,
                                                     verbskip=250, epsilon=1e-9)
        # x2g_, F2g_, G2g_, _, _, _ = accbpg.ABPG_gain(f, h, L, x0, gamma=gamma, maxitrs=N, G0=0.1, ls_inc=1.5,
        #                                              ls_dec=1.5, theta_eq=True, verbskip=250, epsilon=1e-9)

        labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"AIBM"]
        styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

        # y_vals = [F00_, FLS_, F20_, F2e_, F2g_, F_AIBM]
        y_vals = [F00_, FLS_, F20_, F2e_, F_AIBM]
        # ax1 = plt.subplot(1, 2, 1)
        axs[i].set_title(print(f'\gamma: {gamma}'))
        accbpg.plot_comparisons(axs[i], y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[0, 1000],
                                ylim=[1e-6, 200],
                                xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

#         ax2 = plt.subplot(1, 2, 2)
#         accbpg.plot_comparisons(axs[i], y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xscale="log", xlim=[1, 5000],
#                                 ylim=[1e-6, 200],
#                                 xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="lower left",
#                                 linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.savefig("plot.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    start()
