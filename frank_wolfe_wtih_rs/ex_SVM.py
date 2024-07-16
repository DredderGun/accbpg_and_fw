import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import accbpg


def start_svm_exps():
    """
    Experiments compares Frank-Wolfe algorithm with Bregman Proximal gradient method and its accelerated version.
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 500
    ls_ratio = 2.0
    gamma = 2.0

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs = axs.flatten()

    np.random.seed(3)

    lamdas = [0.1, 0.5, 1, 4]
    for i, lamda in zip(range(len(lamdas)), lamdas):
        f, [poly_h, sqL2_h], L, x0, radius = accbpg.svm_digits_ds_divs_ball(lamda=lamda, real_ds=True)

        x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_l2_ball(radius),
                                                                maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio,
                                                                verbskip=100)
        xLS_, FLS_, GLS_, TLS_ = accbpg.BPG(f, poly_h, L, x0, maxitrs=maxitrs, linesearch=True, ls_ratio=1.5, verbskip=100)
        x20_, F20_, G20_, T20_ = accbpg.ABPG(f, poly_h, L, x0, gamma=gamma, maxitrs=maxitrs, theta_eq=False, verbskip=100)
        labels = [r"FW", r"BPG-LS", r"ABPG"]
        styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

        y_vals = [F00_FW, FLS_, F20_]
        # y_vals = [F00_FW, FLS_, F20_, F2e_]
        axs[i].set_title(f'$\lambda$ = {lamda}')
        accbpg.plot_comparisons(axs[i], y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[],
                                ylim=[], xlabel=r"$k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

    plt.subplots_adjust(hspace=0.33)
    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    start_svm_exps()
