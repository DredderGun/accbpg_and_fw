import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import accbpg


def start_iris():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    radius = 500
    maxitrs = 1000
    ls_ratio = 2.0
    gamma = 2.0
    f, h, L, x0 = accbpg.svm_alg_iris_ds(radius=radius)

    x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_notnegative_ball(radius, center=radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    x00, F00, G00, T00 = accbpg.BPG(f, h, L, x0, maxitrs=maxitrs, linesearch=True, ls_ratio=ls_ratio, verbskip=100)
    x20, F20, G20, T20 = accbpg.ABPG(f, h, L, x0, gamma=gamma, maxitrs=maxitrs, theta_eq=True, verbskip=100)

    # Plot the objective gap and estimated gains for triangle scaling
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"FW", r"BPG", r"ABPG"]
    styles = ['r:', 'g-', 'b-.']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    ax1 = plt.subplot(1, 2, 1)
    y_vals = [F00_FW, F00, F20]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    ax2 = plt.subplot(1, 2, 2)
    y_vals = [T00_FW, T00, T20]
    accbpg.plot_comparisons(ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r'$\hat{G}_k$', legendloc="lower right",
                            linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.show()

def start_banknote():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    radius = 1000
    maxitrs = 1000
    ls_ratio = 2.0
    gamma = 2.0
    f, h, L, x0 = accbpg.svm_alg_banknote(radius=radius)

    x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_notnegative_ball(radius, center=radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    x00, F00, G00, T00 = accbpg.BPG(f, h, L, x0, maxitrs=maxitrs, linesearch=True, ls_ratio=ls_ratio, verbskip=100)
    x20, F20, G20, T20 = accbpg.ABPG(f, h, L, x0, gamma=gamma, maxitrs=maxitrs, theta_eq=True, verbskip=100)

    # Plot the objective gap and estimated gains for triangle scaling
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"FW", r"BPG", r"ABPG"]
    styles = ['r:', 'g-', 'b-.']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    ax1 = plt.subplot(1, 2, 1)
    y_vals = [F00_FW, F00, F20]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    ax2 = plt.subplot(1, 2, 2)
    y_vals = [T00_FW, T00, T20]
    accbpg.plot_comparisons(ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r'$\hat{G}_k$', legendloc="lower right",
                            linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.show()


if __name__ == "__main__":
    # start_iris()
    start_banknote()

