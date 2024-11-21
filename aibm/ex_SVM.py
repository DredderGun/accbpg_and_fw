import matplotlib
import matplotlib.pyplot as plt

import accbpg


def start_aibm():
    """
    Adaptive Fast Gradient Method with Inexact Oracle (AdapFGM) VS Adaptive Intermediate Bregman Method (AIBM).
    See the paper "Accelerated Bregman gradient methods for relatively smooth and
    relatively Lipschitz continuous minimization problems".
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    N = 1000
    lamda = 0.001
    epsilon = 1e-5

    fig, ax = plt.subplots(figsize=(10, 8))

    f, [poly_h, sqL2_h], L, x0, radius = accbpg.svm_digits_ds_divs_ball(lamda=lamda, real_ds=True)

    for noise, ax in zip([0.1], [ax]):
        _, F_AIBM, AIBM_G1, _ = accbpg.AIBM(f, poly_h, L, x0, gamma=2.0, maxitrs=N, verbskip=250, epsilon=epsilon,
                                         noise=noise)
        _, F_AdaptFGM, G_AdaptFGM, _ = accbpg.AdaptFGM(f, poly_h, L, x0, maxitrs=N, verbskip=250, epsilon=epsilon,
                                                                           noise=noise)

        labels = [r"AIBM", r"AdaptFGM"]
        styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

        y_vals = [F_AIBM, F_AdaptFGM]
        ax.set_title(f'$\delta$ = {noise}')
        accbpg.plot_comparisons(ax, y_vals, labels, plotdiff=False, yscale="log",
                                xlabel=r"$k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

    plt.show()


if __name__ == "__main__":
    start_aibm()
