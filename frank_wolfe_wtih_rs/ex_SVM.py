import matplotlib
import matplotlib.pyplot as plt

import accbpg


def start_svm_divs():
    """
    Experiments compares Frank-Wolfe algorithms with different divergences and exponents (gamma) in TSE property.
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 500
    ls_ratio = 2.0
    gamma = 2.0

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs = axs.flatten()

    lamdas = [0.1, 0.5, 10, 100]
    for i, lamda in zip(range(len(lamdas)), lamdas):
        f, [poly_h, sqL2_h], L, x0, radius = accbpg.svm_digits_ds_divs_ball(lamda=lamda, real_ds=True)

        x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_l2_ball(radius),
                                                                maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio,
                                                                verbskip=100)
        x00_FW1, F00_FW1, G00_FW1, T00_FW1 = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_l2_ball(radius),
                                                                    maxitrs=maxitrs, gamma=1.5, ls_ratio=ls_ratio,
                                                                    verbskip=100)
        # x00_FW3, F00_FW3, G00_FW3, T00_FW3 = accbpg.FW_alg_div_step(f, sqL2_h, L, x0, lmo=accbpg.lmo_l2_ball(radius),
        #                                                             maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio,
        #                                                             verbskip=100)
        x00_FW4, F00_FW4, G00_FW4, T00_FW4 = accbpg.BPG(f, sqL2_h, L, x0, maxitrs=maxitrs, ls_ratio=ls_ratio,
                                                                    verbskip=100)

        labels = [r"FW-2.0", r"FW-1.5", r"BPG"]
        styles = ['r:', 'g-', 'b-.', 'y:']
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

        # y_vals = [F00_FW, F00_FW1, F00_FW3, F00_FW4]
        y_vals = [F00_FW, F00_FW1, F00_FW4]
        axs[i].set_title(f'$\lambda$ = {lamda}')
        accbpg.plot_comparisons(axs[i], y_vals, labels, x_vals=[], plotdiff=False, yscale="log", xlim=[],
                                ylim=[], xlabel=r"$k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

    plt.subplots_adjust(hspace=0.33)
    plt.show()


if __name__ == "__main__":
    start_svm_divs()
