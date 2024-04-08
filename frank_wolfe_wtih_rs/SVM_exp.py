import matplotlib
import matplotlib.pyplot as plt

import accbpg


def start_svm_divs():
    """
    Experiments compares Frank-Wolfe algorithms with different divergences and exponents (gamma) in TSE property.
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 1000
    ls_ratio = 2.0
    gamma = 2.0
    lamda = 100
    f, [poly_h, sqL2_h, kl], L, x0, radius = accbpg.svm_digits_ds_divs_ball(lamda=lamda)

    x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    x00_FW1, F00_FW1, G00_FW1, T00_FW1 = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=1.5, ls_ratio=ls_ratio, verbskip=100)
    x00_FW3, F00_FW3, G00_FW3, T00_FW3 = accbpg.FW_alg_div_step(f, sqL2_h, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    x00_FW4, F00_FW4, G00_FW4, T00_FW4 = accbpg.FW_alg_div_step(f, kl, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)

    fig, ax = plt.subplots()

    labels = [r"FW-2.0", r"FW-1.5", r"FW-sqL2", r"KL"]
    styles = ['r:', 'g-', 'b-.', 'y:']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    y_vals = [F00_FW, F00_FW1, F00_FW3, F00_FW4]
    accbpg.plot_comparisons(ax, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.show()


if __name__ == "__main__":
    start_svm_divs()

