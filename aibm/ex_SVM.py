import matplotlib
import matplotlib.pyplot as plt

import accbpg


def start_aibm():
    """
    Experiments compares Frank-Wolfe algorithms with different divergences and exponents (gamma) in TSE property.
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 500
    ls_ratio = 2.0
    gamma = 1.5
    lamda = 0.001
    epsilon = 1e-12

    fig, ax = plt.subplots(figsize=(15, 15))

    f, [poly_h, sqL2_h], L, x0, radius = accbpg.svm_digits_ds_divs_ball(lamda=lamda, real_ds=True)

    _, F_AIBM_1, G1, _ = accbpg.AIBM(f, poly_h, L, x0, gamma=gamma, maxitrs=maxitrs, verbskip=250, epsilon=epsilon,
                                    noise=1e-14)
    _, F_AIBM_2, G2, _ = accbpg.AIBM(f, poly_h, L, x0, gamma=gamma, maxitrs=maxitrs, verbskip=250, epsilon=epsilon,
                                    noise=1e-12)
    _, F_AIBM_3, G3, _ = accbpg.AIBM(f, poly_h, L, x0, gamma=gamma, maxitrs=maxitrs, verbskip=250, epsilon=epsilon,
                                    noise=1e-9)
    _, F_AIBM_4, G4, _ = accbpg.AIBM(f, poly_h, L, x0, gamma=gamma, maxitrs=maxitrs, verbskip=250, epsilon=epsilon,
                                    noise=1e-7)

    labels = [r"1e-14", r"1e-12", r"1e-9", r"1e-7"]
    styles = ['r:', 'g-', 'b-.', 'y:']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    y_vals = [F_AIBM_1, F_AIBM_2, F_AIBM_3, F_AIBM_4]
    ax.set_title(f'Different noise level')
    accbpg.plot_comparisons(ax, y_vals, labels, x_vals=[], plotdiff=False, yscale="log", xlim=[],
                            ylim=[], xlabel=r"$k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    plt.subplots_adjust(hspace=0.33)
    plt.show()
    plt.savefig("plot.png", bbox_inches='tight')


if __name__ == "__main__":
    start_aibm()
