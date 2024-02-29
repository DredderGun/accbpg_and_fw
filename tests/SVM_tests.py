import matplotlib
import matplotlib.pyplot as plt

import accbpg


def start_svm():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    # radius = 1000
    maxitrs = 1000
    ls_ratio = 2.0
    gamma = 2.0
    lamda = 100
    f, h, L, x0, radius = accbpg.smv_digits_ds(lamda=lamda)

    x00_FW, F00_FW, G00_FW, T00_FW, alphas = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_l2_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    # x00, F00, G00, T00 = accbpg.BPG(f, h, L, x0, maxitrs=maxitrs, linesearch=True, ls_ratio=ls_ratio, verbskip=100)
    # x20, F20, G20, T20 = accbpg.ABPG(f, h, L, x0, gamma=gamma, maxitrs=maxitrs, theta_eq=True, verbskip=100)

    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"FW", r"BPG", r"ABPG"]
    styles = ['r:', 'g-', 'b-.']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    ax1 = plt.subplot(1, 2, 1)
    # y_vals = [F00_FW, F00, F20]
    y_vals = [F00_FW]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    ax2 = plt.subplot(1, 2, 2)
    # y_vals = [T00_FW, T00, T20]
    # y_vals = [alphas, np.zeros(x00.shape), np.zeros(x20.shape)]
    y_vals = [alphas]

    accbpg.plot_comparisons(ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r'$\alpha_k$', legendloc="lower right",
                            linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.show()


def start_svm_divs():
    """
    Сравнивается алгоритм Франка-Вульфа с дивергенцией и с обычной евклидовой нормой в шаге alpha
    для задачи SVM для которой есть доказательство что она относительно липшицева относительно прокса poly_h

    Эксперимент показал, что на сиклассический шаг FW самый эффективный, но на шаре можно добиться преимущест

    Шар: smv_digits_ds_divs_ball для данных и accbpg.lmo_l2_ball для LMO
    Симплекс: smv_digits_ds_divs_simplex для данных и accbpg.lmo_simplex для LMO
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 1000
    ls_ratio = 2.0
    gamma = 2.0
    lamda = 100
    f, [poly_h, sqL2_h, kl], L, x0, radius = accbpg.svm_digits_ds_divs_ball(lamda=lamda)

    x00_FW, F00_FW, G00_FW, T00_FW, alphas = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    x00_FW1, F00_FW1, G00_FW1, T00_FW1, alphas1 = accbpg.FW_alg_div_step(f, poly_h, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=1.5, ls_ratio=ls_ratio, verbskip=100)
    x00_FW3, F00_FW3, G00_FW3, T00_FW3, alphas3 = accbpg.FW_alg_div_step(f, sqL2_h, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)
    x00_FW4, F00_FW4, G00_FW4, T00_FW4, alphas4 = accbpg.FW_alg_div_step(f, kl, L, x0, lmo=accbpg.lmo_linf_ball(radius),
                                                                    maxitrs=maxitrs, gamma=gamma, ls_ratio=ls_ratio, verbskip=100)

    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"FW-2.0", r"FW-1.5", r"FW-sqL2", r"KL"]
    styles = ['r:', 'g-', 'b-.', 'y:']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    ax1 = plt.subplot(1, 1, 1)
    # y_vals = [F00_FW, F00, F20]
    y_vals = [F00_FW, F00_FW1, F00_FW3, F00_FW4]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    ax2 = plt.subplot(1, 2, 2)
    # y_vals = [T00_FW, T00, T20]
    # y_vals = [alphas, np.zeros(x00.shape), np.zeros(x20.shape)]
    y_vals = [alphas]

    accbpg.plot_comparisons(ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[0, 70],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r'$\alpha_k$', legendloc="lower right",
                            linestyles=styles, linedash=dashes)

    plt.tight_layout(w_pad=4)
    plt.show()


if __name__ == "__main__":
    start_svm_divs()

