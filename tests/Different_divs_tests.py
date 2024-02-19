import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import accbpg


def Poisson_regr_divs():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 1000
    m = 10
    n = 100
    radius = 100
    noise = 0
    ls_ratio = 2
    f, [burg_h, sqL2_h, shannon_h], L, x0, solution = accbpg.Poisson_regr_diff_divs(m, n, radius=radius, noise=noise, lamda=0.001, randseed=1, normalizeA=False)

    x00_burg, F00_burg, G00_burg, T00_burg = accbpg.FW_alg_div_step(f, burg_h, L, x0, lmo=accbpg.lmo_notnegative_ball(radius, is_shifted_pos_ball=True), maxitrs=maxitrs, gamma=2.5,
                                                                    ls_ratio=ls_ratio, verbskip=1000)

    x00_sq, F00_sq, G00_sq, T00_sq = accbpg.FW_alg_div_step(f, sqL2_h, L, x0, lmo=accbpg.lmo_notnegative_ball(radius, is_shifted_pos_ball=True), maxitrs=maxitrs, gamma=2.0,
                                                        ls_ratio=ls_ratio, verbskip=1000)

    x00_sh, F00_sh, G00_sh, T00_sh = accbpg.FW_alg_div_step(f, shannon_h, L, x0, lmo=accbpg.lmo_notnegative_ball(radius, is_shifted_pos_ball=True), maxitrs=maxitrs, gamma=2.0,
                                                            ls_ratio=ls_ratio, verbskip=1000)

    print(np.linalg.norm(solution - x00_burg), "burg")
    print(np.linalg.norm(solution - x00_sq), "sq")
    print(np.linalg.norm(solution - x00_sh), "sh")

    print(F00_burg[F00_burg.shape[0] - 1], "sol burg")
    print(F00_sq[F00_sq.shape[0] - 1], "sol sq")
    print(F00_sh[F00_sh.shape[0] - 1], "sol sh")

    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"BurgNorm", r"SquaredL2Norm", r"Shannon"]
    styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

    ax1 = plt.subplot(1, 2, 1)
    y_vals = [F00_burg, F00_sq, F00_sh]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[-20, 300],
                            ylim=[1e-6, 1e+7],
                            xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)
    plt.show()


if __name__ == "__main__":
    Poisson_regr_divs()
