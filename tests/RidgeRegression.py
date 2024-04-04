import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import accbpg


def start():
    """
    \\ todo
    """
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    maxitrs = 5000
    gamma = 2.0
    lamda = 0.2
    f, [h1, h2], L, x0, radius, solution_x = accbpg.distributed_ridge_regression_problem(500, 200, comp_nmbr=50, noise=0.1, lamda=lamda, randseed=2024)

    # sol = f.get_cvxpy_solution()
    print(np.linalg.norm(solution_x - x0))
    x00_FW, F00_FW, G00_FW, T00_FW, alphas = accbpg.FW_alg_div_step(f, h1, L, x0, lmo=accbpg.lmo_l2_ball(radius), linesearch=False,
                                                                    maxitrs=maxitrs, gamma=gamma, verbskip=100)
    x00_FW_evk, F00_FW_evk, G00_FW_evk, T00_FW_evk, alphas_evk = accbpg.FW_alg_div_step(f, h2, L, x0, lmo=accbpg.lmo_l2_ball(radius),
                                                                    linesearch=False, maxitrs=maxitrs, gamma=gamma, verbskip=100)
    x00_BPG, F00_BPG, G00_BPG, T00_BPG = accbpg.BPG(f, h2, L, x0, linesearch=False, maxitrs=maxitrs, verbskip=100)

    fig, ax = plt.subplots()

    labels = [r"FW-RS", r"FW-RS-evklid-norm", r"Bregman Proximal"]
    styles = ['r:', 'g-', 'b-.', 'y:']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2]]

    y_vals = [F00_FW, F00_FW_evk, F00_BPG]
    accbpg.plot_comparisons(ax, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[],
                            ylim=[], xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)

    # ax2 = plt.subplot(1, 2, 2)
    # y_vals = [alphas, alphas_sq]
    # accbpg.plot_comparisons(ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear", xlim=[],
    #                         ylim=[], xlabel=r"Iteration number $k$", ylabel=r'$\alpha_k$', legendloc="lower right",
    #                         linestyles=styles, linedash=dashes)

    print(np.linalg.norm(solution_x - x00_FW))
    print(np.linalg.norm(solution_x - x00_FW_evk))
    print(np.linalg.norm(solution_x - x00_BPG))
    plt.tight_layout(w_pad=4)
    plt.show()


if __name__ == "__main__":
    start()
