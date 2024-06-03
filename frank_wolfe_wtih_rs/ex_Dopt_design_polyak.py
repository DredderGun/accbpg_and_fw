import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import accbpg
from accbpg import X_equals_one_cnstrnt
from accbpg.algorithms import SwitchingGradientDescent


def polyak_vs_FW_d_opt_design():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    np.random.seed(2024)
    max_itrs = 500
    m = 2000
    n = 1000
    simplex_radius = 1
    tol = 1e-12
    verb_skip = 100
    reg_lamda = 0.01

    f, h, L, x0 = accbpg.D_opt_libsvm('data/housing.txt')

    fig, axs = plt.subplots(figsize=(15, 15))

    x00_fw, F00_fw, G00_fw, T00_fw = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_simplex(),
                                                            maxitrs=max_itrs, gamma=2.0, ls_ratio=2, verbskip=100)
    polyak_step_size_method = SwitchingGradientDescent(f, solution_f=f(x00_fw), cnstrnt_nmbr=10 ** -9,
                                                       constraints=[X_equals_one_cnstrnt(sign=simplex_radius),
                                                                    X_equals_one_cnstrnt(sign=-1 * simplex_radius)],
                                                       M=10**7)
    x_polyak, F_polyak, T_polyak = polyak_step_size_method.solve(x0, epsilon=tol, maxitrs=max_itrs, verbskip=verb_skip)

    labels = [r"FW", r"Polyak"]
    styles = ['k:', 'g-']
    dashes = [[1, 2], []]

    y_vals = [F00_fw, F_polyak]
    accbpg.plot_comparisons(axs, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[],
                            xlabel=r"$k$", ylabel=r"$sol - F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)


    plt.tight_layout()
    plt.savefig('polyak_vs_FW_dopt.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    polyak_vs_FW_d_opt_design()
