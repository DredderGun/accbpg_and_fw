import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import accbpg
from accbpg import X_equals_one_cnstrnt
from accbpg.algorithms import SwitchingGradientDescent


def polyak_vs_FW_poisson_regr_in_simplex():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    np.random.seed(2024)
    max_itrs = 2500
    m = 2000
    n = 1000
    simplex_radius = 1
    tol = 1e-12
    verb_skip = 100
    reg_lamda = 0.01

    h, p_positions = accbpg.Poisson_regr_simplex(m, n, noise=0.01)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs = axs.flatten()

    for i, (key, value) in enumerate(p_positions.items()):
        print(f'Positions: {key}')
        f, L, solution, x0 = value

        x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, h, L, x0, epsilon=tol, lmo=accbpg.lmo_simplex(simplex_radius),
                                                                maxitrs=max_itrs, gamma=2.0, ls_ratio=1.5, verbskip=verb_skip)
        polyak_step_size_method = SwitchingGradientDescent(f, solution_f=f(solution), cnstrnt_nmbr=10**-5,
                                                           constraints=[X_equals_one_cnstrnt(sign=simplex_radius),
                                                                        X_equals_one_cnstrnt(sign=-1*simplex_radius)],
                                                           M=10)
        x_polyak, F_polyak, T_polyak = polyak_step_size_method.solve(x0, epsilon=tol, maxitrs=max_itrs, verbskip=verb_skip)

        labels = [r"FW", r"Polyak"]
        styles = ['k:', 'g-']
        dashes = [[1, 2], []]

        y_vals = [(np.abs(f(solution) - F00_FW)), np.abs(f(solution) - F_polyak[F_polyak != 0.0])]
        axs[i].set_title(key)
        accbpg.plot_comparisons(axs[i], y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[],
                                xlabel=r"$k$", ylabel=r"$sol - F(x_k)$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

    plt.tight_layout()
    plt.savefig('polyak_vs_FW.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    polyak_vs_FW_poisson_regr_in_simplex()
