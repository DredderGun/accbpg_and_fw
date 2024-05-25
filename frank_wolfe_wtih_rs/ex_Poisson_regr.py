import time

import matplotlib
import matplotlib.pyplot as plt
import accbpg


def poisson_regr_in_simplex():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    N = 2500
    m = 2000
    n = 1000
    radius = 1

    h, p_positions = accbpg.Poisson_regr_simplex(m, n, noise=0.001)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs = axs.flatten()

    for i, (key, value) in enumerate(p_positions.items()):
        print(f'Positions: {key}')
        f, L, solution, x0 = value

        x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_simplex(radius), maxitrs=N, gamma=2.0,
                                                            ls_ratio=1.5, verbskip=100)
        x00_, F00_, G00_, T00_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=False, verbskip=1000)
        xLS_, FLS_, GLS_, TLS_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=True, ls_ratio=1.5, verbskip=1000)
        x20_, F20_, G20_, T20_ = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=N, theta_eq=False, verbskip=1000)
        x2e_, F2e_, _, G2e_, T2e_ = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=N, theta_eq=False, Gmargin=1,
                                                     verbskip=1000)
        x2g_, F2g_, G2g_, _, _, _ = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=N, G0=0.1, ls_inc=1.5,
                                                     ls_dec=1.5, theta_eq=True, verbskip=1000)

        labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"FW"]
        styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

        y_vals = [F00_, FLS_, F20_, F2e_, F2g_, F00_FW]
        axs[i].set_title(key)
        accbpg.plot_comparisons(axs[i], y_vals, labels, x_vals=[], plotdiff=False, yscale="log", xlim=[],
                                ylim=[],
                                xlabel=r"$k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

    plt.tight_layout()
    plt.savefig(str('plot') + str(time.time() * 1000)[-4:] + str('.png'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    poisson_regr_in_simplex()
