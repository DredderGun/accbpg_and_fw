import matplotlib
import matplotlib.pyplot as plt
import accbpg


def poisson_regr_in_simplex():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    N = 1000
    m = 2000
    n = 1000
    radius = 1
    f, [h, h_kl, h_sq], L, x0, solution = accbpg.Poisson_regr_simplex(m, n, noise=0.001, randseed=2024)

    x00_FW, F00_FW, G00_FW, T00_FW = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_simplex(radius), maxitrs=N, gamma=2.0,
                                                        ls_ratio=1.5, verbskip=1000)
    x00_, F00_, G00_, T00_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=False, verbskip=1000)
    xLS_, FLS_, GLS_, TLS_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=True, ls_ratio=1.5, verbskip=1000)
    x20_, F20_, G20_, T20_ = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=N, theta_eq=False, verbskip=1000)
    x2e_, F2e_, _, G2e_, T2e_ = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=N, theta_eq=False, Gmargin=1,
                                                 verbskip=1000)
    x2g_, F2g_, G2g_, _, _, _ = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=N, G0=0.1, ls_inc=1.5,
                                                 ls_dec=1.5, theta_eq=True, verbskip=1000)

    fig, ax1 = plt.subplots()

    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"FW"]
    styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

    y_vals = [F00_, FLS_, F20_, F2e_, F2g_, F00_FW]
    accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=False, yscale="log", xlim=[],
                            ylim=[],
                            xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                            linestyles=styles, linedash=dashes)
    plt.tight_layout(w_pad=4)
    plt.show()


if __name__ == "__main__":
    poisson_regr_in_simplex()
