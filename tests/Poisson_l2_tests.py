import matplotlib
import matplotlib.pyplot as plt
import accbpg


def poisson_regr_in_l2_ball():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    N = 5000
    m = 100
    n = 1000
    radius = 100
    f, h, L, x0 = accbpg.Poisson_regrL2_ball(m, n, radius=radius, noise=0.001, lamda=0.001, randseed=1)

    # Solve the problem using BPG w/o line seach and adaptive ABPG with gamma=2 (TSE)
    x00_1, F00_1, G00_1, T00_1 = accbpg.FW_alg_div_step(f, h, L, x0, lmo=accbpg.lmo_positive_ball(radius), maxitrs=N, gamma=2.0,
                                                                  ls_ratio=1.5, verbskip=1)
    x00_, F00_, G00_, T00_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=False, verbskip=1)
    xLS_, FLS_, GLS_, TLS_ = accbpg.BPG(f, h, L, x0, maxitrs=N, linesearch=True, ls_ratio=1.5, verbskip=1)
    x20_, F20_, G20_, T20_ = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=N, theta_eq=False, verbskip=1)
    x2e_, F2e_, _, G2e_, T2e_ = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=N, theta_eq=False, Gmargin=1,
                                                 verbskip=1)
    x2g_, F2g_, G2g_, _, _, _ = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=N, G0=0.1, ls_inc=1.5,
                                                 ls_dec=1.5, theta_eq=True, verbskip=1)

    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"FW-adapt"]
    styles = ['k:', 'g-', 'b-.', 'k-', 'r--', 'y-']
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

    does_print_plots = True
    if does_print_plots:
        ax1 = plt.subplot(1, 2, 1)
        y_vals = [F00_, FLS_, F20_, F2e_, F2g_, F00_1]
        accbpg.plot_comparisons(ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=[-20, 2000],
                                ylim=[1e-6, 1e-1],
                                xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)$", legendloc="upper right",
                                linestyles=styles, linedash=dashes)

        ax2 = plt.subplot(1, 2, 2)
        y_vals = [GLS_, G20_, G2e_, G2g_, G00_1]
        accbpg.plot_comparisons(ax2, y_vals, labels[1:], x_vals=[], plotdiff=False, yscale="log", xlim=[-20, 2000],
                                ylim=[1e-4, 1e3],
                                xlabel=r"Iteration number $k$", ylabel=r'$\hat{G}_k$', legendloc="center right",
                                linestyles=styles[1:], linedash=dashes[1:])

        plt.tight_layout(w_pad=4)
        fig.suptitle('$minimize_{x \in \|x\|_2 \geq 1}  D_KL(b, Ax)$')
        plt.show()


if __name__ == "__main__":
    poisson_regr_in_l2_ball()
