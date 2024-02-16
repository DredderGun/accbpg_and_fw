import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12, 'font.family': 'serif'})

import accbpg


def dopt_design():
    # m = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    n = 1000
    m = [100, 200, 300]
    K = 3
    eps = ['1e-3', '1e-4', '1e-5', '1e-6', '1e-7', '1e-8']

    Nmax = 20000
    Nskip = int(Nmax / 10)

    Ieps = dict()
    Teps = dict()
    for s in eps:
        Ieps[s] = np.zeros((5, len(m), K))
        Teps[s] = np.zeros((5, len(m), K))

    for i in range(len(m)):
        print("\n********** m = {0:d}, n = {1:d} **********".format(m[i], n))
        for k in range(K):
            f, h, L, x0Kh = accbpg.D_opt_design(m[i], n)
            x0KY = accbpg.D_opt_KYinit(f.H)
            x0Mx = (1 - 1e-3) * x0KY + 1e-3 * x0Kh

            _, F_WAKY_TEST, _, T_WAKY_TEST = accbpg.D_opt_FW_RS_adaptive(f, h, L, x0Mx, gamma=2.2, maxitrs=Nmax,
                                                                         linesearch=True, ls_ratio=2, verbskip=Nskip)

            Fmin = F_WAKY_TEST.min()
            F = [F_WAKY_TEST]
            T = [T_WAKY_TEST]
            for s in eps:
                for j in range(len(F)):
                    I_eps = np.nonzero(F[j] - Fmin <= float(s))
                    if len(I_eps[0]) > 0:
                        i_eps = I_eps[0][0]
                        t_eps = T[j][i_eps]
                    else:
                        i_eps = Nmax + 1
                        t_eps = T[j][-1]
                    Ieps[s][j, i, k] = i_eps
                    Teps[s][j, i, k] = t_eps

    s = '1e-3'

    m = np.array(m)
    Igem = np.zeros((5, len(m)))
    Imax = np.zeros((5, len(m)))
    Imin = np.zeros((5, len(m)))
    Tgem = np.zeros((5, len(m)))
    Tmax = np.zeros((5, len(m)))
    Tmin = np.zeros((5, len(m)))

    for i in range(5):
        for j in range(len(m)):
            Igem[i, j] = Ieps[s][i, j].prod() ** (1.0 / K)
            Imax[i, j] = Ieps[s][i, j].max()
            Imin[i, j] = Ieps[s][i, j].min()
            Tgem[i, j] = Teps[s][i, j].prod() ** (1.0 / K)
            Tmax[i, j] = Teps[s][i, j].max()
            Tmin[i, j] = Teps[s][i, j].min()

    # Plot required number of iterations and time
    plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)

    labels = [r"gamma=2"]
    linestyles = ['g-']

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(F_WAKY_TEST)
    # for i in range(5):
    #     idx = np.nonzero(Igem[i] <= Nmax)[0]
    #     if len(idx) > 0:
    #         ax1.errorbar(m[idx], Igem[i,idx], yerr=[Igem[i,idx]-Imin[i,idx], Imax[i,idx]-Igem[i,idx]],
    #                      fmt=linestyles[i], label=labels[i], marker='o', markersize=4, capsize=3)
    ax1.legend()
    plt.title(r"$\gamma=1.5$")
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Номер итерации")
    ax1.set_ylabel(r"F(k)")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(T_WAKY_TEST)
    # for i in range(5):
    #     idx = np.nonzero(Igem[i] <= Nmax)[0]
    #     if len(idx) > 0:
    #         ax2.errorbar(m[idx], Tgem[i,idx], yerr=[Tgem[i,idx]-Tmin[i,idx], Tmax[i,idx]-Tgem[i,idx]],
    #                      fmt=linestyles[i], label=labels[i], marker='o', markersize=4, capsize=3)
    ax2.legend()
    plt.title(r"$\gamma=1.5$")
    ax2.set_yscale('log')
    ax2.set_xlabel(r"Номер итерации")
    ax2.set_ylabel(r"Время")


if __name__ == "__main__":
    dopt_design()
