import matplotlib.pyplot as plt
import numpy as np
import accbpg

def start():
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.random.rand(400)
    M = np.outer(x, x)

    maxitrs = 200
    r = 10
    epsilon = 1e-9
    verbskip = 100
    
    f, constraint_fun, h, L, X0 = accbpg.FrobeniusSymLossExWithLinearCnstrnts(M, r)

    FBPG_, DG_, LS_ = accbpg.PrimalDualSwitchingGradientMethod(f, h, L, constraint_fun, X0, maxitrs, epsilon=epsilon, linesearch=False,
                                      verbose=True, verbskip=verbskip)
    FBPG_LS, DG_LS, LS_LS = accbpg.PrimalDualSwitchingGradientMethod(f, h, L, constraint_fun, X0, maxitrs, epsilon=epsilon, linesearch=True,
                                      verbose=True, verbskip=verbskip)
    
    labels = [r"No LS", r"With LS"]
    styles = ['k:', 'g-', 'b-.']
    dashes = [[1, 2], [], [4, 2, 1, 2]]
    markers = ['o', 's', '^']

    accbpg.plot_comparisons(ax, [DG_, DG_LS], labels, x_vals=[], plotdiff=True,
                            yscale="log", xlim=[], ylim=[], xlabel=r"Номер итерации",
                            ylabel=r"Зазор двойственности", legendloc="upper right",
                            linestyles=styles, linedash=dashes)
        
    # Modify the lines after plotting
    for line, marker in zip(ax.lines, markers):
        line.set_linewidth(2.0)
        line.set_marker(marker)
        line.set_markevery(50)
        line.set_markersize(6)

    # Then modify the legend
    leg = ax.get_legend()
    plt.setp(leg.get_texts(), fontsize=12)  # Increase font size

    # fig.suptitle(f'Степень разреженности M = {matrix_sparsity:.2f}', fontsize=16)
    plt.tight_layout()
    plt.savefig("output_image.png")
    plt.show()

if __name__ == "__main__":
    start()