import h5py
import numpy as np
from matplotlib import pyplot as plt

import accbpg

def start():
    gamma = 2.0
    file_name = "orl.jld"

    try:
        with h5py.File("./descent_and_lipschitz_steps/data/" + file_name, "r") as file:
            M = np.array(file["M"])
            y_true = np.array(file["labels"])
    except FileNotFoundError:
        print(f"Error: Could not find the file './descent_and_lipschitz_steps/data/{file_name}'")
        print("Please ensure that:")
        print("1. The 'data' directory exists in descent_and_lipschitz_steps/")
        print("2. The file 'orl.jld' is present in the descent_and_lipschitz_steps/data directory")
        return
    
    # x = np.random.rand(700)
    # M = np.outer(x, x)

    N = 600
    r = 70
    epsilon = 1e-9
    noise_level = 0  # Set noise level to 0
    verbskip = 10

    print("Noise level", noise_level)
    n = M.shape[0]
    assert r < n, "r should be less than n"

    f, [h, h_euk], L, X0 = accbpg.FrobeniusSymLossResMeasEx(M, r, noise_level)

    _, FW_, _, G_Descent, divergences = accbpg.FW_alg_descent_step(f, h, X0, maxitrs=N, 
                                                                    lmo=accbpg.lmo_linf_ball(radius=1, center=1), 
                                                                    epsilon=epsilon, verbskip=verbskip)
    
    _, FW_desc, _, G_Descent_euk, divergences_euk = accbpg.FW_alg_descent_step(f, h_euk, X0, maxitrs=N, 
                                                                    lmo=accbpg.lmo_linf_ball(radius=1, center=1), 
                                                                    epsilon=epsilon, verbskip=verbskip)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [r"FW-Burg", r"FW-Euk"]
    styles = ['g-', 'k:']
    dashes = [[], [1, 2]]

    y_vals = [FW_, FW_desc]
    accbpg.plot_comparisons(
        ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log",
        xlabel=r"$k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
        linestyles=styles, linedash=dashes
    )

    # last_term_ar_Burg = np.maximum.accumulate(G_Descent)
    # last_term_ar_Burg = last_term_ar_Burg * np.maximum.accumulate(divergences)
    last_term_ar_Burg = G_Descent * divergences

    # last_term_ar_Euk = np.maximum.accumulate(G_Descent_euk)
    # last_term_ar_Euk = last_term_ar_Euk * np.maximum.accumulate(divergences_euk)
    last_term_ar_Euk = G_Descent_euk * divergences_euk
    accbpg.plot_comparisons(
        ax2, [last_term_ar_Burg, last_term_ar_Euk], labels, x_vals=[], plotdiff=False, yscale="log",
        xlabel=r"$k$", ylabel=r'$L_{\text{max_k}} \cdot V_{\text{max_k}}$', legendloc="lower right",
        linestyles=styles, linedash=dashes
    )

    # Modify the lines after plotting
    # markers = ['o', 's',]
    # for line, marker in zip(ax1.lines, markers):
    #     line.set_linewidth(2.0)
    #     line.set_marker(marker)
    #     line.set_markevery(20)
    #     line.set_markersize(6)

    # Ensure the legend is created before modifying it
    ax1.legend(fontsize=12)

    # Save the plot
    plt.savefig("convergence_rates_plot.png", dpi=300, bbox_inches="tight")
    
    plt.show()
    
if __name__ == "__main__":
    start()
