import h5py
import numpy as np
from matplotlib import pyplot as plt

import accbpg

def start():
    gamma = 2.0
    file_name = "orl.jld"

    try:
        with h5py.File("./dual_rs_methods/data/" + file_name, "r") as file:
            M = np.array(file["M"])
            y_true = np.array(file["labels"])
    except FileNotFoundError:
        print(f"Error: Could not find the file './dual_rs_methods/data/{file_name}'")
        print("Please ensure that:")
        print("1. The 'data' directory exists in dual_rs_methods/")
        print("2. The file 'orl.jld' is present in the dual_rs_methods/data directory")
        return

    N = 200
    r = 10
    epsilon = 1e-9
    noise_level = 0  # Set noise level to 0

    print("Noise level", noise_level)
    n = M.shape[0]
    assert r < n, "r should be less than n"

    f, [h, h_dual, h_fw], L, X0 = accbpg.FrobeniusSymLossEx(M, r, noise_level)

    _, FW_, _, TFW_ = accbpg.UniversalGM(f, h_fw, L, X0, noise_level=noise_level, maxitrs=N, verbskip=10, epsilon=epsilon)
    _, FW_desc, TFW_desc = accbpg.FW_alg_descent_step(f, h, X0, maxitrs=N, lmo=accbpg.lmo_linf_ball(radius=1, center=1), epsilon=epsilon, verbskip=10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [r"Universal FW", r"FW-Descent Step"]
    styles = ['g-', 'k:']
    dashes = [[], [1, 2]]

    y_vals = [FW_, FW_desc]
    accbpg.plot_comparisons(
        ax1, y_vals, labels, x_vals=[], plotdiff=False, yscale="log",
        xlabel=r"$k$", ylabel=r"$F(x_k)-F_\star$", legendloc="upper right",
        linestyles=styles, linedash=dashes
    )

    y_vals = [TFW_, TFW_desc]
    accbpg.plot_comparisons(
        ax2, y_vals, labels, x_vals=[], plotdiff=False, yscale="linear",
        xlabel=r"$k$", ylabel=r'CPU Consumption', legendloc="lower right",
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
