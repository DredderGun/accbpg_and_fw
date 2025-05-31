import h5py
import numpy as np
from matplotlib import pyplot as plt

import accbpg


def start():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.flatten()

    file_name = "orl.jld"

    # try:
    #     with h5py.File("./triangle_methods/data/" + file_name, "r") as file:
    #         M = np.array(file["M"])  # Read the "M" dataset
    #         y_true = np.array(file["labels"])  # Read the "labels" dataset
    # except FileNotFoundError:
    #     print(f"Error: Could not find the file './triangle_methods/data/{file_name}'")
    #     print("Please ensure that:")
    #     print("1. The 'data' directory exists in triangle_methods/")
    #     print("2. The file 'orl.jld' is present in the triangle_methods/data directory")
    #     return

    x = np.random.rand(400)
    M = np.outer(x, x)

    matrix_sparsity = (M == 0).sum() / M.size

    N = 200
    r = 1
    epsilon = 1e-9
    verbskip = 200

    for ax, noise_level in zip(axes, np.logspace(-1, 1, 2)):
        
        n = M.shape[0]
        assert r < n, "r should be less than n"

        f, [h, h_dual, h_fw], L, X0 = accbpg.FrobeniusSymLossEx(M, r, noise_level)

        xBPG_, FBPG_, GBPG_, TBPG_ = accbpg.BPG(f, h, L, X0, noise_level=noise_level, maxitrs=N, linesearch=False, ls_ratio=1.5, verbskip=100,
                                                epsilon=epsilon)
        xBPG_LS, FBPG_LS, GBPG_LS, TBPG_LS = accbpg.BPG(f, h, L, X0, noise_level=noise_level, maxitrs=N, linesearch=True, ls_ratio=1.5, verbskip=100, 
                                                        epsilon=epsilon)
        xFW_, FFW_, GFW_, TFW_ = accbpg.UniversalGM(f, h_fw, L, X0, noise_level=noise_level, maxitrs=N, verbskip=100,
                                                    epsilon=epsilon)
        
        labels = [r"FW", r"BPG", r"BPG LS"]
        styles = ['k:', 'g-', 'b-.']
        dashes = [[1, 2], [], [4, 2, 1, 2]]
        markers = ['o', 's', '^']

        ax.set_title(f'$\delta$ = {noise_level:.2f}')

        print('BPG method matrix rank is', np.linalg.matrix_rank(xBPG_.dot(xBPG_.T), tol=1e-5))
        print('BPG-LS method matrix rank is', np.linalg.matrix_rank(xBPG_LS.dot(xBPG_LS.T), tol=1e-5))
        # print(np.linalg.matrix_rank(xBPG_Dual, tol=1e-5))
        print('FW method matrix rank is', np.linalg.matrix_rank(xFW_.dot(xFW_.T), tol=1e-5))

        # First create the plots
        M_norm = np.linalg.norm(M, 'fro')**2
        accbpg.plot_comparisons(ax, [FFW_ / M_norm, FBPG_ / M_norm, FBPG_LS / M_norm], labels, x_vals=[], plotdiff=True,
                                yscale="log", xlim=[0, N], ylim=[], xlabel=r"Номер итерации",
                                ylabel=r"$(F(x_k) - F^*) \ / \ \| M \|_2$", legendloc="upper right",
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

    fig.suptitle(f'Степень разреженности M = {matrix_sparsity:.2f}', fontsize=16)
    plt.tight_layout()
    plt.savefig("output_image.png")
    plt.show()
    

if __name__ == "__main__":
    start()
