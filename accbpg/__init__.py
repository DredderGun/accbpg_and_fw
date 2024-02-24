# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from .functions import *
from .algorithms import BPG, ABPG, ABPG_expo, ABPG_gain, ABDA, FW_alg_div_step
from .applications import D_opt_libsvm, D_opt_design, D_opt_KYinit, Poisson_regrL1, Poisson_regrL2, KL_nonneg_regr, Poisson_regrL2_ball, Poisson_regr_diff_divs, svm_alg_iris_ds, svm_alg_banknote
from .D_opt_alg import D_opt_FW, D_opt_FW_away
from .trianglescaling import plotTSE, plotTSE0
from .plotfigs import plot_comparisons