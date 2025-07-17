# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from .functions import *
from .functions_lmo import *
from .algorithms import (BPG, ABPG, ABPG_expo, ABPG_gain, ABDA, AIBM, AdaptFGM, 
                         UniversalGM, PrimalDualSwitchingGradientMethod)
from .algorithms_fw import (FW_alg_div_step, FW_alg_descent_step, FW_alg_div_step_adapt, FW_alg_l0_l1_step_adapt)
from .applications import (D_opt_libsvm, D_opt_design, D_opt_KYinit,
                           Poisson_regrL1, Poisson_regrL2, KL_nonneg_regr,
                           Poisson_regr_simplex, Poisson_regr_simplex_acc,
                           svm_digits_ds_divs_ball, FrobeniusSymLossExL2Ball, FrobeniusSymLossExWithLinearCnstrnts,
                           FrobeniusSymLossResMeasEx, FrobeniusSymLossExLInfBall, L0L1_FW_log_reg, L0L1_FW_log_reg_a9a)
from .D_opt_alg import D_opt_FW, D_opt_FW_away
from .trianglescaling import plotTSE, plotTSE0
from .plotfigs import plot_comparisons
from .utils import random_point_on_simplex, random_point_in_l2_ball
