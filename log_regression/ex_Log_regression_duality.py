import numpy as np
import matplotlib.pyplot as plt

from accbpg import LogisticRegressionFun, GradientDescentDualityCriteria, ConstantStepSize


def start_logistic_regression_duality():
    mn_X = np.load('./data/X.npy')
    mn_y = np.load('./data/y.npy')

    alpha = 0.001

    gap = LogisticRegressionFun(mn_X, mn_y, alpha=alpha)
    method = GradientDescentDualityCriteria(ConstantStepSize(0.1), gap.f)
    f = LogisticRegressionFun(mn_X, mn_y, alpha=alpha)
    omega0 = np.random.random(mn_X.shape[1])
    method.solve(omega0, f, f.gradient(mn_X, mn_y), max_iter=1000)

    fig, ax = plt.subplots()
    x = []
    y = []
    for (val, i) in method.history:
        x.append(i)
        y.append(val)

    ax.plot(np.array(x), np.array(y))

    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_logistic_regression_duality()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
