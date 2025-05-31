import unittest
import numpy as np
import cvxpy as cp
from accbpg.functions import SumOf2nd4thPowers

class TestSumOf2nd4thPowers(unittest.TestCase):

    def test_prox_map_mx(self):
        alpha = 1.0
        sigma = 1.0
        g = np.array([[1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0]])
        L = 1.0

        sum_of_powers = SumOf2nd4thPowers(alpha, sigma)
        result = sum_of_powers.prox_map(g, L)

        X = cp.Variable(g.shape, pos=True)
        objective = cp.Minimize(cp.vec(g) @ cp.vec(X) +
                                L * (alpha/4 * cp.norm(X, 'fro')**4 + sigma/2 * cp.norm(X, 'fro')**2))
        problem = cp.Problem(objective)
        problem.solve(verbose=True)
        expected_result = X.value

        np.testing.assert_almost_equal(result, expected_result, decimal=3)

    def test_prox_map_mx_big(self):
        alpha = 1.0
        sigma = 1.0
        g = np.random.rand(500, 500)
        L = 1.0

        sum_of_powers = SumOf2nd4thPowers(alpha, sigma)
        result = sum_of_powers.prox_map(g, L)

        X = cp.Variable(g.shape, pos=True)
        objective = cp.Minimize(cp.vec(g) @ cp.vec(X) +
                                L * (alpha/4 * cp.norm(X, 'fro')**4 + sigma/2 * cp.norm(X, 'fro')**2))
        problem = cp.Problem(objective)
        problem.solve()
        expected_result = X.value

        np.testing.assert_almost_equal(result, expected_result, decimal=3)


if __name__ == '__main__':
    unittest.main()
