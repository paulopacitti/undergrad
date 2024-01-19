from undergrad.ops import Softmax, CrossEntropy
import numpy as np
import unittest


class TestStringMethods(unittest.TestCase):
    def test_softmax(self):
        s = Softmax()
        x = np.array([[0, 0],
                      [1, 2],
                      [-3, 2]])

        expected_softmax = np.array([[0.5, 0.5],
                                    [0.26894142, 0.73105858],
                                    [0.00669285, 0.99330715]])
        result_softmax = s(x)
        self.assertTrue((abs(result_softmax - expected_softmax) < 1e-8).all(),
                        f"{expected_softmax} was expected, but {result_softmax} was given")

    def test_cross_entropy(self):
        Y = np.array([[0, 1, 1], [1, 0, 0]])
        Y_pred = np.array([[0, 1, 1], [0.7, 0, 0.3],])

        expected_ce = 0.1783374548265092
        cross_entropy = CrossEntropy()
        ce_result = cross_entropy(Y, Y_pred)
        self.assertTrue(abs(ce_result - expected_ce) < 1e-8,
                        f"{expected_ce} was expected, but {ce_result} was given")

        expected_grad = np.array([[0.,  0.,  0.],
                                  [-0.3,  0.,  0.3]])
        grad = cross_entropy.grad(Y, Y_pred)
        self.assertTrue((abs(grad - expected_grad) < 1e-8).all(),
                        f"{expected_grad} was expected, but {grad} was given")


if __name__ == '__main__':
    unittest.main()
