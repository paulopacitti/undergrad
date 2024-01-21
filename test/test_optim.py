from undergrad import Model, Trainer
from undergrad.ops import ReLU, Softmax, CrossEntropy
from undergrad.optim import SGDOptimizer
import numpy as np
import unittest


class TestOptim(unittest.TestCase):
    def test_softmax(self):
        m = Model([2, 1, 2], [ReLU(), Softmax()])

        X = np.array([[0, 1],
                      [-1, 0]])

        W0 = np.array([[2],
                       [1]], dtype=np.float64)
        b0 = np.array([[1]], dtype=np.float64)
        W1 = np.array([[2, 3]], dtype=np.float64)
        b1 = np.array([[1, -1]], dtype=np.float64)

        m.weights = [W0, W1]
        m.bias = [b0, b1]

        t = Trainer(m, None, CrossEntropy())
        t.batch_size = X.shape[0]

        y = np.array([[0, 1],
                      [1, 0]], dtype=np.float64)
        prediction = m.forward(X)
        grads = t.backward(y)
        opt = SGDOptimizer(m, lr=1)
        opt.step(grads)

        expected_W0 = np.array([[2.],
                                [1.25]])
        expected_b0 = np.array([[1.25]])

        expected_W1 = np.array([[1.5, 3.5]])
        expected_b1 = np.array([[0.80960146, -0.80960146]])

        W0, b0 = m.weights[0], m.bias[0]
        self.assertTrue((abs(expected_W0 - W0) < 1e-8).all(
        ), f"The expected result to W0 after SGD optim step is {expected_W0}, but gives {W0}")
        self.assertTrue((abs(expected_b0 - b0) < 1e-8).all(
        ), f"The expected result to b0 after SGD optim step is  {expected_b0}, but gives {b0}")

        W1, b1 = m.weights[1], m.bias[1]
        self.assertTrue((abs(expected_W1 - W1) < 1e-8).all(
        ), f"The expected result to W1 after SGD optim step is  {expected_W1}, but gives {W1}")
        self.assertTrue((abs(expected_b1 - b1) < 1e-8).all(
        ), f"The expected result to B1 after SGD optim step is {expected_b1}, but gives {b1}")


if __name__ == "__main__":
    unittest.main()
