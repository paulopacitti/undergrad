from undergrad import Model, Trainer
from undergrad.ops import ReLU, Softmax, CrossEntropy
import numpy as np
import unittest


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        m = Model([2, 1, 2], [ReLU(), Softmax()])

        X = np.array([[0, 1],
                      [-1, 0]])

        W0 = np.array([[2],
                       [1]])
        b0 = np.array([[1]])
        W1 = np.array([[2, 3]])
        b1 = np.array([[1, -1]])

        m.weights = [W0, W1]
        m.bias = [b0, b1]

        t = Trainer(m, None, CrossEntropy())
        t.batch_size = X.shape[0]

        y = np.array([[0, 1],
                      [1, 0]])
        prediction = m.forward(X)
        grads = t.backward(y)

        # Deixamos esse valor caso você precise verificar seus resultados
        #
        # expected_dZ1 = np.array([[ 0.5       , -0.5       ],
        #                         [-0.11920292,  0.11920292]])
        #
        # expected_dZ0 = np.array([[-0.5],
        #                          [ 0. ]])
        #
        # y_pred = np.array([[0.5       , 0.5       ],
        #                    [0.88079708, 0.11920292]])

        expected_dW1 = np.array([[0.5, -0.5]])

        expected_db1 = np.array([[0.19039854, -0.19039854]])

        expected_dW0 = np.array([[0.],
                                [-0.25]])

        expected_db0 = np.array([[-0.25]])

        dW1, db1 = grads[1]
        self.assertTrue((abs(expected_dW1 - dW1) <
                        1e-8).all(), f"The expected value to dW1 is {expected_dW1}, but {dW1} was given")
        self.assertTrue((abs(expected_db1 - db1) <
                        1e-8).all(), f"The expected value to  db1 is {expected_db1}, but {db1} was given")

        dW0, db0 = grads[0]
        self.assertTrue((abs(expected_dW0 - dW0) <
                        1e-8).all(), f"The expected value to  dW0 is {expected_dW0}, but {dW0} was given")
        self.assertTrue((abs(expected_db0 - db0) <
                        1e-8).all(), f"The expected value to  db0 is {expected_db0}, but {db0} was given")


if __name__ == "__main__":
    unittest.main()
