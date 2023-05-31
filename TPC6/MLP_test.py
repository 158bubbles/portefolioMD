import unittest
import numpy as np
from scipy import optimize
from mlp import MLP

class TestMLP(unittest.TestCase):
    def setUp(self):
        # Create sample input and label data for testing
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.label = np.array([0, 1, 1])
        self.mlp = MLP(self.X, self.label)

    def test_pesos(self):
        peso1 = np.ones((2, 4))
        peso2 = np.ones((1, 3))
        self.mlp.pesos(peso1, peso2)
        self.assertTrue(np.array_equal(self.mlp.peso1, peso1))
        self.assertTrue(np.array_equal(self.mlp.peso2, peso2))

    def test_construcao_modelo(self):
        self.mlp.construcao_modelo()
        self.assertTrue(np.all(self.mlp.peso1 != np.zeros((2, 4))))
        self.assertTrue(np.all(self.mlp.peso2 != np.zeros((1, 3))))

    def test_normalizar_dados(self):
        self.mlp.normalizar_dados()
        self.assertTrue(self.mlp.normalizado)
        self.assertTrue(np.allclose(np.mean(self.mlp.X[:, 1:], axis=0), np.zeros(3)))
        self.assertTrue(np.allclose(np.std(self.mlp.X[:, 1:], axis=0), np.ones(3)))

    def test_sigmoid(self):
        x = np.array([-1, 0, 1])
        sigmoid_vals = self.mlp.sigmoid(x)
        self.assertTrue(np.allclose(sigmoid_vals, [0.26894142, 0.5, 0.73105858]))

if __name__ == '__main__':
    unittest.main()
