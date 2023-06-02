from mlp import MLP
import numpy as np

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([1, 0, 0, 1])

mlp = MLP(X, Y, 2)
mlp.pesos(np.array([[-30, 20, 20], [10, -20, -20]]), np.array([[-10, 20, 20]]))

print(mlp.previsao(np.array([0, 0])))  # Expected output: close to 1
print(mlp.previsao(np.array([0, 1])))  # Expected output: close to 0
print(mlp.previsao(np.array([1, 0])))  # Expected output: close to 0
print(mlp.previsao(np.array([1, 1])))  # Expected output: close to 1
