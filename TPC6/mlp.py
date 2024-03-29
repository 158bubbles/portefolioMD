"""
O algoritmo MLP é usado para treinar redes neurais artificiais com múltiplas camadas ocultas
"""

import numpy as np
from scipy import optimize

class MLP:
    
    # Initialization of the MLP class
    def __init__(self, X, label, ocultos=2, normalizar=False):

        self.X = X
        self.label = label  # Initialize with input and output data
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))  # Add an additional column of 1's
        
        self.n = ocultos
        self.peso1 = np.zeros([ocultos, self.X.shape[1]])  # peso1 represents the weights of the hidden layer
        self.peso2 = np.zeros([1, ocultos + 1])  # peso2 represents the weights of the output layer
        
        # Normalize the data
        if normalizar:
            self.normalizar_dados()
        else:
            self.normalizado = False

    # Set the weights of the layers
    def pesos(self, peso1, peso2):
        self.peso1 = peso1
        self.peso2 = peso2   

    # Make a prediction for a given input instance
    def previsao(self, instancia):

        x = np.empty([self.X.shape[1]])  # Create a vector x, where the first element is 1 and the rest are from the instance
        x[0] = 1
        x[1:] = np.array(instancia[:self.X.shape[1] - 1])
        
        # If the data is normalized, normalize the instance
        if self.normalizado:
            if np.all(self.desvio_padrao != 0):
                x[1:] = (x[1:] - self.media) / self.desvio_padrao
            else:
                x[1:] = (x[1:] - self.media)
        
        # Perform prediction calculations
        z2 = np.dot(self.peso1, x)
        a2 = np.empty([z2.shape[0] + 1])
        a2[0] = 1
        a2[1:] = self.sigmoid(z2)
        z3 = np.dot(self.peso2, a2)
                        
        return self.sigmoid(z3)

    # Calculate the cost function
    def custo(self, pesos=None):

        if pesos is not None:
            # Update the weights with the provided values
            self.peso1 = pesos[:self.n * self.X.shape[1]].reshape([self.n, self.X.shape[1]])
            self.peso2 = pesos[self.n * self.X.shape[1]:].reshape([1, self.n + 1])
        
        # Perform predictions using the updated weights and return the cost function value
        m = self.X.shape[0]
        z2 = np.dot(self.X, self.peso1.T)
        a2 = np.hstack((np.ones([z2.shape[0], 1]), self.sigmoid(z2)))
        z3 = np.dot(a2, self.peso2.T)
        previsoes = self.sigmoid(z3)
        erro_quadratico = (previsoes - self.label.reshape(m, 1)) ** 2
        res = np.sum(erro_quadratico) / (2 * m)

        return res

    # Build the MLP model
    def construcao_modelo(self):
        tamanho = self.n * self.X.shape[1] + self.n + 1
        pesos_iniciais = np.random.rand(tamanho)  # Initialize random initial weights

        # Use the BFGS optimization algorithm to find the optimal weight values that minimize the cost
        res = optimize.minimize(lambda p: self.custo(p), pesos_iniciais, method='BFGS', 
                                options={"maxiter": 1000, "disp": False})
        pesos = res.x

        # Assign the obtained weight values to peso1 and peso2
        self.peso1 = pesos[:self.n * self.X.shape[1]].reshape([self.n, self.X.shape[1]])
        self.peso2 = pesos[self.n * self.X.shape[1]:].reshape([1, self.n + 1])

    # Normalize the input data using mean and standard deviation
    def normalizar_dados(self):
        self.media = np.mean(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] - self.media
        self.desvio_padrao = np.std(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] / self.desvio_padrao
        self.normalizado = True
      
    # Sigmoid function
    def sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Y=np.array([1, 0, 0, 1])

# mlp = MLP(X, Y, 2)
# mlp.pesos(np.array([[-30, 20, 20], [10, -20, -20]]), 
#               np.array([[-10, 20, 20]]))

# print(mlp.previsao(np.array([0, 0])))
# print(mlp.previsao(np.array([0, 1])))
# print(mlp.previsao(np.array([1, 0])))
# print(mlp.previsao(np.array([1, 1])))
# print(mlp.custo())