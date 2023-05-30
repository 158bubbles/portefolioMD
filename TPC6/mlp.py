# -*- coding: utf-8 -*-
"""MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GgUGzSUk9jcrpsfB7LTID_qTWLtDoZOY

O algoritmo MLP é usado para treinar redes neurais artificiais com múltiplas camadas ocultas
"""

import numpy as np
from scipy import optimize

class Dataset:
    
    # constructor
    def __init__(self, filename = None, X = None, Y = None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None
        
    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:,0:-1]
        self.Y = data[:,-1]
        
    def writeDataset(self, filename, sep = ","):
        fullds = np.hstack( (self.X, self.Y.reshape(len(self.Y),1)))
        np.savetxt(filename, fullds, delimiter = sep)
        
    def getXy (self):
        return self.X, self.Y
    
    def train_test_split(self, p = 0.7):
        from random import shuffle
        ninst = self.X.shape[0]
        inst_indexes = np.array(range(ninst))
        ntr = (int)(p*ninst)
        shuffle(inst_indexes)
        tr_indexes = inst_indexes[1:ntr]
        tst_indexes = inst_indexes[ntr+1:]
        Xtr = self.X[tr_indexes,]
        ytr = self.Y[tr_indexes]
        Xts = self.X[tst_indexes,]
        yts = self.Y[tst_indexes]
        return (Xtr, ytr, Xts, yts) 
    
    def process_binary_y(self):
        y_values = np.unique(self.Y)
        if len(y_values) == 2:
            self.Y = np.where(self.Y == y_values[0], 0, 1)
        else:
            print("Non binary")

class MLP:
    
    #Inicialização da classe MLP
    def __init__(self, dataset, ocultos = 2, normalizar = False):

        self.X, self.label = dataset.getXy()   #inicializar com os dados de entrada e saída do dataset
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))   #adicionamos uma coluna adicional de 1's
        
        self.n = ocultos
        #peso1 e peso2 são matrizes de zeros, que representam os pesos das camadas
        self.peso1 = np.zeros([ocultos, self.X.shape[1]])
        self.peso2 = np.zeros([1, ocultos + 1])
        
        #normaliza os dados
        if normalizar:
            self.normalizar_dados()
        else:
            self.normalizado = False

    #definição os pesos das camadas
    def pesos(self, peso1, peso2):
        self.peso1 = peso1
        self.peso2 = peso2   

    #faz a previsão da saída para uma instância de entrada fornecida
    def previsao(self, instancia):

        #cria um vetor x, onde o primeiro elemento é 1 e o restantes são da instância
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instancia[:self.X.shape[1] - 1])
        
        #se os dados estiverem normalizados, normaliza a instância
        if self.normalizado:
            if np.all(self.desvio_padrao != 0): 
                x[1:] = (x[1:] - self.media) / self.desvio_padrao
            else: x[1:] = (x[1:] - self.media)
        
        #cálculos de previsão
        z2 = np.dot(self.peso1, x)
        a2 = np.empty([z2.shape[0] + 1])
        a2[0] = 1
        a2[1:] = self.sigmoid(z2)
        z3 = np.dot(self.peso2, a2)
                        
        return self.sigmoid(z3)

    #cálculo da função de custo
    def custo(self, pesos = None):

        #os pesos são atualizados com os valores fornecidos
        if pesos is not None:
            self.peso1 = pesos[:self.n * self.X.shape[1]].reshape([self.n, self.X.shape[1]])
            self.peso2 = pesos[self.n * self.X.shape[1]:].reshape([1, self.n + 1])
        
        #cálculos de previsão usando os pesos atualizados e retorna o valor da função de custo
        m = self.X.shape[0]
        z2 = np.dot(self.X, self.peso1.T)
        a2 = np.hstack((np.ones([z2.shape[0], 1]), self.sigmoid(z2)))
        z3 = np.dot(a2, self.peso2.T)
        previsoes = self.sigmoid(z3)
        erro_quadratico = (previsoes - self.label.reshape(m, 1)) ** 2
        res = np.sum(erro_quadratico) / (2 * m)

        return res

    #construção do modelo MLP
    def construcao_modelo(self):
        tamanho = self.n * self.X.shape[1] + self.n + 1
        pesos_iniciais = np.random.rand(tamanho)      

        #usa o algoritmo de otimização BFGS para encontrar os valores ótimos dos pesos que minimizam o custo  
        res = optimize.minimize(lambda p: self.custo(p), pesos_iniciais, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        pesos = res.x

        #os valores dos pesos obtidos são atribuídos a peso1 e peso2
        self.peso1 = pesos[:self.n * self.X.shape[1]].reshape([self.n, self.X.shape[1]])
        self.peso2 = pesos[self.n * self.X.shape[1]:].reshape([1, self.n + 1])

    #normalização dos dados de entrada usando a média e o desvio padrão
    def normalizar_dados(self):
          self.media = np.mean(self.X[:, 1:], axis=0)
          self.X[:, 1:] = self.X[:, 1:] - self.media
          self.desvio_padrao = np.std(self.X[:, 1:], axis=0)
          self.X[:, 1:] = self.X[:, 1:] / self.desvio_padrao
          self.normalizado = True
      
    #função sigmoid
    def sigmoid(self, x):
        return (1 / (np.exp(-x)+1))

dataset = Dataset(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([1, 0, 0, 1]))
mlp = MLP(dataset, 2)
mlp.pesos(np.array([[-30, 20, 20], [10, -20, -20]]), 
              np.array([[-10, 20, 20]]))

print(mlp.previsao(np.array([0, 0])))
print(mlp.previsao(np.array([0, 1])))
print(mlp.previsao(np.array([1, 0])))
print(mlp.previsao(np.array([1, 1])))
print(mlp.custo())