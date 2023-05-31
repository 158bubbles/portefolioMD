import sys
sys.path.append('.\\TPC1')
from dataset import Dataset
import numpy as np
from scipy.stats import f
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

def f_regression(X, y):

    #new column of ones to X
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    #linear regression
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    #the predicted values and residuals
    residuals = y - np.dot(X, beta)

    SST = np.sum((y - np.mean(y)) ** 2)
    SSR = np.sum(residuals ** 2)
    
    n = X.shape[0]
    p = dataset.X.shape[1]  # number of features in the dataset
    

    F_values = (SST - SSR) / p / (SSR / (n-p-1))
    p_values = 1 - f.cdf(F_values, p, (n-p-1))

    return F_values, p_values


iris = load_iris()

dataset = Dataset(X=iris.data,
                  y=iris.target,
                  features=["f1", "f2", "f3", "f4"],
                  label="y")

F_values, p_values = f_regression(iris.data, iris.target)

print("P_values:", p_values)
print("F_values:", F_values)