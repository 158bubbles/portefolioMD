# -*- coding: utf-8 -*-
"""f_classif.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gFPbLbgvceg8KNNnYjiF4rkIepkVR7wf
"""

from dataset import Dataset

import numpy as np
from scipy.stats import f_oneway


def f_classif(dataset: Dataset):

    #group samples/exemples by classes 
    classes = np.unique(dataset.y)
    X_classes = [dataset.X[dataset.y == c] for c in classes]

    #using ANOVA for each feature
    f_values = []
    p_values = []
    for i in range(dataset.X.shape[1]):
        F, P = f_oneway(*[X[:, i] for X in X_classes])
        f_values.append(F)
        p_values.append(P)
    
    return f_values, p_values

dataset = Dataset(X=np.array([[0, 1, 7, 3],
                              [1, 4, 0, 7],
                              [8, 4, 8, 1]]),
                  y=np.array([0, 1, 0]),
                  features=["f1", "f2", "f3", "f4"],
                  label="y")

f_values, p_values = f_classif(dataset)

print("P_values for each feature:", p_values)
print("F_values for each feature:", f_values)