import sys
sys.path.append('.\\TPC1')

from dataset import Dataset
import numpy as np
from scipy.stats import f_oneway


def f_classif(dataset: Dataset):
    # Group samples/exemples by classes 
    classes = np.unique(dataset.y)  # Get unique classes in the target variable
    X_classes = [dataset.X[dataset.y == c] for c in classes]  # Store feature matrix for each class

    # Using ANOVA for each feature
    f_values = []  # List to store F-values for each feature
    p_values = []  # List to store p-values for each feature
    for i in range(dataset.X.shape[1]):  # Iterate over each feature
        # Perform ANOVA test for the i-th feature
        F, P = f_oneway(*[X[:, i] for X in X_classes])
        f_values.append(F)  # Append the F-value to the f_values list
        p_values.append(P)  # Append the p-value to the p_values list
    
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