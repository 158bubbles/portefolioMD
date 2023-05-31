import sys
sys.path.append('.\\TPC1')
from dataset import Dataset
from f_regression import f_regression
import numpy as np
from typing import Callable
from sklearn.datasets import load_breast_cancer

class SelectKBest:
    # Select features according to the k highest scores

    def __init__(self, dataset: Dataset, score_func: Callable, k: int):
        # Initialize the SelectKBest instance
        # dataset: Dataset object
        # score_func: Function that returns an array of scores for each feature (scores, p_values)
        # k: Number of top features to select

        if k <= 0:
            raise ValueError("Sorry, k must be > 0")

        self.score_func = score_func
        self.k = k
        self.dataset = dataset

        self.F = None
        self.p = None

    def fit(self, dataset):
        # Calculate the scores and p-values for each feature

        scores, p_values = self.score_func(dataset)
        self.F = scores
        self.p = p_values

    def transform(self, dataset):
        # Transform the dataset X by selecting the top k features based on their scores

        X = dataset.X
        t = np.argsort(self.p)[:self.k]  # Indices of the top k features based on p-values
        return X[:, t]  # Return the dataset with only the selected features

    def fit_transform(self, dataset):
        # Calculate the scores and transform the dataset by selecting the k highest scoring features

        self.fit(dataset)
        return self.transform(dataset)



data = load_breast_cancer()

dataset = Dataset(X=data.data,
                  y=data.target,
                  features=data.feature_names,
                  label="y")

function = SelectKBest(dataset, score_func = f_regression, k = 1)
res = function.fit_transform(dataset)

print(res)