import sys
sys.path.append('.\\TPC1')
from dataset import Dataset
import numpy as np
from typing import Callable
from f_classif import f_classif
from sklearn.feature_selection import f_regression

class SelectKBest:
#select features according to the k highest scores

    def __init__(self, dataset: Dataset, score_func: Callable, k: int):
        #score_func: function that return an array of scores for each feature (scores, p_values)
        #k: number of top features to select

        if k <= 0:
            raise ValueError("Sorry, k must be > 0")

        self.score_func = score_func
        self.k = k
        self.dataset = dataset

        self.F = None
        self.p = None


    def fit(self):
        #calculate the scores and p values

        self.F, p_values = self.score_func(self.dataset.X, self.dataset.y)


    def transform(self):
        #transforms the dataset X by selecting the top k features based on their scores
        t = np.argsort(self.p)[:self.k]
        return [self.dataset.features[i] for i in t]

    def fit_transform(self):
        #calculates the scores and transforms the dataset by selecting the k highest scoring features
        self.fit()

        return self.transform()


dataset = Dataset(X=np.array([[7, 2, 7],
                              [1, 0, 6],
                              [8, 0, 9]]),
                  y=np.array([1, 8, 3]),
                  features=["f1", "f2", "f3"],
                  label="y")

function = SelectKBest(dataset, score_func = f_regression, k = 1)
res = function.fit_transform()

print(res)