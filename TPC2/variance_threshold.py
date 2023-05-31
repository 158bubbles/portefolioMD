import numpy as np


class VarianceThreshold:
    # Class that can be used to remove features with low variance

    def __init__(self, threshold: float = 0.0):
        # Initialize the VarianceThreshold instance
        # threshold: Minimum threshold for variance to keep a feature

        if threshold < 0:
            raise ValueError("Sorry, Threshold must be >= 0")

        self.threshold = threshold
        self.variance = None

    def fit(self, X) -> 'VarianceThreshold':
        # Calculate the variances of each feature in the input data

        self.variance = np.var(X, axis=0)
        return self

    def transform(self, X):
        # Apply the variance thresholding to the input data

        new_features = self.variance > self.threshold  # Boolean array indicating features with variance > threshold
        X_selected = X[:, new_features]  # Select only the features with variance > threshold
        if X_selected is not None:
            print('Features with variance > threshold:')
            print(X_selected)
        else:
            print('Features with variance <= threshold:')
        return X_selected

    def fit_transform(self, X):
        # Apply both fitting and transforming in only one step

        self.fit(X)
        res = self.transform(X)
        return res


def teste():
  X = np.array([[1, 8, 1, 3],
                  [5, 1, 1, 3],
                  [4, 0, 1, 3]])
  y = np.array([0, 1, 0])

  f = VarianceThreshold()
  f.fit(X)

  new = f.transform(X)

teste()