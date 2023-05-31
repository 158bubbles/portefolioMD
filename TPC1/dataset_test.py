import unittest
import numpy as np
import pandas as pd
from typing import Sequence
from dataset import Dataset as mypd

class DatasetTest(unittest.TestCase):
    
    def test_dataset_creation(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        features = ['feature1', 'feature2']
        types = ['int', 'int']
        label = 'target'
        
        dataset = mypd(X=X, y=y, features=features, types=types, label=label)
        
        self.assertTrue(np.array_equal(dataset.X, X))
        self.assertTrue(np.array_equal(dataset.y, y))
        self.assertEqual(dataset.features, features)
        self.assertEqual(dataset.types, types)
        self.assertEqual(dataset.label, label)
    

    def test_dataset_creation_with_defaults(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        dataset = mypd(X=X, y=y)
        
        self.assertTrue(np.array_equal(dataset.X, X))
        self.assertTrue(np.array_equal(dataset.y, y))
        self.assertEqual(dataset.features, None)
        self.assertEqual(dataset.types, None)
        self.assertEqual(dataset.label, "y")

if __name__ == '__main__':
    unittest.main()
