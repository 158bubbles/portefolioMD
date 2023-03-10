# -*- coding: utf-8 -*-
"""TPC1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ogUrE_hc4TXVOPyUYvJPcAeonb1kJt50
"""

from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from pydoc import locate
from scipy import stats as st

class Dataset:
    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, features: Sequence[str] = None, types: Sequence[str] = None ,  label: str = None):
        """
        Dataset represents a machine learning tabular dataset.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        features=features

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.types = types
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape[::-1]

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None


    def get_classes(self):
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)


    def get_column(self, s: str) -> np.array:
        return self.X[self.features.index(s)]


    def get_type(self, s: str) -> str:
        return self.types[self.features.index(s)]


    def null_counter(self, s:str, o:object) -> int:

      if (self.get_type(s) == 'object'):
        res = np.count_nonzero(self.get_column(s)== '')
      else:
        print(self.get_type(s))
        res = np.count_nonzero(pd.isna(self.get_column(s)))
          
      return res

    
    def null_replace(self, s:str, o:object) -> np.array:
      res = cena.get_column(s)
      if (self.get_type(s) == 'object'):
        if(type(o) != str):
          raise ValueError('Tipo de dados invalido')
        #substituidos por 0 (depois mudar)
        res[res == ''] = o
      else:
        #substituidos pela media
        if(type(o) == str):
          raise ValueError('Tipo de dados invalido')
        else:  
          res[pd.isna(res)] = o
      return res  
        
         

    


    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        res = []
        for f in self.features:
          if(self.get_type(f) != 'object'):
            res.append(np.nanmean(self.get_column(f), axis=0, out=locate(self.get_type(f))))
        
        return res

    def get_variance(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        res = []
        for f in self.features:
          if(self.get_type(f) != 'object'):
            res.append(np.nanvar(self.get_column(f), axis=0, out=locate(self.get_type(f))))
        
        return res

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        res = []
        for f in self.features:
          if(self.get_type(f) != 'object'):
            res.append(np.nanmedian(self.get_column(f), axis=0, out=locate(self.get_type(f))))
        
        return res

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        res = []
        for f in self.features:
          if(self.get_type(f) != 'object'):
            res.append(np.nanmin(self.get_column(f), axis=0, out=locate(self.get_type(f))))
        
        return res

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        res = []
        for f in self.features:
          if(self.get_type(f) != 'object'):
            res.append(np.nanmax(self.get_column(f), axis=0, out=locate(self.get_type(f))))
        
        return res

    def get_moda(self, s:str) -> object:
      
      array = self.get_column(s)
      if('object' == self.get_type(s)):
        array = array[array != '']
      else:  
        array = array[np.logical_not(pd.isna(array))]

      return st.mode(array)[0][0]

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """

        # fazer um dataframe com estas merdas para cada cenas com nrs
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        
        res = []
        for f in self.features:
          if(self.get_type(f) != 'object'):
            res.append(f)

        return pd.DataFrame.from_dict(data, orient="index", columns=res)
  
        

    @classmethod
    def from_csv(cls, path: str, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset

        
        """
        
        df = pd.read_csv(path)
        features = []
        types = []
        
            
        if label:
            y = df[label].to_numpy()
            df = df.drop(label, axis=1)
        else:
            y = None

        
        for d in df.columns:
              if (df[d].dtype == object):
                df[d] = df[d].replace(np.nan, '')
              features.append(d)
              types.append(str(df[d].dtype))
        X = df.transpose().to_numpy()      

        return cls(X, y, features=features,types=types, label=label)

    def to_csv(self, name: str = 'dataset.csv'):
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        


        if self.y is None:
            df = pd.DataFrame(self.X.transpose(), columns=self.features)
        else:
            df = pd.DataFrame(self.X.transpose(), columns=self.features)
            df[self.label] = self.y.transpose()

        df.to_csv(name, index = False)

from typing import Tuple, Sequence
from pydoc import locate
from scipy import stats as st

import numpy as np
import pandas as pd


cena = Dataset.from_csv('/content/titanic_dataset.csv')

cena.to_csv()

cena1 = Dataset.from_csv('/content/dataset.csv')

from google.colab import drive
drive.mount('/content/drive')