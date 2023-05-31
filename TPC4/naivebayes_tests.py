import unittest
import pandas as pd
import numpy as np
from naivebayes import NaiveBayesClassifier

# Import the NaiveBayesClassifier class here

# Define the unit tests
class NaiveBayesClassifierTests(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
            'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
            'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
            'windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'true'],
            'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
        }
        self.df = pd.DataFrame(data)

    def test_fit(self):
        # Create an instance of NaiveBayesClassifier
        model = NaiveBayesClassifier()

        # Prepare the data for training
        X = self.df.drop('play', axis=1)
        y = self.df['play']

        # Fit the model
        model.fit(X, y)

        # Assert that the class priors, feature probabilities, and classes are set
        self.assertIsNotNone(model.class_priors_)
        self.assertIsNotNone(model.feature_probs_)
        self.assertIsNotNone(model.classes_)

    def test_predict(self):
        # Create an instance of NaiveBayesClassifier
        model = NaiveBayesClassifier()

        # Prepare the data for training
        X = self.df.drop('play', axis=1)
        y = self.df['play']

        # Fit the model
        model.fit(X, y)

        # Prepare the data for prediction
        test_data = pd.DataFrame({
            'outlook': ['sunny', 'overcast', 'rainy'],
            'temperature': ['hot', 'mild', 'cool'],
            'humidity': ['high', 'normal', 'high'],
            'windy': ['false', 'true', 'false']
        })

        # Perform prediction
        y_pred = model.predict(test_data)

        # Assert that the predicted labels match the expected values
        expected_labels = ['no', 'yes', 'yes']
        self.assertEqual(y_pred, expected_labels)

# Run the unit tests
if __name__ == '__main__':
    unittest.main()
