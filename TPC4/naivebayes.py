import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
    # Initialize the class priors to None
        self.class_priors_ = None

        # Initialize the feature probabilities to None
        self.feature_probs_ = None

        # Initialize the classes to None
        self.classes_ = None

        
    def fit(self, X, y):
        # Calculate the prior probability of each class
        class_priors = {}
        total_instances = len(X)

        # Iterate over unique classes in y
        for c in y.unique():
            # Count instances in the current class
            instances_in_class = len(X[y == c])
            # Calculate the log prior probability of the class
            class_priors[c] = np.log((instances_in_class) / (total_instances))

        # Calculate the conditional probability of each feature given each class
        feature_probs = {}

        # Iterate over unique classes in y
        for c in y.unique():
            # Get instances of the current class
            class_instances = X[y == c]
            total_instances_in_class = len(class_instances)
            feature_probs[c] = {}

            # Iterate over columns (features) in X
            for feature in X.columns:
                # Calculate the log conditional probability of the feature given the class
                feature_probs[c][feature] = np.log(class_instances[feature].value_counts(normalize=True))

        # Assign the calculated probabilities and classes to the instance variables
        self.class_priors_ = class_priors
        self.feature_probs_ = feature_probs
        self.classes_ = y.unique()



        
    def predict(self, X):
        # Initialize an empty list to store the predicted labels
        y_pred = []

        # Iterate over instances in X
        for _, instance in X.iterrows():
            # Initialize a dictionary to store the scores for each class
            scores = {c: self.class_priors_[c] for c in self.classes_}

            # Iterate over features and values in the current instance
            for feature, value in instance.items():
                # Iterate over classes
                for c in self.classes_:
                    # Check if the value exists in the feature probabilities for the class
                    if value in self.feature_probs_[c][feature]:
                        # Add the conditional probability to the score of the class
                        scores[c] += self.feature_probs_[c][feature][value]

            # Append the predicted label (class with the maximum score) to the y_pred list
            y_pred.append(max(scores, key=scores.get))

            # Print the instance and its predicted probabilities
            print('Instance:', instance.tolist(), '\nProbabilities:', {k: np.exp(v) for k, v in scores.items()})

        # Return the list of predicted labels
        return y_pred


# Test the model on a new instance
x_new = pd.DataFrame({
    'outlook': ['Sunny'],
    'temp': ['Cool'],
    'humidity': ['High'],
    'wind': ['Strong']
    })

# my_model.predict(x_new)

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Load the dataset
df = pd.read_csv('TPC4\play_tennis.csv')
df = df.drop('day', axis=1) # Drop the 'day' column


# separate the target variable from the input features
# Split the data into features and labels
X = df.drop('play', axis=1)
y = df['play']

# encode the categorical variables as integers using label encoding
encoder = LabelEncoder()
X = X.apply(encoder.fit_transform)
# create the CategoricalNB classifier object
clf = CategoricalNB()

# fit the classifier to the data
clf.fit(X.values, y)

# define a new data point to be predicted {'outlook': ['Sunny'],'temp': ['Cool'],'humidity': ['High'],'wind': ['Strong']}
new_data = np.array([[2, 0, 0, 0]])

# make a prediction for the new data point
prediction = clf.predict(new_data)

# print the predicted value
# print(prediction[0])