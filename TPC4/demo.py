import pandas as pd
from naivebayes import NaiveBayesClassifier

# Load the dataset
df = pd.read_csv('TPC4\play_tennis.csv')
df = df.drop('day', axis=1) # Drop the 'day' column

model = NaiveBayesClassifier()

# separate the target variable from the input features
# Split the data into features and labels
X = df.drop('play', axis=1)
y = df['play']

# Fit the model
model.fit(X, y)

# Test data
x_new = pd.DataFrame({
    'outlook': ['Sunny'],
    'temp': ['Cool'],
    'humidity': ['High'],
    'wind': ['Strong']
    })

# Make predictions
predictions = model.predict(x_new)

print(predictions)