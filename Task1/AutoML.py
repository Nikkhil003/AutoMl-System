# Step 1: Import required libraries
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

# Step 2: Load the Iris dataset
url = '/Users/gaurav/MYProjects/Extion DS Tasks/AutoMl System/Task1/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, header=None, names=column_names)

# Split features and labels
X = data.drop('species', axis=1)
y = data['species']


# Step 3: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize TPOTClassifier with a fixed number of generations and population size
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

# Step 5: Fit the TPOT model to the training data
tpot.fit(X_train, y_train)

# Step 6: Evaluate the best model on the test set
y_pred = tpot.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the best model: {accuracy * 100:.2f}%")

# Step 7: Export the best model and pipeline found by TPOT
tpot.export('best_model_pipeline.py')

# Step 8: Report the best model and hyperparameters
print("Best pipeline found by TPOT:")
print(tpot.fitted_pipeline_)
