# Import libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
cancer = datasets.load_breast_cancer()

# Split into features and labels
X = cancer.data
y = cancer.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Logistic Regression classifier with its hyperparameters
clf = LogisticRegression()
param_grid = {'solver': ['liblinear'], 'C': np.logspace(-2, 2, 5)}

# Define a k-fold cross-validation object
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Use GridSearchCV to perform grid search
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit the grid search object on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and the best score
best_clf = grid_search.best_estimator_
best_score = grid_search.best_score_

# Print the best estimator and the best score
print('Best estimator:', best_clf)
print('Best score:', best_score)

# Evaluate the best estimator on the testing data
y_pred = best_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Classification report:', classification_report(y_test, y_pred))

# Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate the specificity and sensitivity
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Print the specificity and sensitivity
print('Specificity:', specificity)
print('Sensitivity:', sensitivity)