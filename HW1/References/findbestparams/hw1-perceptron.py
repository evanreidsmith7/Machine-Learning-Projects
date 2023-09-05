#************************************************************************************
# Evan Smith
# ML – HW#1
# Filename: hw1-perceptron.py
# Due: Sept. 6, 2023
#
# Objective:
# Use a scikit-learn perceptron model to classify all labels
# Use the “STG” and “PEG” features to train, predict, and plot the classification.
# Print out the training and test accuracy values to a text file
# Plot the classification outcome, save it as an image for submission.
#*************************************************************************************
# Import libraries
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# Load the traing data from the xls dataset file to a panda dataframe
training_pd = pd.read_excel("C:/Users/Owner/OneDrive - Texas State University/Desktop/EE4331ML/HW/HW1/Data_User_Modeling_Dataset.xls", sheet_name='Training_Data')

# Load the test data from the xls dataset file to a panda dataframe
test_pd = pd.read_excel("C:/Users/Owner/OneDrive - Texas State University/Desktop/EE4331ML/HW/HW1/Data_User_Modeling_Dataset.xls", sheet_name='Test_Data')

# drop the data we do not need
training_pd.drop(columns=['SCG', 'STR', 'LPR', 'Unnamed: 6', 'Unnamed: 7', 'Attribute Information:'])
test_pd.drop(columns=['SCG', 'STR', 'LPR', 'Unnamed: 6', 'Unnamed: 7', 'Attribute Information:'])

# extract the features we need
X_test_df = test_pd.filter(['STG', 'PEG']).copy()
X_train_df = training_pd.filter(['STG', 'PEG']).copy()

# merge the label strings into one column
training_pd['label'] = training_pd[' UNS']
test_pd['label'] = test_pd[' UNS']

# convert the label strings to numbers
training_pd['label'] = training_pd['label'].map({'very_low': 0, 'Low': 1, 'Middle': 2, 'High': 3})
test_pd['label'] = test_pd['label'].map({'Very Low': 0, 'Low': 1, 'Middle': 2, 'High': 3})

# extract the labels we need into a panda dataframe 
y_train_df = training_pd['label'].copy()
y_test_df = test_pd['label'].copy()

# convert the panda dataframes to numpy arrays
X_test = X_test_df.to_numpy()
y_test = y_test_df.to_numpy()
X_train = X_train_df.to_numpy()
y_train = y_train_df.to_numpy()

# scale the training data
sc = StandardScaler()
sc.fit(X_train) # fit the scaler to the training data
sc.fit(X_test) # fit the scaler to the test data
X_train_std = sc.transform(X_train) # transform the training data
X_test_std = sc.transform(X_test) # transform the test data

# create paremeter grid to search through
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [1000, 2000],
    'tol': [1e-3, 1e-4],
    'eta0': [0.1, 0.01],
    'random_state': [42]  # Fixed random state for reproducibility
}

# create an empty list to store the results
results = []

# perform grid search with cross validation
for i in range(10):
    # create perceptron
    ppn = Perceptron()

    # create grid search object
    grid_search_object = GridSearchCV(ppn, param_grid, cv=5, scoring='accuracy')

    # fit the grid search object to the training data
    grid_search_object.fit(X_train_std, y_train)

    # get the best esiimator and its test accuracy
    best_estimator = grid_search_object.best_estimator_
    y_test_pred = best_estimator.predict(X_test_std)
    test_acc = accuracy_score(y_test, y_test_pred)

    # record the results in a dictionary
    # Log the results in a dictionary
    result = {
        'Test Number': i + 1,
        'Best Parameters': grid_search_object.best_params_,
        'Test Accuracy': test_acc
    }
    
    # Append the result dictionary to the list
    results.append(result)

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results)

# print out the results
print(results_df)

# Find the highest test accuracy and corresponding parameters
best_result = results_df.loc[results_df['Test Accuracy'].idxmax()]
print("Best Test Accuracy:")
print(best_result)
print('\n\n\n\n\n')
# ******************************************************when done finding best parameters************************************
# create perceptron with the best parameters found
ppn = Perceptron(alpha=.01, max_iter=1000, eta0=0.1, random_state=42)

# train the perceptron
ppn.fit(X_train_std, y_train)

# predict the labels / testing the model data
y_train_pred = ppn.predict(X_train_std)
y_test_pred = ppn.predict(X_test_std)

# print the accuracy of the model
train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
print("Training Acccuracy :", train_acc)
print("Testing Acccuracy  :", test_acc)
# ******************************************************when done finding best parameters************************************