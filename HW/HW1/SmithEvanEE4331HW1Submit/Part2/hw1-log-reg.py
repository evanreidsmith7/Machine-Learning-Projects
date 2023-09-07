#************************************************************************************
# Evan Smith
# ML – HW#1
# Filename: hw1-log-reg.py
# Due: Sept. 6, 2023
#
# Objective:
# Use a scikit-learn logistic regression model to classify all labels.  
# Use the “SCG” and “STR” features to train, predict, and plot the classification. 
# In your program, print out the training and test accuracy values to a text file 
# Plot the classification image, save it as an image for submission. 
# Generate and log your excel sheet to record the parameter changes vs. test accuracy (min 
# 10 test performed to find best accuracy) 
# to Find the highest test accuracy by tuning the model parameter
#*************************************************************************************
# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# create a variable to store the location of the dataset and output files
# working directroy is SmithEvanEE4331HW1
DATASET_FILE_LOCATION =     r"Data_User_Modeling_Dataset.xls"
RESULTS_LOG_LOCATION =      r"ResultFiles/Part2Log.xlsx"
PLOT_PNG_LOCATION =         r"ResultFiles/Perceptron.png"
RESULTS_ACCURACY_LOCATION = r"ResultFiles/results.txt"

# Load the traing data from the xls dataset file to a panda dataframe
training_pd = pd.read_excel(DATASET_FILE_LOCATION, sheet_name='Training_Data')
test_pd = pd.read_excel(DATASET_FILE_LOCATION, sheet_name='Test_Data')


# drop the data we do not need
training_pd.drop(columns=['STG', 'LPR', 'PEG', 'Unnamed: 6', 'Unnamed: 7', 'Attribute Information:'])
test_pd.drop(columns=['STG', 'LPR', 'PEG', 'Unnamed: 6', 'Unnamed: 7', 'Attribute Information:'])

# extract the features we need
X_test_df = test_pd.filter(['SCG', 'STR']).copy()
X_train_df = training_pd.filter(['SCG', 'STR']).copy()

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

# define a list of parameter combinations to try
parameter_combos = [
    {'penalty': 'l1', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 100},
    {'penalty': 'l2', 'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000},
    {'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear', 'max_iter': 500},
    {'penalty': 'l2', 'C': 10.0, 'solver': 'liblinear', 'max_iter': 1000},
    {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 1000},
    {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000},
    {'penalty': 'l2', 'C': 0.01, 'solver': 'lbfgs', 'max_iter': 500},
    {'penalty': 'l2', 'C': 0.1, 'solver': 'saga', 'max_iter': 1000},
    {'penalty': 'l1', 'C': 0.01, 'solver': 'saga', 'max_iter': 100},
    {'penalty': 'l1', 'C': 0.1, 'solver': 'saga', 'max_iter': 1000},
]
# create an empoty list to store results
results = []
# current iteration number
iteration = 1
# loop through the parameter combinations
for paremter_combo in parameter_combos:
    # create a perceptron model with the current parameter combination
    lr = LogisticRegression(
        penalty=paremter_combo['penalty'],
        C=paremter_combo['C'], 
        solver=paremter_combo['solver'], 
        max_iter=paremter_combo['max_iter'], 
        random_state=42)
    # train the model
    lr.fit(X_train_std, y_train)
    # predict the labels of the training data
    y_pred_train = lr.predict(X_train_std)
    # predict the labels of the test data
    y_pred_test = lr.predict(X_test_std)
    # calculate the training accuracy of the model
    train_acc = accuracy_score(y_train, y_pred_train)
    # calculate the test accuracy of the model
    test_acc = accuracy_score(y_test, y_pred_test)
    # Log the results in a dictionary
    result = {
        'Test Number': iteration,
        'Parameters': paremter_combo,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc
    }
    # increment the iteration number
    iteration += 1
    # append the result to the results list
    results.append(result)

# create a dataframe from the results list
results_df = pd.DataFrame(results)

# sort the dataframe by the test accuracy
results_df = results_df.sort_values(by=['Test Accuracy'], ascending=False)

# keep the top 10 results of the dataframe
results_df = results_df.head(10)

# print the top 10 results
print(results_df)

# define the parameter grid to search
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'solver': ['liblinear', 'saga'],  # Solver algorithms
    'max_iter': [100, 500, 1000, 2000]  # Maximum number of iterations
}

# lets use the grid search to find the best parameters
lr1 = LogisticRegression(random_state=42)

# create a grid search object
grid_search_object = GridSearchCV(lr1, param_grid, cv=5, scoring='accuracy')

# fit the grid search object to the training data
grid_search_object.fit(X_train_std, y_train)

# get the best esiimator and its test accuracy
best_estimator = grid_search_object.best_estimator_

# get the best parameters
best_parameters = grid_search_object.best_params_

# lets give lr1 the best parameters
lr1 = LogisticRegression(
    penalty=best_parameters['penalty'],
    C=best_parameters['C'], 
    solver=best_parameters['solver'], 
    max_iter=best_parameters['max_iter'], 
    random_state=42)

# print the best parameters
print(best_parameters)

# train the model
lr1.fit(X_train_std, y_train)

# predict the labels of the training data
y_pred_train = lr1.predict(X_train_std)

# predict the labels of the test data
y_pred_test = lr1.predict(X_test_std)


# ****************************************plot below****************************************************************************************************************************************************************************************

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^')  # Add a marker for the fourth class
    colors = ('red', 'blue', 'lightgreen', 'purple')  # Add a color for the fourth class
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all the samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)

    # Highlight test samples if test_idx is provided
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr1)

#plot with labels and legend
plt.xlabel(X_train_df.columns[0])
plt.ylabel(X_train_df.columns[1])
plt.legend()

#Save plot ---> LEAP requires this part
plt.savefig(PLOT_PNG_LOCATION)
# ****************************************plot above****************************************************************************************************************************************************************************************



# ****************************************print to txt below****************************************************************************************************************************************************************************************

#Print out the training and test accuracy values to a text file
print('Misclassified samples: %d' % (y_test != y_pred_test).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_test))
print('Training Accuracy: %.2f' % lr1.score(X_train_std, y_train))
print('Test Accuracy: %.2f' % lr1.score(X_test_std, y_test))

#Name of the text file
with open(RESULTS_ACCURACY_LOCATION, "w") as f:
    f.write('Misclassified samples: %d' % (y_test != y_pred_test).sum())
    f.write('\n')
    f.write('Training Accuracy: %.2f' % lr1.score(X_train_std, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % lr1.score(X_test_std, y_test))
    f.write('\n')
# ************************************************print to txt above*********************************************************************************************************************************************************



# ***********************************************print to excel below********************************************************************************************************************************************************
# Define the header information and metadata
header_info = [
    {"item": "Name", "label": "Evan"},
    {"item": "Last Name", "label": "Smith"},
    {"item": "Homework", "label": "1"},
    {"item": "Due", "label": "9/7/2022"},
    {"item": "technique", "label": "log reg"}
]

# Create a DataFrame for header information
header_df = pd.DataFrame(header_info)

# add the top 10 results to the header dataframe
header_df = pd.concat([header_df, results_df], ignore_index=True)

with pd.ExcelWriter(RESULTS_LOG_LOCATION) as writer:
    header_df.to_excel(writer, index=False)
# print to excel ABOVE ***************************************************************************************************************************************************************************************************************************