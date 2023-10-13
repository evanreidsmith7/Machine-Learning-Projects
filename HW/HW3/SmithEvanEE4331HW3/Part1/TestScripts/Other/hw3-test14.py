#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-logreg.py
# Due: , 2023
#
# Objective:
#*************************************************************************************
# Import libraries
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import seaborn as sns
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# Data Preprocessing
###########################################################################################################################

main_directory = r'Datasets/Measurements_Upload/'



# Define the list of paths
paths = [
    r'Datasets/Measurements_Upload/Corridor_rm155_7.1',
    r'Datasets/Measurements_Upload/Lab139_7.1',
    r'Datasets/Measurements_Upload/Main_Lobby_7.1',
    r'Datasets/Measurements_Upload/Sport_Hall_7.1'
]

# Number of subdirectories to traverse (you can adjust this as needed)
num_subdirectories_to_traverse = 300  # Set to None to traverse all subdirectories

# Function to process a directory and return combined data
def process_directory(directory_path, label):
    combined_data = pd.DataFrame()
    
    for root, dirs, files in os.walk(directory_path):
        for subdir in dirs[:num_subdirectories_to_traverse]:
            subdir_path = os.path.join(root, subdir)
            
            for filename in os.listdir(subdir_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(subdir_path, filename)
                    data = pd.read_csv(file_path)
                    data = data['# Version 1.00'].str.split(';', expand=True)
                    data = data.drop([0, 1])
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    combined_data['label'] = label
    return combined_data

# Create a list of dataframes for each path
data_frames = []

for path in paths:
    label = os.path.basename(path)
    data_frame = process_directory(path, label)
    data_frames.append(data_frame)

# Concatenate all dataframes into one
combined_data = pd.concat(data_frames, ignore_index=True)


#combined_data.to_csv("Datasets/combined_data_all.csv", index=False)

# drop the 5th collumn
combined_data = combined_data.drop(columns=[5])

# Assuming you have loaded your data into the combined_data DataFrame
# If not, you should load your data here

# Handling missing values by dropping rows with missing values
combined_data.dropna(inplace=True)

# Splitting into X (features) and y (labels)
X = combined_data.drop(columns=['label'])  # Assuming 'label' is the column containing your labels
# Convert empty strings to NaN
X[X == ''] = np.nan

# Convert all values to float
X = X.astype(float)

#y = combined_data.drop(columns=[0,1,2,3,4])
y = combined_data['label']

# encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
##########################################################################################################################
# PIPELINES
###########################################################################################################################
# Create a pipeline
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid1 = {
    'reduce_dim__n_components': [1, 2],  # Number of components for PCA
    'classifier__penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid2 = {
    'reduce_dim__n_components': [1, 2],  # Number of components for LDA
    'classifier__penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}

pipe3 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', KernelPCA(kernel='rbf')),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid3 = {
    'reduce_dim__n_components': [1, 2],  # Number of components for LDA
    'classifier__penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}

pipe4 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid4 = {
    'reduce_dim__n_components': [1, 2],  # Number of components for PCA
    'classifier__penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}

pipe5 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid5 = {
    'reduce_dim__n_components': [1, 2],  # Number of components for LDA
    'classifier__penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}

pipe6 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', KernelPCA(kernel='rbf')),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid6 = {
    'reduce_dim__n_components': [1, 2],  # Number of components for LDA
    'classifier__penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag','lbfgs', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}
##########################################################################################################################
# Grid Search
##########################################################################################################################
gs = GridSearchCV(estimator=pipe4, param_grid=param_grid4, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print("\n\n\n")
print("\ngs1.best_score_:")
print(gs.best_score_)
print("\ngs1.best_params_:")
print(gs.best_params_)
print("\ngs1.best_estimator_:")
print(gs.best_estimator_)
print("\n\n\n")
best_model = gs.best_estimator_
best_params = gs.best_params_
'''
gs1 = GridSearchCV(estimator=pipe1, param_grid=param_grid1, scoring='accuracy', cv=5, n_jobs=-1)
gs1.fit(X_train, y_train)
print("\n\n\n")
print("\ngs1.best_score_:")
print(gs1.best_score_)
print("\ngs1.best_params_:")
print(gs1.best_params_)
print("\ngs1.best_estimator_:")
print(gs1.best_estimator_)
print("\n\n\n")

gs2 = GridSearchCV(estimator=pipe2, param_grid=param_grid2, scoring='accuracy', cv=5, n_jobs=-1)
gs2.fit(X_train, y_train)
print("\n\n\n")
print("\ngs2.best_score_:")
print(gs2.best_score_)
print("\ngs2.best_params_:")
print(gs2.best_params_)
print("\ngs2.best_estimator_:")
print(gs2.best_estimator_)
print("\n\n\n")

gs3 = GridSearchCV(estimator=pipe3, param_grid=param_grid3, scoring='accuracy', cv=5, n_jobs=-1)
gs3.fit(X_train, y_train)
print("\n\n\n")
print("\ngs3.best_score_:")
print(gs3.best_score_)
print("\ngs3.best_params_:")
print(gs3.best_params_)
print("\ngs3.best_estimator_:")
print(gs3.best_estimator_)
print("\n\n\n")

gs4 = GridSearchCV(estimator=pipe4, param_grid=param_grid4, scoring='accuracy', cv=5, n_jobs=-1)
gs4.fit(X_train, y_train)
print("\n\n\n")
print("\ngs4.best_score_:")
print(gs4.best_score_)
print("\ngs4.best_params_:")
print(gs4.best_params_)
print("\ngs4.best_estimator_:")
print(gs4.best_estimator_)
print("\n\n\n")

gs5 = GridSearchCV(estimator=pipe5, param_grid=param_grid5, scoring='accuracy', cv=5, n_jobs=-1)
gs5.fit(X_train, y_train)
print("\n\n\n")
print("\ngs5.best_score_:")
print(gs5.best_score_)
print("\ngs5.best_params_:")
print(gs5.best_params_)
print("\ngs5.best_estimator_:")
print(gs5.best_estimator_)
print("\n\n\n")

gs6 = GridSearchCV(estimator=pipe6, param_grid=param_grid6, scoring='accuracy', cv=5, n_jobs=-1)
gs6.fit(X_train, y_train)
print("\n\n\n")
print("\ngs6.best_score_:")
print(gs6.best_score_)
print("\ngs6.best_params_:")
print(gs6.best_params_)
print("\ngs6.best_estimator_:")
print(gs6.best_estimator_)
print("\n\n\n")
##########################################################################################################################
# find best model
###########################################################################################################################

# set best_model to the best_estimator_ from GridSearchCV that has the best score 
# Compare the best scores from each grid search
best_score1 = gs1.best_score_
best_score2 = gs2.best_score_
best_score3 = gs3.best_score_
best_score4 = gs4.best_score_
best_score5 = gs5.best_score_
best_score6 = gs6.best_score_

# Find the index of the grid search with the highest score
best_index = np.argmax([best_score1, best_score2, best_score3, best_score4, best_score5, best_score6])

# Set best_model to the best estimator from the grid search with the highest score
if best_index == 0:
    g = "gs1"
    best_model = gs1.best_estimator_
    best_params = gs1.best_params_
elif best_index == 1:
    g = "gs2"
    best_model = gs2.best_estimator_
    best_params = gs2.best_params_    
elif best_index == 2:
    g = "gs3"
    best_model = gs3.best_estimator_
    best_params = gs3.best_params_
elif best_index == 3:
    g = "gs4"
    best_model = gs4.best_estimator_
    best_params = gs4.best_params_
elif best_index == 4:
    g = "gs5"
    best_model = gs5.best_estimator_
    best_params = gs5.best_params_
else:
    g = "gs6"
    best_model = gs6.best_estimator_
    best_params = gs6.best_params_

'''
# Make predictions on the test data
y_pred_train = best_model.predict(X_train)
y_pred = best_model.predict(X_test)

# Calculate and print the accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n\n\n\nBest Model Train Accuracy: {train_accuracy:.2f}")
print(f"Best Model Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}\n")
print(f"Recall: {recall:.2f}\n")
print(f"F1: {f1:.2f}\n")
print("best_model")
print(best_model)
print("best_params:")
print(best_params)
print("\n\n\n")
print(X.shape)
print(y_encoded.shape)


##########################################################################################################################
# txt file
###########################################################################################################################

# print to a txt file
with open('Part1/results/tuning/resultspipe4300f.txt', 'w') as file:
    file.write(f"Best Model Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Model Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision: {precision:.2f}\n")
    file.write(f"\nRecall: {recall:.2f}\n")
    file.write(f"\nF1: {f1:.2f}\n")
    file.write("\nbest_model:\n")
    file.write(str(best_model))
    file.write("\nbest_params:\n")
    file.write(str(best_params))