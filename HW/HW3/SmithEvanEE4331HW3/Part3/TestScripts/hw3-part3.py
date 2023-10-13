#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-tree.py
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
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# Data Preprocessing
###########################################################################################################################
##########################################################################################################################
# Data Preprocessing
###########################################################################################################################
# Set the path to the main directory containing subdirectories with CSV files
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
print(X)
print(X.shape)
print(y_encoded)
print(y_encoded.shape)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

##########################################################################################################################
# PIPELINES
###########################################################################################################################
pipe1 = Pipeline([
 # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', DecisionTreeClassifier())
])

param_grid = {
    'classifier__criterion': ['gini', 'entropy'],  # Split criterion
    'classifier__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for split
}

##########################################################################################################################
# GridSearchCV
'''
gs = GridSearchCV(estimator=pipe_svc,
 ... param_grid=param_grid,
 ... scoring='accuracy',
 ... cv=5,
 ... n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
0.984615384615
print(gs.best_params_)
'''
##########################################################################################################################

gs1 = GridSearchCV(estimator=pipe1, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs1.fit(X_train, y_train)
print("\n\n\n")
print("\ngs1.best_score_:")
print(gs1.best_score_)
print("\ngs1.best_params_:")
print(gs1.best_params_)
print("\ngs1.best_estimator_:")
print(gs1.best_estimator_)
print("\n\n\n")

##########################################################################################################################
# find best model
###########################################################################################################################

best_score1 = gs1.best_score_
best_model = gs1.best_estimator_
best_params = gs1.best_params_

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
print("\n\n\n\n")
print(X.shape)
print(y_encoded.shape)
##########################################################################################################################
# txt file
###########################################################################################################################

# print to a txt file
with open('3top3pipes300f.txt', 'w') as file:
    file.write(f"\nBest Model Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Model Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision: {precision:.2f}\n")
    file.write(f"\nRecall: {recall:.2f}\n")
    file.write(f"\nF1: {f1:.2f}\n")
    file.write("\nbest_model:\n")
    file.write(str(best_model))
    file.write("\nbest_params:\n")
    file.write(str(best_params))

