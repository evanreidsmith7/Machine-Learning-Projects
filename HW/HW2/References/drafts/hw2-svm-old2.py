#************************************************************************************
# Evan Smith
# ML – HW#2
# Filename: hw2-svm.py
# Due: Sept. 20, 2023
#
# Objective:
'''
• Use a scikit-learn SVM model to classify all targets.
• Determine which features and parameters will provide the best outcomes using PCA or LDA
• From the dataset:
    o For each location, choose a single loc_number folder (use this one for all models).
    o Within the folder, combine all the CSV files into a single file for that location.
        § Label the data accordingly to the dataset information.
    o You can either keep it separate or join all samples to a large master CSV file.
• In your program, print out the training and test accuracy values and the best features values via
  the PCA or LDA to a text file from the best model experiment.
• Generate the t-SNE and UMAP images.
• Generate the training and testing Plot Decision image.
'''
#*************************************************************************************
# Import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# * labels *
# corridor_rm155_7.1
# Lab139_7.1
# Main_Lobby7.1
# Sport_Hall_7.1


###########################################################################################################################
# Data Preprocessing
###########################################################################################################################

# Set the path of each of the folders we want to excract from
Corridor_rm155_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Lab139_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Main_Lobby71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Sport_Hall_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
RESULTS_ACCURACY_LOCATION = 'HW\HW2\SmithEvanEE4331HW2\Part1\part1results.txt'
RESULTS_DIR = 'HW\HW2\SmithEvanEE4331HW2\Part1\results'

# Create a dataframe to store the data
combined_data = pd.DataFrame()

# Loop through each file in the corridor folder and read the data into a dataframe
for filename in os.listdir(Corridor_rm155_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Corridor_rm155_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data1 = pd.concat([combined_data, data], ignore_index=True)

# add the labels to the dataframe
combined_data1['label'] = 'Corridor'
combined_data1.to_csv("datasets\combined_data_Corridor_rm155_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Lab139_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data2 = pd.concat([combined_data, data], ignore_index=True)

combined_data2['label'] = 'Lab139'
combined_data2.to_csv("datasets\combined_data_Lab139_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Main_Lobby71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data3 = pd.concat([combined_data, data], ignore_index=True)

combined_data3['label'] = 'Main_Lobby'
combined_data3.to_csv("datasets\combined_data_Main_Lobby71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Sport_Hall_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data4 = pd.concat([combined_data, data], ignore_index=True)

combined_data4['label'] = 'Sport_Hall'
combined_data4.to_csv("datasets\combined_data_Sport_Hall_71_loc0000.csv", index=False)

# List of CSV files to combine
csv_files = [
    'datasets\combined_data_Corridor_rm155_71_loc0000.csv',
    'datasets\combined_data_Lab139_71_loc0000.csv',
    'datasets\combined_data_Main_Lobby71_loc0000.csv',
    'datasets\combined_data_Sport_Hall_71_loc0000.csv'
]

# combine all combined data into a single dataframe
data_frames = [combined_data1, combined_data2, combined_data3, combined_data4]

combined_data = pd.concat(data_frames, ignore_index=True)

combined_data.to_csv("combined_data_all.csv", index=False)

# drop the 5th collumn
combined_data = combined_data.drop(columns=[5])

# Assuming you have loaded your data into the combined_data DataFrame
# If not, you should load your data here

# Handling missing values by dropping rows with missing values
combined_data.dropna(inplace=True)

# Splitting into X (features) and y (labels)
X = combined_data.drop(columns=['label'])  # Assuming 'label' is the column containing your labels

#y = combined_data.drop(columns=[0,1,2,3,4])
y = combined_data['label']

# encode y
le = LabelEncoder()
y = le.fit_transform(y)

###########################################################################################################################
# standardize and normalize
###########################################################################################################################

# standardize X: X_std
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# normalize X: X_norm
mms = MinMaxScaler()
X_norm = mms.fit_transform(X)

# Splitting into training and testing sets for std
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Splitting into training and testing sets for norm
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y, test_size=0.2, random_state=42)

###########################################################################################################################
# lets find the best parameters for the SVM model first using standardized data
###########################################################################################################################

# define the parameters to use in the grid search
linearsvm_param_grid = {
    'C': [0.1, 1, 10, 100, 1000]
}
# define the parameters to use in the grid search
nonlinesvm_param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.0001,0.01]
}
linear_svc = SVC(random_state=42, kernel='linear')
nonlinear_svc = SVC(random_state=42, kernel='rbf')
'''
# create a grid search object for linear
grid_search_linear_std = GridSearchCV(linear_svc, linearsvm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#create a grid search object for nonlinear
grid_search_nonlinear_std = GridSearchCV(nonlinear_svc, nonlinesvm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# fit the grid search objects to the data
grid_search_linear_std.fit(X_train_std, y_train_std)
grid_search_nonlinear_std.fit(X_train_std, y_train_std)

best_linear_model_std = grid_search_linear_std.best_estimator_
best_nonlinear_model_std = grid_search_nonlinear_std.best_estimator_

y_test_pred_linear_std = best_linear_model_std.predict(X_test_std)
y_test_pred_nonlinear_std = best_nonlinear_model_std.predict(X_test_std)

# print the accuracy scores
print('Linear SVM test accuracy std: %.4f' % accuracy_score(y_test_std, y_test_pred_linear_std))
print('Nonlinear SVM test accuracy std: %.4f' % accuracy_score(y_test_std, y_test_pred_nonlinear_std))
'''

###########################################################################################################################
# lets use normilized
###########################################################################################################################

# create a grid search object for linear
grid_search_linear_norm = GridSearchCV(linear_svc, linearsvm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#create a grid search object for nonlinear
grid_search_nonlinear_norm = GridSearchCV(nonlinear_svc, nonlinesvm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# fit the grid search objects to the data
grid_search_linear_norm.fit(X_train_norm, y_train_norm)
grid_search_nonlinear_norm.fit(X_train_norm, y_train_norm)

best_linear_model_norm = grid_search_linear_norm.best_estimator_
best_nonlinear_model_norm = grid_search_nonlinear_norm.best_estimator_

y_test_pred_linear_norm = best_linear_model_norm.predict(X_test_norm)
y_test_pred_nonlinear_norm = best_nonlinear_model_norm.predict(X_test_norm)

# print the accuracy scores
print('Linear SVM test accuracy norm: %.4f' % accuracy_score(y_test_norm, y_test_pred_linear_norm))
print('Nonlinear SVM test accuracy norm: %.4f' % accuracy_score(y_test_norm, y_test_pred_nonlinear_norm))

# found the best model
best_linear_model = best_linear_model_norm
best_nonlinear_model = best_nonlinear_model_norm

###########################################################################################################################
# find the best pca params
###########################################################################################################################

pca_param_grid = [1,2,3,4,5]
for params in pca_param_grid:
    pca = PCA(n_components=params)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)

    best_linear_model.fit(X_train_pca, y_train_norm)
    best_nonlinear_model.fit(X_train_pca, y_train_norm)

    y_test_pca_linear_pred = best_linear_model.predict(X_test_pca)
    y_test_pca_nonlinear_pred = best_nonlinear_model.predict(X_test_pca)
    print('n_components: ', params)
    print('Linear SVM test accuracy pca: %.4f' % accuracy_score(y_test_norm, y_test_pca_linear_pred))
    print('Nonlinear SVM test accuracy pca: %.4f' % accuracy_score(y_test_norm, y_test_pca_nonlinear_pred))

###########################################################################################################################
# find the best lda params
###########################################################################################################################
lda_param_grid = [1,2,3]
for params in lda_param_grid:
    lda = LDA(n_components=params)
    X_train_lda = lda.fit_transform(X_train_norm, y_train_norm)
    X_test_lda = lda.transform(X_test_norm)

    best_linear_model.fit(X_train_lda, y_train_norm)
    best_nonlinear_model.fit(X_train_lda, y_train_norm)

    y_test_lda_linear_pred = best_linear_model.predict(X_test_lda)
    y_test_lda_nonlinear_pred = best_nonlinear_model.predict(X_test_lda)
    print('n_components: ', params)
    print('Linear SVM test accuracy lda: %.4f' % accuracy_score(y_test_norm, y_test_lda_linear_pred))
    print('Nonlinear SVM test accuracy lda: %.4f' % accuracy_score(y_test_norm, y_test_lda_nonlinear_pred))

###########################################################################################################################
# find the best k pca params
###########################################################################################################################
kpca_param_grid = [1,2,3]
for params in lda_param_grid:
    kpca = KernelPCA(n_components=params, kernel='rbf', gamma=15)
    X_train_kpca = kpca.fit_transform(X_train_norm, y_train_norm)
    X_test_kpca = kpca.transform(X_test_norm)

    best_linear_model.fit(X_train_kpca, y_train_norm)
    best_nonlinear_model.fit(X_train_kpca, y_train_norm)

    y_test_kpca_linear_pred = best_linear_model.predict(X_test_kpca)
    y_test_kpca_nonlinear_pred = best_nonlinear_model.predict(X_test_kpca)
    print('n_components: ', params)
    print('Linear SVM test accuracy kpca: %.4f' % accuracy_score(y_test_norm, y_test_kpca_linear_pred))
    print('Nonlinear SVM test accuracy kpca: %.4f' % accuracy_score(y_test_norm, y_test_kpca_nonlinear_pred))
