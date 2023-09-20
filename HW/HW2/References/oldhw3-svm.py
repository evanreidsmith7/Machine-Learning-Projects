#************************************************************************************
# Evan Smith
# ML – HW#2
# Filename: hw2-svm.py
# Due: Sept. 14, 2023
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
# Dataset analysis
###########################################################################################################################


# standardize X
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA with two components to the training data
n_components_pca = 2  # Adjust the number of components as needed
pca = PCA(n_components=n_components_pca)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

###########################################################################################################################
# TRAINING AND TESTING THE PCA svc MODEL
###########################################################################################################################

# define the parameters to use in the grid search
param_grid = {
    'C': [0.1, 1, 10, 100, 1000]
}

# use the grid search to find the best parameters
model = SVC(random_state=42)

# create a grid search object
grid_search_object = GridSearchCV(model, param_grid, cv=5)

# fit the grid search object to the training data
grid_search_object.fit(X_train_pca, y_train)

# get the best esiimator and its test accuracy
model1 = grid_search_object.best_estimator_

'''
# get the best parameters
best_parameters = grid_search_object.best_params_

# lets give scv1 the best parameters
model1 = SVC(
    C=best_parameters['C'], kernel='linear', random_state=42)

'''

# train the model
model1.fit(X_train_pca, y_train)
y_train_pred = model1.predict(X_train_pca)
y_pred = model1.predict(X_test_pca)

# Train the best model on the entire training dataset (X_train_scaled, y_train)
best_model = model1  # Change this to your best model

# Make predictions on the entire training and testing datasets
y_train_pred = best_model.predict(X_train_pca)  # Change to X_train_lda if using LDA
y_test_pred = best_model.predict(X_test_pca)    # Change to X_test_lda if using LDA

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print and save the accuracy values to a text file
accuracy_text = f"Training Accuracy: {train_accuracy:.2f}\nTesting Accuracy: {test_accuracy:.2f}\n"
print(accuracy_text)


# Access the principal components (eigenvectors) and their importance
principal_components = pca.components_
explained_variance = pca.explained_variance_ratio_

feature_names = ['cooridor', 'lab139', 'main_lobby', 'sport_hall']

# Print the principal components and their importance
for i, component in enumerate(principal_components):
    print(f"Principal Component {i + 1}:")
    print(component)
    print(f"Explained Variance: {explained_variance[i]:.4f}\n")
    for(feature) in component:
        print(f"Feature {feature}")
    









best_pcComps = f"Best PCA Components: {pca.components_}\n"


# Save the accuracy and feature information to a text file
with open('HW\HW2\SmithEvanEE4331HW2\Part1\part1results.txt', 'w') as file:
    file.write(accuracy_text)
    file.write(best_pcComps)

#best features 1 and 3


'''
# Train the best model on the entire training dataset (X_train_scaled, y_train)
best_model = svm_model_pca  # Change this to your best model

# Make predictions on the entire training and testing datasets
y_train_pred = best_model.predict(X_train_pca)  # Change to X_train_lda if using LDA
y_test_pred = best_model.predict(X_test_pca)    # Change to X_test_lda if using LDA

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print and save the accuracy values to a text file
accuracy_text = f"Training Accuracy: {train_accuracy:.2f}\nTesting Accuracy: {test_accuracy:.2f}\n"
print(accuracy_text)

# Save the best features from PCA or LDA to a text file
if isinstance(best_model, PCA):
    best_features = f"Best PCA Components: {best_model.components_}\n"
elif isinstance(best_model, LDA):
    best_features = f"Best LDA Components: {best_model.scalings_}\n"
else:
    best_features = "No feature information available for this model.\n"

print(best_features)

# Save the accuracy and feature information to a text file
with open('experiment_results.txt', 'w') as file:
    file.write(accuracy_text)
    file.write(best_features)

'''

'''

# PCA: X_pca_2pc
pca_2pc = PCA(n_components=2)
X_pca_2pc = pca_2pc.fit_transform(X_std)

colors = ['r', 'b', 'g', 'y']
markers = ['s', 'x', 'o', '^']
for l, c, m in zip(np.unique(y), colors, markers):
    plt.scatter(X_pca_2pc[y == l, 0], \
                X_pca_2pc[y == l, 1], \
                c=c, label=l, marker=m)
    
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.savefig('HW\HW2\SmithEvanEE4331HW2\Part1\Results\pca_2pc.png')

'''